"""Streaming speech-to-text over WebSocket, wrapping Nemotron 3.5 ASR (cache-aware FastConformer-RNNT).

Protocol, one WebSocket per call at /v1/stream?language=en-US ("auto" detects the language):
  client -> binary   16 kHz mono PCM16 little-endian audio, any frame size
  client -> text     {"type": "end"}  the utterance is over; flush it
  server -> text     {"type": "partial", "text": ...}  full text so far, as words are recognized
  server -> text     {"type": "final", "text": ...}    once per "end"; the socket stays open for the next one
"""

import asyncio
import json
import logging
import os
import queue
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from transformers import AutoModelForRNNT, AutoProcessor, TextStreamer

MODEL_PATH = os.environ.get("STT_MODEL_PATH", "/models/nemotron-3.5-asr")
# Right context in 80 ms frames (0, 3, 6 or 13 in Transformers): lower = faster finals, higher = fewer word errors.
LOOKAHEAD = int(os.environ.get("STT_LOOKAHEAD", "3"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stt")

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForRNNT.from_pretrained(MODEL_PATH).to("cuda").eval()
processor.set_num_lookahead_tokens(LOOKAHEAD)
SR = processor.feature_extractor.sampling_rate
HOP = processor.feature_extractor.hop_length
N_FFT = processor.feature_extractor.n_fft


class Partials(TextStreamer):
    """Collects the text model.generate() streams out and reports it word by word."""

    def __init__(self, emit):
        super().__init__(processor.tokenizer, skip_special_tokens=True)
        self.emit, self.text = emit, ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text.strip():
            self.text += text
            self.emit({"type": "partial", "text": self.text.strip()})


def transcribe(audio_q: queue.Queue, language: str, emit) -> str:
    """Runs in a worker thread: feeds audio from audio_q (None = end) to the model, returns the final text."""
    buf, ended = np.zeros(0, np.float32), False

    def have(n: int) -> bool:  # block until buf holds n samples, or the utterance ended first
        nonlocal buf, ended
        while len(buf) < n and not ended:
            item = audio_q.get()
            if item is None:  # one chunk of silence gives the last words their right context
                ended, item = True, np.zeros(processor.num_samples_per_audio_chunk, np.float32)
            buf = np.concatenate([buf, item])
        return len(buf) >= n

    def features(samples: np.ndarray, first: bool):
        return processor(samples, sampling_rate=SR, is_streaming=True, is_first_audio_chunk=first,
                         language=language, return_tensors="pt").to(model.device, dtype=model.dtype)

    n_first = processor.num_samples_first_audio_chunk
    if not have(n_first):
        buf = np.pad(buf, (0, n_first - len(buf)))
    first = features(buf[:n_first], first=True)

    def chunks():  # same chunking as the model card's streaming example, but audio arrives live
        yield first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel, size = processor.num_mel_frames_first_audio_chunk, processor.num_samples_per_audio_chunk
        while True:
            start = mel * HOP - N_FFT // 2
            if not have(start + size):
                return
            yield features(buf[start : start + size], first=False).input_features
            mel += processor.num_mel_frames_per_audio_chunk

    streamer = Partials(emit)
    # shortcut: one generate() per stream, no batching across calls; fine for tens of calls,
    # move to NeMo batched cache-aware streaming if one GPU must serve hundreds.
    with torch.inference_mode():
        model.generate(**{**first, "input_features": chunks()}, streamer=streamer)
    return streamer.text.strip()


def warm_up():
    q = queue.Queue()
    for _ in range(10):
        q.put(np.zeros(SR // 10, np.float32))
    q.put(None)
    start = time.perf_counter()
    transcribe(q, "en-US", lambda _: None)
    log.info("warm-up done in %.0f ms, chunk %d ms", (time.perf_counter() - start) * 1000,
             processor.streaming_latency_ms)


warm_up()
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/v1/stream")
async def stream(ws: WebSocket, language: str = "en-US"):
    await ws.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()
    emit = lambda msg: loop.call_soon_threadsafe(outbox.put_nowait, msg)  # called from the worker thread

    async def send_all():
        while True:
            await ws.send_json(await outbox.get())

    sender = asyncio.create_task(send_all())
    audio_q, job = None, None
    try:
        while (msg := await ws.receive())["type"] != "websocket.disconnect":
            if msg.get("bytes"):
                if audio_q is None:
                    audio_q = queue.Queue()
                    job = asyncio.create_task(asyncio.to_thread(transcribe, audio_q, language, emit))
                audio_q.put(np.frombuffer(msg["bytes"], np.int16).astype(np.float32) / 32768)
            elif msg.get("text") and json.loads(msg["text"]).get("type") == "end":
                text = ""
                if audio_q is not None:
                    audio_q.put(None)
                    text = await job
                    audio_q, job = None, None
                outbox.put_nowait({"type": "final", "text": text})
    finally:
        if audio_q is not None:
            audio_q.put(None)  # let the worker thread finish
        sender.cancel()
