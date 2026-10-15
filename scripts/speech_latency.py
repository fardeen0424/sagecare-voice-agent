"""Measures TTS and STT latency. Runs inside the stt container (it has numpy and websockets):
    docker compose -f infra/docker-compose.yml exec -T stt python - [N] < scripts/speech_latency.py

TTS: time to first audio byte (TTFA) and real-time factor for short agent-style sentences.
STT: streams that TTS audio in real time (80 ms frames), then measures "end" -> final transcript.
Exits 1 if STT does not hear back most of the words TTS said, so it doubles as a round-trip smoke test.
"""

import asyncio
import json
import re
import statistics
import sys
import time
import urllib.request

import numpy as np
import websockets

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TTS = "http://tts:8080/v1/tts"
STT = "ws://localhost:8001/v1/stream?language=en-US"
SENTENCES = [
    "Sure, I can help you book that appointment.",
    "I found an opening next Tuesday at ten thirty in the morning.",
    "Could you confirm your date of birth for me, please?",
    "Thanks, I've sent a six digit code to your phone.",
    "You're all set with Doctor Patel on Tuesday.",
]


def p(values, q):  # nearest-rank percentile
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def tts(text: str) -> tuple[float, float, np.ndarray, int]:
    body = json.dumps({"text": text, "format": "wav", "streaming": True, "seed": 7}).encode()
    req = urllib.request.Request(TTS, body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        header = resp.read(44)  # streaming WAV: header first, then raw PCM16 chunks
        first = resp.read(1024)
        ttfa = time.perf_counter() - start
        rest = resp.read()
    total = time.perf_counter() - start
    rate = int.from_bytes(header[24:28], "little")  # sample rate field of the WAV header
    pcm = np.frombuffer(first + rest, np.int16)
    return ttfa * 1000, total / (len(pcm) / rate), pcm, rate


def recall(said: str, heard: str) -> float:  # share of spoken words that came back
    words = lambda t: re.findall(r"[a-z]+", t.lower())
    heard_words = set(words(heard))
    return sum(w in heard_words for w in words(said)) / len(words(said))


def to_16k(pcm: np.ndarray, rate: int) -> bytes:
    x = pcm.astype(np.float32)
    t = np.arange(0, len(x) / rate, 1 / 16000)
    return np.interp(t, np.arange(len(x)) / rate, x).astype(np.int16).tobytes()


async def stt(audio: bytes) -> tuple[float, str]:
    frame = 16000 * 2 * 80 // 1000  # 80 ms of PCM16
    async with websockets.connect(STT) as ws:
        for i in range(0, len(audio), frame):
            await ws.send(audio[i : i + frame])
            await asyncio.sleep(0.08)  # real time, like a caller
        end = time.perf_counter()
        await ws.send(json.dumps({"type": "end"}))
        while (msg := json.loads(await ws.recv()))["type"] != "final":
            pass
        return (time.perf_counter() - end) * 1000, msg["text"]


tts("Warm up.")
ttfa, rtf, finals, misheard = [], [], [], 0
for i in range(N):
    text = SENTENCES[i % len(SENTENCES)]
    ms, factor, pcm, rate = tts(text)
    ttfa.append(ms)
    rtf.append(factor)
    final_ms, heard = asyncio.run(stt(to_16k(pcm, rate)))
    finals.append(final_ms)
    misheard += recall(text, heard) < 0.7
    print(f"tts {ms:4.0f} ms  rtf {factor:.2f}  stt final {final_ms:4.0f} ms  | {text!r} -> {heard!r}")

print(f"\nTTS time to first audio  p50={statistics.median(ttfa):.0f}ms  p95={p(ttfa, 0.95):.0f}ms  "
      f"min={min(ttfa):.0f}ms  RTF p50={statistics.median(rtf):.2f}")
print(f"STT end -> final         p50={statistics.median(finals):.0f}ms  p95={p(finals, 0.95):.0f}ms  "
      f"min={min(finals):.0f}ms")

if misheard:
    sys.exit(f"{misheard} of {N} sentences were not heard back correctly")
