"""End-to-end check and latency benchmark for the LLM, TTS and STT services.

Runs inside the stt container, which has numpy and websockets and reaches the others by service name:
    docker compose -f infra/docker-compose.yml exec -T stt python - [ROUNDS] < scripts/check.py

Fails (exit 1) unless the LLM replies without thinking, the LLM calls a tool, and STT hears back
what TTS said. Over ROUNDS sentences it measures LLM time to first token, TTS time to first audio,
and STT time from end of speech to the final transcript.
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

LLM = "http://llm:8000/v1/chat/completions"
TTS = "http://tts:8002/v1/audio/speech"
STT = "ws://localhost:8001/v1/stream?language=en-US"
SYSTEM = "You are a friendly clinic scheduling assistant. Reply in one or two sentences."
SENTENCES = [
    "Sure, I can help you book that appointment.",
    "I found an opening next Tuesday at ten thirty in the morning.",
    "Could you confirm your date of birth for me, please?",
    "Thanks, I've sent a six digit code to your phone.",
    "You're all set with Doctor Patel on Tuesday.",
]
FIND_SLOTS = {
    "type": "function",
    "function": {
        "name": "find_slots",
        "description": "Find open appointment slots",
        "parameters": {
            "type": "object",
            "required": ["specialty"],
            "properties": {
                "specialty": {"type": "string"},
                "date": {"type": "string"},
                "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "evening"]},
            },
        },
    },
}


def post(url: str, body: dict):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60)


def chat(system: str, user: str, **extra) -> dict:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    body = {"model": "qwen3.8-27b", "max_tokens": 200, "temperature": 0, "messages": messages, **extra}
    with post(LLM, body) as resp:
        return json.load(resp)["choices"][0]


def llm_ttft() -> float:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hi, I'd like to book an appointment."}]
    start = time.perf_counter()
    with post(LLM, {"model": "qwen3.8-27b", "max_tokens": 40, "stream": True, "messages": messages}) as resp:
        for line in resp:
            if line.startswith(b"data: {") and json.loads(line[6:])["choices"][0]["delta"].get("content"):
                return (time.perf_counter() - start) * 1000
    raise RuntimeError("LLM stream ended without content")


def tts(text: str) -> tuple[float, np.ndarray, int]:
    """Streams raw PCM16; returns time to first audio (ms), the audio and its sample rate."""
    start = time.perf_counter()
    with post(TTS, {"input": text, "voice": "default", "stream": True, "response_format": "pcm"}) as resp:
        first = resp.read1()
        ttfa = (time.perf_counter() - start) * 1000
        pcm = np.frombuffer(first + resp.read(), np.int16)
        return ttfa, pcm, int(resp.headers["X-Sample-Rate"])


def to_16k(pcm: np.ndarray, rate: int) -> bytes:
    t = np.arange(0, len(pcm) / rate, 1 / 16000)
    return np.interp(t, np.arange(len(pcm)) / rate, pcm).astype(np.int16).tobytes()


async def stt(audio: bytes) -> tuple[float, str]:
    """Sends audio at real-time pace in 80 ms frames; returns end of speech -> final (ms) and the text."""
    frame = 16000 * 2 * 80 // 1000
    async with websockets.connect(STT) as ws:
        for i in range(0, len(audio), frame):
            await ws.send(audio[i : i + frame])
            await asyncio.sleep(0.08)
        end = time.perf_counter()
        await ws.send(json.dumps({"type": "end"}))
        while (msg := json.loads(await ws.recv()))["type"] != "final":
            pass
        return (time.perf_counter() - end) * 1000, msg["text"]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def summary(name: str, values: list[float]) -> str:
    p95 = sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]
    return f"{name:<24} p50={statistics.median(values):4.0f}ms  p95={p95:4.0f}ms  min={min(values):4.0f}ms"


def main(rounds: int) -> None:
    reply = chat(SYSTEM, "Hi, I need to see a doctor about my knee.")["message"]
    assert reply["content"].strip() and not reply.get("reasoning"), f"bad LLM reply: {reply}"
    print("ok   llm reply:", reply["content"].strip()[:80])

    choice = chat("You are a clinic scheduling assistant. Use tools to find appointment slots.",
                  "Can you find me an orthopedics slot next Tuesday morning?", tools=[FIND_SLOTS])
    assert choice["finish_reason"] == "tool_calls", f"no tool call: {choice}"
    call = choice["message"]["tool_calls"][0]["function"]
    print("ok   llm tool call:", call["name"], call["arguments"])

    tts("Warm up.")
    ttft, ttfa, final = [], [], []
    for i in range(rounds):
        said = SENTENCES[i % len(SENTENCES)]
        ttft.append(llm_ttft())
        ms, pcm, rate = tts(said)
        ttfa.append(ms)
        final_ms, heard = asyncio.run(stt(to_16k(pcm, rate)))
        final.append(final_ms)
        recall = sum(w in set(words(heard)) for w in words(said)) / len(words(said))
        assert recall >= 0.7, f"STT heard {heard!r} for {said!r}"
        print(f"ok   round trip: tts {ms:4.0f} ms, stt {final_ms:3.0f} ms | {said!r} -> {heard!r}")

    print("\n" + "\n".join([
        summary("LLM first token", ttft),
        summary("TTS first audio", ttfa),
        summary("STT end -> final", final),
    ]))


main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
