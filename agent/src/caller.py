"""Simulated callers: perceived latency (end of caller speech -> first agent audio) over N turns.

Runs inside the agent container:  python caller.py [TURNS] [CALLS]
Each call joins a fresh room (the worker dispatches the agent to it), speaks TTS-generated patient
lines in real time and listens for the reply. With CALLS > 1 the calls run in parallel, starting
spread over CALLS * 0.3 s, as a load test. Prints perceived latency, missed replies, failed calls, and
the agent's own e2e_latency and stage timings for the same rooms from $TURN_LOG.
"""

import asyncio
import json
import os
import random
import statistics
import sys
import time
import uuid

import aiohttp
import numpy as np
from livekit import api, rtc

LIVEKIT_URL = "ws://127.0.0.1:7880"
TTS_URL = "http://127.0.0.1:8002/v1/audio/speech"
TURN_LOG = os.environ.get("TURN_LOG", "/logs/turns.jsonl")
CALLER_VOICE = 21  # a different seed from the agent's voice, so caller and agent are easy to tell apart
SR = 48000
FRAME = SR // 100  # 10 ms
LOUD = 500  # int16 peak above which a frame counts as speech
REPLY_TIMEOUT = 15  # s without an agent reply counts as a missed turn
LINES = [
    "Hi, I'd like to book an appointment with a cardiologist.",
    "Next Tuesday morning works best for me.",
    "My date of birth is March third, nineteen eighty.",
    "Yes, please book the earliest one.",
    "No, that's everything. Thank you.",
]


class Ears:
    """Tracks when the agent's audio is audible."""

    def __init__(self) -> None:
        self.last_voice = 0.0
        self.armed_at: float | None = None
        self.reply = asyncio.Event()
        self.reply_at = 0.0
        self.stream: rtc.AudioStream | None = None

    async def listen(self, track: rtc.Track) -> None:
        self.stream = rtc.AudioStream(track, sample_rate=SR, num_channels=1)
        async for ev in self.stream:
            if np.abs(np.frombuffer(ev.frame.data, np.int16)).max(initial=0) > LOUD:
                self.last_voice = time.perf_counter()
                if self.armed_at is not None and not self.reply.is_set():
                    self.reply_at = self.last_voice
                    self.reply.set()

    async def wait_quiet(self, seconds: float = 1.0) -> None:
        while not self.last_voice or time.perf_counter() - self.last_voice < seconds:
            await asyncio.sleep(0.05)


async def tts_frames(http: aiohttp.ClientSession, text: str) -> list[rtc.AudioFrame]:
    body = {"input": text, "seed": CALLER_VOICE, "stream": True, "response_format": "pcm"}
    async with http.post(TTS_URL, json=body) as resp:
        pcm, rate = await resp.read(), int(resp.headers["X-Sample-Rate"])
    resampler = rtc.AudioResampler(input_rate=rate, output_rate=SR, num_channels=1)
    out = resampler.push(rtc.AudioFrame(pcm[: len(pcm) // 2 * 2], rate, 1, len(pcm) // 2)) + resampler.flush()
    data = b"".join(bytes(f.data) for f in out)
    step = FRAME * 2
    return [rtc.AudioFrame(data[i : i + step].ljust(step, b"\0"), SR, 1, FRAME) for i in range(0, len(data), step)]


async def call(lines: list[list[rtc.AudioFrame]], turns: int, delay: float, verbose: bool) -> dict:
    """One simulated call. Returns its room, per-turn latencies (ms) and missed replies."""
    await asyncio.sleep(delay)
    room_name = f"sim-{uuid.uuid4().hex[:8]}"
    token = (api.AccessToken().with_identity("sim-caller")
             .with_grants(api.VideoGrants(room_join=True, room=room_name)).to_jwt())
    room, ears, source = rtc.Room(), Ears(), rtc.AudioSource(SR, 1)
    room.on("track_subscribed", lambda track, *_: asyncio.ensure_future(ears.listen(track))
            if track.kind == rtc.TrackKind.KIND_AUDIO else None)
    result = {"room": room_name, "latencies": [], "missed": 0, "error": None}
    try:
        await room.connect(LIVEKIT_URL, token)
        await room.local_participant.publish_track(  # the agent only listens to microphone tracks
            rtc.LocalAudioTrack.create_audio_track("mic", source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        await asyncio.wait_for(ears.wait_quiet(), timeout=60)  # the agent joins and greets
        for i in range(turns):
            for frame in lines[i % len(lines)]:
                await source.capture_frame(frame)
            await source.wait_for_playout()
            ears.armed_at, ears.reply = time.perf_counter(), asyncio.Event()
            try:
                await asyncio.wait_for(ears.reply.wait(), timeout=REPLY_TIMEOUT)
            except asyncio.TimeoutError:
                result["missed"] += 1
                ears.armed_at = None
                continue
            result["latencies"].append((ears.reply_at - ears.armed_at) * 1000)
            if verbose:
                print(f"turn {i + 1}: {result['latencies'][-1]:5.0f} ms  | caller: {LINES[i % len(LINES)]!r}", flush=True)
            ears.armed_at = None
            await asyncio.wait_for(ears.wait_quiet(), timeout=60)
    except Exception as e:  # a failed call is a result of the load test, not a crash of it
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if ears.stream:
            await ears.stream.aclose()  # before the event loop closes, or the SDK's FFI handles complain
        await room.disconnect()
        await source.aclose()
    return result


def stats(name: str, values: list[float]) -> str:
    if not values:
        return f"{name:<26} n= 0"
    p95 = sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]
    return f"{name:<26} n={len(values):3d}  p50={statistics.median(values):5.0f}ms  p95={p95:5.0f}ms  max={max(values):5.0f}ms"


async def main(turns: int, calls: int) -> None:
    async with aiohttp.ClientSession() as http:
        lines = [await tts_frames(http, line) for line in LINES]
    started = time.perf_counter()
    results = await asyncio.gather(*(call(lines, turns, random.uniform(0, calls * 0.3) if calls > 1 else 0,
                                          verbose=calls == 1) for _ in range(calls)))
    perceived = [ms for r in results for ms in r["latencies"]]
    failed = [r for r in results if r["error"]]
    print(f"\ncalls={calls} turns/call={turns} wall={time.perf_counter() - started:.0f}s  "
          f"answered={len(perceived)}/{calls * turns}  missed={sum(r['missed'] for r in results)}  "
          f"failed calls={len(failed)}")
    for r in failed[:5]:
        print("  failed:", r["room"], r["error"][:150])

    rooms = {r["room"] for r in results}
    with open(TURN_LOG) as f:
        rows = [r for r in map(json.loads, f) if r["room"] in rooms]
    stage = lambda key: [r["metrics"][key] * 1000 for r in rows if key in r["metrics"]]
    print(stats("perceived (caller side)", perceived))
    print(stats("agent e2e_latency", stage("e2e_latency")))
    for key in ("transcription_delay", "end_of_turn_delay", "llm_node_ttft", "llm_node_ttfs", "tts_node_ttfb"):
        print(stats(f"  {key}", stage(key)))


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5, int(sys.argv[2]) if len(sys.argv) > 2 else 1))
