"""Simulated caller: measures perceived latency, end of caller speech -> first agent audio, over N turns.

Runs inside the agent container:  python caller.py [TURNS]
Joins a fresh room (the worker dispatches the agent to it), speaks TTS-generated patient lines in real
time, and listens for the reply. Prints per-turn latency, p50/p95, and the agent's own e2e_latency
for the same turns from $TURN_LOG.
"""

import array
import asyncio
import json
import os
import statistics
import sys
import time
import uuid

import aiohttp
from livekit import api, rtc

LIVEKIT_URL = "ws://127.0.0.1:7880"
TTS_URL = "http://127.0.0.1:8002/v1/audio/speech"
TURN_LOG = os.environ.get("TURN_LOG", "/logs/turns.jsonl")
SR = 48000
FRAME = SR // 100  # 10 ms
LOUD = 500  # int16 peak above which a frame counts as speech
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
            if max(map(abs, array.array("h", bytes(ev.frame.data))), default=0) > LOUD:
                self.last_voice = time.perf_counter()
                if self.armed_at is not None and not self.reply.is_set():
                    self.reply_at = self.last_voice
                    self.reply.set()

    async def wait_quiet(self, seconds: float = 1.0) -> None:
        while not self.last_voice or time.perf_counter() - self.last_voice < seconds:
            await asyncio.sleep(0.05)


async def tts_frames(http: aiohttp.ClientSession, text: str) -> list[rtc.AudioFrame]:
    body = {"input": text, "voice": "default", "stream": True, "response_format": "pcm"}
    async with http.post(TTS_URL, json=body) as resp:
        pcm, rate = await resp.read(), int(resp.headers["X-Sample-Rate"])
    resampler = rtc.AudioResampler(input_rate=rate, output_rate=SR, num_channels=1)
    out = resampler.push(rtc.AudioFrame(pcm[: len(pcm) // 2 * 2], rate, 1, len(pcm) // 2)) + resampler.flush()
    data = b"".join(bytes(f.data) for f in out)
    step = FRAME * 2
    return [rtc.AudioFrame(data[i : i + step].ljust(step, b"\0"), SR, 1, FRAME) for i in range(0, len(data), step)]


async def main(turns: int) -> None:
    room_name = f"sim-{uuid.uuid4().hex[:8]}"
    token = (api.AccessToken().with_identity("sim-caller")
             .with_grants(api.VideoGrants(room_join=True, room=room_name)).to_jwt())
    room, ears = rtc.Room(), Ears()
    room.on("track_subscribed", lambda track, *_: asyncio.ensure_future(ears.listen(track))
            if track.kind == rtc.TrackKind.KIND_AUDIO else None)
    await room.connect(LIVEKIT_URL, token)
    source = rtc.AudioSource(SR, 1)
    await room.local_participant.publish_track(  # the agent only listens to microphone tracks
        rtc.LocalAudioTrack.create_audio_track("mic", source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    latencies = []
    async with aiohttp.ClientSession() as http:
        lines = [await tts_frames(http, LINES[i % len(LINES)]) for i in range(turns)]
        await ears.wait_quiet()  # the agent's greeting
        for i, frames in enumerate(lines):
            for frame in frames:
                await source.capture_frame(frame)
            await source.wait_for_playout()
            ears.armed_at, ears.reply = time.perf_counter(), asyncio.Event()
            await asyncio.wait_for(ears.reply.wait(), timeout=15)
            latencies.append((ears.reply_at - ears.armed_at) * 1000)
            print(f"turn {i + 1}: {latencies[-1]:5.0f} ms  | caller: {LINES[i % len(LINES)]!r}", flush=True)
            ears.armed_at = None
            await ears.wait_quiet()
    if ears.stream:
        await ears.stream.aclose()  # before the event loop closes, or the SDK's FFI handles complain
    await room.disconnect()
    await source.aclose()

    with open(TURN_LOG) as f:
        e2e = [r["metrics"]["e2e_latency"] * 1000 for r in map(json.loads, f)
               if r["room"] == room_name and r["role"] == "assistant" and "e2e_latency" in r["metrics"]]
    for name, values in [("perceived (caller side)", latencies), ("agent e2e_latency", e2e)]:
        if values:
            p95 = sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]
            print(f"{name:<24} n={len(values):2d}  p50={statistics.median(values):4.0f}ms  p95={p95:4.0f}ms")


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
