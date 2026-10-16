"""LiveKit TTS plugin for Fish S2 Pro on SGLang-Omni (/v1/audio/speech, streamed raw PCM16).

One request per text chunk; the session's StreamAdapter feeds it clause by clause, and audio
is pushed to the room as soon as the first bytes arrive.
"""

import json

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, tts, utils

SAMPLE_RATE = 44100  # S2 Pro's DAC codec
VOICE_FILE = "/voices/sage.json"  # reference voice as VQ codes (infra/tts/encode_voice.py)


class FishTTS(tts.TTS):
    def __init__(self, url: str = "http://127.0.0.1:8002/v1/audio/speech", voice_file: str = VOICE_FILE) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=SAMPLE_RATE, num_channels=1)
        # Every request carries the same reference: without one S2 Pro samples a new speaker each time.
        with open(voice_file) as f:
            self._url, self._references = url, [json.load(f)]

    @property
    def model(self) -> str:
        return "fish-s2-pro"

    @property
    def provider(self) -> str:
        return "self-hosted"

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "FishStream":
        return FishStream(tts=self, input_text=text, conn_options=conn_options)


class FishStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        body = {"input": self._input_text, "references": self._tts._references, "stream": True, "response_format": "pcm"}
        async with utils.http_context.http_session().post(self._tts._url, json=body) as resp:
            resp.raise_for_status()
            rate = int(resp.headers["X-Sample-Rate"])
            assert rate == SAMPLE_RATE, f"TTS sent {rate} Hz, expected {SAMPLE_RATE}"
            output_emitter.initialize(
                request_id=utils.shortuuid(), sample_rate=rate, num_channels=1, mime_type="audio/pcm"
            )
            async for chunk in resp.content.iter_any():
                output_emitter.push(chunk)
            output_emitter.flush()
