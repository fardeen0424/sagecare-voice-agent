"""LiveKit STT plugin for our Nemotron streaming service (protocol in infra/stt/server.py).

The service finalizes an utterance only when told to, so the agent calls end_utterance() when VAD
reports that the caller stopped speaking; interim text streams in the meantime. A partial that ends a
sentence (Nemotron writes punctuation) is sent as a preflight transcript: an eager end of turn that lets
preemptive generation start before VAD has heard enough silence.
"""

import asyncio
import json
import weakref

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr


class NemotronSTT(stt.STT):
    def __init__(self, url: str = "ws://127.0.0.1:8001/v1/stream", language: str = "en-US") -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True, offline_recognize=False)
        )
        self._url, self._language = url, language
        self._streams: weakref.WeakSet[NemotronStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return "nemotron-3.5-asr-streaming-0.6b"

    @property
    def provider(self) -> str:
        return "self-hosted"

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options) -> stt.SpeechEvent:
        raise NotImplementedError("NemotronSTT is streaming-only")

    def stream(
        self, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "NemotronStream":
        stream = NemotronStream(stt=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def end_utterance(self) -> None:
        for stream in self._streams:
            stream.flush()


class NemotronStream(stt.RecognizeStream):
    def __init__(self, *, stt: NemotronSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)  # frames arrive resampled
        self._nemotron = stt

    async def _run(self) -> None:
        url = f"{self._nemotron._url}?language={self._nemotron._language}"
        async with utils.http_context.http_session().ws_connect(url) as ws:

            async def send() -> None:
                async for item in self._input_ch:
                    if isinstance(item, self._FlushSentinel):
                        await ws.send_str(json.dumps({"type": "end"}))
                    else:
                        await ws.send_bytes(bytes(item.data))
                await ws.close()

            async def receive() -> None:
                async for msg in ws:
                    data = json.loads(msg.data)
                    text = data["text"]
                    if data["type"] == "final":
                        if not text:
                            continue  # "end" with no speech in it
                        kind = stt.SpeechEventType.FINAL_TRANSCRIPT
                    elif text.endswith((".", "?", "!")):
                        kind = stt.SpeechEventType.PREFLIGHT_TRANSCRIPT
                    else:
                        kind = stt.SpeechEventType.INTERIM_TRANSCRIPT
                    alt = stt.SpeechData(language=self._nemotron._language, text=text)
                    self._event_ch.send_nowait(stt.SpeechEvent(type=kind, alternatives=[alt]))

            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
