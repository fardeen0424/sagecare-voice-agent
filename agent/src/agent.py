"""Care-navigation voice agent: a LiveKit Agents worker whose STT, LLM and TTS all run on this GPU server.

Run: python -m livekit.agents start agent.py   (LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET from env)
Every finished turn appends one JSON line to $TURN_LOG with the framework's latency metrics.
"""

import json
import logging
import os
import time

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, tts
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.english import EnglishModel

from clauses import ClauseTokenizer
from fish_tts import FishTTS
from nemotron_stt import NemotronSTT

TURN_LOG = os.environ.get("TURN_LOG", "/logs/turns.jsonl")
log = logging.getLogger("agent")
INSTRUCTIONS = (
    "You are Sage, a friendly care-navigation assistant for a clinic, speaking with a patient on a call. "
    "You help them book appointments. Reply in one or two short spoken sentences: plain words, no lists, "
    "no markdown, no emojis."
)


def setup(proc: JobProcess) -> None:
    # loaded once per worker process, not per call. A short silence closes the speech segment early,
    # which finalizes STT (see _finalize_stt) so preemptive LLM + TTS can start; the turn detector
    # and endpointing below still decide when the caller is really done.
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.2)


server = AgentServer(setup_fnc=setup)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    stt = NemotronSTT()
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        llm=openai.LLM(base_url="http://127.0.0.1:8000/v1", api_key="local", model="qwen3.8-27b"),
        tts=tts.StreamAdapter(tts=FishTTS(), sentence_tokenizer=ClauseTokenizer()),
        turn_handling={
            "turn_detection": EnglishModel(),  # semantic end-of-turn model, local ONNX on CPU
            # Reply 0.3 s after speech the turn detector calls complete; when it calls the speech unfinished,
            # wait at most 2 s (default 3 s, which made pausing callers wait too long).
            "endpointing": {"min_delay": 0.3, "max_delay": 2.0},
            "interruption": {"mode": "vad"},  # barge-in on voice activity; "adaptive" calls LiveKit Cloud
            "preemptive_generation": {"enabled": True, "preemptive_tts": True},
        },
    )

    @session.on("user_state_changed")
    def _finalize_stt(ev) -> None:
        log.info("user %s -> %s", ev.old_state, ev.new_state)
        if ev.old_state == "speaking":
            stt.end_utterance()

    @session.on("user_input_transcribed")
    def _log_transcript(ev) -> None:
        if ev.is_final:
            log.info("user said: %s", ev.transcript)

    @session.on("conversation_item_added")
    def _log_turn(ev) -> None:
        item = ev.item
        if getattr(item, "role", None) not in ("user", "assistant"):
            return
        record = {"ts": time.time(), "room": ctx.room.name, "role": item.role,
                  "text": item.text_content, "metrics": dict(item.metrics)}
        with open(TURN_LOG, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    await session.start(room=ctx.room, agent=Agent(instructions=INSTRUCTIONS))
    # A fixed greeting: no LLM call, and Qwen's chat template rejects a request with no user message.
    await session.say("Hi, this is Sage from the clinic. How can I help you today?")
