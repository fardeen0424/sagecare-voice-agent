"""Encodes the reference clip into S2 Pro's VQ codes once, with S2 Pro's own encoder.

Requests then carry the codes, so the server skips encoding the clip: sglang-omni 0.1.8 does that on
CPU for every request (~11 s). Re-run only when sage.wav changes, in the tts container:
    docker compose -f infra/docker-compose.yml exec -T tts /opt/sglang/bin/python3 - \
        < infra/tts/encode_voice.py > infra/tts/voices/sage.json
"""

import json
from pathlib import Path

from sglang_omni.models.fishaudio_s2_pro.stages import FishReferenceEncodeHook, load_codec

CHECKPOINT = "/models/fish-s2-pro"
hook = FishReferenceEncodeHook(codec=load_codec(CHECKPOINT, "cpu"), checkpoint_id=CHECKPOINT)
codes = hook.encode_one(hook.normalize_input({"audio_path": "/voices/sage.wav"}))
print(json.dumps({"text": Path("/voices/sage.txt").read_text().strip(), "vq_codes": codes.tolist()}))
