#!/usr/bin/env bash
# Downloads model weights into /models on the GPU server, in parallel, with Xet high-performance mode.
set -euo pipefail

download () {   # $1 = Hugging Face repo, $2 = folder under /models
  docker run --rm --env-file /srv/voice/.env \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e HF_XET_NUM_CONCURRENT_RANGE_GETS=32 \
    -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
    -v /models:/models python:3.12-slim \
    sh -c "pip install -q -U huggingface_hub hf_xet && hf download $1 --local-dir /models/$2"
}

download Qwen/Qwen3.8-27B-FP8 qwen3.8-27b-fp8 &
# TTS and STT downloads get added here in Phases 4–5, each ending with &, so all run in parallel
wait
echo "All downloads finished."