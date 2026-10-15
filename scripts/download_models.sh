#!/usr/bin/env bash
# Downloads model weights into /models on the GPU server, in parallel, with Xet high-performance mode.
set -euo pipefail

download () {   # $1 = Hugging Face repo, $2 = folder under /models, $3 = optional extra hf args
  [ -f "/models/$2/.complete" ] && { echo "$2 already downloaded"; return; }
  docker run --rm --env-file /srv/voice/.env \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e HF_XET_NUM_CONCURRENT_RANGE_GETS=32 \
    -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
    -v /models:/models python:3.12-slim \
    sh -c "pip install -q -U huggingface_hub hf_xet && hf download $1 ${3:-} --local-dir /models/$2 && touch /models/$2/.complete"
}

download Qwen/Qwen3.8-27B-FP8 qwen3.8-27b-fp8 &
download fishaudio/s2-pro fish-s2-pro &
# Transformers weights only; skips the .nemo and .gguf copies of the same model
download nvidia/nemotron-3.5-asr-streaming-0.6b nemotron-3.5-asr "--include '*.json' model.safetensors" &
wait
echo "All downloads finished."