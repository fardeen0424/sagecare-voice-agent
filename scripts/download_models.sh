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
    sh -c "pip install -q -U huggingface_hub hf_xet && for try in 1 2 3 4 5; do
             timeout 900 hf download $1 ${3:-} --local-dir /models/$2 && touch /models/$2/.complete && exit 0
             echo \"$2: attempt \$try stalled or failed, resuming\"
           done; exit 1"   # Xet transfers can hang without erroring; a retry resumes where it stopped
}

download Qwen/Qwen3.8-27B-FP8 qwen3.8-27b-fp8 &
download fishaudio/s2-pro fish-s2-pro &
# Transformers weights only: skip the .nemo and .gguf copies of the same model
download nvidia/nemotron-3.5-asr-streaming-0.6b nemotron-3.5-asr "--exclude '*.nemo' --exclude '*.gguf' --exclude '*.png'" &
for job in $(jobs -p); do wait "$job"; done   # a failed download fails the script
echo "All downloads finished."