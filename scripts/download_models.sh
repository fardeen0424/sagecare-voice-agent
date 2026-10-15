#!/usr/bin/env bash
# Downloads model weights into /models on the GPU server.
# Runs the Hugging Face CLI inside a throwaway container so nothing is installed on the host.
set -euo pipefail

download () {   # $1 = Hugging Face repo, $2 = folder under /models
  docker run --rm --env-file /srv/voice/.env -v /models:/models python:3.12-slim \
    sh -c "pip install -q -U huggingface_hub && hf download $1 --local-dir /models/$2"
}

download Qwen/Qwen3.8-27B-FP8 qwen3.8-27b-fp8
# TTS and STT get added in Phases 4 and 5
