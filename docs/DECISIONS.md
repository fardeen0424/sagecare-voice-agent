# Decisions

## 012 – Pin every image and package of the model stack

Status: accepted (2026-10-15)

Context: `vllm/vllm-openai:latest` resolved to a different build on each pull, and the SGLang-Omni image
clones its `main` branch on every start. A new server (the planned 2× H200) must come up with the exact
stack that was measured here, without debugging.

Decision: pin images by tag plus digest and Python packages by exact version. Verified working
together on 1× H100 80 GB (driver 580.178.04), deployed from an empty server by `deploy.sh`.

| Part | Pinned to | Where |
|---|---|---|
| LLM server | `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd…` (CUDA 13.0, sm_90 + sm_120 kernels) | `infra/docker-compose.yml` |
| TTS base | `lmsysorg/sglang:v0.5.21@sha256:b1259f3e…` (CUDA 13) | `infra/tts/Dockerfile` |
| TTS serving | `sglang-omni==0.1.8`, `descript-audiotools==0.7.2`, `descript-audio-codec==1.0.0`, protobuf 6.x | `infra/tts/Dockerfile` |
| STT base | `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime@sha256:3c0b174a…` | `infra/stt/Dockerfile` |
| STT packages | `transformers==5.20.0`, `fastapi==0.143.2`, `uvicorn[standard]==0.54.0`, `librosa==1.0.0` | `infra/stt/requirements.txt` |
| Downloader | `python:3.12-slim@sha256:a6e34c59…`, `huggingface_hub==2.2.0`, `hf_xet==1.7.0` | `scripts/download_models.sh` |
| GPU smoke test | `nvidia/cuda:12.8.0-base-ubuntu24.04@sha256:e778509d…` | `scripts/bootstrap.sh` |
| Weights | Hugging Face revisions: Qwen3.8-27B-FP8 `017b9c7a`, s2-pro `1de9996b`, nemotron-3.5-asr `ea30d66d` | `scripts/download_models.sh` |
| Host | Ubuntu 24.04, NVIDIA driver ≥ 580 (CUDA 13.0 images), Docker 29, NVIDIA Container Toolkit 1.20 | `scripts/bootstrap.sh` |

Why: tags move, digests do not. All three model images share CUDA 13, so one host driver rule (≥ 580)
covers them, and `bootstrap.sh` refuses older drivers instead of failing later inside a container.

Alternatives: track `latest` (rejected: the first deploy hit a different vLLM than the one tested);
SGLang-Omni's prebuilt `hongccc/sglang-omni:dev` image (rejected: its SGLang 0.5.19 breaks
sglang-omni 0.1.8 with `ImportError: SpawnRanks`); Fish's own API server (rejected: 688 ms p50 to first
audio and one request at a time, against 332 ms on SGLang-Omni).

Consequences: upgrades are deliberate: change the pin, deploy, and `deploy.sh` must end in `Deploy OK`
(its `scripts/check.py` round trip) before the new pin is kept. A model repo changing upstream no
longer changes what a new server downloads.
