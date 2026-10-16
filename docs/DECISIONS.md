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

## 013 – Latency tuning of the voice loop

Status: accepted (2026-10-16)

Context: perceived latency (caller stops speaking → first agent audio) was 1314 ms p50 on the first
working agent. The target is 300–500 ms on US hardware.

Decision, measured with `agent/src/caller.py` (10 simulated turns, server-side, no WAN):

| Change | Why | Effect |
|---|---|---|
| STT finalizes on VAD end of speech; Silero `min_silence_duration` 0.55 → 0.2 s | the transcript waited for 0.55 s of silence | transcript 605 → 254 ms |
| Endpointing `min_delay` 0.3 s, `max_delay` 3 → 2 s | the semantic turn detector still guards early replies | p95 3128 → 2148 ms |
| Preemptive generation with preemptive TTS | LLM + TTS start on the transcript, before the turn is confirmed | overlaps endpointing |
| MTP speculative decoding, 3 draft tokens | the checkpoint's own MTP head; 2.9 accepted per step | decode 83 → 172 tok/s, first clause 179 → 124 ms; KV cache 329K → 178K tokens |
| Clause-level TTS input (`, ; : . ! ?`) | TTS starts on "Sure," instead of the whole sentence | |
| TTS chunks 7 then 14 tokens (default 40/45) | first audio sooner; a smaller first chunk (5/15) stalled ~100 ms at p95 | first audio 349 → 322 ms, no stalls |

Result: 836 ms p50 perceived (agent `e2e_latency` 783 ms): transcript 254 + LLM first clause 124 + TTS
first audio 313 ms. p95 2148 ms comes from turns the turn detector judges unfinished.

Alternatives: shorter VAD silence (0.15 s; more mid-sentence splits for ~50 ms), TTS first chunk of 5
(rejected: audible stalls), filler phrases (deferred: they mask latency rather than remove it).

Consequences: TTS is now the largest stage; reaching 500 ms needs a faster TTS path, not more tuning
here. US callers add ~100–130 ms of network from Helsinki.
