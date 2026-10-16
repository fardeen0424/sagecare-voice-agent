# Setup log

What was built on the GPU server (Verda 1× H100 80 GB, Helsinki), in order, with the errors hit and
their fixes. Every step now runs from `scripts/bootstrap.sh` + `scripts/deploy.sh`.

## 2026-10-15 – LLM: vLLM + Qwen3.8-27B FP8

- First config (`--max-model-len 32768`, `--gpu-memory-utilization 0.50`) crash-looped 23 times:
  `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (126). Each decode sequence
  requires one Mamba cache block`. Qwen3.8 is hybrid: each running sequence needs one recurrent-state
  block. Fix: `--max-num-seqs 32`, `--max-model-len 16384`, no image/video budget, 60% of the GPU.
- Pinned `vllm/vllm-openai:v0.31.0` by digest (`latest` had resolved to it by luck).
- Measured: TTFT 45 ms; decode 83 tokens/s, 172 tokens/s with MTP (see ADR 013).

## 2026-10-15 – Automation

- `bootstrap.sh` (root, new server): packages, auto-updates, fail2ban, NVIDIA driver ≥ 580 check,
  Docker, NVIDIA Container Toolkit, user, deploy key, `.env`, ufw (`open_ports.sh`), SSH hardening,
  then starts `deploy.sh`.
- `deploy.sh`: pull, fill missing `.env` values, download weights, `docker compose up --wait`,
  `scripts/check.py`, a simulated call.
- Verified by wiping the server (containers, 186 GB of images, weights, checkout) and redeploying:
  `Deploy OK` after 21 minutes with no manual step. Not yet verified: bootstrap's root half on a
  fresh instance.

## 2026-10-15 – STT: Nemotron 3.5 ASR streaming

- No streaming server exists, so `infra/stt/server.py` wraps the Transformers 5.20 streaming API
  in a WebSocket (PCM16 in, partial/final text out).
- Errors: PEP 668 `externally-managed-environment` in the PyTorch image (fix:
  `PIP_BREAK_SYSTEM_PACKAGES=1`); missing `librosa`; `device_map` needs `accelerate` (fix: `.to("cuda")`);
  `num_lookahead_tokens=1 is not supported. Supported values: [3, 0, 6, 13]` (fix: 3, 320 ms chunks).
- Download filter `--include '*.json' model.safetensors` fetched only the weights; fixed with excludes.
- Measured: final transcript 28 ms after end of speech.

## 2026-10-15 – TTS: Fish S2 Pro

- Fish's own server: 688 ms p50 to first audio, one request at a time. Moved to SGLang-Omni 0.1.8.
- `hongccc/sglang-omni:dev` ships SGLang 0.5.19: `ImportError: cannot import name 'SpawnRanks'`.
  Fix: build on `lmsysorg/sglang:v0.5.21`, the base sglang-omni pins.
- `Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.18`: the fraction is
  of memory free at TTS start, not of the GPU. Fix: 0.5, and start TTS after STT so that is stable.
- Xet download of `model-00001` hung silently at 4.09 GB; `download_models.sh` now retries with a
  timeout and resumes.
- Measured: 322 ms to first audio with 7/14-token chunks and no playback stall (ADR 013).

## 2026-10-15 – LiveKit, Caddy, agent

- LiveKit 1.13.9 and Caddy 2.11.6 on host networking; ufw opens 80, 443, 7881/tcp, 50000–60000/udp.
  Let's Encrypt failed (`tls-alpn-01 ... Timeout`) until the ports were opened, then succeeded.
- Agent on LiveKit Agents 1.8.8 with our STT/TTS plugins. Errors: `livekit.agents.tokenize` resolves to
  a function (import from `livekit.agents.tokenize.tokenizer`); greeting via `generate_reply` failed with
  `No user query found in messages` (Qwen template), now `session.say(...)`; the simulated caller was
  unheard until its track was published as `SOURCE_MICROPHONE`.
- Measured perceived latency: 1314 ms p50 at first, 836 ms p50 after tuning (ADR 013).

## 2026-10-16 – Voice, short replies, latency

- Mixed voices in a browser test: no fixed reference voice (ADR 014). The uploaded-voice API took 11 s per
  request (CPU re-encode); fixed by sending precomputed VQ codes.
- One-word replies ("No.", "Hi.") vanished: streaming Nemotron emitted only `<pad>` tokens for them, while
  an offline pass got them right. `server.py` now re-decodes an utterance offline when streaming is empty;
  `check.py` asserts "No." is heard. More trailing silence did not help and cost 75 ms.
- MTP speculative decoding: 83 → 172 tokens/s. TTS chunks 7/14: 322 ms, no stall (`check.py` measures
  stalls now). Perceived latency 1000 → 816 ms p50 (ADR 013).

## 2026-10-16 – First load test (1× H100, `scripts/loadtest.sh`, 4 turns per call)

Bugs found and fixed before the numbers meant anything:
- Parallel utterances crashed STT (`'Nemotron3_5AsrForRNNT' object has no attribute '_symbols_at_frame'`):
  Transformers keeps generate()'s state on the model object. Fixed with a shallow model copy per utterance;
  `check.py` now runs 10 parallel streams.
- `Too many open files` from 25 calls: container default nofile 1024. Raised to 65536 for livekit and agent.
- Agent flushed STT streams that had already ended (`NemotronStream input ended`); now skipped.

Result after the fixes:

| Calls | Answered | Perceived p50 / p95 | Transcript delay p50 | LLM TTFT p50 | TTS first audio p50 |
|---|---|---|---|---|---|
| 1 | 100% | 816 / 2128 ms | 254 ms | 76 ms | 346 ms |
| 10 | 26/40 | 4137 / 7906 ms | 3557 ms | 74 ms | 352 ms |
| 25 | 0/100 | — | — | — | 455 ms (greetings) |
| 50 | 0/200 | — | — | — | 515 ms, p95 1071 ms (greetings) |

Where it breaks: STT, at under 10 calls. One generate() thread per utterance contends for the Python GIL
(STT CPU ~2 cores while the GPU idles ~10%), so transcripts fall seconds behind real time; every thread
also adds CUDA workspace memory, and at 50 calls the GPU filled (STT 3 → 11.9 GB) and STT failed with
`torch.OutOfMemoryError`. LLM latency stayed flat (TTFT ~75 ms) and never queued. TTS degrades under 50
simultaneous greetings (p95 1.1 s) and is the next limit to measure once STT scales.
