# Model weights (not in git)

Weights are downloaded on the GPU server only, into /models, and mounted read-only into containers.

/models/
  qwen3.8-27b-fp8/     LLM, served by vLLM             (~28 GB)
  fish-s2-pro/         TTS, served by SGLang-Omni      (~9 GB)
  nemotron-3.5-asr/    STT, served by our wrapper      (~2.5 GB)

Locally this folder stays empty. Download commands go in scripts/download_models.sh.