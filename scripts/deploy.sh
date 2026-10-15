#!/usr/bin/env bash
# Deploys the latest main on the GPU server: pull, download missing weights, build and start the
# services, wait until all are healthy, then run the end-to-end check. Safe to re-run.
set -euo pipefail
cd /srv/voice

# Pull first, then re-exec so the freshly pulled version of this script is the one that runs.
if [ "${1:-}" != "--pulled" ]; then
  git pull --ff-only
  exec "$0" --pulled
fi

[ -f .env ] || { echo ".env missing: copy .env.example to .env and fill it in."; exit 1; }
mkdir -p .cache/vllm .cache/tts # compile caches; Docker would create them root-owned
./scripts/download_models.sh

compose="docker compose -f infra/docker-compose.yml"
echo "Starting services (first start on a new GPU compiles kernels for several minutes)..."
$compose up -d --build --remove-orphans --wait --wait-timeout 1800 \
  || { $compose ps -a; $compose logs --tail 60; echo "A service did not become healthy."; exit 1; }

$compose exec -T stt python - 1 < scripts/check.py
echo "Deploy OK: $(git log --oneline -1)"
