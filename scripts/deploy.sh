#!/usr/bin/env bash
# Deploys the latest main on the GPU server: pull, download missing weights, start services,
# wait until healthy, smoke test. Safe to re-run. Run as the app user from /srv/voice.
set -euo pipefail
cd /srv/voice

# Pull first, then restart this script so the freshly pulled version is the one that runs.
if [ "${1:-}" != "--pulled" ]; then
  git pull --ff-only
  exec "$0" --pulled
fi

[ -f .env ] || { echo ".env missing: copy .env.example to .env and fill it in."; exit 1; }
mkdir -p .cache/vllm   # vLLM compile cache; Docker would create it root-owned otherwise

./scripts/download_models.sh

compose="docker compose -f infra/docker-compose.yml"
$compose up -d --remove-orphans

echo "Waiting for services to become healthy (first start on a new GPU takes several minutes)..."
for _ in $(seq 1 120); do   # 120 x 10 s = 20 min
  states=$(docker inspect -f '{{.Name}} {{.State.Health.Status}} {{.RestartCount}}' $($compose ps -aq))
  if echo "$states" | awk '$3 > 0 || $2 == "unhealthy" {bad=1} END {exit !bad}'; then
    echo "$states"; $compose logs --tail 60; echo "A service crashed or is unhealthy."; exit 1
  fi
  if ! echo "$states" | grep -qv ' healthy '; then
    echo "$states"; break
  fi
  sleep 10
done
echo "$states" | grep -qv ' healthy ' && { echo "$states"; echo "Timed out waiting for health."; exit 1; }

./scripts/smoke_test.sh
echo "Deploy OK: $(git log --oneline -1)"
