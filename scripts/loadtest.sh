#!/usr/bin/env bash
# Load test: CALLS simulated calls of TURNS turns each (agent/src/caller.py), while sampling GPU, CPU and
# LLM queue depth every 2 s into logs/loadtest-*.tsv. Run on the GPU server:  ./scripts/loadtest.sh 50 [4]
# Note: the simulated callers run inside the agent container, so agent CPU includes theirs.
set -euo pipefail
cd /srv/voice
calls=${1:?usage: loadtest.sh CALLS [TURNS]}
turns=${2:-4}
compose="docker compose -f infra/docker-compose.yml"
log="logs/loadtest-$(date +%Y%m%d-%H%M%S)-${calls}calls.tsv"

sample () {
  printf 'time\tgpu_util_%%\tgpu_mem_mb\tload_1m\tllm_running\tllm_waiting\tagent_cpu_%%\ttts_cpu_%%\tstt_cpu_%%\n'
  while true; do
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr , '\t')
    llm=$(curl -s localhost:8000/metrics | awk '/^vllm:num_requests_running/ {r=$2} /^vllm:num_requests_waiting/ {w=$2} END {print r "\t" w}')
    cpu=$(docker stats --no-stream --format '{{.CPUPerc}}' agent tts stt | tr -d % | paste -sd '\t')
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%T)" "$gpu" "$(cut -d' ' -f1 /proc/loadavg)" "$llm" "$cpu"
    sleep 2
  done
}
sample > "$log" &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

$compose exec -T agent python caller.py "$turns" "$calls"

kill "$sampler"
trap - EXIT
python3 - "$log" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
print(f"\nsystem during the test ({len(rows)} samples, {sys.argv[1]}):")
for col in list(rows[0])[1:]:
    values = [float(r[col]) for r in rows if r[col]]
    print(f"  {col:<14} mean={sum(values) / len(values):8.1f}  max={max(values):8.1f}")
EOF
