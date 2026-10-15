#!/usr/bin/env bash
# Checks each model service gives a correct answer, not just a 200. Run on the GPU server.
set -euo pipefail
LLM=http://127.0.0.1:8000

chat () { curl -fsS "$LLM/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"; }

curl -fsS "$LLM/health" >/dev/null && echo "ok   llm health"

chat '{"model": "qwen3.8-27b", "max_tokens": 60, "messages": [
  {"role": "system", "content": "You are a friendly clinic scheduling assistant. Reply in one or two sentences."},
  {"role": "user", "content": "Hi, I need to see a doctor about my knee."}]}' \
| python3 -c '
import json, sys
m = json.load(sys.stdin)["choices"][0]["message"]
text = m.get("content") or ""
assert text.strip(), "empty reply"
assert "<think>" not in text and not m.get("reasoning"), "thinking is on"
print("ok   llm reply:", text.strip()[:80])'

chat '{"model": "qwen3.8-27b", "max_tokens": 200, "messages": [
  {"role": "system", "content": "You are a clinic scheduling assistant. Use tools to find appointment slots."},
  {"role": "user", "content": "Can you find me an orthopedics slot next Tuesday morning?"}],
  "tools": [{"type": "function", "function": {"name": "find_slots", "description": "Find open appointment slots",
    "parameters": {"type": "object", "required": ["specialty"], "properties": {
      "specialty": {"type": "string"}, "date": {"type": "string"},
      "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "evening"]}}}}}]}' \
| python3 -c '
import json, sys
c = json.load(sys.stdin)["choices"][0]
assert c["finish_reason"] == "tool_calls", c
call = c["message"]["tool_calls"][0]["function"]
assert call["name"] == "find_slots", call
print("ok   llm tool call:", call["name"], json.loads(call["arguments"]))'

curl -fsS http://127.0.0.1:8080/v1/health >/dev/null && echo "ok   tts health"
curl -fsS http://127.0.0.1:8001/health >/dev/null && echo "ok   stt health"
# TTS speaks a sentence, STT must hear it back
out=$(docker compose -f /srv/voice/infra/docker-compose.yml exec -T stt python - 1 < /srv/voice/scripts/speech_latency.py)
echo "ok   speech round trip: ${out%%$'\n'*}"
