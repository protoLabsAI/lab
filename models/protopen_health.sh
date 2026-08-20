#!/usr/bin/env bash
# ExecStartPost gate for vllm-protopen.service. /health returns 200 on a wedged engine
# (2026-08-11 KV lesson), so gate on a REAL completion. Probes THINKING-OFF so a reasoning
# model can't burn the token budget before emitting content (the [[feedback_no_thinking_off]]
# / thinking-eats-budget trap). Retries within a ~20min deadline to ride out NVFP4 cold JIT.
set -uo pipefail
PORT="${PORT:-8050}"
URL="http://localhost:${PORT}/v1"
DEADLINE=$(( $(date +%s) + 1200 ))

until curl -s -m 3 "$URL/models" >/dev/null 2>&1; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "protopen: /models never came up"; exit 1; }
  sleep 10
done

# retry the completion until a non-empty answer or deadline (cold first-inference can be slow)
while :; do
  RESP=$(curl -s -m 120 "$URL/chat/completions" -H 'Content-Type: application/json' -d '{
    "model":"protopen",
    "messages":[{"role":"user","content":"Reply with the single word: ok"}],
    "max_tokens":64,"temperature":0,
    "chat_template_kwargs":{"enable_thinking":false}}' 2>/dev/null)
  if echo "$RESP" | python3 -c '
import sys,json
d=json.load(sys.stdin); m=d["choices"][0]["message"]
txt=(m.get("content") or m.get("reasoning_content") or "").strip()
assert txt, "empty"
print("protopen: healthy ->", txt[:60])
' 2>/dev/null; then exit 0; fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "protopen: no valid completion before deadline"; exit 1; }
  echo "protopen: warming (retry in 15s)..."; sleep 15
done
