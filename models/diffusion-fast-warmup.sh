#!/bin/bash
# Warm the DiffusionGemma fast lane: pre-trigger Triton JIT for the common input/output
# shapes so first-hit compiles happen here, not as mid-request stalls in the harness.
# (DG's startup graph-capture doesn't cover the runtime token-shapes -> jit_monitor spikes.)
# Runs detached from the systemd unit (ExecStartPost ... &); polls readiness then sweeps.
URL=http://localhost:8002/v1/chat/completions
for i in $(seq 1 120); do
  curl -s --max-time 3 http://localhost:8002/v1/models 2>/dev/null | grep -q local-fast && break
  sleep 5
done
filler() { yes "The quarterly report noted regional output rose while costs held steady." 2>/dev/null | head -n "$1" | tr '\n' ' '; }
echo "[warmup] sweeping shapes $(date -u +%H:%M:%S)"
# input sizes x output sizes covering short/med/long (lines ~12 tok each)
for inlines in 2 40 350 1400; do
  prompt="Summarize then continue:\n$(filler $inlines)\nWrite a few sentences."
  for mt in 16 128 512; do
    curl -s --max-time 120 "$URL" -H "Content-Type: application/json" \
      -d "$(printf '{"model":"local-fast","messages":[{"role":"user","content":%s}],"max_tokens":%d,"temperature":0.2}' \
            "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$prompt")" "$mt")" \
      >/dev/null 2>&1
  done
done
echo "[warmup] done $(date -u +%H:%M:%S)"
