#!/usr/bin/env bash
# Validate an MTP checkpoint: serve OFF-GATEWAY, measure acceptance + decode tok/s,
# and point at the lossless eval. Never touches production (:8000/:8003).
#
# Prereq: free GPU1 first  ->  sudo systemctl stop vllm-replica-b
#         restore after     ->  sudo systemctl reset-failed vllm-replica-b && sudo systemctl start vllm-replica-b
#
# Usage:
#   bash validate.sh <checkpoint-dir> [port] [num_spec_tokens]
#   bash validate.sh /mnt/data/checkpoints/ornith-9b-mtp 8005 1
set -euo pipefail

CKPT="${1:?checkpoint dir}"; PORT="${2:-8005}"; NSPEC="${3:-1}"
SERVED="mtp-validate"
source ~/dev/vllm-env/bin/activate

# Blackwell sm120 required env (see CLAUDE.md / vLLM 0.22.1 migration)
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=1   # GPU1 (after stopping replica-b)

echo "[validate] serving $CKPT on :$PORT with mtp spec-decode (num_speculative_tokens=$NSPEC)"
vllm serve "$CKPT" \
  --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED" \
  --max-model-len 65536 --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}" \
  > /tmp/mtp-validate.log 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true' EXIT

echo "[validate] waiting for health..."
for i in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  sleep 5
  kill -0 $VPID 2>/dev/null || { echo "server died -- see /tmp/mtp-validate.log"; tail -30 /tmp/mtp-validate.log; exit 1; }
done

# --- speed + acceptance probe ---
echo "[validate] running speed probe..."
for n in 1 2 3 4 5; do
  curl -sf "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\":\"$SERVED\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed 400-word explanation of how spec decoding works.\"}],
    \"max_tokens\":600,\"temperature\":0.7}" >/dev/null
done

echo "[validate] spec-decode metrics:"
curl -sf "http://localhost:$PORT/metrics" | grep -E "spec_decode|num_accepted|num_draft|acceptance" || true

python - "$PORT" <<'PY'
import sys, urllib.request, re
m = urllib.request.urlopen(f"http://localhost:{sys.argv[1]}/metrics").read().decode()
def total(name):
    v = 0.0
    for line in m.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            v += float(line.split()[-1])
    return v
acc = total("vllm:spec_decode_num_accepted_tokens_total")
draft = total("vllm:spec_decode_num_draft_tokens_total")
if draft:
    print(f"\n[validate] acceptance rate = {acc/draft:.3f}  ({acc:.0f}/{draft:.0f} draft tokens)")
else:
    print("\n[validate] no draft-token metric found -- check vLLM metric names in /metrics output above")
PY

cat <<EOF

[validate] decode tok/s: run  bash ~/dev/lab/models/speed-test.sh 5   against :$PORT
[validate] LOSSLESS eval (expect ~identical to plain-9B challenger row):
  cd ~/dev/lab/evals
  ./run.sh --local custom --suite coding --model $SERVED --gateway-url http://localhost:$PORT/v1 --thinking --trials 1
  ./run.sh --local function-call --model $SERVED --gateway-url http://localhost:$PORT/v1 --all-suites --trials 1

server still running (pid $VPID) on :$PORT. Ctrl-C to stop.
EOF
wait $VPID
