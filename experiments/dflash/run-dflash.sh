#!/bin/bash
# Serve a target model + (optional) dFlash/MTP draft for A/B testing.
# Prod-faithful by default (mirrors vllm.service flags) so GPU1:8003 numbers transfer to prod.
# Runs on a spare GPU/port so it doesn't touch the prod lanes (GPU0 :8000 smart, GPU1 :8002 fast).
#
# Usage:
#   MODE=dflash   bash run-dflash.sh    # target + dFlash draft
#   MODE=mtp      bash run-dflash.sh    # target + native MTP (baseline)
#   MODE=baseline bash run-dflash.sh    # target alone, no spec decode
#   O3=0 MODE=dflash bash run-dflash.sh # disable -O3 (default O3=1, matches prod)
#
# Env: TARGET DRAFT GPU PORT NUM_SPEC MAXLEN UTIL SEQS NAME MODE O3
set -euo pipefail

TARGET="${TARGET:-Qwen/Qwen3.6-27B-FP8}"
DRAFT="${DRAFT:-z-lab/Qwen3.6-27B-DFlash}"
GPU="${GPU:-1}"
PORT="${PORT:-8003}"
NUM_SPEC="${NUM_SPEC:-10}"
MAXLEN="${MAXLEN:-230400}"      # prod: 225K
UTIL="${UTIL:-0.90}"            # prod: 0.90
SEQS="${SEQS:-512}"            # prod: 512
NAME="${NAME:-dflash-test}"
MODE="${MODE:-dflash}"
O3="${O3:-1}"                  # prod: -O3 on

source ~/dev/vllm-env/bin/activate
export HF_HOME=/mnt/models/huggingface
export VLLM_USE_FLASHINFER_SAMPLER=0          # required on sm120
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"

SPEC_ARGS=()
case "$MODE" in
  dflash)   SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NUM_SPEC}") ;;
  mtp)      SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}') ;;
  baseline) SPEC_ARGS=() ;;
  *) echo "unknown MODE=$MODE (want dflash|mtp|baseline)"; exit 2 ;;
esac

O3_ARG=(); [ "$O3" = "1" ] && O3_ARG=(-O3)

echo ">> MODE=$MODE O3=$O3 TARGET=$TARGET DRAFT=$DRAFT GPU=$GPU PORT=$PORT NUM_SPEC=$NUM_SPEC MAXLEN=$MAXLEN SEQS=$SEQS"
set -x
exec vllm serve "$TARGET" \
  "${O3_ARG[@]}" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$NAME" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$SEQS" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --gpu-memory-utilization "$UTIL" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 8 \
  "${SPEC_ARGS[@]}"
