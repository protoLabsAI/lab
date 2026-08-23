#!/usr/bin/env bash
# protoPen security lane — abliterated ThinkingCap-Qwen3.6-27B-heretic NVFP4 on GPU0 :8050.
# Serves `protopen` + `heretic` for the gateway alias protolabs/protopen (homelab-iac#210,
# PR #249). NVFP4 (not bf16) is the deliberate choice for a SHARED card — co-resides with
# daria-lane (:8045) + embed-b (:8004). sm120 recipe env REQUIRED; NVFP4 needs marlin.
#
# Env matches ~/dev/atelier/security-dataset/serve-heretic-smart-nvfp4.sh (vllm-024-test,
# the env this lane was validated on). Overridable knobs below.
set -euo pipefail

VENV="${VENV:-/home/ava/dev/vllm-024-test}"
MODEL="${MODEL:-/mnt/models/quantized/ThinkingCap-Qwen3.6-27B-heretic-NVFP4}"
GPU="${GPU:-0}"
PORT="${PORT:-8050}"
MAXLEN="${MAXLEN:-131072}"
MAXSEQS="${MAXSEQS:-8}"
UTIL="${UTIL:-0.55}"                       # fraction of the WHOLE card; GPU0 is shared
SERVED_NAMES="${SERVED_NAMES:-protopen heretic}"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_TRITON_FP8_GEMM=1
export HF_HOME=/mnt/models/huggingface
export HF_HUB_CACHE=/mnt/models/huggingface/hub
export MAX_JOBS=4

exec "$VENV/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --served-model-name $SERVED_NAMES \
  --trust-remote-code --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
  --gpu-memory-utilization "$UTIL" --kv-cache-dtype fp8 \
  --enable-chunked-prefill --enable-prefix-caching \
  --linear-backend marlin
