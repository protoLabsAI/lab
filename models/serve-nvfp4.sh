#!/usr/bin/env bash
# serve-nvfp4.sh — serve a compressed-tensors NVFP4 checkpoint on sm120.
#
# Wraps the two env stacks NVFP4 needs on this box, which are easy to forget:
#  - FlashInfer sm120 JIT recipe (experiments/quantize/FLASHINFER-SM120-RECIPE.md)
#  - Blackwell belt-and-suspenders (sampler off, Triton FP8 GEMM)
#
# Usage: bash serve-nvfp4.sh <model_dir> [port] [gpu] [served_name]
set -euo pipefail

MODEL="${1:?usage: serve-nvfp4.sh <model_dir> [port] [gpu] [served_name]}"
PORT="${2:-8011}"
GPU="${3:-1}"
NAME="${4:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"

CU13=/home/ava/dev/vllm-env/lib/python3.12/site-packages/nvidia/cu13
source /home/ava/dev/vllm-env/bin/activate

CUDA_VISIBLE_DEVICES=$GPU \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_USE_TRITON_FP8_GEMM=1 \
CUDA_HOME=$CU13 \
PATH=$CU13/bin:$PATH \
FLASHINFER_CUDA_ARCH_LIST="12.0f" \
NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" \
MAX_JOBS=4 \
exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$NAME" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --reasoning-parser qwen3
