#!/usr/bin/env bash
# CANONICAL DSV4-Flash smart lane — jasl fork build (cutover 2026-08-10, Josh-approved).
# vs stock 0.25.1 lane: +71% C1 decode (DSpark K=5), C8 neutral, 384K ctx (effort:max
# unlocked), honest KV accounting + kv-cache-memory reclaim → 951K-token pool (2.42x@384K).
# Full characterization: evals/results/DSV4-JASL-TEST-2026-08-10.md
# KNOWN DELTAS vs stock: default thinking = adaptive-ON (short); reasoning_hard −0.11
# pending effort-routing mitigation at the gateway (send reasoning_effort xhigh/max for
# deliberation-heavy aliases). Accepted debt: fork supply-chain review outstanding.
# Rebuild env: models/build-vllm-jasl.sh (pin aa0d513027). Rollback: stock unit backup
# ~/dev/.vllm-bump-review/unit-backups/vllm-smart.service.pre-jasl-*.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/data/models-cold/DeepSeek-V4-Flash-0731}
PORT=${PORT:-8041}
MAXLEN=${MAXLEN:-393216}
UTIL=${UTIL:-0.92}
MNBT=${MNBT:-4096}
MAXSEQS=${MAXSEQS:-16}
GENCFG=${GENCFG:-'{"top_p": 0.95}'}
SERVED_NAMES=${SERVED_NAMES:-smart reasoning coder deepseek-v4-flash}
# DSpark block size is 5: K MUST be >=5 (fork validates; smaller = incorrect output).
SPEC=${SPEC-'{"method":"dspark","num_speculative_tokens":5}'}
# 2026-08-11 (later): vision lane retired from this node (MiniCPM moves to ava), embed-B back
# on GPU0 → KV pool raised from the 5.25 GiB vision-era value. Tenants are symmetric again
# (embed-B GPU0 :8004 / embed-A GPU1 :8001, ~1.8 GB each), which matters under TP=2: the pool
# is sharded evenly, so it is bounded by the TIGHTER card and a lopsided tenant strands
# headroom on the other one.
#
# 8 GiB, NOT the 11859195904 from 2026-08-10. That value allocates fine (951,437 tok, 2.42x)
# but then dies at inference: torch.OutOfMemoryError on GPU1 wanting 256 MiB of transient
# activation with ~245 MiB free — the engine wedges while /health still returns 200. Vision
# only ever constrained GPU0; GPU1's tenancy (embed-A 1.81 GiB) is unchanged, so GPU1 could
# never hold an 11.3 GiB pool. `--kv-cache-memory` is NOT profiled — vLLM allocates exactly
# what you ask for and discovers the shortfall at runtime, so leave real headroom here.
# 8 GiB → ~689K tokens (1.75x @ 384K), ~3 GiB runtime cushion on GPU1. Raise only with a
# concurrent load test, not a boot check.
KVMEM=${KVMEM-8589934592}

VENV=$HOME/dev/vllm-jasl
export HF_HOME=/mnt/models/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# sm120 house recipe (venv-own 13.0 toolchain)
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_TRITON_FP8_GEMM=1
export CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export MAX_JOBS=4
export NCCL_P2P_DISABLE=1

SPEC_ARGS=()
[ -n "$SPEC" ] && SPEC_ARGS=(--speculative-config "$SPEC")

# shellcheck disable=SC2086
exec "$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name $SERVED_NAMES \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --disable-custom-all-reduce \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens "$MNBT" \
  --max-num-seqs "$MAXSEQS" \
  --override-generation-config "$GENCFG" \
  --gpu-memory-utilization "$UTIL" \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tokenizer-mode deepseek_v4 \
  "${SPEC_ARGS[@]}" \
  ${KVMEM:+--kv-cache-memory "$KVMEM"} \
  --trust-remote-code
