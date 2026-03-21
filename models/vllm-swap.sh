#!/bin/bash
# Usage: ./vllm-swap.sh <model-name>
# Gracefully stops vLLM, waits for port release, starts new model.
#
# === Single GPU (GPU 0) ===
#   qwen-27b-int4    — Qwen3.5-27B-GPTQ-Int4, 128K ctx (daily driver)
#   qwen-27b-int4-160k — Qwen3.5-27B-GPTQ-Int4, 160K ctx (max single GPU)
#   qwen-122b-int4-1gpu — Qwen3.5-122B-A10B-GPTQ-Int4, 64K ctx
#   qwen-27b         — Qwen3.5-27B bf16, 128K ctx (baseline)
#   qwen-35b         — Qwen3.5-35B-A3B MoE bf16, 64K ctx
#   llama-70b        — Llama-3.3-70B-Instruct-AWQ, 128K ctx (creative/fast)
#   qwen-27b-opus-v2 — Qwen3.5-27B-Opus-v2 bf16, 128K ctx
#   omnicoder        — OmniCoder-9B, 262K ctx (fine-tune base)
#
# === Dual GPU (TP=2, needs UPS) ===
#   qwen-122b        — Qwen3.5-122B-A10B-FP8, 64K ctx
#   qwen-122b-128k   — Qwen3.5-122B-A10B-FP8, 128K ctx
#   qwen-122b-int4   — Qwen3.5-122B-A10B-GPTQ-Int4, 128K ctx
#   qwen-35b-tp2      — Qwen3.5-35B-A3B MoE, 128K ctx
#   qwen-27b-int4-tp2 — Qwen3.5-27B-GPTQ-Int4, 256K ctx

set -euo pipefail

PORT=8000
VLLM_BIN="$HOME/dev/vllm-env/bin/vllm"
LOG_DIR="/mnt/scratch/logs"

export HF_HOME="/mnt/models/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

usage() {
    echo "Single GPU:"
    echo "  $0 {qwen-27b-int4|qwen-27b-int4-160k|qwen-122b-int4-1gpu|qwen-27b|qwen-35b|llama-70b|omnicoder}"
    echo ""
    echo "Dual GPU (TP=2):"
    echo "  $0 {qwen-122b|qwen-122b-128k|qwen-122b-int4|qwen-27b-int4-tp2}"
    exit 1
}

stop_vllm() {
    echo "Stopping vLLM..."
    pkill -f "vllm serve" 2>/dev/null || true
    for i in $(seq 1 30); do
        if ! ss -tlnp | grep -q ":${PORT} "; then
            echo "Port ${PORT} released."
            return
        fi
        sleep 1
    done
    echo "Port still held, force killing..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
}

wait_ready() {
    echo "Waiting for model to load..."
    for i in $(seq 1 60); do
        if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" | grep -q "model"; then
            echo "Ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Model failed to start within 5 minutes."
    tail -20 "${LOG_DIR}/vllm-swap.log"
    return 1
}

[[ $# -lt 1 ]] && usage

stop_vllm

case "$1" in
    # ─── Single GPU configs ────────────────────────────────────────

    qwen-27b-int4)
        echo "Starting Qwen3.5-27B-GPTQ-Int4 (GPU 0, 128K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-27b-int4-160k)
        echo "Starting Qwen3.5-27B-GPTQ-Int4 (GPU 0, 160K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 163840 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-122b-int4-1gpu)
        echo "Starting Qwen3.5-122B-A10B-GPTQ-Int4 (GPU 0, 64K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 65536 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --enforce-eager \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-27b)
        echo "Starting Qwen3.5-27B bf16 (GPU 0, 128K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-27B \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.85 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-35b)
        echo "Starting Qwen3.5-35B-A3B MoE (GPU 0, 64K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-35B-A3B \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 65536 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.85 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    llama-70b)
        echo "Starting Llama-3.3-70B-AWQ (GPU 0, 128K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve casperhansen/llama-3.3-70b-instruct-awq \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --enable-auto-tool-choice --tool-call-parser llama3_json \
            --gpu-memory-utilization 0.90 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-27b-opus-v2)
        echo "Starting Qwen3.5-27B-Opus-v2 bf16 (GPU 0, 128K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --tokenizer Qwen/Qwen3.5-27B \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.85 \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    omnicoder)
        echo "Starting OmniCoder-9B (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Tesslate/OmniCoder-9B \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --enable-auto-tool-choice --tool-call-parser hermes \
            --gpu-memory-utilization 0.90 \
            --trust-remote-code \
            --language-model-only \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;

    # ─── Dual GPU configs (TP=2) ───────────────────────────────────

    qwen-122b)
        echo "Starting Qwen3.5-122B-A10B-FP8 (TP=2, 64K)..."
        $VLLM_BIN serve Qwen/Qwen3.5-122B-A10B-FP8 \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 65536 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --enforce-eager \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-122b-128k)
        echo "Starting Qwen3.5-122B-A10B-FP8 (TP=2, 128K)..."
        $VLLM_BIN serve Qwen/Qwen3.5-122B-A10B-FP8 \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.95 \
            --disable-custom-all-reduce \
            --enforce-eager \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-122b-int4)
        echo "Starting Qwen3.5-122B-A10B-GPTQ-Int4 (TP=2, 128K)..."
        $VLLM_BIN serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --enforce-eager \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-35b-tp2)
        echo "Starting Qwen3.5-35B-A3B MoE (TP=2, 250K)..."
        $VLLM_BIN serve Qwen/Qwen3.5-35B-A3B \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 253952 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-27b-int4-tp2)
        echo "Starting Qwen3.5-27B-GPTQ-Int4 (TP=2, 256K)..."
        $VLLM_BIN serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 262144 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --enforce-eager \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    *)
        usage
        ;;
esac

wait_ready
