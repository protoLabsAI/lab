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
#   qwen-9b          — Qwen3.5-9B bf16, 262K ctx (fine-tune base)
#   cydonia-24b      — Cydonia-24B-v4.3 (Mistral base), 32K ctx (creative/roleplay)
#   mistral-119b     — Mistral-Small-4-119B AWQ-4bit, 32K ctx (MoE)
#
# === Dual GPU (TP=2, needs UPS) ===
#   qwen-122b        — Qwen3.5-122B-A10B-FP8, 64K ctx
#   qwen-122b-128k   — Qwen3.5-122B-A10B-FP8, 128K ctx
#   qwen-122b-int4   — Qwen3.5-122B-A10B-GPTQ-Int4, 128K ctx
#   qwen-35b-tp2      — Qwen3.5-35B-A3B MoE, 128K ctx
#   qwen-27b-int4-tp2 — Qwen3.5-27B-GPTQ-Int4, 256K ctx
#   minimax-reap     — MiniMax-M2.5-REAP-139B-A10B-AWQ, 64K ctx

set -euo pipefail

PORT=8000
VLLM_BIN="$HOME/dev/vllm-env/bin/vllm"
LOG_DIR="/mnt/scratch/logs"

export HF_HOME="/mnt/models/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

usage() {
    echo "Single GPU:"
    echo "  $0 {qwen-27b-int4|qwen-27b-int4-160k|qwen-122b-int4-1gpu|qwen-27b|qwen-35b|llama-70b|qwen-9b}"
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
    qwen-27b-int4-opt)
        echo "Starting Qwen3.5-27B-GPTQ-Int4 OPTIMIZED (GPU 0, 128K, FP8 KV)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --async-scheduling \
            --enable-prefix-caching \
            --performance-mode interactivity \
            --kv-cache-dtype fp8 \
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
    qwen-35b-opt)
        echo "Starting Qwen3.5-35B-A3B MoE OPTIMIZED (GPU 0, 64K, FP8 KV + MoE FP8)..."
        CUDA_VISIBLE_DEVICES=0 \
        VLLM_USE_FLASHINFER_MOE_FP8=1 \
        VLLM_FLASHINFER_MOE_BACKEND=latency \
        $VLLM_BIN serve Qwen/Qwen3.5-35B-A3B \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 65536 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.85 \
            --async-scheduling \
            --enable-prefix-caching \
            --performance-mode interactivity \
            --kv-cache-dtype fp8 \
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
    qwen-9b)
        echo "Starting Qwen3.5-9B (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-9B \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-4b-int4)
        echo "Starting Qwen3.5-4B-AWQ-4bit (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve cyankiwi/Qwen3.5-4B-AWQ-4bit \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-4b-int4-opt)
        echo "Starting Qwen3.5-4B-AWQ-4bit OPTIMIZED (GPU 0, 262K, FP8 KV)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve cyankiwi/Qwen3.5-4B-AWQ-4bit \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --async-scheduling \
            --enable-prefix-caching \
            --performance-mode interactivity \
            --kv-cache-dtype fp8 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-4b)
        echo "Starting Qwen3.5-4B (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-4B \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-2b)
        echo "Starting Qwen3.5-2B (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-2B \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-0.8b)
        echo "Starting Qwen3.5-0.8B (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-0.8B \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;

    cydonia-24b)
        echo "Starting Cydonia-24B-v4.3 (GPU 0, 32K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve TheDrummer/Cydonia-24B-v4.3 \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 32768 \
            --served-model-name local \
            --enable-auto-tool-choice --tool-call-parser mistral \
            --gpu-memory-utilization 0.90 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    mistral-119b)
        echo "Starting Mistral-Small-4-119B AWQ-4bit (GPU 0, 32K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 32768 \
            --served-model-name local \
            --enable-auto-tool-choice --tool-call-parser mistral \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;

    # ─── MTP configs (speculative decoding, NO tool calling) ────────
    # MTP breaks tool calls (issue #36872) — use for creative/roleplay/summarization only

    qwen-35b-mtp)
        echo "Starting Qwen3.5-35B-A3B MoE + MTP (GPU 0, 64K, NO TOOLS)..."
        CUDA_VISIBLE_DEVICES=0 \
        VLLM_USE_FLASHINFER_MOE_FP8=1 \
        VLLM_FLASHINFER_MOE_BACKEND=latency \
        $VLLM_BIN serve Qwen/Qwen3.5-35B-A3B \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 65536 \
            --gpu-memory-utilization 0.85 \
            --async-scheduling \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    cydonia-24b-mtp)
        echo "Starting Cydonia-24B + MTP (GPU 0, 32K, NO TOOLS)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve TheDrummer/Cydonia-24B-v4.3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.90 \
            --async-scheduling \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
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
    qwen-122b-opt)
        echo "Starting Qwen3.5-122B-A10B-FP8 OPTIMIZED (TP=2, 64K, NCCL tuned)..."
        # Note: VLLM_USE_FLASHINFER_MOE_FP8 breaks on 122B FP8 (unsupported quant scheme)
        NCCL_ALGO=Ring \
        NCCL_PROTO=Simple \
        NCCL_MIN_NCHANNELS=4 \
        NCCL_MAX_NCHANNELS=8 \
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
            --async-scheduling \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
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
    qwen-35b-tp2-opt)
        echo "Starting Qwen3.5-35B-A3B MoE OPTIMIZED (TP=2, 250K, NCCL tuned)..."
        NCCL_ALGO=Ring \
        NCCL_PROTO=Simple \
        NCCL_MIN_NCHANNELS=4 \
        NCCL_MAX_NCHANNELS=8 \
        VLLM_USE_FLASHINFER_MOE_FP8=1 \
        VLLM_FLASHINFER_MOE_BACKEND=latency \
        $VLLM_BIN serve Qwen/Qwen3.5-35B-A3B \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 253952 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --async-scheduling \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
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
    minimax-reap)
        echo "Starting MiniMax-M2.5-REAP-139B-A10B-AWQ (TP=2, 64K)..."
        $VLLM_BIN serve cassettesgoboom/MiniMax-M2.5-REAP-139B-A10B-AWQ-w4g128-int4all \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 65536 \
            --enable-auto-tool-choice --tool-call-parser minimax_m2 \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --enforce-eager \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    *)
        usage
        ;;
esac

wait_ready
