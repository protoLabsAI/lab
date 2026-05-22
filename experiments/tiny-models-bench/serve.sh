#!/usr/bin/env bash
# tiny-models-bench / serve.sh
#
# Launches one tiny-model under test on GPU 1 alongside the running
# `vllm-fast.service` (Gemma 4 26B MoE judge). All configs:
#   - CUDA_VISIBLE_DEVICES=1
#   - --host 0.0.0.0 --port 8003
#   - --served-model-name local-bench
#   - low --gpu-memory-utilization (0.10–0.18 depending on size)
#
# Pre-requisite (when running 9 B FP8 or larger): trim protolabs/fast
# via:
#   sudo sed -i 's/--gpu-memory-utilization 0.72/--gpu-memory-utilization 0.55/;s/--max-model-len 131072/--max-model-len 16384/' /etc/systemd/system/vllm-fast.service
#   sudo systemctl daemon-reload && sudo systemctl restart vllm-fast
# Revert post-bench.
#
# Usage:
#   bash experiments/tiny-models-bench/serve.sh <model_key>
#   bash experiments/tiny-models-bench/serve.sh stop
#
# Model keys map to entries in tiny-models-bench/PLAN.md.

set -euo pipefail

BENCH_PORT=8003
VLLM_BIN="$HOME/dev/vllm-env/bin/vllm"
LOG_DIR="/mnt/scratch/logs"
mkdir -p "$LOG_DIR"

export HF_HOME="/mnt/models/huggingface"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# vLLM doesn't always read ~/.cache/huggingface/token for gated repos at
# serve time. Surface it as HF_TOKEN explicitly.
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

stop_bench() {
    echo "Stopping bench server on :${BENCH_PORT}..."
    # Kill the API server (listening on port) AND any EngineCore worker
    # children — vLLM forks the engine into a separate process that survives
    # a port kill.
    pkill -f -- '--served-model-name local-bench' 2>/dev/null || true
    fuser -k "${BENCH_PORT}/tcp" 2>/dev/null || true
    for i in $(seq 1 30); do
        if ! ss -tlnp | grep -q ":${BENCH_PORT} " \
           && ! pgrep -f -- '--served-model-name local-bench' >/dev/null; then
            echo "Port ${BENCH_PORT} released, bench processes cleared."
            sleep 2  # let VRAM reaper catch up before next launch
            return
        fi
        sleep 1
    done
    echo "Force-killing remaining bench processes..."
    pkill -9 -f -- '--served-model-name local-bench' 2>/dev/null || true
    fuser -k -9 "${BENCH_PORT}/tcp" 2>/dev/null || true
    sleep 3
}

wait_ready() {
    echo "Waiting for model to load on :${BENCH_PORT}..."
    for i in $(seq 1 90); do
        if curl -s --max-time 3 "http://localhost:${BENCH_PORT}/v1/models" | grep -q "local-bench"; then
            echo "Ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Model failed to start within 7.5 minutes."
    tail -50 "${LOG_DIR}/bench-serve.log"
    return 1
}

usage() {
    cat <<'EOF'
Usage: serve.sh <model_key>

Sub-1B:
  smollm2-135m, smollm2-360m, functiongemma-270m, functiongemma-ft,
  llama-3.2-1b, gemma-3-1b, qwen-0.8b-base

1–3B:
  smollm2-1.7b, qwen-2b-base, gemma-3-4b, gemma-3-4b-fp8,
  gemma-4-e2b, gemma-4-e2b-fp8, llama-3.2-3b, phi-4-mini, phi-4-mini-fp8,
  olmoe

3–9B:
  gemma-4-e4b, gemma-4-e4b-fp8, qwen-4b, qwen-4b-fp8, qwen-4b-int4,
  granite-4.1-8b, qwen-9b, qwen-9b-fp8

Separate (not vLLM):
  bitnet  — CPU-only via bitnet.cpp; this script doesn't launch it

Control:
  stop    — kill anything on :8003
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage

case "$1" in
    stop) stop_bench; exit 0 ;;
esac

stop_bench

case "$1" in
    # ─── Sub-1B ─────────────────────────────────────────────────
    smollm2-135m)
        echo "SmolLM2-135M-Instruct (135M, bf16)"
        $VLLM_BIN serve HuggingFaceTB/SmolLM2-135M-Instruct \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 8192 --dtype bfloat16 \
            --gpu-memory-utilization 0.05 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    smollm2-360m)
        echo "SmolLM2-360M-Instruct (360M, bf16)"
        $VLLM_BIN serve HuggingFaceTB/SmolLM2-360M-Instruct \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 8192 --dtype bfloat16 \
            --gpu-memory-utilization 0.10 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    functiongemma-270m)
        echo "FunctionGemma-270M base (270M, bf16, fine-tune target — not a dialogue model)"
        $VLLM_BIN serve google/functiongemma-270m-it \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 8192 --dtype bfloat16 \
            --gpu-memory-utilization 0.10 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    functiongemma-ft)
        echo "FunctionGemma-270M fine-tuned for mobile actions (270M, bf16)"
        $VLLM_BIN serve litert-community/functiongemma-270m-ft-mobile-actions \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 8192 --dtype bfloat16 \
            --gpu-memory-utilization 0.10 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    llama-3.2-1b)
        echo "Llama-3.2-1B-Instruct (1B, bf16)"
        $VLLM_BIN serve meta-llama/Llama-3.2-1B-Instruct \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --gpu-memory-utilization 0.08 \
            --enable-auto-tool-choice --tool-call-parser llama3_json \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    gemma-3-1b)
        echo "Gemma 3 1B-it (1B, bf16)"
        $VLLM_BIN serve google/gemma-3-1b-it \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --gpu-memory-utilization 0.08 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    qwen-0.8b-base)
        echo "Qwen3.5-0.8B (800M, bf16, chat-tuned, 262K ctx, multimodal native)"
        $VLLM_BIN serve Qwen/Qwen3.5-0.8B \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --language-model-only \
            --gpu-memory-utilization 0.10 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;

    # ─── 1–3B ───────────────────────────────────────────────────
    smollm2-1.7b)
        echo "SmolLM2-1.7B-Instruct (1.7B, bf16)"
        $VLLM_BIN serve HuggingFaceTB/SmolLM2-1.7B-Instruct \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 8192 --dtype bfloat16 \
            --gpu-memory-utilization 0.10 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    qwen-2b-base)
        echo "Qwen3.5-2B (2B, bf16, chat-tuned, 262K ctx)"
        $VLLM_BIN serve Qwen/Qwen3.5-2B \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --language-model-only \
            --gpu-memory-utilization 0.14 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    gemma-3-4b)
        echo "Gemma 3 4B-it (4B, bf16, language-model-only)"
        $VLLM_BIN serve google/gemma-3-4b-it \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --gpu-memory-utilization 0.20 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    gemma-3-4b-fp8)
        echo "Gemma 3 4B-it + on-the-fly FP8 (4B, dynamic FP8)"
        $VLLM_BIN serve google/gemma-3-4b-it \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 --quantization fp8 \
            --gpu-memory-utilization 0.16 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    gemma-4-e2b)
        echo "Gemma 4 E2B-it (2.3B effective, bf16)"
        $VLLM_BIN serve google/gemma-4-E2B-it \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --language-model-only \
            --gpu-memory-utilization 0.16 \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    gemma-4-e2b-fp8)
        echo "Gemma 4 E2B-it + on-the-fly FP8"
        $VLLM_BIN serve google/gemma-4-E2B-it \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 --quantization fp8 \
            --language-model-only \
            --gpu-memory-utilization 0.16 \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    llama-3.2-3b)
        echo "Llama-3.2-3B-Instruct (3.2B, bf16)"
        $VLLM_BIN serve meta-llama/Llama-3.2-3B-Instruct \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --gpu-memory-utilization 0.12 \
            --enable-auto-tool-choice --tool-call-parser llama3_json \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    phi-4-mini)
        echo "Phi-4-Mini-Instruct (3.8B, bf16)"
        $VLLM_BIN serve microsoft/Phi-4-mini-instruct \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --gpu-memory-utilization 0.14 \
            --enable-auto-tool-choice --tool-call-parser phi4_mini_json \
            --trust-remote-code \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    phi-4-mini-fp8)
        echo "Phi-4-Mini-Instruct + on-the-fly FP8"
        $VLLM_BIN serve microsoft/Phi-4-mini-instruct \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 --quantization fp8 \
            --gpu-memory-utilization 0.10 \
            --enable-auto-tool-choice --tool-call-parser phi4_mini_json \
            --trust-remote-code \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    olmoe)
        echo "OLMoE-1B-7B-0125-Instruct (1.3B active / 6.9B total MoE, bf16)"
        $VLLM_BIN serve allenai/OLMoE-1B-7B-0125-Instruct \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 4096 --dtype bfloat16 \
            --gpu-memory-utilization 0.16 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;

    # ─── 3–9B ───────────────────────────────────────────────────
    gemma-4-e4b)
        echo "Gemma 4 E4B-it (4.5B effective, bf16)"
        $VLLM_BIN serve google/gemma-4-E4B-it \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --language-model-only \
            --gpu-memory-utilization 0.22 \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    gemma-4-e4b-fp8)
        echo "Gemma 4 E4B-it + on-the-fly FP8"
        $VLLM_BIN serve google/gemma-4-E4B-it \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 --quantization fp8 \
            --language-model-only \
            --gpu-memory-utilization 0.20 \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    qwen-4b)
        echo "Qwen3.5-4B (4B, bf16, chat-tuned, 262K ctx)"
        $VLLM_BIN serve Qwen/Qwen3.5-4B \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --language-model-only \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.18 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    qwen-4b-fp8)
        echo "Qwen3.5-4B + on-the-fly FP8"
        $VLLM_BIN serve Qwen/Qwen3.5-4B \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 --quantization fp8 \
            --language-model-only \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.14 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    qwen-4b-int4)
        echo "Qwen3.5-4B-Instruct-AWQ (4B INT4)"
        $VLLM_BIN serve Qwen/Qwen3.5-4B-Instruct-AWQ \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.10 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    granite-4.1-8b)
        echo "IBM Granite 4.1 8B FP8 (8B native FP8)"
        $VLLM_BIN serve ibm-granite/granite-4.1-8b-fp8 \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 \
            --gpu-memory-utilization 0.18 \
            --enable-auto-tool-choice --tool-call-parser granite \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    qwen-9b)
        echo "Qwen3.5-9B (9B, bf16, 262K ctx)  — NOTE: needs protolabs/fast trim for headroom"
        $VLLM_BIN serve Qwen/Qwen3.5-9B \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 \
            --language-model-only \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.28 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;
    qwen-9b-fp8)
        echo "Qwen3.5-9B + on-the-fly FP8"
        $VLLM_BIN serve Qwen/Qwen3.5-9B \
            --host 0.0.0.0 --port $BENCH_PORT --served-model-name local-bench \
            --max-model-len 16384 --dtype bfloat16 --quantization fp8 \
            --language-model-only \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.18 \
            >> "${LOG_DIR}/bench-serve.log" 2>&1 &
        ;;

    bitnet)
        echo "BitNet b1.58 2B is CPU-only via bitnet.cpp — not launched from this script."
        echo "See microsoft/BitNet GitHub for the inference harness."
        exit 1
        ;;
    *)
        usage
        ;;
esac

wait_ready
