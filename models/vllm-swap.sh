#!/bin/bash
# Usage: ./vllm-swap.sh <model-name>
# Gracefully stops vLLM, waits for port release, starts new model.
#
# === Production (Single GPU) ===
#   qwen-35b           — speed king MoE FP8 (Qwen official), 262K ctx (180 tok/s)
#   qwen-27b-int4      — daily driver, agentic work (53 tok/s)
#   qwen-27b-int4-mtp  — daily driver + MTP, chat/creative (70 tok/s, T08 regresses)
#   qwen-9b-fp8        — on-the-fly FP8 (140 tok/s, +53% vs bf16)
#   qwen-9b-mtp        — fine-tune base + MTP (112 tok/s, tools safe)
#   qwen-4b-fp8        — on-the-fly FP8 edge (140 tok/s)
#   qwen-4b-int4       — edge deploy, fastest absolute (297 tok/s)
#
# === Training / LoRA ===
#   qwen-9b            — fine-tune base, no MTP (92 tok/s)
#   qwen-4b            — LoRA base bf16 (155 tok/s)
#   qwen-2b / qwen-0.8b — training experiments
#
# === Creative / Experimental ===
#   cydonia-24b        — Mistral 24B creative/roleplay (no tool calling)
#   llama-70b          — Llama 70B AWQ creative (38 tok/s)
#
# === TP=2 (Both GPUs) ===
#   qwen-122b-fp8      — quality ceiling FP8 (Qwen official), 112 tok/s
#   qwen-122b-int4     — quality ceiling INT4 (122 tok/s, faster on PCIe)
#   qwen-27b-fp8-tp2   — on-the-fly FP8 TP=2 (70 tok/s, 131K ctx)
#   qwen-35b-tp2-opt   — 250K context, prefix caching (220 tok/s)
#   qwen-27b-int4-tp2  — 256K context
#
# === Legacy / Eval-only (suffix -opt adds P1+P2 flags) ===
#   qwen-27b-int4-opt, qwen-4b-int4-opt, qwen-35b-opt, qwen-122b-opt
set -euo pipefail
PORT=8000
VLLM_BIN="$HOME/dev/vllm-env/bin/vllm"
LOG_DIR="/mnt/scratch/logs"
export HF_HOME="/mnt/models/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
# Shared flags for Qwen3.5 with prefix caching (hybrid Mamba/attention)
# --mamba-block-size requires --enable-prefix-caching
QWEN35_PREFIX_FLAGS="--enable-prefix-caching --mamba-cache-mode align --mamba-block-size 8"
# -O3: graph optimizations + kernel fusions (5-15% throughput gain)
O3="-O3"
usage() {
    echo "Production:"
    echo "  $0 {qwen-35b|qwen-9b-fp8|qwen-4b-fp8|qwen-27b-int4|qwen-27b-int4-mtp|qwen-9b-mtp|qwen-4b-int4|context-1}"
    echo ""
    echo "Training / LoRA:"
    echo "  $0 {qwen-9b|qwen-4b|qwen-2b|qwen-0.8b}"
    echo ""
    echo "Creative:"
    echo "  $0 {cydonia-24b|llama-70b}"
    echo ""
    echo "TP=2 (both GPUs):"
    echo "  $0 {qwen-122b-fp8|qwen-122b-int4|qwen-27b-fp8-tp2|qwen-35b-tp2-opt|qwen-27b-int4-tp2}"
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
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --enable-chunked-prefill \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-27b-int4-mtp)
        echo "Starting Qwen3.5-27B-GPTQ-Int4 + MTP (GPU 0, 128K, 70 tok/s)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --async-scheduling \
            --enable-chunked-prefill \
            --kv-cache-dtype fp8 \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
            $QWEN35_PREFIX_FLAGS \
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
    qwen-35b)
        echo "Starting Qwen3.5-35B-A3B MoE FP8 official (GPU 0, 262K, 180 tok/s)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-35B-A3B-FP8 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 262144 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-35b-bf16)
        echo "Starting Qwen3.5-35B-A3B MoE bf16 (GPU 0, 64K, 171 tok/s)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-35B-A3B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 65536 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.85 \
            --enable-chunked-prefill \
            $QWEN35_PREFIX_FLAGS \
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
    llama-8b)
        echo "Starting Llama-3.1-8B-AWQ (GPU 0, 128K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --enable-auto-tool-choice --tool-call-parser llama3_json \
            --gpu-memory-utilization 0.90 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    hermes-70b)
        echo "Starting Hermes-3-Llama-3.1-70B-FP8 (GPU 0, 128K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve NousResearch/Hermes-3-Llama-3.1-70B-FP8 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --enable-auto-tool-choice --tool-call-parser hermes \
            --gpu-memory-utilization 0.90 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-9b-fp8)
        echo "Starting Qwen3.5-9B + on-the-fly FP8 (GPU 0, 262K, 140 tok/s)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-9B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.85 \
            --language-model-only \
            --enable-chunked-prefill \
            --quantization fp8 \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-4b-fp8)
        echo "Starting Qwen3.5-4B + on-the-fly FP8 (GPU 0, 262K, 140 tok/s)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-4B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.85 \
            --language-model-only \
            --enable-chunked-prefill \
            --quantization fp8 \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-9b)
        echo "Starting Qwen3.5-9B bf16 (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-9B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-9b-mtp)
        echo "Starting Qwen3.5-9B + MTP (GPU 0, 262K, 112 tok/s)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.5-9B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --async-scheduling \
            --enable-chunked-prefill \
            --kv-cache-dtype fp8 \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-4b-int4)
        echo "Starting Qwen3.5-4B-AWQ-4bit (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve cyankiwi/Qwen3.5-4B-AWQ-4bit \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --max-model-len 262144 \
            --served-model-name local \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            $QWEN35_PREFIX_FLAGS \
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
    # ─── Context-1 (Chroma retrieval agent) ────────────────────────────
    context-1)
        echo "Starting Context-1 20B MoE retrieval agent (GPU 0, 32K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve chromadb/context-1 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.90 \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    context-1-gpu1)
        echo "Starting Context-1 20B MoE retrieval agent (GPU 1, port 8001, 32K)..."
        CUDA_VISIBLE_DEVICES=1 $VLLM_BIN serve chromadb/context-1 \
            --host 0.0.0.0 --port 8001 \
            --served-model-name context-1 \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.90 \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap-ctx1.log" 2>&1 &
        ;;
    # ─── Qwen3-Coder configs ─────────────────────────────────────────
    qwen3-coder-30b)
        echo "Starting Qwen3-Coder-30B-A3B FP8 (GPU 0, 256K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --enable-auto-tool-choice --tool-call-parser qwen3_coder \
            --gpu-memory-utilization 0.90 \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen3-coder-next)
        echo "Starting Qwen3-Coder-Next FP8 (TP=2, 131K)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_CUMEM_ENABLE=0 \
        $VLLM_BIN serve Qwen/Qwen3-Coder-Next-FP8 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --tensor-parallel-size 2 \
            --max-model-len 131072 \
            --enable-auto-tool-choice --tool-call-parser qwen3_coder \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --async-scheduling \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    # ─── Experimental MTP configs (no tool calling) ─────────────────
    # These are for creative/roleplay/summarization benchmarking only.
    # MTP hurts MoE models — these exist for testing, not production.
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
            --kv-cache-dtype fp8 \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    # ─── Dual GPU configs (TP=2) ───────────────────────────────────
    # NCCL_P2P_DISABLE=1 required for stable CUDA graphs on Blackwell PCIe
    # (ACS on PCIe bridges corrupts P2P during graph replay)
    qwen-27b-fp8-tp2)
        echo "Starting Qwen3.5-27B FP8 official (TP=2, 131K, 70 tok/s, vision)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_ALGO=Ring NCCL_PROTO=Simple \
        NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=8 \
        $VLLM_BIN serve Qwen/Qwen3.5-27B-FP8 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --tensor-parallel-size 2 \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --enable-chunked-prefill \
            --disable-custom-all-reduce \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-122b-fp8)
        echo "Starting Qwen3.5-122B-A10B FP8 official (TP=2, 65K, 112 tok/s, vision)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_ALGO=Ring NCCL_PROTO=Simple \
        NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=8 \
        $VLLM_BIN serve Qwen/Qwen3.5-122B-A10B-FP8 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --tensor-parallel-size 2 \
            --max-model-len 65536 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --enable-chunked-prefill \
            --disable-custom-all-reduce \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-122b-int4)
        echo "Starting Qwen3.5-122B-A10B-GPTQ-Int4 (TP=2, 128K, 122 tok/s)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_CUMEM_ENABLE=0 \
        $VLLM_BIN serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 131072 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --async-scheduling \
            --enable-chunked-prefill \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-35b-tp2)
        echo "Starting Qwen3.5-35B-A3B MoE (TP=2, 250K, 205 tok/s)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_CUMEM_ENABLE=0 \
        $VLLM_BIN serve Qwen/Qwen3.5-35B-A3B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 253952 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --async-scheduling \
            --enable-chunked-prefill \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-27b-int4-tp2)
        echo "Starting Qwen3.5-27B-GPTQ-Int4 (TP=2, 256K)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_CUMEM_ENABLE=0 \
        $VLLM_BIN serve Qwen/Qwen3.5-27B-GPTQ-Int4 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size 2 \
            --served-model-name local \
            --max-model-len 262144 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --disable-custom-all-reduce \
            --async-scheduling \
            --enable-chunked-prefill \
            $QWEN35_PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    *)
        usage
        ;;
esac
wait_ready
