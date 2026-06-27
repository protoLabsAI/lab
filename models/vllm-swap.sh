#!/bin/bash
# Usage: ./vllm-swap.sh <model-name>
# Gracefully stops vLLM, waits for port release, starts new model.
#
# === Qwen 3.6 Dual-GPU Production Setup ===
#   dual                              — start both models at once (recommended)
#   GPU 0 (:8000) — qwen36-27b-fp8   — 27B FP8, thinking+planning, 225K (no MTP)
#   GPU 1 (:8002) — gemma4-moe-fast  — Gemma 4 26B-A4B MoE FP8, instruct, 131K
#
# === Qwen 3.6 (Single GPU) ===
#   qwen-36b-fp8       — Qwen3.6-35B-A3B MoE + on-the-fly FP8 (single GPU, 262K)
#   qwen-36b-fast      — Qwen3.6-35B-A3B MoE FP8, NO thinking, fast execution (GPU 1, :8002, 131K)
#   gemma4-moe-fast    — Gemma 4 26B-A4B MoE FP8, instruct mode (GPU 1, :8002, 131K, local-fast)
#   qwen-36b-bf16      — Qwen3.6-35B-A3B MoE bf16 (single GPU, 64K)
#   qwen36-27b-fp8     — Qwen3.6-27B FP8 official (single GPU, 256K)
#   qwen36-27b-fp8-mtp — Qwen3.6-27B FP8 + MTP spec decode (single GPU, 225K) — now the always-on default (in vllm.service)
#
# === Qwen 3.6 (TP=2) ===
#   qwen-36b-tp2       — Qwen3.6-35B-A3B MoE bf16 (TP=2, 250K)
#   qwen36-27b-fp8-tp2 — Qwen3.6-27B FP8 official (TP=2, 131K)
#
# === Qwen3-Coder (parked — coding-agent work out of scope, see brand pivot) ===
#   qwen3-coder-30b    — Qwen3-Coder-30B-A3B FP8 (single GPU, 131K)
#
# === Gemma 4 (text-only via --language-model-only) ===
#   gemma4-moe         — 26B-A4B MoE bf16, 128K (single GPU)
#   gemma4-moe-fp8     — 26B-A4B MoE on-the-fly FP8 (175 tok/s, 128K)
#   gemma4-e4b         — E4B edge bf16, 128K (single GPU)
#   gemma4-e4b-fp8     — E4B edge on-the-fly FP8 (128K)
#   gemma4-moe-tp2     — Gemma 4 26B-A4B MoE FP8 TP=2 (131K, both GPUs)
#
# === Ornith-1.0 (DeepReinforce agentic-coding eval candidates, Qwen3.5 base) ===
#   ornith-9b          — Ornith-1.0-9B dense bf16 (single GPU, 131K)
#   ornith-35b         — Ornith-1.0-35B MoE bf16 + on-the-fly FP8 (single GPU, 262K, no -O3)
#                        (published -35B-FP8 repo reported broken — quantize bf16 ourselves)
#
# === Creative / Experimental ===
#   cydonia-24b        — Mistral 24B creative/roleplay (no tool calling)
#   llama-70b          — Llama 70B AWQ creative (38 tok/s)
#   context-1          — Chroma Context-1 20B MoE retrieval agent (32K)
set -euo pipefail
PORT=8000
VOICE_PORT=8002
VLLM_BIN="$HOME/dev/vllm-env/bin/vllm"
LOG_DIR="/mnt/scratch/logs"
NONTHINKING_TEMPLATE="$HOME/dev/lab/models/qwen3.5-tool-calling-nothink.jinja"
export HF_HOME="/mnt/models/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
# sm120/Blackwell + vLLM 0.22.1: flashinfer sampler JIT rejects sm120 — force native sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
# FlashInfer-sm120 build recipe (lets cutlass fused-MoE / JIT kernels compile on Blackwell).
# See experiments/quantize/FLASHINFER-SM120-RECIPE.md. lib64/.so symlinks are already persistent.
_CU13="$HOME/dev/vllm-env/lib/python3.12/site-packages/nvidia/cu13"
if [ -d "$_CU13" ]; then
    export CUDA_HOME="$_CU13"
    export PATH="$_CU13/bin:$PATH"               # CUDA-13.3 nvcc, not system 12.8
    export FLASHINFER_CUDA_ARCH_LIST="12.0f"     # emit compute_120f past broken auto-detect
    export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"  # nvcc13.3 vs cudart13.0 skew
    export MAX_JOBS=4                             # cutlass compiles OOM at full parallelism
fi
# Shared flags for hybrid Mamba/attention models (Qwen 3.6) with prefix caching
# --mamba-block-size requires --enable-prefix-caching
PREFIX_FLAGS="--enable-prefix-caching --mamba-cache-mode align --mamba-block-size 8"
# -O3: graph optimizations + kernel fusions (5-15% throughput gain)
O3="-O3"
usage() {
    echo "Qwen 3.6 (single GPU):"
    echo "  $0 {qwen-36b-fp8|qwen-36b-bf16|qwen-36b-voice|qwen36-27b-fp8|qwen36-27b-fp8-mtp}"
    echo ""
    echo "Qwen3-Coder (parked, kept for reference):"
    echo "  $0 {qwen3-coder-30b}"
    echo ""
    echo "Gemma 4:"
    echo "  $0 {gemma4-moe|gemma4-moe-fp8|gemma4-e4b|gemma4-e4b-fp8}"
    echo ""
    echo "Creative / Experimental:"
    echo "  $0 {cydonia-24b|cydonia-24b-mtp|llama-70b|llama-8b|hermes-70b|context-1|context-1-gpu1}"
    echo ""
    echo "Dual GPU (one model per GPU):"
    echo "  $0 dual                — 27B thinking (GPU 0) + Gemma 4 26B MoE fast (GPU 1)"
    echo ""
    echo "TP=2 (both GPUs):"
    echo "  $0 {qwen-36b-tp2|qwen36-27b-fp8-tp2|gemma4-moe-tp2}"
    exit 1
}
stop_vllm() {
    echo "Stopping vLLM (port ${PORT})..."
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    for i in $(seq 1 30); do
        if ! ss -tlnp | grep -q ":${PORT} "; then
            echo "Port ${PORT} released."
            return
        fi
        sleep 1
    done
    echo "Port ${PORT} still held after 30s, force killing..."
    fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
    sleep 3
}
stop_all() {
    echo "Stopping all vLLM services..."
    sudo systemctl stop vllm 2>/dev/null || true
    sudo systemctl stop vllm-fast 2>/dev/null || true
    # Also kill any ad-hoc processes not managed by systemd
    stop_vllm
    stop_vllm_voice
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
stop_vllm_voice() {
    echo "Stopping voice vLLM (port ${VOICE_PORT})..."
    fuser -k "${VOICE_PORT}/tcp" 2>/dev/null || true
    for i in $(seq 1 30); do
        if ! ss -tlnp | grep -q ":${VOICE_PORT} "; then
            echo "Port ${VOICE_PORT} released."
            return
        fi
        sleep 1
    done
    echo "Port ${VOICE_PORT} still held after 30s."
}
wait_ready_voice() {
    echo "Waiting for voice model on port ${VOICE_PORT}..."
    for i in $(seq 1 60); do
        if curl -s --max-time 3 "http://localhost:${VOICE_PORT}/v1/models" | grep -q "model"; then
            echo "Voice model ready!"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Voice model failed to start within 5 minutes."
    tail -20 "${LOG_DIR}/vllm-voice.log"
    return 1
}
[[ $# -lt 1 ]] && usage
# Voice/dual configs manage their own stop — don't kill blindly
case "$1" in
    qwen-36b-fast)    ;; # fast only touches GPU 1 / :8002
    gemma4-moe-fast)  ;; # fast only touches GPU 1 / :8002
    ornith-9b-fast)   ;; # fast only touches GPU 1 / :8002
    dual)            ;; # dual does stop_all itself
    *) stop_vllm ;;
esac
case "$1" in
    # ─── Dual GPU split (one model per GPU) via systemd ──────────────
    dual)
        echo "=== Dual-GPU setup: 27B thinking (GPU 0) + 35B no-thinking (GPU 1) ==="
        stop_all

        echo ""
        echo "Starting vllm.service (27B FP8, GPU 0, port ${PORT})..."
        sudo systemctl start vllm
        wait_ready

        echo ""
        echo "Starting vllm-fast.service (Gemma 4 26B MoE FP8, GPU 1, port ${VOICE_PORT})..."
        sudo systemctl start vllm-fast
        wait_ready_voice

        echo ""
        echo "=== Both models ready ==="
        echo "  GPU 0 :${PORT}        — local       (27B FP8, thinking)"
        echo "  GPU 1 :${VOICE_PORT}  — local-fast   (Gemma 4 26B MoE FP8, instruct)"
        exit 0
        ;;
    qwen-36b-fast)
        # Heretic FP8 — uncensored 35B-A3B, manual swap target for protolabs/fast.
        # NOT the current live fast lane — vllm-fast.service runs Gemma 4 26B MoE
        # (briefly reverted to heretic and back 2026-05-30; Gemma stays default).
        # Use this target to put heretic on :8002 on demand. Heretic was originally
        # retired
        # 2026-04-29 after the model card's recommended logit_bias on
        # <think>/</think> tokens corrupted generation (1-token role-marker
        # garbage on thinking-engaging prompts). Never use that logit_bias.
        #
        # --reasoning-parser qwen3 IS required (CLAUDE.md "Thinking models"):
        # heretic has trained-in always-thinks behavior that fires on some
        # turns even under the no-think template. The parser routes that into
        # reasoning_content instead of leaking raw <think>...</think> into
        # content (the protoCLI leaked-release-notes incident). Verified no
        # leak across factual/reasoning/creative/tool-call prompts 2026-05-30.
        # Callers must allow 32K max output (gateway protoMaker#3830) so the
        # think can close — too small a budget leaves content empty mid-think;
        # the gateway thinking_normalizer salvages reasoning -> content if so.
        echo "Starting heretic FP8 (GPU 1, port ${VOICE_PORT}, 131K)..."
        stop_vllm_voice
        (
            export CUDA_VISIBLE_DEVICES=1
            exec $VLLM_BIN serve /mnt/models/quantized/Qwen3.6-35B-A3B-uncensored-heretic-FP8 \
                --host 0.0.0.0 --port $VOICE_PORT \
                --served-model-name local-fast \
                --max-model-len 131072 \
                --max-num-seqs 512 \
                --chat-template "$NONTHINKING_TEMPLATE" \
                --enable-auto-tool-choice --tool-call-parser qwen3_xml \
                --reasoning-parser qwen3 \
                --gpu-memory-utilization 0.73 \
                --language-model-only \
                --enable-chunked-prefill \
                --async-scheduling \
                --performance-mode interactivity \
                $PREFIX_FLAGS \
                >> "${LOG_DIR}/vllm-voice.log" 2>&1
        ) &
        wait_ready_voice
        exit 0
        ;;
    ornith-9b-fast)
        # Ornith-1.0-9B (dense bf16) as the FAST lane: GPU 1, :8002, local-fast,
        # full 256K + VISION. util 0.72 leaves room for Fish TTS + embed co-residence.
        echo "Starting Ornith-1.0-9B — FAST lane (GPU 1, :${VOICE_PORT}, local-fast, 256K + vision)..."
        stop_vllm_voice
        (
            export CUDA_VISIBLE_DEVICES=1
            exec $VLLM_BIN serve deepreinforce-ai/Ornith-1.0-9B \
                --host 0.0.0.0 --port $VOICE_PORT \
                --served-model-name local-fast \
                --max-model-len 262144 \
                --max-num-seqs 256 \
                --reasoning-parser qwen3 \
                --enable-auto-tool-choice --tool-call-parser qwen3_xml \
                --gpu-memory-utilization 0.72 \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --limit-mm-per-prompt '{"image": 4, "video": 0}' \
                --trust-remote-code \
                >> "${LOG_DIR}/vllm-voice.log" 2>&1
        ) &
        wait_ready_voice
        exit 0
        ;;
    gemma4-moe-fast)
        echo "Starting Gemma 4 26B-A4B MoE FP8 (GPU 1, port ${VOICE_PORT}, 131K, local-fast)..."
        stop_vllm_voice
        (
            export CUDA_VISIBLE_DEVICES=1
            exec $VLLM_BIN serve google/gemma-4-26B-A4B-it \
                $O3 \
                --host 0.0.0.0 --port $VOICE_PORT \
                --served-model-name local-fast \
                --max-model-len 131072 \
                --dtype bfloat16 \
                --quantization fp8 \
                --gpu-memory-utilization 0.72 \
                --language-model-only \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --async-scheduling \
                --generation-config auto \
                --enable-auto-tool-choice --tool-call-parser gemma4 \
                --reasoning-parser gemma4 \
                >> "${LOG_DIR}/vllm-voice.log" 2>&1
        ) &
        wait_ready_voice
        exit 0
        ;;
    qwen-36b-fp8)
        echo "Starting Qwen3.6-35B-A3B MoE + on-the-fly FP8 (GPU 0, 262K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.6-35B-A3B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 262144 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --quantization fp8 \
            $PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-36b-bf16)
        echo "Starting Qwen3.6-35B-A3B MoE bf16 (GPU 0, 64K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.6-35B-A3B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 65536 \
            --max-num-seqs 512 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.85 \
            --language-model-only \
            --enable-chunked-prefill \
            $PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen-36b-tp2)
        echo "Starting Qwen3.6-35B-A3B MoE (TP=2, 250K)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_CUMEM_ENABLE=0 \
        $VLLM_BIN serve Qwen/Qwen3.6-35B-A3B \
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
            $PREFIX_FLAGS \
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
    # ─── Gemma 4 configs ─────────────────────────────────────────────
    # On-the-fly FP8 works on our build despite #39049 reports — verified coherent
    # Heterogeneous head dims force Triton attention fallback (#38887) — expect slower than Qwen
    # Tool calling parsers are experimental — test before relying on them
    gemma4-moe)
        echo "Starting Gemma 4 26B-A4B MoE bf16 (GPU 0, 128K, 4B active)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve google/gemma-4-26B-A4B-it \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --dtype bfloat16 \
            --kv-cache-dtype fp8 \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --async-scheduling \
            --generation-config auto \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            --reasoning-parser gemma4 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    gemma4-moe-fp8)
        echo "Starting Gemma 4 26B-A4B MoE + on-the-fly FP8 (GPU 0, 128K, 175 tok/s)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve google/gemma-4-26B-A4B-it \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --dtype bfloat16 \
            --quantization fp8 \
            --kv-cache-dtype fp8 \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --async-scheduling \
            --generation-config auto \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            --reasoning-parser gemma4 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    gemma4-e4b)
        echo "Starting Gemma 4 E4B dense bf16 (GPU 0, 128K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve google/gemma-4-E4B-it \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --async-scheduling \
            --generation-config auto \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            --reasoning-parser gemma4 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    gemma4-e4b-fp8)
        echo "Starting Gemma 4 E4B + on-the-fly FP8 (GPU 0, 128K)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve google/gemma-4-E4B-it \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --dtype bfloat16 \
            --quantization fp8 \
            --kv-cache-dtype fp8 \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --async-scheduling \
            --generation-config auto \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            --reasoning-parser gemma4 \
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
    # ─── Experimental MTP configs (no tool calling) ─────────────────
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
    qwen36-27b-fp8)
        echo "Starting Qwen3.6-27B FP8 official (GPU 0, 256K, thinking/planning)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.6-27B-FP8 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 262144 \
            --max-num-seqs 512 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --enable-chunked-prefill \
            $PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    qwen36-27b-fp8-mtp)
        echo "Starting Qwen3.6-27B FP8 + MTP (GPU 0, 225K, thinking/planning) — +48% (50→74 tok/s), 79% accept..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve Qwen/Qwen3.6-27B-FP8 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 230400 \
            --max-num-seqs 512 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --enable-chunked-prefill \
            $PREFIX_FLAGS \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    # ─── Ornith-1.0 (DeepReinforce agentic-coding, Qwen3.5 base) ────
    # Eval candidates vs daily drivers (2026-06-26). Qwen3.5 VLM arch
    # (vision_config present) → --language-model-only for text serving.
    # Reasoning models (<think>), official recipe: qwen3_xml + qwen3 reasoning
    # + prefix-caching + trust-remote-code. Recommended sampling temp 0.6 /
    # top_p 0.95 / top_k 20. 9B dense (bf16), 35B MoE (compressed-tensors
    # FP8-Dynamic, Triton path — NO -O3 on MoE). 397B too big for 192GB.
    ornith-9b)
        echo "Starting Ornith-1.0-9B dense bf16 (GPU 0, 131K, agentic coding)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve deepreinforce-ai/Ornith-1.0-9B \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --max-num-seqs 512 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    ornith-35b-fp8-vision)
        # Our static FP8 Ornith-35B as the SMART lane: FULL 256K context + VISION
        # enabled (NO --language-model-only → loads the vision tower for image/video).
        # quant_method=fp8 auto-detected → Triton fused-MoE. For thorough hands-on test.
        echo "Starting Ornith-1.0-35B static FP8 — SMART lane, 256K + VISION (GPU 0)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve /mnt/data/quantized/Ornith-1.0-35B-FP8 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 262144 \
            --max-num-seqs 256 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.92 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --limit-mm-per-prompt '{"image": 4, "video": 0}' \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    ornith-35b-fp8-static)
        # Our static block-wise [128,128] FP8 quant of Ornith-1.0-35B (linear_attn/SSM
        # kept bf16). Built via experiments/quantize/quantize_native_fp8_lowmem.py.
        # quant_method=fp8 auto-detected from config → Triton fused-MoE (no flashinfer).
        # Candidate for protoLabsAI/Ornith-1.0-35B-FP8 (upstream FP8 reportedly broken).
        echo "Starting Ornith-1.0-35B static FP8 (GPU 0, 131K, our quant) — no -O3 (MoE)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve /mnt/data/quantized/Ornith-1.0-35B-FP8 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 131072 \
            --max-num-seqs 512 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    ornith-35b)
        # bf16 repo + on-the-fly --quantization fp8 (vLLM dynamic FP8, Triton fused-MoE).
        # Why not the published -35B-FP8 repo: reported broken (2026-06-26). Why not raw
        # bf16: 0.22.1 routes unquantized MoE to flashinfer-cutlass → sm120 wall. On-the-fly
        # FP8 is our proven 35B-A3B path (qwen-36b-fp8), quality-neutral on MoE, sidesteps both.
        # NO -O3 (MoE regresses ~25%). Raw-bf16 truer run possible via the sm120 recipe env
        # already exported above (drop --quantization fp8) if a no-quant eval is wanted.
        echo "Starting Ornith-1.0-35B bf16 + on-the-fly FP8 (GPU 0, 262K, agentic coding) — no -O3 (MoE)..."
        CUDA_VISIBLE_DEVICES=0 $VLLM_BIN serve deepreinforce-ai/Ornith-1.0-35B \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --max-model-len 262144 \
            --max-num-seqs 512 \
            --dtype bfloat16 \
            --quantization fp8 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --trust-remote-code \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    # ─── Dual GPU configs (TP=2) ───────────────────────────────────
    # NCCL_P2P_DISABLE=1 required for stable CUDA graphs on Blackwell PCIe
    # (ACS on PCIe bridges corrupts P2P during graph replay)
    qwen36-27b-fp8-tp2)
        echo "Starting Qwen3.6-27B FP8 official (TP=2, 131K)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_ALGO=Ring NCCL_PROTO=Simple \
        NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=8 \
        $VLLM_BIN serve Qwen/Qwen3.6-27B-FP8 \
            $O3 \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local \
            --tensor-parallel-size 2 \
            --max-model-len 131072 \
            --max-num-seqs 512 \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --gpu-memory-utilization 0.90 \
            --enable-chunked-prefill \
            --disable-custom-all-reduce \
            $PREFIX_FLAGS \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    gemma4-moe-tp2)
        echo "Starting Gemma 4 26B-A4B MoE FP8 (TP=2, 131K, both GPUs)..."
        NCCL_P2P_DISABLE=1 \
        NCCL_CUMEM_ENABLE=0 \
        $VLLM_BIN serve google/gemma-4-26B-A4B-it \
            --host 0.0.0.0 --port $PORT \
            --served-model-name local-fast \
            --tensor-parallel-size 2 \
            --max-model-len 131072 \
            --dtype bfloat16 \
            --quantization fp8 \
            --gpu-memory-utilization 0.90 \
            --language-model-only \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --async-scheduling \
            --generation-config auto \
            --disable-custom-all-reduce \
            --enable-auto-tool-choice --tool-call-parser gemma4 \
            --reasoning-parser gemma4 \
            >> "${LOG_DIR}/vllm-swap.log" 2>&1 &
        ;;
    *)
        usage
        ;;
esac
wait_ready
