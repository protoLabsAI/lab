#!/bin/bash
# Run fine-tuning experiments A/B/C with eval after each
# Usage: nohup bash training/run_experiments.sh > /mnt/scratch/logs/ft-experiments.log 2>&1 &

set -euo pipefail

LLAMA_FACTORY="$HOME/dev/LLaMA-Factory"
LAB="$HOME/dev/lab"
SWAP="$LAB/models/vllm-swap.sh"
VLLM_OMNI_ENV="$HOME/dev/vllm-omni-env"
VLLM_ENV="$HOME/dev/vllm-env"

export HF_HOME="/mnt/models/huggingface"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

merge_and_eval() {
    local adapter_path="$1"
    local merged_path="$2"
    local label="$3"

    echo ""
    echo ">>> Merging $label..."
    cd "$LLAMA_FACTORY" && source .venv/bin/activate
    llamafactory-cli export \
        --model_name_or_path Qwen/Qwen3.5-9B \
        --adapter_name_or_path "$adapter_path" \
        --template qwen3 \
        --finetuning_type lora \
        --trust_remote_code true \
        --export_dir "$merged_path" \
        --export_size 5 \
        --export_legacy_format false

    echo ">>> Serving $label for WildBench eval..."
    # Kill any existing vLLM
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 5
    fuser -k /dev/nvidia0 2>/dev/null || true
    sleep 3

    "$VLLM_ENV/bin/vllm" serve "$merged_path" \
        --host 0.0.0.0 --port 8000 \
        --served-model-name local \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --language-model-only \
        --tokenizer Qwen/Qwen3.5-9B \
        --trust-remote-code &

    # Wait for model to load
    for i in $(seq 1 60); do
        curl -s --max-time 3 http://localhost:8000/v1/models | grep -q model && break
        sleep 5
    done

    echo ">>> Running WildBench 10-task eval for $label..."
    cd "$LAB/evals"
    source "$LAB/.venv/bin/activate"
    bash run.sh wildbench infer --model local --limit 10 --gateway-url http://localhost:8000/v1
    # Save with unique name
    cp results/wildbench/local.json "results/wildbench/${label}.json"
    bash run.sh wildbench judge --results "results/wildbench/${label}.json" --judge claude-sonnet-4-6

    echo ""
    echo "=== $label COMPLETE ==="
}

echo "=========================================="
echo "Fine-Tuning Experiments — $(date)"
echo "=========================================="

# ─── Method A: ORPO ────────────────────────────────────
echo ""
echo "############################################"
echo "### METHOD A: ORPO (single-stage)        ###"
echo "############################################"

# Kill vLLM to free GPU
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3

cd "$LLAMA_FACTORY" && source .venv/bin/activate
llamafactory-cli train "$LAB/training/configs/qwen9b_orpo_wildbench.yaml"

merge_and_eval \
    "/mnt/data/training/qwen9b-orpo" \
    "/mnt/data/training/qwen9b-orpo-merged" \
    "9b-orpo"

# ─── Method B: SFT → DPO ──────────────────────────────
echo ""
echo "############################################"
echo "### METHOD B: SFT then DPO               ###"
echo "############################################"

# Phase 1: SFT
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3

cd "$LLAMA_FACTORY" && source .venv/bin/activate
echo ">>> Phase 1: SFT on chosen responses..."
llamafactory-cli train "$LAB/training/configs/qwen9b_sft_wildbench.yaml"

# Phase 2: DPO from SFT checkpoint
echo ">>> Phase 2: DPO from SFT checkpoint..."
llamafactory-cli train "$LAB/training/configs/qwen9b_dpo_v2_wildbench.yaml"

merge_and_eval \
    "/mnt/data/training/qwen9b-sft-dpo" \
    "/mnt/data/training/qwen9b-sft-dpo-merged" \
    "9b-sft-dpo"

# ─── Method C: DPO only (corrected hyperparams) ───────
echo ""
echo "############################################"
echo "### METHOD C: DPO v2 (corrected params)  ###"
echo "############################################"

pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3

# Create config for standalone DPO (no SFT adapter)
cd "$LLAMA_FACTORY" && source .venv/bin/activate

# Modify config to not use SFT adapter
sed 's|adapter_name_or_path: /mnt/data/training/qwen9b-sft|# adapter_name_or_path: none|;s|create_new_adapter: true|# create_new_adapter: false|;s|output_dir: /mnt/data/training/qwen9b-sft-dpo|output_dir: /mnt/data/training/qwen9b-dpo-v2|;s|logging_dir: /mnt/data/training/qwen9b-sft-dpo/logs|logging_dir: /mnt/data/training/qwen9b-dpo-v2/logs|' \
    "$LAB/training/configs/qwen9b_dpo_v2_wildbench.yaml" > /tmp/qwen9b_dpo_v2_standalone.yaml

llamafactory-cli train /tmp/qwen9b_dpo_v2_standalone.yaml

merge_and_eval \
    "/mnt/data/training/qwen9b-dpo-v2" \
    "/mnt/data/training/qwen9b-dpo-v2-merged" \
    "9b-dpo-v2"

# ─── Restore daily driver ─────────────────────────────
echo ""
echo ">>> Restoring daily driver..."
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3
bash "$SWAP" qwen-27b-int4

echo ""
echo "=========================================="
echo "All Experiments Complete — $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  A (ORPO):     evals/results/wildbench/9b-orpo.scored.json"
echo "  B (SFT+DPO):  evals/results/wildbench/9b-sft-dpo.scored.json"
echo "  C (DPO v2):   evals/results/wildbench/9b-dpo-v2.scored.json"
echo ""
echo "Compare with: bash evals/run.sh wildbench show --results <file>"
