#!/bin/bash
# Overnight model evaluation batch — runs quick profile on multiple models
# Usage: nohup bash run-overnight.sh > /mnt/scratch/logs/overnight-eval.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWAP="$HOME/dev/lab/models/vllm-swap.sh"
LOG="/mnt/scratch/logs/overnight-eval.log"

echo "=========================================="
echo "Overnight Eval Batch — $(date)"
echo "=========================================="

# ─── Cloud models (no swap needed, run through gateway) ───
echo ""
echo ">>> Running cloud models through gateway..."

for model in gpt-oss-20b gpt-oss-120b grok-4.1-fast; do
    echo ""
    echo "--- $model ($(date)) ---"
    cd "$SCRIPT_DIR" && bash run.sh profile --name quick --model "$model" || echo "FAILED: $model"
done

# ─── Local models (need vLLM swap) ───
echo ""
echo ">>> Running local models (vLLM swap)..."

for config_model in "cydonia-24b:Cydonia 24B" "mistral-119b:Mistral-Small-4 119B"; do
    config="${config_model%%:*}"
    label="${config_model##*:}"
    echo ""
    echo "--- $label ($(date)) ---"
    echo "Swapping to $config..."
    bash "$SWAP" "$config" || { echo "FAILED to start $config"; continue; }
    cd "$SCRIPT_DIR" && bash run.sh profile --name quick --model local || echo "FAILED: $label"
done

# ─── Restore daily driver ───
echo ""
echo ">>> Restoring daily driver (qwen-27b-int4)..."
bash "$SWAP" qwen-27b-int4

echo ""
echo "=========================================="
echo "Overnight Eval Complete — $(date)"
echo "=========================================="
