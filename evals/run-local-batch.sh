#!/bin/bash
# Run quick profile on all local models sequentially
# Usage: nohup bash run-local-batch.sh > /mnt/scratch/logs/local-batch.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWAP="$HOME/dev/lab/models/vllm-swap.sh"

echo "=========================================="
echo "Local Model Batch — $(date)"
echo "=========================================="

for config_label in \
  "qwen-27b-int4:Qwen 27B INT4" \
  "qwen-27b-int4-mtp:Qwen 27B INT4 MTP" \
  "qwen-35b:Qwen 35B MoE" \
  "qwen-4b-int4:Qwen 4B INT4"; do

  config="${config_label%%:*}"
  label="${config_label##*:}"

  echo ""
  echo "--- $label ($config) — $(date) ---"
  bash "$SWAP" "$config" || { echo "FAILED to start $config"; continue; }
  cd "$SCRIPT_DIR" && bash run.sh profile --name quick --model local || echo "FAILED eval: $label"
done

# Restore daily driver
echo ""
echo ">>> Restoring daily driver..."
bash "$SWAP" qwen-27b-int4

echo ""
echo "=========================================="
echo "Local Batch Complete — $(date)"
echo "=========================================="
