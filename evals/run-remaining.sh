#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWAP="$HOME/dev/lab/models/vllm-swap.sh"

echo "=========================================="
echo "Remaining Models — $(date)"
echo "=========================================="

for config_label in \
  "qwen-9b-mtp:Qwen 9B MTP" \
  "hermes-70b:Hermes 3 70B FP8" \
  "llama-8b:Llama 3.1 8B AWQ"; do

  config="${config_label%%:*}"
  label="${config_label##*:}"

  echo ""
  echo "--- $label ($config) — $(date) ---"
  bash "$SWAP" "$config" || { echo "FAILED to start $config"; continue; }
  cd "$SCRIPT_DIR" && bash run.sh profile --name quick --model local || echo "FAILED eval: $label"
done

echo ""
echo ">>> Restoring daily driver..."
bash "$SWAP" qwen-27b-int4

echo "=========================================="
echo "Remaining Models Complete — $(date)"
echo "=========================================="
