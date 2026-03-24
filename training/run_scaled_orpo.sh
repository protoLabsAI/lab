#!/bin/bash
# Overnight: wait for dataset, train ORPO, merge, eval on full WildBench
# Usage: nohup bash training/run_scaled_orpo.sh > /mnt/scratch/logs/scaled-orpo.log 2>&1 &

set -euo pipefail

LAB="$HOME/dev/lab"
LLAMA="$HOME/dev/LLaMA-Factory"
SWAP="$LAB/models/vllm-swap.sh"
VLLM_ENV="$HOME/dev/vllm-env"

export HF_HOME="/mnt/models/huggingface"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=========================================="
echo "Scaled ORPO Pipeline — $(date)"
echo "=========================================="

# Step 1: Wait for dataset build to finish
echo ">>> Waiting for dataset build..."
while ! [ -f "$LAB/training/data/dpo_scaled_9b.json" ]; do
    sleep 60
    echo "  $(date +%H:%M): waiting for dpo_scaled_9b.json..."
done
PAIRS=$($LAB/.venv/bin/python -c "import json; print(len(json.load(open('$LAB/training/data/dpo_scaled_9b.json'))))")
echo "Dataset ready: $PAIRS pairs"

# Step 2: Kill vLLM to free GPU
echo ""
echo ">>> Freeing GPU for training..."
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3

# Step 3: Train ORPO
echo ""
echo ">>> Training ORPO on scaled dataset ($PAIRS pairs)..."
cd "$LLAMA" && source .venv/bin/activate
llamafactory-cli train "$LAB/training/configs/qwen9b_orpo_scaled.yaml"

# Step 4: Merge LoRA
echo ""
echo ">>> Merging LoRA adapter..."
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3.5-9B \
    --adapter_name_or_path /mnt/data/training/qwen9b-orpo-scaled \
    --template qwen3 \
    --finetuning_type lora \
    --trust_remote_code true \
    --export_dir /mnt/data/training/qwen9b-orpo-scaled-merged \
    --export_size 5 \
    --export_legacy_format false

# Step 5: Serve merged model
echo ""
echo ">>> Serving merged model for eval..."
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3

"$VLLM_ENV/bin/vllm" serve /mnt/data/training/qwen9b-orpo-scaled-merged \
    --host 0.0.0.0 --port 8000 \
    --served-model-name local \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --language-model-only \
    --tokenizer Qwen/Qwen3.5-9B \
    --trust-remote-code &

for i in $(seq 1 60); do
    curl -s --max-time 3 http://localhost:8000/v1/models | grep -q model && break
    sleep 5
done

# Step 6: Full 1024 WildBench eval
echo ""
echo ">>> Running full WildBench 1024 eval..."
cd "$LAB/evals"
source "$LAB/.venv/bin/activate"
bash run.sh wildbench infer --model local --gateway-url http://localhost:8000/v1 \
    --output results/wildbench/9b-orpo-scaled-full.json

echo ""
echo ">>> Judging with Sonnet..."
bash run.sh wildbench judge --results results/wildbench/9b-orpo-scaled-full.json --judge claude-sonnet-4-6

echo ""
echo ">>> Results:"
bash run.sh wildbench show --results results/wildbench/9b-orpo-scaled-full.scored.json

# Step 7: Restore daily driver
echo ""
echo ">>> Restoring daily driver..."
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3
bash "$SWAP" qwen-27b-int4

echo ""
echo "=========================================="
echo "Scaled ORPO Pipeline Complete — $(date)"
echo "=========================================="
echo ""
echo "Compare:"
echo "  Base 9B:         +3.54 WB-Score"
echo "  ORPO (997 pairs): +3.05 WB-Score"
echo "  ORPO (scaled):   see above"
