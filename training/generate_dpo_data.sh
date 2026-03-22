#!/bin/bash
# Generate DPO training data: chosen (GPT-5.4-mini) + rejected (9B, 4B)
# Run this after the current inference jobs finish, OR run standalone.
#
# Usage: nohup bash training/generate_dpo_data.sh > /mnt/scratch/logs/dpo-datagen.log 2>&1 &

set -euo pipefail

EVALS_DIR="$HOME/dev/lab/evals"
TRAINING_DIR="$HOME/dev/lab/training"
SWAP="$HOME/dev/lab/models/vllm-swap.sh"
WB_DIR="$EVALS_DIR/results/wildbench"

echo "=========================================="
echo "DPO Data Generation — $(date)"
echo "=========================================="

# Step 1: Wait for GPT-5.4-mini "chosen" data if still running
CHOSEN="$WB_DIR/gpt-5.4-mini.json"
if [ ! -f "$CHOSEN" ] || ! python3 -c "import json; d=json.load(open('$CHOSEN')); assert len(d) >= 1000" 2>/dev/null; then
    echo "Waiting for GPT-5.4-mini full inference to complete..."
    while ! python3 -c "import json; d=json.load(open('$CHOSEN')); assert len(d) >= 1000" 2>/dev/null; do
        sleep 60
    done
fi
echo "Chosen data ready: $(python3 -c "import json; print(len(json.load(open('$CHOSEN'))))" ) responses"

# Step 2: Generate 9B rejected data (if not already done)
REJECTED_9B="$WB_DIR/qwen-9b-mtp.json"
if [ ! -f "$REJECTED_9B" ] || ! python3 -c "import json; d=json.load(open('$REJECTED_9B')); assert len(d) >= 1000" 2>/dev/null; then
    # Check if local.json has 9B data from current run
    if python3 -c "import json; d=json.load(open('$WB_DIR/local.json')); assert len(d) >= 1000" 2>/dev/null; then
        cp "$WB_DIR/local.json" "$REJECTED_9B"
        echo "Copied existing 9B data from local.json"
    else
        echo "Generating 9B rejected data..."
        bash "$SWAP" qwen-9b-mtp
        cd "$EVALS_DIR" && bash run.sh wildbench infer --model local --gateway-url http://localhost:8000/v1 --output "$REJECTED_9B"
    fi
fi
echo "9B rejected data ready: $(python3 -c "import json; print(len(json.load(open('$REJECTED_9B'))))" ) responses"

# Step 3: Generate 4B rejected data
REJECTED_4B="$WB_DIR/qwen-4b-int4.json"
if [ ! -f "$REJECTED_4B" ] || ! python3 -c "import json; d=json.load(open('$REJECTED_4B')); assert len(d) >= 1000" 2>/dev/null; then
    echo "Generating 4B rejected data..."
    bash "$SWAP" qwen-4b-int4
    cd "$EVALS_DIR" && bash run.sh wildbench infer --model local --gateway-url http://localhost:8000/v1 --output "$REJECTED_4B"
fi
echo "4B rejected data ready: $(python3 -c "import json; print(len(json.load(open('$REJECTED_4B'))))" ) responses"

# Step 4: Build DPO datasets
echo ""
echo "Building DPO datasets..."
cd "$TRAINING_DIR"

python3 build_dpo_dataset.py \
    --chosen "$CHOSEN" \
    --rejected "$REJECTED_9B" \
    --output data/dpo_wildbench_9b.json

python3 build_dpo_dataset.py \
    --chosen "$CHOSEN" \
    --rejected "$REJECTED_4B" \
    --output data/dpo_wildbench_4b.json

# Step 5: Restore daily driver
echo ""
echo "Restoring daily driver..."
bash "$SWAP" qwen-27b-int4

echo ""
echo "=========================================="
echo "DPO Data Generation Complete — $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  cd ~/dev/LLaMA-Factory"
echo "  source .venv/bin/activate"
echo "  llamafactory-cli train ~/dev/lab/training/configs/qwen9b_dpo_wildbench.yaml"
