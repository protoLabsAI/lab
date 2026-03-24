#!/bin/bash
# Wait for ORPO inference, then judge both ORPO and base, compare results
# Usage: nohup bash evals/run-wb-compare.sh > /mnt/scratch/logs/wb-compare.log 2>&1 &

set -euo pipefail

EVALS="$HOME/dev/lab/evals"
WB="$EVALS/results/wildbench"

echo "=========================================="
echo "WildBench Full 1024 Comparison — $(date)"
echo "=========================================="

# Wait for ORPO inference to finish
echo "Waiting for ORPO inference to complete..."
while ! python3 -c "import json; d=json.load(open('$WB/9b-orpo-full.json')); assert len(d) >= 1024" 2>/dev/null; do
    count=$(python3 -c "import json; print(len(json.load(open('$WB/9b-orpo-full.json'))))" 2>/dev/null || echo 0)
    echo "  $(date +%H:%M): $count/1024"
    sleep 120
done
echo "ORPO inference done: $(python3 -c "import json; print(len(json.load(open('$WB/9b-orpo-full.json'))))" ) responses"

# Copy base 9B responses
echo ""
echo ">>> Judging base 9B (1024 tasks)..."
cp "$WB/local.json" "$WB/9b-base-full.json" 2>/dev/null || true
cd "$EVALS" && bash run.sh wildbench judge --results "$WB/9b-base-full.json" --judge claude-sonnet-4-6

echo ""
echo ">>> Judging ORPO 9B (1024 tasks)..."
cd "$EVALS" && bash run.sh wildbench judge --results "$WB/9b-orpo-full.json" --judge claude-sonnet-4-6

# Compare
echo ""
echo "=========================================="
echo "COMPARISON — $(date)"
echo "=========================================="
echo ""
echo ">>> Base 9B:"
cd "$EVALS" && bash run.sh wildbench show --results "$WB/9b-base-full.scored.json"
echo ""
echo ">>> ORPO 9B:"
cd "$EVALS" && bash run.sh wildbench show --results "$WB/9b-orpo-full.scored.json"

# Restore daily driver
echo ""
echo ">>> Restoring daily driver..."
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3
bash "$HOME/dev/lab/models/vllm-swap.sh" qwen-27b-int4

echo ""
echo "=========================================="
echo "Done — $(date)"
echo "=========================================="
