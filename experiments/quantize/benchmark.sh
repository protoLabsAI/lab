#!/bin/bash
# Quantize → Serve → Speed Test → Eval — full pipeline
#
# Usage:
#   bash benchmark.sh Qwen/Qwen3.5-9B fp8          # FP8 quant + benchmark
#   bash benchmark.sh Qwen/Qwen3.5-9B gptq         # GPTQ INT4 + benchmark
#   bash benchmark.sh Qwen/Qwen3.5-9B fp8 --eval   # + run quick eval profile
#
# Requires: vllm-env activated, llmcompressor installed

set -euo pipefail

MODEL="${1:?Usage: benchmark.sh <model> <method> [--eval]}"
METHOD="${2:?Usage: benchmark.sh <model> <method> [--eval]}"
RUN_EVAL="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUANT_OUTPUT="/mnt/models/quantized"
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')

SUFFIX_MAP_fp8="FP8"
SUFFIX_MAP_gptq="GPTQ-Int4"
SUFFIX_MAP_awq="AWQ-Int4"
eval "SUFFIX=\$SUFFIX_MAP_${METHOD}"

OUTPUT_DIR="${QUANT_OUTPUT}/${MODEL_SHORT}-${SUFFIX}"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/${MODEL_SHORT}-${SUFFIX}_${TIMESTAMP}.json"

echo "============================================"
echo "  protoLabs Quantization Benchmark"
echo "============================================"
echo "  Model:    ${MODEL}"
echo "  Method:   ${METHOD}"
echo "  Output:   ${OUTPUT_DIR}"
echo ""

# ── Step 1: Quantize ─────────────────────────────────────────
echo "[1/4] Quantizing..."
source ~/dev/vllm-env/bin/activate

python "$SCRIPT_DIR/quantize.py" \
    --model "$MODEL" \
    --method "$METHOD" \
    --output "$OUTPUT_DIR"

echo ""

# ── Step 2: Stop current vLLM, serve quantized model ────────
echo "[2/4] Serving quantized model..."
bash "$LAB_DIR/models/vllm-swap.sh" stop 2>/dev/null || true

# Start vLLM with quantized model
CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$OUTPUT_DIR" \
    --host 0.0.0.0 --port 8000 \
    --served-model-name local \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    > /mnt/scratch/logs/vllm.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# Wait for ready
echo "  Waiting for model to load..."
for i in $(seq 1 120); do
    if curl -s --max-time 2 http://localhost:8000/v1/models | grep -q "model"; then
        echo "  Ready! (${i}s)"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "  TIMEOUT — vLLM failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo ""

# ── Step 3: Speed test ──────────────────────────────────────
echo "[3/4] Running speed test (5 runs)..."
bash "$LAB_DIR/models/speed-test.sh" 5 long | tee /tmp/speed_result.txt

echo ""

# ── Step 4: Eval (optional) ────────────────────────────────
if [ "$RUN_EVAL" = "--eval" ]; then
    echo "[4/4] Running quick eval profile..."
    cd "$LAB_DIR/evals"
    ./run.sh --local profile --name quick --model local
else
    echo "[4/4] Eval skipped (pass --eval to run)"
fi

# ── Save results ─────────────────────────────────────────────
python3 -c "
import json
from pathlib import Path

quant_result = {}
qr = Path('$OUTPUT_DIR/quantize_result.json')
if qr.exists():
    quant_result = json.loads(qr.read_text())

print(json.dumps({
    'model': '$MODEL',
    'method': '$METHOD',
    'output_dir': '$OUTPUT_DIR',
    'quantization': quant_result,
    'timestamp': '$TIMESTAMP',
}, indent=2))
" > "$RESULT_FILE"

echo ""
echo "============================================"
echo "  Results saved to: $RESULT_FILE"
echo "  Quantized model:  $OUTPUT_DIR"
echo "============================================"
