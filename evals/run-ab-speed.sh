#!/bin/bash
# A/B Speed Test — compare baseline vs optimized vLLM configs
# Uses vLLM /metrics endpoint for accurate TTFT, TPOT, and throughput
# Usage: bash run-ab-speed.sh <model-base> [runs]
# Example: bash run-ab-speed.sh qwen-4b-int4 5

set -euo pipefail

MODEL="${1:?Usage: $0 <model-base> [runs]}"
RUNS="${2:-5}"
SWAP="$HOME/dev/lab/models/vllm-swap.sh"
OPT="${MODEL}-opt"
URL="http://localhost:8000"

echo "========================================"
echo "A/B Speed Test: ${MODEL} vs ${OPT}"
echo "Runs per config: ${RUNS}"
echo "========================================"

get_metrics() {
    # Snapshot vLLM metrics — returns "ttft_sum ttft_count tpot_sum tpot_count gen_tokens"
    curl -s "${URL}/metrics" 2>/dev/null | python3 -c "
import sys
metrics = {}
for line in sys.stdin:
    line = line.strip()
    if line.startswith('#') or not line:
        continue
    # Parse: metric_name{labels} value
    parts = line.split(' ')
    if len(parts) >= 2:
        key = parts[0].split('{')[0]
        val = parts[-1]
        metrics[key] = val

ttft_sum = metrics.get('vllm:time_to_first_token_seconds_sum', '0')
ttft_count = metrics.get('vllm:time_to_first_token_seconds_count', '0')
tpot_sum = metrics.get('vllm:request_time_per_output_token_seconds_sum', '0')
tpot_count = metrics.get('vllm:request_time_per_output_token_seconds_count', '0')
gen_tokens = metrics.get('vllm:generation_tokens_total', '0')
print(f'{ttft_sum} {ttft_count} {tpot_sum} {tpot_count} {gen_tokens}')
"
}

run_speed_test() {
    local label="$1"
    local n="$2"

    echo ""
    echo "--- ${label} ---"

    # Warmup (3 short requests)
    for _ in 1 2 3; do
        curl -s "${URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"local","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20,"chat_template_kwargs":{"enable_thinking":false}}' > /dev/null 2>&1
    done
    sleep 1

    # Snapshot metrics BEFORE
    BEFORE=$(get_metrics)

    # Run N generation requests
    local total_wall=0
    for i in $(seq 1 "$n"); do
        START=$(date +%s%N)
        curl -s "${URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"local","messages":[{"role":"user","content":"Write a detailed 500-word essay about the history of computing from Babbage to modern AI. Do not think, just write."}],"max_tokens":800,"chat_template_kwargs":{"enable_thinking":false}}' > /dev/null 2>&1
        END=$(date +%s%N)
        wall=$(python3 -c "print(round(($END - $START) / 1e9, 2))")
        echo "  Run $i: ${wall}s wall"
        total_wall=$(python3 -c "print($total_wall + $wall)")
    done

    # Snapshot metrics AFTER
    sleep 1
    AFTER=$(get_metrics)

    # Calculate deltas
    python3 -c "
before = '${BEFORE}'.split()
after = '${AFTER}'.split()

ttft_sum_delta = float(after[0]) - float(before[0])
ttft_count_delta = float(after[1]) - float(before[1])
tpot_sum_delta = float(after[2]) - float(before[2])
tpot_count_delta = float(after[3]) - float(before[3])
gen_tokens_delta = float(after[4]) - float(before[4])
total_wall = ${total_wall}

avg_ttft = (ttft_sum_delta / ttft_count_delta * 1000) if ttft_count_delta > 0 else 0
avg_tpot = (tpot_sum_delta / tpot_count_delta * 1000) if tpot_count_delta > 0 else 0
tok_per_sec = gen_tokens_delta / total_wall if total_wall > 0 else 0
decode_tok_s = (1000 / avg_tpot) if avg_tpot > 0 else 0

print(f'  ────────────────────────────')
print(f'  Requests:     {int(ttft_count_delta)}')
print(f'  Gen tokens:   {int(gen_tokens_delta)}')
print(f'  Avg TTFT:     {avg_ttft:.0f} ms')
print(f'  Avg TPOT:     {avg_tpot:.2f} ms')
print(f'  Decode tok/s: {decode_tok_s:.1f} (1/TPOT)')
print(f'  Wall tok/s:   {tok_per_sec:.1f} (total tokens / wall time)')
print(f'  Total wall:   {total_wall:.1f}s')

# Output machine-readable line for comparison
print(f'RESULT:{decode_tok_s:.1f}:{avg_ttft:.0f}:{avg_tpot:.2f}:{tok_per_sec:.1f}')
"
}

# --- Baseline ---
echo ""
echo ">>> Starting BASELINE: ${MODEL}"
bash "$SWAP" "$MODEL"
sleep 3
BASE_OUT=$(run_speed_test "BASELINE (${MODEL})" "$RUNS")
echo "$BASE_OUT"
BASE_LINE=$(echo "$BASE_OUT" | grep "^RESULT:" | tail -1)

# --- Optimized ---
echo ""
echo ">>> Starting OPTIMIZED: ${OPT}"
bash "$SWAP" "$OPT"
sleep 3
OPT_OUT=$(run_speed_test "OPTIMIZED (${OPT})" "$RUNS")
echo "$OPT_OUT"
OPT_LINE=$(echo "$OPT_OUT" | grep "^RESULT:" | tail -1)

# --- Summary ---
python3 -c "
base = '${BASE_LINE}'.split(':')
opt = '${OPT_LINE}'.split(':')

b_decode, b_ttft, b_tpot, b_wall = float(base[1]), float(base[2]), float(base[3]), float(base[4])
o_decode, o_ttft, o_tpot, o_wall = float(opt[1]), float(opt[2]), float(opt[3]), float(opt[4])

print()
print('========================================')
print('COMPARISON: BASELINE vs OPTIMIZED')
print('========================================')
print(f'                  Baseline    Optimized   Change')
def pct(a, b): return f'{((b/a)-1)*100:>+.1f}%' if a > 0 else 'N/A'
print(f'  Decode tok/s:   {b_decode:>8.1f}    {o_decode:>8.1f}    {pct(b_decode, o_decode)}')
print(f'  Avg TTFT (ms):  {b_ttft:>8.0f}    {o_ttft:>8.0f}    {pct(b_ttft, o_ttft)}')
print(f'  Avg TPOT (ms):  {b_tpot:>8.2f}    {o_tpot:>8.2f}    {pct(b_tpot, o_tpot)}')
print(f'  Wall tok/s:     {b_wall:>8.1f}    {o_wall:>8.1f}    {pct(b_wall, o_wall)}')
print()
"

# Restore baseline config
echo ">>> Restoring baseline: ${MODEL}"
bash "$SWAP" "$MODEL"
