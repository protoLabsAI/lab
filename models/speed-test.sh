#!/bin/bash
# Quick speed test for the currently loaded vLLM model
# Uses vLLM /metrics for accurate TTFT, TPOT, and decode tok/s
#
# Usage:
#   bash models/speed-test.sh           # 5 runs (default)
#   bash models/speed-test.sh 10        # 10 runs
#   bash models/speed-test.sh 3 short   # 3 short runs (200 tokens)

set -euo pipefail

RUNS="${1:-5}"
MODE="${2:-long}"
URL="http://localhost:8000"

# Check vLLM is running
if ! curl -s --max-time 3 "${URL}/v1/models" | grep -q "model"; then
    echo "ERROR: vLLM not running on ${URL}"
    exit 1
fi

MODEL=$(curl -s "${URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

if [ "$MODE" = "short" ]; then
    PROMPT="List 5 programming languages and one strength of each."
    MAX_TOKENS=200
else
    PROMPT="Write a detailed 500-word essay about the history of computing from Babbage to modern AI. Do not think, just write."
    MAX_TOKENS=800
fi

echo "Model:   ${MODEL}"
echo "Runs:    ${RUNS}"
echo "Tokens:  ${MAX_TOKENS} max"
echo ""

# Snapshot metrics helper
get_metrics() {
    curl -s "${URL}/metrics" 2>/dev/null | python3 -c "
import sys
m = {}
for line in sys.stdin:
    line = line.strip()
    if line.startswith('#') or not line: continue
    parts = line.split(' ')
    if len(parts) >= 2:
        m[parts[0].split('{')[0]] = parts[-1]
print(f\"{m.get('vllm:time_to_first_token_seconds_sum','0')} {m.get('vllm:time_to_first_token_seconds_count','0')} {m.get('vllm:request_time_per_output_token_seconds_sum','0')} {m.get('vllm:request_time_per_output_token_seconds_count','0')} {m.get('vllm:generation_tokens_total','0')}\")
"
}

# Warmup
echo "Warming up..."
for _ in 1 2 3; do
    curl -s "${URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"local","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20,"chat_template_kwargs":{"enable_thinking":false}}' > /dev/null 2>&1
done
sleep 1

# Snapshot BEFORE
BEFORE=$(get_metrics)

# Run
total_wall=0
for i in $(seq 1 "$RUNS"); do
    START=$(date +%s%N)
    curl -s "${URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":${MAX_TOKENS},\"chat_template_kwargs\":{\"enable_thinking\":false}}" > /dev/null 2>&1
    END=$(date +%s%N)
    wall=$(python3 -c "print(round(($END - $START) / 1e9, 2))")
    echo "  Run $i: ${wall}s"
    total_wall=$(python3 -c "print($total_wall + $wall)")
done

# Snapshot AFTER
sleep 1
AFTER=$(get_metrics)

# Results
python3 -c "
before = '${BEFORE}'.split()
after = '${AFTER}'.split()

ttft_d = float(after[0]) - float(before[0])
ttft_n = float(after[1]) - float(before[1])
tpot_d = float(after[2]) - float(before[2])
tpot_n = float(after[3]) - float(before[3])
gen_tok = float(after[4]) - float(before[4])
wall = ${total_wall}

avg_ttft = (ttft_d / ttft_n * 1000) if ttft_n > 0 else 0
avg_tpot = (tpot_d / tpot_n * 1000) if tpot_n > 0 else 0
decode = (1000 / avg_tpot) if avg_tpot > 0 else 0
wall_tps = gen_tok / wall if wall > 0 else 0

print()
print(f'  Model:        ${MODEL}')
print(f'  Requests:     {int(ttft_n)}')
print(f'  Gen tokens:   {int(gen_tok)}')
print(f'  ──────────────────────────')
print(f'  Decode tok/s: {decode:.1f}')
print(f'  Wall tok/s:   {wall_tps:.1f}')
print(f'  Avg TTFT:     {avg_ttft:.0f} ms')
print(f'  Avg TPOT:     {avg_tpot:.2f} ms')
print(f'  Total wall:   {wall:.1f}s')
"
