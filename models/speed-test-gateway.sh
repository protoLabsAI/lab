#!/bin/bash
# Speed test for models behind the LiteLLM gateway
# Measures wall-clock TTFT and tok/s (no vLLM /metrics available)
#
# Usage:
#   bash models/speed-test-gateway.sh minimax-m2.7
#   bash models/speed-test-gateway.sh minimax-m2.7-highspeed 10
#   bash models/speed-test-gateway.sh gpt-5.4-mini 5 short

set -euo pipefail

MODEL="${1:?Usage: speed-test-gateway.sh <model> [runs] [short|long]}"
RUNS="${2:-5}"
MODE="${3:-long}"
GATEWAY_URL="${GATEWAY_URL:-http://100.101.189.45:4000/v1}"
API_KEY="${GATEWAY_API_KEY:-not-needed}"

if [ "$MODE" = "short" ]; then
    PROMPT="List 5 programming languages and one strength of each."
    MAX_TOKENS=200
else
    PROMPT="Write a detailed 500-word essay about the history of computing from Babbage to modern AI. Do not use thinking tags or reasoning blocks."
    MAX_TOKENS=800
fi

echo "Model:   ${MODEL}"
echo "Gateway: ${GATEWAY_URL}"
echo "Runs:    ${RUNS}"
echo "Tokens:  ${MAX_TOKENS} max"
echo ""

# Warmup
echo "Warming up..."
for _ in 1 2; do
    curl -s "${GATEWAY_URL}/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello briefly.\"}],\"max_tokens\":20}" > /dev/null 2>&1
done
sleep 1

# Run with streaming to measure TTFT
total_tokens=0
total_wall=0
total_ttft=0

for i in $(seq 1 "$RUNS"); do
    # Use streaming to measure TTFT (time to first chunk)
    START_NS=$(date +%s%N)
    FIRST_CHUNK_NS=""
    RESPONSE=""

    while IFS= read -r line; do
        if [ -z "$FIRST_CHUNK_NS" ] && [ -n "$line" ] && echo "$line" | grep -q '"delta"'; then
            FIRST_CHUNK_NS=$(date +%s%N)
        fi
        RESPONSE="${RESPONSE}${line}"
    done < <(curl -sN "${GATEWAY_URL}/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":${MAX_TOKENS},\"stream\":true}" 2>/dev/null)

    END_NS=$(date +%s%N)

    # Extract token count from usage in final chunk
    tokens=$(echo "$RESPONSE" | grep -oP '"completion_tokens":\s*\K\d+' | tail -1)
    tokens=${tokens:-0}

    wall=$(python3 -c "print(round(($END_NS - $START_NS) / 1e9, 2))")

    if [ -n "$FIRST_CHUNK_NS" ]; then
        ttft=$(python3 -c "print(round(($FIRST_CHUNK_NS - $START_NS) / 1e9, 3))")
    else
        ttft="N/A"
    fi

    if [ "$tokens" -gt 0 ]; then
        toks=$(python3 -c "print(round($tokens / $wall, 1))")
    else
        toks="?"
    fi

    echo "  Run $i: ${wall}s | TTFT ${ttft}s | ${tokens} tokens | ${toks} tok/s"

    total_wall=$(python3 -c "print($total_wall + $wall)")
    total_tokens=$(python3 -c "print($total_tokens + $tokens)")
    if [ "$ttft" != "N/A" ]; then
        total_ttft=$(python3 -c "print($total_ttft + $ttft)")
    fi
done

echo ""
echo "─────────────────────────────────"
python3 -c "
runs = $RUNS
wall = $total_wall
tokens = $total_tokens
ttft = $total_ttft

avg_wall = wall / runs
avg_tokens = tokens / runs if tokens > 0 else 0
avg_toks = tokens / wall if wall > 0 else 0
avg_ttft = ttft / runs if ttft > 0 else 0

print(f'Model:     $MODEL')
print(f'Avg wall:  {avg_wall:.2f}s')
print(f'Avg TTFT:  {avg_ttft:.3f}s')
print(f'Avg tok/s: {avg_toks:.1f} (wall)')
print(f'Avg tokens: {avg_tokens:.0f} per response')
"
