#!/usr/bin/env bash
# speed-test-v2.sh — regime × concurrency serving benchmark.
#
# The field standard (InferenceMAX / MLPerf / GuideLLM) measures a matrix of
# ISL/OSL regimes swept across concurrency, reporting client-side TTFT/TPOT
# percentiles + aggregate throughput + goodput under an SLO — not a single
# short-prompt single-stream number. This wraps `vllm bench serve` (in our
# pinned 0.22.1) to do exactly that. speed-test.sh v1 remains for continuity
# with historical single-stream numbers.
#
# Usage:
#   bash speed-test-v2.sh quick            # 2 regimes × C{1,8}, ~10 min (release gate)
#   bash speed-test-v2.sh full             # 4 regimes × C{1,4,8,16,32}, ~60-90 min
#   bash speed-test-v2.sh full 8003 myrun  # custom port + label
#   bash speed-test-v2.sh quick 8011 x /path/to/model   # tokenizer path when the
#                                          # served name isn't a resolvable HF id
#
# SLO for goodput (MLPerf-style headline): TTFT ≤ 2000 ms AND TPOT ≤ 50 ms.

set -euo pipefail

MODE="${1:-quick}"
PORT="${2:-8000}"
LABEL="${3:-}"
TOKENIZER="${4:-}"

BASE_URL="http://localhost:${PORT}"
MODEL=$(curl -s "${BASE_URL}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
[ -n "$LABEL" ] || LABEL=$(echo "$MODEL" | tr '/' '_')

TS=$(date +%Y%m%d-%H%M%S)
OUTDIR="$(dirname "$0")/../evals/results/speed-v2/${LABEL}-${TS}"
mkdir -p "$OUTDIR"

# regime name → ISL OSL  (chat / context-heavy agentic / generation-heavy thinking / v1 legacy)
declare -A REGIMES=(
  [chat]="1024 1024"
  [context]="8192 1024"
  [gen]="1024 8192"
  [legacy]="128 800"
)

if [ "$MODE" = "quick" ]; then
  REGIME_LIST=(chat context)
  CONCURRENCIES=(1 8)
elif [ "$MODE" = "depth" ]; then
  # decode-at-depth ladder (llama-bench -d convention, serving-side):
  # TPOT per rung = decode speed at that KV depth; TTFT per rung = prefill scaling.
  # Server must have max-model-len > deepest rung + OSL.
  REGIMES=([d4k]="4096 256" [d16k]="16384 256" [d32k]="32768 256" [d64k]="63488 256")
  REGIME_LIST=(d4k d16k d32k d64k)
  CONCURRENCIES=(1 4)
else
  REGIME_LIST=(chat context gen legacy)
  CONCURRENCIES=(1 4 8 16 32)
fi

echo "model=${MODEL} mode=${MODE} port=${PORT} → ${OUTDIR}"

source "$HOME/dev/vllm-env/bin/activate"

for regime in "${REGIME_LIST[@]}"; do
  read -r ISL OSL <<< "${REGIMES[$regime]}"
  for C in "${CONCURRENCIES[@]}"; do
    # enough prompts to saturate the concurrency level without unbounded runtime
    NP=$(( C * 8 )); [ "$NP" -lt 16 ] && NP=16; [ "$NP" -gt 128 ] && NP=128
    echo "--- ${regime} (${ISL}/${OSL}) C=${C} n=${NP}"
    vllm bench serve \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      ${TOKENIZER:+--tokenizer "$TOKENIZER"} \
      --dataset-name random \
      --random-input-len "$ISL" \
      --random-output-len "$OSL" \
      --num-prompts "$NP" \
      --max-concurrency "$C" \
      --ignore-eos \
      --seed 42 \
      --percentile-metrics ttft,tpot,itl \
      --metric-percentiles 50,99 \
      --goodput ttft:2000 tpot:50 \
      --save-result \
      --result-dir "$OUTDIR" \
      --result-filename "${regime}-c${C}.json" \
      > "$OUTDIR/${regime}-c${C}.log" 2>&1 || echo "    FAILED (see log)"
  done
done

# house-style summary table
python3 - "$OUTDIR" <<'EOF'
import json, sys, glob, os
outdir = sys.argv[1]
rows = []
for fp in sorted(glob.glob(os.path.join(outdir, "*.json"))):
    r = json.load(open(fp))
    name = os.path.basename(fp)[:-5]
    regime, c = name.rsplit("-c", 1)
    rows.append((regime, int(c),
        r.get("p50_ttft_ms", r.get("median_ttft_ms", 0)),
        r.get("p99_ttft_ms", 0),
        r.get("p50_tpot_ms", r.get("median_tpot_ms", 0)),
        r.get("p99_tpot_ms", 0),
        r.get("output_throughput", 0),
        r.get("request_goodput") or 0,
        r.get("completed", 0)))
rows.sort(key=lambda x: (x[0], x[1]))
print()
print(f"{'regime':<9} {'C':>3} {'ttft p50':>9} {'ttft p99':>9} {'tpot p50':>9} {'tpot p99':>9} {'agg tok/s':>10} {'goodput':>8} {'ok':>4}")
print(f"{'-'*9} {'-'*3} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*8} {'-'*4}")
for regime, c, t50, t99, p50, p99, thr, gp, comp in rows:
    print(f"{regime:<9} {c:>3} {t50:>8.0f}ms {t99:>8.0f}ms {p50:>8.1f}ms {p99:>8.1f}ms {thr:>10.1f} {gp:>8.2f} {comp:>4}")
print(f"\nresults: {outdir}")
EOF
