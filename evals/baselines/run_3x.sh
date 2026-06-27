#!/usr/bin/env bash
# Standing baseline at 3x with expanded coverage (policy: every entry 3 trials + ± band).
#
# Usage: run_3x.sh <served-model-name> <gateway-url> <outdir> [--claw]
#   run_3x.sh local        http://localhost:8000/v1 baselines/runs/35b
#   run_3x.sh ornith-9b    http://localhost:8005/v1 baselines/runs/9b --claw
#
# Judged suites go through ./run.sh --local (Infisical injects the gateway judge key).
# Deterministic suites (FC temp 0, quant-sensitivity) need no judge.
# claw is opt-in (--claw): it's the multi-hour long pole (35 tasks x3 + sandbox).
set -uo pipefail

M="${1:?served model name}"; URL="${2:?gateway url}"; OUT="${3:?outdir}"
CLAW="${4:-}"
cd "$(dirname "$0")/.."   # evals/
mkdir -p "$OUT"

echo "############ 3x baseline: model=$M url=$URL out=$OUT ############"

# --- deterministic (temp 0 / match grader, no judge noise) ---
echo "### function-call (temp 0, exact-match) ###"
./run.sh --local function-call --model "$M" --gateway-url "$URL" --all-suites --trials 3 --temperature 0 \
  --output-dir "$OUT/fc" 2>&1 | tee "$OUT/fc.txt" | grep -E "Results:|Per-trial|^  (external|inprocess|untagged)"

echo "### quant-sensitivity (deterministic grading, thinking-ON so reasoning is reliable) ###"
# Match-graded (exact), but thinking-ON: the model must reason to compute the chains
# reliably at full precision — then quant drift (vs an FP8/INT4 variant) is the clean signal.
# Thinking-off confounds it (the 35B flubs multi-step chains even at bf16).
./run.sh --local custom --suite quant_sensitivity --model "$M" --gateway-url "$URL" --thinking --trials 3 \
  --output-dir "$OUT/quant" 2>&1 | tee "$OUT/quant.txt" | grep -E "Results:|Mean score:"

# --- judged custom suites (thinking-on, serving temp) ---
for S in coding reasoning structured_output protolabs/tool_reliability protolabs/routing; do
  echo "### custom: $S (thinking-on, judged) ###"
  ./run.sh --local custom --suite "$S" --model "$M" --gateway-url "$URL" --thinking --trials 3 \
    --output-dir "$OUT/$(echo "$S" | tr / _)" 2>&1 | tee "$OUT/$(echo "$S" | tr / _).txt" | grep -E "Results:|Mean score:"
done

# --- long-context needle (multi-length recall; validates 256K ctx / FP8-KV / prefix cache) ---
echo "### context needle (4k/16k/64k/128k) ###"
./venv/bin/python tasks/context_window/needle_in_haystack.py --model "$M" --gateway-url "$URL" \
  --lengths 4096,16384,65536,131072 2>&1 | tee "$OUT/needle.txt" | grep -E "Total:|Length|tok"

# --- claw agentic (judged + sandbox) — opt-in, the long pole ---
if [ "$CLAW" = "--claw" ]; then
  echo "### claw (agentic, judged + sandbox, 3x) ###"
  ./run.sh --local claw --model "$M" --gateway-url "$URL" --all-tasks --trials 3 --sandbox \
    --output-dir "$OUT/claw" 2>&1 | tee "$OUT/claw.txt" | grep -E "Results:|Total:|passed"
fi

echo "############ done: $OUT ############"
