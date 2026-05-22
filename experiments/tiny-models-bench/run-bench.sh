#!/usr/bin/env bash
# tiny-models-bench / run-bench.sh
#
# Run a tiered suite set against one model key from serve.sh. Brings up
# the model on :8003, runs each suite, captures results, stops the server.
#
# Results land under evals/results/tiny-bench/<model_key>/<suite>_<timestamp>/.
#
# Usage:
#   bash run-bench.sh <model_key>  [tier]
#
# Tiers:
#   smoke   refusal(simple_safety, n=20)                    — pipeline smoke
#   core    refusal(simple_safety) + function_call + structured_output
#                                                            — fastest useful pass
#   tier1   function_call(trials=3) + claw-eval                 — agentic
#   tier2   coding + reasoning + structured_output + summarization
#           + instruction_following + refusal(xstest + simple_safety)
#   all     tier1 + tier2 + creative_writing + roleplay + svg_generation + research
#
# Claw-eval task scope is controlled via CLAW_TASKS env var:
#   CUSTOM_MAX_TOKENS=4096      — max output tokens for custom suite (default fits 8K ctx)
#   CLAW_TASKS=""               — default 6-task EN sample (T02,T04,T06,T08,T10,T12)
#   CLAW_TASKS="all"            — full 52-task pass^3 run (~80 min/model)
#   CLAW_TASKS="T02,T16,T28"    — explicit task list (even numbers = English)

set -euo pipefail
MODEL_KEY="${1:?usage: $0 <model_key> [tier]}"
TIER="${2:-core}"

REPO="$HOME/dev/lab"
BENCH_DIR="$REPO/experiments/tiny-models-bench"
EVAL_DIR="$REPO/evals"
SERVE_SH="$BENCH_DIR/serve.sh"
# Source venv so PATH includes .venv/bin (claw-eval is a console-script
# entry point that subprocess.run("claw-eval", ...) looks up via PATH).
# shellcheck source=/dev/null
source "$REPO/.venv/bin/activate"
PY="$REPO/.venv/bin/python"

GATEWAY_URL="http://localhost:8003/v1"
JUDGE_URL="http://localhost:8002/v1"
RESULTS_ROOT="$EVAL_DIR/results"
BENCH_RESULTS_DIR="$RESULTS_ROOT/tiny-bench/$MODEL_KEY"
mkdir -p "$BENCH_RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Move whatever results dirs got created during this run into the bench tree.
collect_results() {
    local before_marker="$1"
    find "$RESULTS_ROOT" -maxdepth 1 -type d -newer "$before_marker" \
        \( -name "refusal_local-bench_*" -o -name "function_call_local-bench_*" \
           -o -name "custom_local-bench_*" -o -name "*_local-bench_*" \) \
        -print 2>/dev/null \
        | while read -r d; do
            mv -v "$d" "$BENCH_RESULTS_DIR/" 2>/dev/null || true
        done
}

# Cleanup on exit — always stop the server.
cleanup() {
    log "Stopping bench server..."
    bash "$SERVE_SH" stop || true
}
trap cleanup EXIT

# Start the model
log "Bringing up model_key=$MODEL_KEY"
bash "$SERVE_SH" "$MODEL_KEY"
if ! curl -s --max-time 3 "$GATEWAY_URL/models" | grep -q local-bench; then
    log "ERROR: bench server failed to come up"
    exit 1
fi

cd "$EVAL_DIR"
MARKER="$(mktemp)"

case "$TIER" in
    smoke)
        log "[smoke] refusal: simple_safety, n=20"
        $PY -m runners.run_refusal --model local-bench \
            --gateway-url "$GATEWAY_URL" --judge-url "$JUDGE_URL" \
            --dataset simple_safety --sample 20
        ;;

    core)
        log "[core/1] refusal: simple_safety"
        $PY -m runners.run_refusal --model local-bench \
            --gateway-url "$GATEWAY_URL" --judge-url "$JUDGE_URL" \
            --dataset simple_safety || log "refusal FAILED"

        log "[core/2] function_call: all suites, trials=1"
        $PY -m runners.run_function_call --model local-bench \
            --gateway-url "$GATEWAY_URL" \
            --all-suites --trials 1 || log "function_call FAILED"

        log "[core/3] custom: structured_output, trials=1"
        $PY -m runners.run_custom --model local-bench \
            --gateway-url "$GATEWAY_URL" \
            --max-tokens "${CUSTOM_MAX_TOKENS:-4096}" \
            --suite structured_output --trials 1 || log "structured_output FAILED"
        ;;

    tier1)
        log "[tier1/1] function_call: all suites, trials=3"
        $PY -m runners.run_function_call --model local-bench \
            --gateway-url "$GATEWAY_URL" \
            --all-suites --trials 3 || log "function_call FAILED"

        # English claw tasks are even-numbered (T01zh/T03zh/T05zh are Chinese variants)
        CLAW_TASKS="${CLAW_TASKS:-T02,T04,T06,T08,T10,T12}"
        if [ "$CLAW_TASKS" = "all" ]; then
            log "[tier1/2] claw: --all-tasks, trials=3 (~80 min)"
            $PY -m runners.run_claw --model local-bench \
                --gateway-url "$GATEWAY_URL" \
                --all-tasks --trials 3 --port-offset 200 || log "claw FAILED"
        else
            log "[tier1/2] claw: tasks=$CLAW_TASKS, trials=3"
            $PY -m runners.run_claw --model local-bench \
                --gateway-url "$GATEWAY_URL" \
                --tasks "$CLAW_TASKS" --trials 3 --port-offset 200 || log "claw FAILED"
        fi
        ;;

    tier2)
        for suite in coding reasoning structured_output summarization instruction_following; do
            log "[tier2] custom: $suite, trials=1"
            $PY -m runners.run_custom --model local-bench \
                --gateway-url "$GATEWAY_URL" \
                --suite "$suite" --trials 1 || log "$suite FAILED"
        done
        log "[tier2] refusal: xstest + simple_safety"
        $PY -m runners.run_refusal --model local-bench \
            --gateway-url "$GATEWAY_URL" --judge-url "$JUDGE_URL" \
            --dataset xstest,simple_safety || log "refusal FAILED"
        ;;

    all)
        # Reuse tier1 + tier2 + appendix (skip core which overlaps function_call).
        "$0" "$MODEL_KEY" tier1
        "$0" "$MODEL_KEY" tier2
        for suite in creative_writing roleplay svg_generation research; do
            log "[app] custom: $suite, trials=1"
            $PY -m runners.run_custom --model local-bench \
                --gateway-url "$GATEWAY_URL" \
                --suite "$suite" --trials 1 || log "$suite FAILED"
        done
        ;;

    *)
        log "Unknown tier: $TIER. Valid: smoke|core|tier1|tier2|all"
        exit 1
        ;;
esac

collect_results "$MARKER"
rm -f "$MARKER"
log "Done. Results in: $BENCH_RESULTS_DIR"
