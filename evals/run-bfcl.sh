#!/bin/bash
# Run BFCL v4 benchmark against local vLLM endpoint
#
# Usage:
#   ./run-bfcl.sh generate google/gemma-4-26B-A4B-it multi_turn
#   ./run-bfcl.sh evaluate google/gemma-4-26B-A4B-it multi_turn
#   ./run-bfcl.sh all google/gemma-4-26B-A4B-it all_scoring
#   ./run-bfcl.sh scores

set -euo pipefail

BFCL_DIR="$HOME/dev/gorilla-bfcl/berkeley-function-call-leaderboard"
BFCL_VENV="$BFCL_DIR/.venv"

export REMOTE_OPENAI_BASE_URL="${VLLM_URL:-http://localhost:8000/v1}"
export REMOTE_OPENAI_API_KEY="${VLLM_KEY:-EMPTY}"
export HF_HOME=/mnt/models/huggingface

source "$BFCL_VENV/bin/activate"
cd "$BFCL_DIR"

CMD="${1:-help}"
MODEL="${2:-google/gemma-4-26B-A4B-it}"
CATEGORY="${3:-non_live}"

case "$CMD" in
    generate)
        bfcl generate \
            --model "$MODEL" \
            --test-category "$CATEGORY" \
            --skip-server-setup \
            --include-input-log \
            --num-threads 4
        ;;
    evaluate)
        bfcl evaluate \
            --model "$MODEL" \
            --test-category "$CATEGORY"
        ;;
    all)
        bfcl generate \
            --model "$MODEL" \
            --test-category "$CATEGORY" \
            --skip-server-setup \
            --include-input-log \
            --num-threads 4
        bfcl evaluate \
            --model "$MODEL" \
            --test-category "$CATEGORY"
        ;;
    scores)
        bfcl scores
        ;;
    *)
        echo "Usage: $0 {generate|evaluate|all|scores} [model] [category]"
        echo ""
        echo "Categories: all_scoring, non_live, live, multi_turn, memory, web_search, agentic"
        echo "Models: google/gemma-4-26B-A4B-it, google/gemma-4-31B-it"
        ;;
esac
