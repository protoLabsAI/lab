#!/bin/bash
# protoLabs Evals — Run with Infisical secret injection
#
# Usage:
#   ./run.sh profile --name smoke                              # ~5 min sanity check
#   ./run.sh profile --name quick                              # ~15 min pre-merge
#   ./run.sh profile --name full                               # ~60-90 min release gate
#   ./run.sh claw --tasks T01,T02                              # specific claw tasks
#   ./run.sh custom --suite coding                             # single custom suite
#   ./run.sh function-call --all-suites                        # all FC suites
#   ./run.sh profile --name quick --model protolabs/fast       # test a different model
#   ./run.sh wildbench infer                                   # generate responses (1,024 tasks)
#   ./run.sh wildbench judge --results X --judge gpt-5.4       # score with cloud judge (~$5)
#   ./run.sh compare results/model_a results/model_b
#   ./run.sh --setup       # configure machine identity (first time)
#   ./run.sh --local ...   # use local .env instead of Infisical
#
# Secrets injected:
#   GATEWAY_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IDENTITY_FILE="${HOME}/.config/gateway/identity"  # shared with gateway
INFISICAL_URL="${INFISICAL_URL:-https://secrets.proto-labs.ai}"

# ─── Colors ───────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[ok]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
fail() { echo -e "${RED}[fail]${NC} $1"; }

# ─── Activate venv ────────────────────────────────────────
activate_venv() {
    if [ ! -d "$SCRIPT_DIR/venv" ]; then
        fail "No venv found. Run: ./scripts/setup.sh"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/venv/bin/activate"
}

# ─── Run with Infisical ──────────────────────────────────
run_with_infisical() {
    if [ ! -f "$IDENTITY_FILE" ]; then
        fail "No identity file at ${IDENTITY_FILE}"
        echo "Run: ./run.sh --setup"
        echo "Or use the gateway setup: cd ~/dev/gateway && ./start.sh --setup"
        exit 1
    fi

    # shellcheck disable=SC1090
    source "$IDENTITY_FILE"

    INFISICAL_TOKEN=$(infisical login \
        --method=universal-auth \
        --client-id="$INFISICAL_CLIENT_ID" \
        --client-secret="$INFISICAL_CLIENT_SECRET" \
        --domain="${INFISICAL_URL}/api" \
        --silent --plain 2>/dev/null) || {
        fail "Could not authenticate with Infisical at ${INFISICAL_URL}"
        exit 1
    }

    ok "Authenticated with Infisical"

    activate_venv

    cd "$SCRIPT_DIR"
    INFISICAL_TOKEN="$INFISICAL_TOKEN" \
    infisical run \
        --projectId="$INFISICAL_PROJECT_ID" \
        --env="$INFISICAL_ENV" \
        --token="$INFISICAL_TOKEN" \
        --domain="${INFISICAL_URL}/api" \
        -- python -m runners.cli "$@"
}

# ─── Run with local .env ─────────────────────────────────
run_local() {
    if [ -f "$SCRIPT_DIR/.env" ]; then
        set -a
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/.env"
        set +a
    else
        warn "No .env file — secrets must be set in environment"
    fi

    activate_venv
    cd "$SCRIPT_DIR"
    python -m runners.cli "$@"
}

# ─── Setup (reuse gateway identity) ──────────────────────
cmd_setup() {
    if [ -f "$IDENTITY_FILE" ]; then
        ok "Identity already configured at ${IDENTITY_FILE} (shared with gateway)"
        echo "To reconfigure, run: cd ~/dev/gateway && ./start.sh --setup"
        return
    fi

    echo "=== Evals Identity Setup ==="
    echo ""
    echo "The evals suite shares the gateway's Infisical identity."
    echo "Run the gateway setup first:"
    echo ""
    echo "  cd ~/dev/gateway && ./start.sh --setup"
    echo ""
    echo "This creates ${IDENTITY_FILE} which both gateway and evals use."
}

# ─── Main ─────────────────────────────────────────────────
case "${1:-}" in
    --setup)
        cmd_setup
        ;;
    --local)
        shift
        run_local "$@"
        ;;
    --help|-h)
        echo "Usage: $0 [--local|--setup|--help] <command> [args...]"
        echo ""
        echo "Commands (passed to proto-eval CLI):"
        echo "  claw      Run claw-eval benchmark"
        echo "  custom    Run custom eval tasks"
        echo "  compare   Compare results across models"
        echo ""
        echo "Options:"
        echo "  --local   Use local .env instead of Infisical"
        echo "  --setup   Configure Infisical identity"
        echo "  --help    Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 claw --model local --tasks T01,T02"
        echo "  $0 custom --suite tool_use --model local --submit-langfuse"
        echo "  $0 --local custom --suite tool_use --model local"
        ;;
    *)
        run_with_infisical "$@"
        ;;
esac
