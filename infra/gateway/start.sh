#!/bin/bash
# protoLabs AI Gateway — Secure startup with Infisical secret injection
#
# Usage:
#   ./start.sh              # fetch secrets from Infisical, then start
#   ./start.sh --local      # use local .env file instead (dev only)
#   ./start.sh --stop       # stop the gateway
#
# Requires:
#   - Machine identity credentials at ~/.config/gateway/identity (not in repo)
#   - Infisical instance reachable at INFISICAL_URL
#
# First-time setup:
#   1. Create a Machine Identity in Infisical UI (Universal Auth)
#   2. Scope it to the gateway project with 'reader' role on /prod/
#   3. Run: ./start.sh --setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IDENTITY_DIR="${HOME}/.config/gateway"
IDENTITY_FILE="${IDENTITY_DIR}/identity"
INFISICAL_URL="${INFISICAL_URL:-https://secrets.proto-labs.ai}"

# ─── Colors ───────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[ok]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
fail() { echo -e "${RED}[fail]${NC} $1"; }

# ─── Commands ─────────────────────────────────────────────

cmd_stop() {
    cd "$SCRIPT_DIR"
    docker compose down
    ok "Gateway stopped"
}

cmd_setup() {
    echo "=== Gateway Identity Setup ==="
    echo ""
    echo "Create a Machine Identity in Infisical:"
    echo "  1. Go to ${INFISICAL_URL} > Organization > Machine Identities"
    echo "  2. Create identity 'gateway-ava-ai' with Universal Auth"
    echo "  3. Attach to gateway project with 'reader' role"
    echo ""

    mkdir -p "$IDENTITY_DIR"
    chmod 700 "$IDENTITY_DIR"

    read -rp "Client ID: " client_id
    read -rsp "Client Secret: " client_secret
    echo ""

    read -rp "Project ID: " project_id
    read -rp "Environment [prod]: " env_name
    env_name="${env_name:-prod}"

    cat > "$IDENTITY_FILE" <<IDEOF
# Gateway Machine Identity — DO NOT COMMIT
# Generated $(date -Iseconds)
INFISICAL_CLIENT_ID=${client_id}
INFISICAL_CLIENT_SECRET=${client_secret}
INFISICAL_PROJECT_ID=${project_id}
INFISICAL_ENV=${env_name}
IDEOF

    chmod 600 "$IDENTITY_FILE"
    ok "Identity saved to ${IDENTITY_FILE} (mode 600)"
    echo ""
    echo "Test with: ./start.sh"
}

cmd_start_infisical() {
    # Load machine identity
    if [ ! -f "$IDENTITY_FILE" ]; then
        fail "No identity file at ${IDENTITY_FILE}"
        echo "Run: $0 --setup"
        exit 1
    fi

    # Source identity (contains CLIENT_ID, CLIENT_SECRET, PROJECT_ID, ENV)
    # shellcheck disable=SC1090
    source "$IDENTITY_FILE"

    echo "Fetching secrets from Infisical..."

    # Get a token using universal auth
    INFISICAL_TOKEN=$(infisical login \
        --method=universal-auth \
        --client-id="$INFISICAL_CLIENT_ID" \
        --client-secret="$INFISICAL_CLIENT_SECRET" \
        --domain="${INFISICAL_URL}/api" \
        --silent --plain 2>/dev/null) || {
        fail "Could not authenticate with Infisical at ${INFISICAL_URL}"
        echo "Is Infisical running? Check: curl ${INFISICAL_URL}/api/status"
        exit 1
    }

    ok "Authenticated with Infisical"

    # Export secrets and start docker compose
    cd "$SCRIPT_DIR"
    INFISICAL_TOKEN="$INFISICAL_TOKEN" \
    infisical run \
        --projectId="$INFISICAL_PROJECT_ID" \
        --env="$INFISICAL_ENV" \
        --token="$INFISICAL_TOKEN" \
        --domain="${INFISICAL_URL}/api" \
        -- docker compose up -d

    ok "Gateway started with Infisical secrets"
    echo ""
    echo "  API:    http://localhost:4000/v1"
    echo "  Admin:  http://localhost:4000/ui"
}

cmd_start_local() {
    cd "$SCRIPT_DIR"
    if [ ! -f .env ]; then
        warn "No .env file found — copying from .env.example"
        cp .env.example .env
        warn "Edit .env with your API keys before using cloud providers"
    fi
    docker compose up -d
    ok "Gateway started with local .env"
    echo ""
    echo "  API:    http://localhost:4000/v1"
    echo "  Admin:  http://localhost:4000/ui"
}

# ─── Main ─────────────────────────────────────────────────

case "${1:-}" in
    --stop)
        cmd_stop
        ;;
    --setup)
        cmd_setup
        ;;
    --local)
        cmd_start_local
        ;;
    --help|-h)
        echo "Usage: $0 [--local|--stop|--setup|--help]"
        echo ""
        echo "  (no args)   Fetch secrets from Infisical, start gateway"
        echo "  --local     Use local .env file (dev only)"
        echo "  --stop      Stop the gateway"
        echo "  --setup     Configure Infisical machine identity"
        echo "  --help      Show this help"
        ;;
    *)
        cmd_start_infisical
        ;;
esac
