#!/usr/bin/env bash
# Symlink the tuned MoE kernel configs into the vLLM venv's
# fused_moe/configs/ dir. Run this once after a fresh vLLM install or
# upgrade — symlinks survive in-place edits but get clobbered when pip
# replaces the configs directory wholesale.
#
# Usage: bash install-moe-configs.sh
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/moe-configs" && pwd)"
DST_DIR="${VLLM_VENV:-$HOME/dev/vllm-env}/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs"

if [ ! -d "$DST_DIR" ]; then
    echo "vLLM configs dir not found at: $DST_DIR" >&2
    echo "Set VLLM_VENV to override (e.g. VLLM_VENV=/path/to/venv bash $0)" >&2
    exit 1
fi

count=0
for src in "$SRC_DIR"/*.json; do
    [ -f "$src" ] || continue
    name="$(basename "$src")"
    rm -f "$DST_DIR/$name"
    ln -s "$src" "$DST_DIR/$name"
    count=$((count + 1))
done

echo "linked $count MoE config(s) into $DST_DIR"
