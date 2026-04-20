#!/bin/bash
# Download LibriSpeech train sets and run attribute extraction sequentially
# Run on GPU 1 while training continues on GPU 0
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash run_extraction_pipeline.sh

set -euo pipefail

DATA_DIR="/mnt/data/salm-duplex/data"
OUTPUT_DIR="/mnt/data/salm-duplex/data"
EXTRACT_SCRIPT="$HOME/dev/lab/experiments/salm-duplex/extract_attributes.py"
VENV="$HOME/dev/moe-train-env"

source "$VENV/bin/activate"

download_and_extract() {
    local name=$1
    local url=$2
    local flat_dir="${DATA_DIR}/${name}-flat"

    echo "=== $(date): Downloading ${name} ==="
    cd "$DATA_DIR"
    if [ ! -f "${name}.tar.gz" ]; then
        wget -q --show-progress "$url" -O "${name}.tar.gz"
    else
        echo "  Already downloaded"
    fi

    echo "  Extracting..."
    tar xzf "${name}.tar.gz"

    echo "  Linking to flat dir..."
    mkdir -p "$flat_dir"
    find "$DATA_DIR/LibriSpeech/${name}" -name "*.flac" -exec ln -sf {} "$flat_dir/" \;
    local count=$(ls "$flat_dir"/*.flac 2>/dev/null | wc -l)
    echo "  ${count} files linked"

    echo "  Running attribute extraction..."
    python "$EXTRACT_SCRIPT" \
        --audio-dir "$flat_dir" \
        --output "${OUTPUT_DIR}/${name}-attributes.jsonl" \
        --skip-llm

    local lines=$(wc -l < "${OUTPUT_DIR}/${name}-attributes.jsonl")
    echo "=== $(date): ${name} complete — ${lines} samples ==="
    echo ""
}

# train-clean-100 (~6.3GB, ~28K utterances, ~100 hours)
download_and_extract "train-clean-100" "https://www.openslr.org/resources/12/train-clean-100.tar.gz"

# train-clean-360 (~23GB, ~105K utterances, ~360 hours)
download_and_extract "train-clean-360" "https://www.openslr.org/resources/12/train-clean-360.tar.gz"

# train-other-500 (~30GB, ~149K utterances, ~500 hours) — noisier, good for robustness
download_and_extract "train-other-500" "https://www.openslr.org/resources/12/train-other-500.tar.gz"

echo "=== ALL DONE ==="
echo "Total extracted:"
wc -l "${OUTPUT_DIR}"/*-attributes.jsonl
