#!/usr/bin/env bash
# Setup openWakeWord training environment
# Venv on /mnt/data (root drive has <3GB free)
set -euo pipefail

ENV_DIR="/mnt/data/training/wake-word/env"
OWW_DIR="/mnt/data/training/wake-word/openWakeWord"
DATA_DIR="/mnt/data/training/wake-word"

echo "=== Setting up openWakeWord training environment ==="
echo "Venv: $ENV_DIR"
echo "OWW:  $OWW_DIR"

# Create venv
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating venv at $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# Point pip cache to scratch drive to avoid filling root
export PIP_CACHE_DIR="/mnt/scratch/cache/pip"
mkdir -p "$PIP_CACHE_DIR"

# Clone openWakeWord if not present
if [ ! -d "$OWW_DIR" ]; then
    echo "Cloning openWakeWord..."
    git clone https://github.com/dscripka/openWakeWord.git "$OWW_DIR"
fi

# Install openWakeWord with deps
echo "Installing openWakeWord..."
pip install --upgrade pip
pip install -e "$OWW_DIR"

# Training dependencies
# torch/torchaudio: augmentation pipeline
# tensorflow: classifier training
# speechbrain: RIR convolution + audio I/O in data.py
# acoustics/pronouncing: noise generation + phoneme-based adversarial text
# torchinfo/torchmetrics: model summary + training metrics
echo "Installing training dependencies..."
pip install \
    torch torchaudio \
    tensorflow \
    speechbrain \
    pronouncing \
    acoustics \
    audiomentations \
    torch_audiomentations \
    torchinfo \
    torchmetrics \
    "scipy<1.15" \
    mutagen \
    pydub \
    tqdm \
    requests \
    soundfile \
    librosa \
    huggingface_hub \
    pyyaml

# Create data directories
echo "Creating data directories..."
mkdir -p "$DATA_DIR"/{mit_rirs,output}

# Symlink data dir into experiment
EXPERIMENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -L "$EXPERIMENT_DIR/data" ] || [ ! -e "$EXPERIMENT_DIR/data" ]; then
    rm -rf "$EXPERIMENT_DIR/data"
    ln -sf "$DATA_DIR" "$EXPERIMENT_DIR/data"
    echo "Symlinked $EXPERIMENT_DIR/data -> $DATA_DIR"
fi

echo ""
echo "=== Environment ready ==="
echo "Activate with: source $ENV_DIR/bin/activate"
echo "Data directory: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. bash scripts/download_data.sh    # download negative data + RIRs"
echo "  2. python scripts/generate_clips.py  # generate synthetic clips with Fish Audio"
echo "  3. bash scripts/train.sh             # augment + train + export"
