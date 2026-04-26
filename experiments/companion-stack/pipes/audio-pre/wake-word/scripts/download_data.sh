#!/usr/bin/env bash
# Download negative training data, validation features, and RIRs
# Stored on /mnt/data/training/wake-word/ (NVMe)
set -euo pipefail

DATA_DIR="/mnt/data/training/wake-word"
mkdir -p "$DATA_DIR"

echo "=== Downloading openWakeWord training data ==="
echo "Target: $DATA_DIR"
echo ""

# 1. ACAV100M pre-computed negative features (~2000 hrs, ~17GB)
ACAV_FILE="$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
if [ ! -f "$ACAV_FILE" ]; then
    echo "[1/3] Downloading ACAV100M features (~17GB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='davidscripka/openwakeword_features',
    filename='openwakeword_features_ACAV100M_2000_hrs_16bit.npy',
    repo_type='dataset',
    local_dir='$DATA_DIR',
)
print('Done: ACAV100M features')
"
else
    echo "[1/3] ACAV100M features already downloaded, skipping"
fi

# 2. Validation features (~11 hrs, ~177MB)
VAL_FILE="$DATA_DIR/validation_set_features.npy"
if [ ! -f "$VAL_FILE" ]; then
    echo "[2/3] Downloading validation features (~177MB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='davidscripka/openwakeword_features',
    filename='validation_set_features.npy',
    repo_type='dataset',
    local_dir='$DATA_DIR',
)
print('Done: validation features')
"
else
    echo "[2/3] Validation features already downloaded, skipping"
fi

# 3. MIT Room Impulse Responses (270 rooms, 16kHz)
RIR_DIR="$DATA_DIR/mit_rirs"
if [ ! -d "$RIR_DIR/16khz" ] || [ -z "$(ls -A "$RIR_DIR/16khz" 2>/dev/null)" ]; then
    echo "[3/3] Downloading MIT environmental impulse responses..."
    mkdir -p "$RIR_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='davidscripka/MIT_environmental_impulse_responses',
    repo_type='dataset',
    local_dir='$RIR_DIR',
)
print('Done: MIT RIRs')
"
else
    echo "[3/3] MIT RIRs already downloaded, skipping"
fi

# Background noise is generated synthetically during augmentation via
# the acoustics library (colored noise). No external audio files needed.

echo ""
echo "=== All data downloaded ==="
du -sh "$DATA_DIR"/*
echo ""
echo "Total:"
du -sh "$DATA_DIR"
