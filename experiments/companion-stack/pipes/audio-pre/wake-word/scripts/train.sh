#!/usr/bin/env bash
# End-to-end training pipeline for "hey orbis" wake word model
#
# Prerequisites:
#   bash scripts/setup_env.sh
#   bash scripts/download_data.sh
#
# Pipeline:
#   1. Generate clips with Fish Audio → {output_dir}/{model_name}/positive_train/ etc.
#   2. Augment clips → extract features → .npy files
#   3. Train classifier → export ONNX + tflite
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="/mnt/data/training/wake-word/env"
OWW_DIR="/mnt/data/training/wake-word/openWakeWord"
DATA_DIR="/mnt/data/training/wake-word"
CONFIG="$EXPERIMENT_DIR/configs/hey_orbis.yml"

source "$ENV_DIR/bin/activate"

echo "=== hey orbis — openWakeWord Training Pipeline ==="
echo "Config: $CONFIG"
echo "Data:   $DATA_DIR"
echo "OWW:    $OWW_DIR"
echo ""

# Step 1: Generate synthetic clips with Fish Audio (if not already done)
CLIPS_DIR="$DATA_DIR/output/hey_orbis"
if [ ! -d "$CLIPS_DIR/positive_train" ] || [ -z "$(ls -A "$CLIPS_DIR/positive_train" 2>/dev/null)" ]; then
    echo "=== Step 1: Generating synthetic clips with Fish Audio ==="
    python "$SCRIPT_DIR/generate_clips.py" \
        --n-positive 5000 \
        --n-adversarial 2000 \
        --output-dir "$DATA_DIR/output" \
        --model-name hey_orbis
else
    echo "=== Step 1: Synthetic clips already exist, skipping ==="
    echo "  positive_train: $(ls "$CLIPS_DIR/positive_train" | wc -l) clips"
    echo "  positive_test:  $(ls "$CLIPS_DIR/positive_test" | wc -l) clips"
    echo "  negative_train: $(ls "$CLIPS_DIR/negative_train" | wc -l) clips"
    echo "  negative_test:  $(ls "$CLIPS_DIR/negative_test" | wc -l) clips"
fi

# Step 2: Augment clips + extract features
echo ""
echo "=== Step 2: Augmenting clips + extracting features ==="
python "$OWW_DIR/openwakeword/train.py" \
    --training_config "$CONFIG" \
    --augment_clips

# Step 3: Train the model
echo ""
echo "=== Step 3: Training model ==="
python "$OWW_DIR/openwakeword/train.py" \
    --training_config "$CONFIG" \
    --train_model

# Step 4: Copy outputs to experiment dir
echo ""
echo "=== Step 4: Collecting outputs ==="
mkdir -p "$EXPERIMENT_DIR/models"

for ext in onnx tflite; do
    model_file="$DATA_DIR/output/hey_orbis.$ext"
    if [ -f "$model_file" ]; then
        cp "$model_file" "$EXPERIMENT_DIR/models/"
        echo "  Copied: hey_orbis.$ext ($(du -h "$model_file" | cut -f1))"
    fi
done

echo ""
echo "=== Training complete ==="
echo "Models saved to: $EXPERIMENT_DIR/models/"
ls -lh "$EXPERIMENT_DIR/models/"
