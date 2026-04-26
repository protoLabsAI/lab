#!/usr/bin/env bash
# Upload trained hey_orbis model to HuggingFace
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$EXPERIMENT_DIR/models"
REPO_ID="protoLabsAI/hey-orbis-wakeword"

if [ ! -f "$MODELS_DIR/hey_orbis.onnx" ]; then
    echo "ERROR: No trained model found at $MODELS_DIR/hey_orbis.onnx"
    echo "Run scripts/train.sh first."
    exit 1
fi

echo "=== Publishing hey_orbis to HuggingFace ==="
echo "Repo: $REPO_ID"
echo "Files:"
ls -lh "$MODELS_DIR/"

# Create model card
cat > "$MODELS_DIR/README.md" << 'EOF'
---
tags:
  - wake-word
  - openwakeword
  - voice-assistant
  - orbis
license: apache-2.0
library_name: openwakeword
pipeline_tag: audio-classification
---

# hey-orbis-wakeword

Custom [openWakeWord](https://github.com/dscripka/openWakeWord) model trained to detect the wake phrase **"hey orbis"**.

## Usage

```python
import openwakeword
from openwakeword.model import Model

# Download and load
openwakeword.utils.download_models(["hey_orbis"], target_directory="./models")
model = Model(wakeword_models=["./models/hey_orbis.onnx"])

# Feed 80ms audio frames (1280 samples @ 16kHz)
prediction = model.predict(audio_frame)
print(prediction["hey_orbis"])  # 0.0 - 1.0
```

## Training Details

- **Positive data:** ~5,000 synthetic clips generated with Fish Audio S2 Pro (5 voice clones)
- **Negative data:** ~2,000 hours ACAV100M pre-computed features
- **Augmentation:** MIT room impulse responses + colored noise (via acoustics), 2 rounds
- **Architecture:** DNN classifier (layer_size=32) on frozen Google speech embeddings
- **Target:** ≤0.2 false accepts/hr, ≥0.95 recall

## Part of protoLabs

Built for [ORBIS](https://github.com/protoLabsAI/ORBIS), the protoLabs voice companion.
Blog: [protolabs.studio](https://protolabs.studio)
EOF

# Upload
python3 -c "
from huggingface_hub import HfApi
api = HfApi()

api.create_repo('$REPO_ID', exist_ok=True, private=True)

api.upload_folder(
    folder_path='$MODELS_DIR',
    repo_id='$REPO_ID',
    commit_message='Upload hey_orbis openWakeWord model',
)
print(f'Uploaded to https://huggingface.co/$REPO_ID')
print('Note: repo is PRIVATE — flip to public after blog post')
"
