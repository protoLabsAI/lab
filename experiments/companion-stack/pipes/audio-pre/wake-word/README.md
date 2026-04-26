# wake-word — Custom "hey orbis" openWakeWord Model

**Goal.** Train a custom wake word detection model for "hey orbis" using
openWakeWord's training pipeline with Fish Audio S2 Pro for high-quality
synthetic speech generation. Publish to HuggingFace for deployment across
all protoLabs systems (ORBIS, protoVoice, Home Assistant).

---

## Architecture

openWakeWord uses a frozen Google speech embedding model as a feature
extractor, with a tiny classification head (~few hundred KB) trained on
top. Audio is processed in 80ms frames, outputting a 0–1 confidence score.

```
mic audio → melspectrogram (ONNX) → speech embeddings (frozen) → classifier (trained) → 0.0–1.0
```

The classifier is what we train. Everything else is pre-trained and frozen.

## Data Pipeline

| Data Type | Source | Count |
|-----------|--------|-------|
| **Positive (synthetic)** | Fish Audio S2 Pro, 5 voices + varied prosody | ~5,000 clips |
| **Adversarial negative** | Fish Audio, phonetically similar phrases | ~2,000 clips |
| **Generic negative** | ACAV100M pre-computed features | ~2,000 hours |
| **Background noise** | Colored noise via `acoustics` (generated during augmentation) | synthetic |
| **Room impulse responses** | MIT environmental impulse responses | 270 rooms |

### Why Fish Audio instead of Piper TTS

The standard openWakeWord pipeline uses Piper TTS (libritts_r-medium).
We replace it with Fish Audio S2 Pro (4.4B params, already running on
`:8092`) because:

- **Voice diversity** — 5 saved voice references + varied prosody/temperature
- **Quality** — 0.40 RTF with `--half --compile`, 44.1kHz native
- **Prosody control** — 15,000+ inline control tags for varied emphasis
- **Already deployed** — running as Docker sidecar in protoVoice stack

## Target Metrics

- False-accept rate: ≤ 0.2 per hour
- False-reject rate: < 5% (recall ≥ 0.95)
- Inference latency: < 5ms per frame

## Deployment

Published to `protoLabsAI/hey-orbis-wakeword` on HuggingFace.
Consumed via openWakeWord runtime in:
- **ORBIS** — pipecat wake word filter
- **protoVoice** — pipecat wake phrase strategy
- **Home Assistant** — openWakeWord add-on

## Structure

```
wake-word/
├── README.md           # this file
├── RESULTS.md          # training results and metrics (post-training)
├── configs/
│   └── hey_orbis.yml   # openWakeWord training config
├── scripts/
│   ├── setup_env.sh         # environment + dependency setup
│   ├── download_data.sh     # negative data + RIRs download
│   ├── generate_clips.py    # Fish Audio synthetic data generation
│   ├── train.sh             # augment → train → export (end-to-end)
│   └── publish.sh           # upload model + card to HuggingFace
├── models/             # trained model outputs (ONNX, tflite)
└── data/               # gitignored, symlink to /mnt/data/training/wake-word/
```
