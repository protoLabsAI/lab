# wake-word — "hey orbis" Experiment Log

## v0 (2026-04-27, openWakeWord + Fish Audio S2 Pro)

**Goal.** Train a custom wake word model for "hey orbis" using
openWakeWord's pipeline, replacing Piper TTS with Fish Audio S2 Pro
for higher-quality synthetic speech. Deploy via ONNX/TFLite for
real-time streaming inference in ORBIS, protoVoice, and Home Assistant.

### Architecture

```
mic audio (16kHz int16)
  → melspectrogram (frozen ONNX, 80ms frames)
  → Google speech embeddings (frozen ONNX, 96-dim)
  → classifier (trained, tiny DNN)
  → sigmoid [0.0–1.0] per frame
```

The classifier takes 16 consecutive embedding frames (~1.3s window)
and outputs a single wake word probability. Everything upstream is
frozen — only the classifier head is trained.

### Data pipeline

| Data type | Source | Count |
|-----------|--------|------:|
| Positive (synthetic) | Fish Audio S2 Pro, 5 voices × 5 temp/top_p combos | ~7,000 clips |
| Adversarial negative | Fish Audio, phonetically similar phrases | ~2,000 clips |
| Generic negative | ACAV100M pre-computed features (downloaded) | ~2,000 hrs |
| Room impulse responses | MIT environmental impulse responses | 270 rooms |
| Background noise | Colored noise via `acoustics` (generated during augmentation) | synthetic |

Fish Audio S2 Pro (already running at `:8092` for ORBIS TTS) replaced
Piper TTS for voice diversity and prosody control. 5 saved voice
references with varied temperature (0.5–0.9) and top_p (0.5–0.8)
produced natural variation across clips.

### Training

- **Steps:** 50,000 training + 5,000 validation + 5,000 false-positive
  validation
- **Features:** 2 augmentation rounds per clip (RIR convolution +
  background noise + colored noise), CPU-only extraction at ~3.3
  clips/s
- **Export:** ONNX (opset 18) + TFLite (float32 + float16)
- **Wall time:** ~10 min total (feature extraction ~6 min, training
  ~4 min on CPU)

### Model artifacts

| Format | Path | Size |
|--------|------|-----:|
| ONNX | `output/hey_orbis.onnx` | 199 KB |
| TFLite float32 | `output/hey_orbis_tflite/hey_orbis_float32.tflite` | 203 KB |
| TFLite float16 | `output/hey_orbis_tflite/hey_orbis_float16.tflite` | 105 KB |

Input: `[1, 16, 96]` (16 frames × 96-dim embeddings).
Output: `[1, 1]` sigmoid probability.

---

### Results — pre-extracted features (direct ONNX inference)

Evaluating the ONNX model directly on the augmented feature `.npy`
files (the exact representations it was trained on):

| Set | N | Detected @0.5 | Rate | Mean | Median | Min | Max |
|-----|--:|:-------------:|-----:|-----:|-------:|----:|----:|
| **Positive train** | 4,000 | 3,111 | **77.8%** | 0.765 | 0.983 | 0.000 | 0.993 |
| **Positive test** | 1,000 | 767 | **76.7%** | 0.754 | 0.982 | 0.000 | 0.993 |
| **Negative train** | 1,600 | 0 | **0.0%** | 0.000 | 0.000 | 0.000 | 0.004 |
| **Negative test** | 400 | 15 | **3.8%** | 0.038 | 0.000 | 0.000 | 0.985 |

Positive test score distribution:

| Threshold | Count | Rate |
|----------:|------:|-----:|
| ≥ 0.1 | 809/1000 | 80.9% |
| ≥ 0.3 | 783/1000 | 78.3% |
| ≥ 0.5 | 767/1000 | 76.7% |
| ≥ 0.7 | 751/1000 | 75.1% |
| ≥ 0.9 | 713/1000 | 71.3% |

The bimodal score distribution (scores cluster near 0.0 or 0.98+)
means threshold selection between 0.3–0.7 has minimal impact on
recall. The ~23% of positive samples scoring near 0 likely correspond
to heavily augmented clips where the wake word became unintelligible.

### Results — streaming audio validation (real `.wav` clips)

The more meaningful test: feed raw `.wav` clips through openWakeWord's
full inference pipeline (`Model.predict()`) in a simulated streaming
context (2s silence → clip → 1s silence, 80ms chunks):

| Set | N | Detected @0.5 | Rate | Mean | Median | Min | Max |
|-----|--:|:-------------:|-----:|-----:|-------:|----:|----:|
| **Positive test** | 1,000 | 924 | **92.4%** | 0.898 | 0.982 | 0.000 | 0.992 |
| **Negative test** | 400 | 27 | **6.8%** | 0.068 | 0.000 | 0.000 | 0.987 |
| **Positive train** | 100 | 89 | **89.0%** | — | — | — | — |

Positive test distribution (streaming):

| Threshold | Count | Rate |
|----------:|------:|-----:|
| ≥ 0.1 | 951/1000 | 95.1% |
| ≥ 0.3 | 937/1000 | 93.7% |
| ≥ 0.5 | 924/1000 | 92.4% |
| ≥ 0.7 | 909/1000 | 90.9% |
| ≥ 0.9 | 854/1000 | 85.4% |

Negative test distribution (streaming):

| Threshold | Count | Rate |
|----------:|------:|-----:|
| ≥ 0.1 | 39/400 | 9.8% |
| ≥ 0.3 | 30/400 | 7.5% |
| ≥ 0.5 | 27/400 | 6.8% |
| ≥ 0.7 | 22/400 | 5.5% |
| ≥ 0.9 | 16/400 | 4.0% |

Streaming recall is **higher** than pre-extracted features (92.4% vs
76.7%) because the streaming pipeline applies the same embedding model
end-to-end with natural temporal context, while pre-extracted features
include heavily-augmented distorted versions.

---

### Ablation study

Trained 5 variants modifying `max_negative_weight` (controls FA
penalty during training) and `layer_size` (model capacity). All
evaluated on the full 1000 positive + 400 negative test clips in
streaming mode. Raw results persisted in
`/mnt/data/training/wake-word/ablation_results/ablation_results.json`.

| Name | Rcl@.5 | FA@.5 | Rcl@.7 | FA@.7 | Rcl@.9 | FA@.9 | Size |
|------|:------:|:-----:|:------:|:-----:|:------:|:-----:|-----:|
| **v0-baseline** (neg1500) | **94.4%** | 7.2% | **92.3%** | 5.8% | **89.3%** | 4.8% | 199KB |
| v0-original (pre-patch) | 92.4% | 6.8% | 90.9% | 5.5% | 86.0% | 4.0% | 199KB |
| v1-neg3000 | 89.6% | 6.5% | 88.2% | 5.2% | — | — | 14KB |
| v2-neg6000 | 83.2% | **4.5%** | 81.3% | **4.5%** | — | — | 14KB |
| v3-layer64 | 94.9% | 8.8% | 94.1% | 7.2% | 92.0% | 6.2% | 15KB |
| v4-layer64-neg3000 | 94.6% | 8.5% | 93.6% | 8.0% | 91.1% | 6.0% | 15KB |

**Findings:**

1. **`max_negative_weight` is the only lever that moves FA rate** —
   but it trades recall 1:1. neg6000 achieves 4.5% FA but drops
   recall to 83.2%. The score distribution is bimodal (near 0 or
   near 1), so threshold tuning moves you along the curve cheaply.

2. **Larger model (layer64) makes FA worse.** More capacity memorizes
   adversarial negatives as positives. The 32-dim default is right-sized.

3. **`model.eval()` patch matters.** v0-baseline (with patch) beats
   v0-original (without) by 2.0% recall at @0.5 and 3.3% at @0.9.
   BatchNorm running stats are meaningful even at this scale.

### Threshold operating curve (v0-baseline, full test set)

1000 positive + 400 adversarial negative clips, streaming mode:

| Threshold | Recall | FA Rate | Precision | F1 |
|:---------:|:------:|:-------:|:---------:|:--:|
| 0.30 | 95.5% | 9.2% | 96.3% | 95.9 |
| 0.50 | 94.4% | 7.2% | 97.0% | 95.7 |
| 0.70 | 92.3% | 5.8% | 97.6% | 94.9 |
| **0.80** | **90.9%** | **5.0%** | **97.8%** | **94.2** |
| 0.90 | 89.3% | 4.8% | 97.9% | 93.4 |
| 0.95 | 85.6% | 3.0% | 98.6% | 91.6 |

**Recommended production config:** threshold=0.80 + patience=3 in
streaming. Gives 90.9% recall / 5.0% FA on adversarial negatives.
Real-world FA will be lower since adversarials are intentionally
confusable phrases.

---

### Summary

| Metric | v0-baseline @0.5 | v0-baseline @0.8 |
|--------|:----------------:|:----------------:|
| **Recall (streaming)** | **94.4%** | **90.9%** |
| **FA rate (adversarial)** | 7.2% | **5.0%** |
| Model size (ONNX) | 199 KB | — |
| Model size (TFLite fp16) | 105 KB | — |
| Input window | ~1.3s (16 × 80ms) | — |

### Bugs hit during training

1. **`model.eval()` not called before ONNX export** — openwakeword's
   `train.py` exports the model in training mode (BatchNorm uses
   batch stats instead of running stats). Patched line 428 to add
   `model_to_save.eval()` before `torch.onnx.export()`.

2. **Streaming test produced all-zero scores** — initial test fed
   short isolated clips (~0.7–1.2s) directly through `Model.predict()`.
   The 16-frame embedding buffer never filled. Root cause: clips are
   shorter than the model's ~1.3s input window. Fix: embed clips in
   a longer audio stream with leading silence (simulating real
   microphone use). This is how the model is designed to operate —
   continuous streaming, not isolated clip classification.

3. **`onnxscript` missing** — first training run completed but ONNX
   export crashed. No checkpoint saved (model only in memory). Full
   re-run required after `pip install onnxscript`.

4. **Partial feature files from crashed run** — second run detected
   existing `positive_features_train.npy` from the crashed first run
   and skipped augmentation. Delete partial files before re-running.

5. **`onnx_tf` not installed** — openWakeWord's TFLite conversion
   uses `onnx-tf`. Used `onnx2tf` as drop-in replacement.

### Limitations and next steps

1. **All test data is synthetic.** Both positive and negative clips
   were generated by Fish Audio S2 Pro. Real-world performance
   (actual humans saying "hey orbis" into a microphone, in varied
   acoustic environments) is untested. Record a small held-out set
   of real speech before trusting these numbers for production.

2. **6.8% FA rate on adversarial negatives** is acceptable for a
   first model but leaves room for improvement. The `patience`
   parameter in openWakeWord (requiring N consecutive frames above
   threshold) can reduce FA at the cost of latency. Recommended:
   `patience={"hey_orbis": 3}` for production.

3. **Negative test set is small (400 clips).** FA rate estimate has
   wide confidence interval. Real-world FA rate against ambient
   speech/music/noise will differ.

4. **No comparison baseline.** Unlike audio-tags (which had majority
   class + linear probe + v0→v5 progression), this is a single-shot
   training run. Useful ablations: vary clip count (1k vs 3k vs 7k),
   augmentation rounds (1 vs 2 vs 4), training steps (25k vs 50k vs
   100k).

### Deployment

**For ORBIS (pipecat):**
```python
from openwakeword.model import Model

model = Model(
    wakeword_models=["path/to/hey_orbis.onnx"],
    inference_framework="onnx"
)

# In audio callback (80ms chunks, 1280 samples at 16kHz):
prediction = model.predict(audio_chunk)
if prediction["hey_orbis"] > 0.7:
    # Wake word detected
    ...
```

**For Home Assistant:**
Copy `hey_orbis.tflite` to the openWakeWord add-on's custom model
directory.

**Publish target:** `protoLabsAI/hey-orbis-wakeword` on HuggingFace.
