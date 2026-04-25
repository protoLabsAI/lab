# audio-tags — tiny audio-understanding model for ORBIS

**Goal.** A small, fast model that turns the same audio frames Whisper-STT
sees into a structured JSON tag dict (mood, energy, speaker traits,
acoustic environment) for injection into ORBIS's LLM context. Drop-in,
sub-50 ms additional latency, runs alongside ORBIS's existing Whisper
pass on the same GPU.

This is a **classification** experiment, not generative. The Phase 1 SALM
work (Qwen3.5-4B + Canary + adapter) is the inspiration; this is its
distilled, ORBIS-shaped descendant.

---

## ORBIS integration target

ORBIS pipeline (per its README):

```
Browser  ──WebRTC──▶  Pipecat: STT (Whisper) ──▶  ORBIS LLM ──▶  TTS  ──▶  Browser
                                                       │
                                                       └──▶ SQLite (mood, facts)
```

We tap the audio frame just upstream of (or in parallel with) the Whisper
call, run the tag model, then inject results in two places:

1. **Pre-LLM context message** — a single line prepended to user input,
   e.g. `[audio_context: mood=warm, energy=low, environment=quiet_indoor,
   speaker=female_adult]`.
2. **`mood` table writer** — feed valence/arousal predictions into
   ORBIS's existing short-term mood state.

Wire format: an OpenAI-style HTTP service (`POST /tag`) takes raw PCM
bytes (or a base64 wav), returns:

```json
{
  "mood": {"valence": 0.62, "arousal": 0.31, "label": "calm_positive"},
  "energy": "low",
  "speaker": {"gender": "female", "approx_age": "adult"},
  "acoustic": {"environment": "indoor_quiet", "snr_db": 38.2,
               "volume": "normal", "pitch": "medium",
               "speaking_speed": "normal"},
  "confidence": {"mood": 0.81, "speaker.gender": 0.94, ...}
}
```

---

## Architecture

```
audio (16 kHz PCM, 1-30 s window)
  ↓
Whisper-tiny encoder (frozen, ~10M params, 384-dim hidden)
  ↓
mean-pool over time  (and/or attention-pool variant)
  ↓
shared 256-dim trunk
  ↓
multi-head outputs:
  ├─ mood_class       (Linear → softmax over 7 classes)
  ├─ valence          (Linear → tanh)        ── regression
  ├─ arousal          (Linear → tanh)        ── regression
  ├─ gender           (Linear → softmax 3)
  ├─ age_class        (Linear → softmax 4)
  ├─ environment      (Linear → softmax ~6)
  ├─ volume           (Linear → softmax 3)
  ├─ pitch            (Linear → softmax 3)
  ├─ speaking_speed   (Linear → softmax 3)
  └─ snr_db           (Linear)               ── regression
```

**Why Whisper-tiny encoder, not Canary.** ORBIS already loads Whisper.
Whisper-tiny is 39 M params total (encoder ~10 M); we attach heads to
the existing forward pass — zero marginal model load cost in the ORBIS
process. Canary-1b-flash would be 100× larger and is overkill for
classification. We can revisit larger backbones (Whisper-base, Distil-
Whisper, our existing Canary v2 features) as ablations.

**Why frozen encoder v0.** Fastest to iterate, and we're hypothesizing
that pre-trained Whisper representations already discriminate these
attributes. If heads underfit, unfreeze last 2 encoder layers (LoRA or
full FT) as v1.

---

## Data

**315 k LibriSpeech samples** with rule-extracted attributes already on
disk at `/mnt/data/salm-duplex/data/*-attributes.jsonl`:

| Source | Samples | Hours |
|---|---:|---:|
| train-clean-100 | 28,539 | ~100 |
| train-clean-360 | 104,014 | ~360 |
| train-other-500 | 148,688 | ~500 |
| test-clean (held-out) | 2,620 | ~5 |
| subset-31k (mixed) | 31,159 | ~100 |

**Existing labels (from `extract_attributes.py` rule pass):**
- `volume` ∈ {normal, loud}  ← no "quiet" class on LibriSpeech (clean recordings)
- `pitch` ∈ {low, medium, high}  ← severe imbalance; "low" is 0.2% of data
- `speaking_speed` ∈ {slow, normal, fast}
- `snr_db`, `rms`, `duration`, `sample_rate`

**Missing — must mine or label fresh:**
- mood / valence / arousal
- gender, age class
- acoustic environment

**Mining strategy for the missing tags** (pick per-tag based on signal):

1. **Description prose extraction.** The 315 k samples also have rich
   DeSTA2-generated descriptions (`/mnt/data/salm-duplex/manifests/full-960h-described/`)
   with prose like *"clear, steady tone"*, *"educational narration"*, *"high
   signal-to-noise ratio"*. A one-shot Qwen extraction pass over each
   description → structured tags. Cheap and reuses prior compute.
2. **Off-the-shelf classifiers as ground truth.** Run `emotion2vec`
   (emotion), `speechbrain/spkrec-xvect-voxceleb` style gender model, or
   a pre-trained age classifier over the audio once. These become the
   training targets; our small student then matches them at a fraction
   of the cost.
3. **Hybrid.** Best per attribute. Gender → off-the-shelf classifier
   (high confidence). Environment → description prose. Mood → either,
   but lean on emotion2vec for v0 since LibriSpeech audiobook prose
   is mood-flat.

**Caveat — LibriSpeech is read audiobook speech.** Mood/energy
distributions will be narrow, skewed toward calm/neutral. For the ORBIS
use case (conversational, varied affect), we need to add either:
- **MELD / IEMOCAP / RAVDESS** (small but actually emotional)
- **DailyTalk** (already on disk, conversational)
- **Common Voice** (varied speakers, accents — gender labels included)

LibriSpeech is the volume substrate; affect-rich corpora are the
finetune diet.

---

## Training plan

### Phase 0 — label coverage audit
- Per-jsonl class distributions (done inline; pitch + volume class
  imbalance noted).
- One-shot Qwen extraction pilot on 100 descriptions → structured tag
  yield rate.
- Decide per-attribute mining route from the matrix in §Data.

### Phase 1 — taxonomy v0 freeze
- Lock head set + class lists.
- Define eval splits: held-out on test-clean + a small (5-10 hr)
  affect-rich set from MELD/RAVDESS as the real benchmark for mood.

### Phase 2 — label generation
- Build `labels/build_labels.py` → emits `labels.parquet` keyed by
  audio_path with all v0 tags, plus per-tag confidence/source flag.
- Start with the 31 k subset (fast iteration), expand to 315 k.

### Phase 3 — baseline training
- Whisper-tiny encoder (frozen) + multi-head MLP.
- Per-head loss (CE for classification, MSE for regression), summed.
- Train on GPU 1, AdamW, cosine LR, ~30 min on 31 k.

### Phase 4 — eval + iterate
- Per-head accuracy / F1 / MAE on test-clean.
- Latency: forward pass on Blackwell vs CPU (ORBIS supports both).
- Failure analysis → unfreeze last 2 encoder layers if underfitting.

### Phase 5 — serve
- FastAPI server, OpenAI-style endpoint.
- systemd unit on ava node, Tailscale-only port.
- Document wire format for ORBIS integration.

### Phase 6 — ORBIS integration sketch (no code into ORBIS yet)
- Identify Pipecat hook point (frame processor between transport and STT?).
- Document context-injection format.
- Hand off to ORBIS PR review.

---

## Open questions

1. **Pool over what window?** Per-utterance (post-VAD) vs sliding 1-2 s
   windows. Per-utterance simpler; sliding gives streaming mood.
2. **Whisper-tiny vs Distil-Whisper-small** as encoder. Tiny is smallest;
   distil-small is 2× bigger but better representations. Ablate.
3. **Multi-task loss weighting.** Naive sum biases toward regression
   heads with larger gradient magnitudes. Plan to use uncertainty
   weighting (Kendall et al.) if naive sum underperforms.
4. **Where do we draw mood ground truth?** emotion2vec is the simplest;
   it's also a 300 M param model — fine for offline labelling, not for
   online inference. We label once, distil into our tiny student.
5. **Calibration.** Confidence values matter for ORBIS (it shouldn't
   inject low-confidence tags). Plan: temperature-scaled softmax.

---

## Layout

```
experiments/audio-tags/
├── PLAN.md                  ← this file
├── labels/
│   ├── audit.py             ← class distributions, prose mineability
│   ├── build_labels.py      ← unified label generator
│   └── extractors/          ← per-attribute miners
├── training/
│   ├── model.py             ← encoder + heads
│   ├── data.py              ← Lhotse / parquet loaders
│   └── train.py
├── eval/
│   ├── test_clean.py
│   └── latency_bench.py
├── serve/
│   └── server.py            ← FastAPI tag service
└── scripts/
    └── ...                  ← one-off utilities
```
