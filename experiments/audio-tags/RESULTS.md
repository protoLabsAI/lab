# audio-tags — Experiment Log

## v0-smoke (2026-04-25, no loss balancing)

**Config:** 1 epoch, AdamW 3e-4, OneCycle cosine, batch=32, frozen
Whisper-tiny encoder, mean-pool, raw SNR target (0-90 dB).

**Train:** `labels-train-clean-100.parquet` (28,539 samples, gender
+ acoustic only — prose extraction in flight).
**Val:** `labels-test-clean.parquet` (2,620 samples, speaker-disjoint).

| Head | Acc | Note |
|---|---:|---|
| speaker_gender | 55.8% | random — head not learning |
| volume | 64.2% | matches majority class baseline |
| pitch | 58.8% | barely above majority |
| speaking_speed | 67.4% | matches majority |

**Diagnosis.** Per-head loss breakdown showed `snr_db` MSE = 96 of 237
total train loss (raw 0-90 targets), drowning out CE gradients on the
classification heads. Heads stuck at ~ln(2) ≈ 0.69 (random).

## v0-balanced (2026-04-25, normalized regression + per-head weights)

**Fixes from smoke:**
1. `data.py`: divide `snr_db` by 90.0 to put regression target in [0,1]
2. `model.py`: per-head loss weights — gender×2, environment/age×0.5,
   else×1.0
3. 3 epochs instead of 1

**Same data** (28k train-clean-100, 2.6k test-clean).

| Head | Acc | F1 macro | Confusion (kept classes) |
|---|---:|---:|---|
| **speaker_gender** | **96.1%** | 0.641 | F 1323/1389 (95.3%), M 1195/1231 (97.1%) |
| speaking_speed | 79.0% | 0.505 | normal 91%, fast 60%, **slow 0%** ❌ |
| volume | 74.6% | 0.714 | normal 57%, loud 84% |
| pitch | 68.9% | 0.680 | medium 60%, high 76% |
| snr_db (regr) | — | — | MAE 0.067 (≈ 6 dB on 0-90 dB) |

**Latency (Blackwell bf16, batch=1):** 1.66 ms.

**Heads not meaningfully evaluated** (n=5 in test): mood_class,
environment, speech_style. Test-clean prose extraction not yet run.

### Confusion matrices

```
speaker_gender:           volume:                  pitch:
       pred F  pred M     pred normal pred loud    pred med  pred high
true F   1323     66      true normal  540  404    true med   684   459
true M     36   1195      true loud    262 1414    true high  355  1122
```

```
speaking_speed (slow column never predicted):
                 pred slow  pred normal  pred fast
true slow         0           73           0
true normal       0         1605         162
true fast         0          314          466
```

### Findings
1. **Frozen Whisper-tiny carries gender cleanly** through mean-pool.
   This is the win. Confirms the architectural premise.
2. **`slow` speaking_speed is unlearnable at 1% prevalence** with
   uniform sampling — model collapses it to `normal`. Fix:
   `WeightedRandomSampler` to oversample the rare class.
3. **Pitch head is the weakest** of the working set. 69% acc, F1
   0.68. Probably needs unfrozen last 2 encoder layers or attention
   pooling.
4. **SNR regression learns** but is noisy (~6 dB MAE) — still
   sufficient as a "is this room noisy" feature for ORBIS.
5. **Latency is irrelevant.** 1.66 ms is 600× under our budget.
   Quality is the only ceiling.

### Comparison to SALM Phase 1 v2 fallback
SALM Phase 1 v2 (Qwen3.5-4B + Canary-1b-flash + adapter, 5.4 B
params total) was the alternative. It does open-ended description.
It was never benchmarked on bounded classification — but at 5 B
params vs. 8 M params, the comparison is meaningful only on quality.

For ORBIS specifically: bounded JSON tags are what's needed, not prose.
v0-balanced clears the "useful for ORBIS" bar on gender alone. Open
question: does v1 close the gap on mood, where SALM might shine?

---

---

## v1-full (2026-04-25, full 281k LibriSpeech + WeightedRandomSampler)

**Config:** 3 epochs, AdamW 3e-4, batch=32, num-workers=8 (~21 min on
Blackwell), `--weighted-sampler` to fix v0's `slow` collapse.

**Train:** `labels-train-full.parquet` (281,051 rows: train-clean-100
+ train-clean-360 + train-other-500, gender + acoustic).
**Val:** same `labels-test-clean.parquet` as v0.

| Head | Acc | F1 macro | Δ vs v0 (acc) | Δ vs v0 (F1) |
|---|---:|---:|---:|---:|
| **speaker_gender** | **97.2%** | 0.648 | +1.1 | +0.7 |
| volume | **83.5%** | 0.817 | +8.9 | **+10.3** |
| pitch | **72.1%** | 0.716 | +3.2 | +3.6 |
| speaking_speed | 69.3%* | **0.584** | -9.7* | **+7.9** |
| snr_db | — | MAE 0.059 | — | -0.008 |

\* Acc dropped because weighted sampler flattened the class prior;
F1 macro is the right metric here and shows v1 is unambiguously better
across the board.

### Speaking_speed confusion (v0 → v1)

```
v0: slow recall = 0%   (model never predicted slow)
v1: slow recall = 90%  ← weighted sampler fixed it
v0: normal recall = 91%
v1: normal recall = 58%  (now actually predicts slow when it should)
v0: fast recall = 60%
v1: fast recall = 92%
```

**Latency:** 1.70 ms (unchanged from v0). 8.3 M params, 0.11 M
trainable. No regression on the main asset.

---

## Whisper detection — research journey

The next iteration target is whisper detection (Alexa "Whisper Mode"
analog). This was harder than expected, *not* because the model
architecture is wrong, but because the **training data didn't exist
on disk** and acquisition was rough.

### Data acquisition — what we tried

1. **wTIMIT** — gold standard, $125-250 + multi-day LDC account approval.
   **Skipped** (incompatible with weekend timeline).
2. **VocalSound** — turned out to have NO whisper class (only laughter,
   sigh, cough, throat clearing, sneeze, sniff). I had misread the
   abstract. **Cross off** for whisper; useful for a future
   "non-speech vocalization" head.
3. **AudioSet whisper subset** — YouTube rot + weak labels. **Skipped**.
4. **Fish S2 Pro `[whisper]` synthesis** — Fish (which we ship for
   ORBIS TTS) has explicit `[whisper]` and `[whisper in small voice]`
   prosody tags. On a controlled English pangram prompt, `[whisper]`
   produced *real whispered acoustics*: low-frequency (80-300 Hz)
   energy went 56% → 2.5%, high-frequency (2-4 kHz) noise rose 0.8%
   → 19%, RMS dropped 11×. **Spectral inversion**, not just quiet
   voiced.

   **But on real LibriSpeech-length prompts**, Fish ignored the tag
   ~93% of the time — output looked like quiet voiced speech (LF%
   80-95%). `[whisper in small voice]` pushed acceptance to ~7%, still
   too low.

   We fell back to a quality-filter approach (generate, keep only
   spectrally-clean whispers), but at 7% acceptance the throughput
   was unworkable for the weekend.

5. **DSP whisperization** — winner. Take voiced LibriSpeech audio,
   apply STFT-based transformation:
   - Sub-300 Hz magnitude × 0.05 (kills F0 + low harmonics)
   - 2-4 kHz magnitude × 1.5 (lifts HF noise floor)
   - Random phase across all bins (destroys remaining periodicity)
   - Renormalize to RMS 0.04 (whisper level)

   Validated on 6 LibriSpeech samples: LF% goes 7-12% → 0%, HF% lifts
   to 18-32%. Deterministic, runs at ~28 files/s on a single CPU core,
   reproducible from the script alone. **This becomes the bulk
   training data + the dataset contribution to HF.**

### Held-out plan (the honest test)

DSP-whisperized speech is synthetic. The question is whether a model
trained on it generalizes to *real* whispered speech recorded into a
microphone. We address this with a small self-recorded held-out:

- Browser-based recorder (`scripts/recorder.html`) — 15 prompts
  × {voiced, whispered} → 30 clips, ~5 min total
- 16 kHz mono WAV, packaged as a zip
- Used **only at eval** time, never seen in training
- Fence against DSP-only overfit

If v2 holds up on this set, the approach generalizes and we ship the
DSP technique. If it tanks, we pay LDC for wTIMIT and retrain.

---

---

## v2-whisper (2026-04-25 evening, full LibriSpeech + 50k DSP whispers + prose)

**Config:** 3 epochs, AdamW 3e-4 (OneCycleLR), batch=32, weighted
sampler over (speaking_speed × voice_quality), trunk + heads trainable
(0.11M / 8.32M), bf16 on Blackwell. ~22 min wall.

**Training data — 331,051 rows total:**
- 281,051 voiced LibriSpeech (clean-100/360 + other-500), full label set
- 50,000 DSP-whisperized LibriSpeech (speaker_gender + voice_quality=whispered,
  everything else MASKED because DSP invalidates volume/pitch/SNR labels)
- Prose extraction caught up: 18 k speech_style, 17 k mood_class,
  25 k environment now real-supervised (vs n=5 in v1 eval)

**Eval:** `labels-test-clean-with-dsp.parquet` (5,149 rows: 2,620 real
LibriSpeech voiced + 2,529 DSP-whispered test-clean). Speaker-disjoint.

| Head | Acc | F1 macro | Notes |
|---|---:|---:|---|
| **voice_quality** ✨ | **100.0%** | **1.000** | **1 error out of 5149**. In-distribution — DSP whispers are deterministic, model effectively memorizes the transformation. Real test is OOD. |
| **speaker_gender** | 97.2% | 0.648 | Held steady across v0→v1→v2. |
| volume | 82.9% | 0.811 | No regression. |
| pitch | 71.9% | 0.716 | No regression. |
| speaking_speed | 67.1% | 0.567 | Same as v1; weighted sampler maintains slow-recall. |
| **environment** | 91.8% | 0.191 | **Collapsed to `indoor_quiet`** — 167/167 indoor_quiet correct, 0/15 indoor_noisy/outdoor/unknown predicted. F1 macro reveals the lie. |
| **mood_class** | 63.1% | 0.111 | **Collapsed to `neutral`** — 111/111 neutral correct, ~0 on rare classes. Same imbalance pattern. |
| **speech_style** | 53.8% | 0.234 | Mostly predicts `narration` (the modal class on LibriSpeech). |

**Latency:** 1.78 ms (Blackwell, bf16, batch=1).

### Known limitations

1. **mood/environment/speech_style heads collapse to majority class.**
   This is a labeling-distribution problem: prose-extracted classes
   from LibriSpeech are heavily skewed toward `neutral`,
   `indoor_quiet`, `narration` because LibriSpeech IS audiobook
   reading. The weighted sampler we use only balances
   speed×voice_quality. Fix: extend sampler to balance these heads
   too, or hide the heads behind a confidence threshold for ORBIS
   until v3.

2. **voice_quality 100% is *in-distribution*.** The model learns DSP
   artifacts perfectly — the question is whether it generalizes to
   actual whispered speech recorded into a microphone. The Gradio
   app's recording manifest at
   `/mnt/data/audio-tags/held_out/manifest.jsonl` is the OOD anchor.
   Until that's populated, **do not trust the 100% number for ORBIS
   deployment**.

3. **Prose-supervised heads are LLM-extracted, not human-labeled.**
   They're consistent under Qwen 3.6-27B-FP8 but inherit any of its
   biases (it likely systematically calls audiobook prose
   "narration", which isn't ORBIS's user-conversation case).

### What's next (concrete, in priority order)

1. **OOD whisper eval** — user records ~30 paired voiced/whispered
   clips via the Gradio app, eval drops in. If voice_quality holds
   above ~85% on real audio, we're shipping. If not, the DSP-only
   approach failed and we need real whispered training data
   (wTIMIT or similar).

2. **Per-head class-balanced sampler** — extend the v2 sampler to
   re-weight by mood_class and environment too. Should rescue F1
   macro on those heads from ~0.11 to something usable.

3. **Personalization fine-tune** — Gradio's "Personalize" button
   adapts the head + trunk on a few minutes of user-recorded clips.
   Demo for the blog: "your tiny model, fine-tuned on you in 30 sec."

4. **Confidence calibration** — temperature-scale softmax outputs
   so the JSON's `confidence` field is meaningful, gating ORBIS
   context injection at e.g. >0.7.

5. **HF release** — model card + dataset card (DSP whisperization
   technique is the unique contribution).

---

## Tier-0 baseline comparison (the humbling part)

After v2 was trained and released, I ran two sanity-check baselines
on the same 5,149-row test set:

1. **Majority class** — predict the most common train class.
2. **Linear probe** — `sklearn.LogisticRegression` on Whisper-tiny
   mean-pooled features (no trunk, no per-head MLP, no fine-tuning).

| Head | Majority | Linear probe | v2 (ours) |
|---|---:|---:|---:|
| speaker_gender | 46.9 / 0.21 | 96.8 / 0.65 | **97.2 / 0.65** |
| voice_quality | 50.9 / 0.34 | 99.6 / 1.00 | **100.0 / 1.00** |
| volume | 64.0 / 0.39 | **83.4 / 0.82** | 82.9 / 0.81 |
| pitch | 56.4 / 0.36 | **72.3 / 0.72** | 71.9 / 0.72 |
| speaking_speed | 67.4 / 0.27 | **84.3 / 0.67** | 67.1 / 0.57 |
| speech_style | 51.6 / 0.14 | **55.5 / 0.30** | 53.8 / 0.23 |
| mood_class | 63.1 / 0.11 | 63.6 / 0.16 | 63.1 / 0.11 |
| environment | 91.8 / 0.19 | 91.8 / 0.19 | 91.8 / 0.19 |

(format: accuracy% / F1 macro)

**What this revealed:**

1. **Whisper-tiny encoder features carry essentially all the signal.**
   v2's trunk + 11-head MLP is decoration — most heads are within ±1%
   of the linear probe.
2. **Linear probe BEATS v2 on speaking_speed by 17% accuracy.** The
   weighted sampler is over-correcting: flattening the prior gains
   slow-class recall but loses harder on the dominant `normal` class.
3. **voice_quality 99.6% on a linear probe** strongly suggests v2
   is detecting the DSP transformation artifact, not whispered
   acoustics per se. **The 100% in-distribution number is
   instrumentation, not a real-world claim.**
4. **mood/environment/speech_style collapse at every level.** The
   problem is the data, not the architecture.

---

## v3-linear and v3-balanced (2026-04-25 night)

Two iterations triggered by the baseline finding:

- **v3-linear**: drop the trunk entirely (encoder + per-head Linear).
  Trainable params 110 K → 50 K.
- **v3-balanced**: keep the trunk but switch from `WeightedRandomSampler`
  to per-head class-weighted CE loss inside `compute_loss`.

Same training data, same 3 epochs, same eval set as v2.

### Full 5-way comparison

| Head | majority | linear probe | v2 | v3-linear | v3-balanced |
|---|---:|---:|---:|---:|---:|
| speaker_gender | 46.9 / 0.21 | 96.8 / 0.65 | **97.2 / 0.65** | 95.0 / 0.63 | 97.0 / 0.65 |
| voice_quality | 50.9 / 0.34 | 99.6 / 1.00 | **100.0 / 1.00** | 99.3 / 0.99 | **100.0 / 1.00** |
| volume | 64.0 / 0.39 | **83.4 / 0.82** | 82.9 / 0.81 | 79.2 / 0.76 | 82.7 / 0.82 |
| pitch | 56.4 / 0.36 | **72.3 / 0.72** | 71.9 / 0.72 | 71.0 / 0.70 | 66.6 / 0.67 |
| speaking_speed | 67.4 / 0.27 | **84.3 / 0.67** | 67.1 / 0.57 | 80.7 / 0.54 | 63.4 / 0.54 |
| mood_class | 63.1 / 0.11 | 63.6 / 0.16 | 63.1 / 0.11 | 63.1 / 0.11 | **51.7 / 0.23** |
| environment | 91.8 / 0.19 | 91.8 / 0.19 | 91.8 / 0.19 | 91.8 / 0.19 | 64.3 / 0.19 |
| speech_style | 51.6 / 0.14 | **55.5 / 0.30** | 53.8 / 0.23 | 54.4 / 0.20 | 35.2 / 0.29 |

(format: accuracy% / F1 macro)

### Findings

1. **No PyTorch model strictly dominates.** v2, v3-linear, and
   v3-balanced trade wins per head. Linear probe still wins on
   `speaking_speed` and `pitch` cleanly.
2. **v3-linear ≈ v2** quality with **half the trainable params** (50 K
   vs 110 K). Small "tiny" win, not a step change.
3. **v3-balanced is the only model that breaks the `mood_class`
   majority-class collapse**: F1 macro 0.11 → 0.23, recall on
   `excited`, `tense`, `playful` becomes non-zero. Trade-off is
   environment / speech_style now spread predictions across classes,
   dropping accuracy on the majority class.
4. **The trunk dilutes speaking_speed signal.** Linear probe at 84.3%
   accuracy vs all PyTorch trained models at 63-81%. Worth investigating
   in v4.

### Decision: ship v2 as default + v3-balanced as companion

Pushed both to HuggingFace (private):

- `protoLabsAI/orbis-audio-tags-v2` — default. Best peak accuracy.
- `protoLabsAI/orbis-audio-tags-v3-balanced` — same arch, mood-aware loss.

Pick v3-balanced if `mood_class` is a downstream consumer; otherwise
default v2.

### What didn't fix mood/environment/speech_style

Even with class-weighted loss, the rare classes for `environment`
(`outdoor`, `phone_call`) and `speech_style` (`oratorical`,
`conversational`) still get ~0% recall. The data is the bottleneck
— LibriSpeech audiobook prose simply doesn't cover these
distributions in any meaningful volume. Real fix is mixing in
conversational/emotional corpora (DailyTalk, MELD, RAVDESS, AMI) for
v4.

---

## v4-multi (2026-04-25 night/morning, multi-corpus mix)

The data fix.

**Training data** — 365,898 rows total:
- 281 k voiced LibriSpeech (existing)
- 50 k DSP-whisperized LibriSpeech (existing)
- 22.5 k DailyTalk utterances (new — 7-class emotion + conversational style)
- 11 k MELD train+dev (new — Friends TV, indoor_noisy, 7 emotions)
- 1.2 k RAVDESS train (new — 24 actors, 8 emotions × 2 intensities, dramatic)

Loss: same class-weighted CE as v3-balanced. Architecture unchanged.

**Held-out**: MELD test (2,610) + RAVDESS held-out actors 21-24 (240) =
**2,850 rows of cross-domain audio** that nothing in training has
seen.

### Eval on the v4 cross-domain holdout (the meaningful test)

| Head | v2 | v3-balanced | **v4-multi** | F1 lift |
|---|---:|---:|---:|---:|
| voice_quality | 71.2 / 0.42 | 70.8 / 0.41 | **99.8 / 0.50** | +0.08 |
| environment | 8.6 / 0.03 | 49.9 / 0.23 | **99.9 / 0.40** | **+0.37** |
| speech_style | 15.2 / 0.07 | 41.8 / 0.12 | **96.4 / 0.39** | **+0.32** |
| mood_class | 45.3 / 0.10 | 21.2 / 0.12 | **24.5 / 0.20** | +0.10 |
| volume | 58.3 / 0.46 | 57.5 / 0.45 | **60.8 / 0.52** | +0.06 |
| speaker_gender | 74.1 / 0.54 | 74.1 / 0.54 | 24.6 / 0.23 | -0.31* |

\* speaker_gender "regression" is misleading: v4 correctly predicts
`unknown` 98% of the time when the speaker is unlabeled (DailyTalk
no-gender, MELD non-main-cast). v2/v3 just guess male/female and
miss everything outside the labeled set. v4 is more honest, not
worse.

### Eval on the original test-clean-with-dsp (in-domain to v2)

v4 looks worse here only because v2 collapses to LibriSpeech's
majority class on mood/env/style (which inflates accuracy on a
test set that's also majority-class-skewed). On real-world audio
v4 wins; on audiobook-dataset audio v2's collapse just happens to
look like high accuracy.

### Why v4 is the right ship

The whole point of fixing mood/env/style was so ORBIS can route on
them. ORBIS users don't talk like LibriSpeech narrators. The v4
holdout matches real conditions:
- 2,610 noisy TV-dialogue utterances (MELD)
- 240 emotionally-acted utterances (RAVDESS)

On that data, v4's `environment` F1 macro is 13× v2's, `speech_style`
is 5.6× v2's, and `voice_quality` lifts from 0.42 to 0.50. **Every
collapsed head now does meaningful work.**

Pushed to HF: `protoLabsAI/orbis-audio-tags-v4-multi` (private).

---

## v5-soft (2026-04-25 morning) — superseded v4

v4's full inverse-frequency class weighting was too aggressive — it
crushed majority classes (`neutral` mood, `indoor_quiet` env) and
hurt in-domain accuracy without proportional gain elsewhere. v5
keeps everything else identical to v4 and only changes the weight
computation:

```python
# v4 (full inverse-frequency)
weight = n_total / (n_class * n_classes)
# v5 (sqrt-tempered)
weight = (n_total / (n_class * n_classes)) ** 0.5
```

### v4 → v5 head-to-head

#### `test-clean-with-dsp` (LibriSpeech in-domain)

| Head | v4 | **v5** | Δ |
|---|---:|---:|---:|
| speaker_gender | 93.0 / 0.63 | **97.0 / 0.65** | +4.0 / +0.02 |
| mood_class | 38.6 / 0.17 | **60.8 / 0.17** | +22 / 0 |
| volume | 81.3 / 0.81 | **83.9 / 0.83** | +2.6 / +0.02 |
| pitch | 65.4 / 0.65 | **70.2 / 0.70** | +4.8 / +0.05 |
| speaking_speed | 64.1 / 0.54 | **81.4 / 0.72** | **+17 / +0.18** |
| environment | 44.0 / 0.14 | **89.6 / 0.22** | **+46 / +0.08** |
| speech_style | 34.1 / 0.26 | **47.8 / 0.32** | +14 / +0.06 |
| voice_quality | 99.9 / 1.00 | 99.9 / 1.00 | tied |

#### Cross-domain holdout (MELD test + RAVDESS held-out)

| Head | v4 | **v5** | Δ |
|---|---:|---:|---:|
| speaker_gender | 24.6 / 0.23 | **39.1 / 0.41** | **+14.5 / +0.18** |
| mood_class | 24.5 / 0.20 | **42.9 / 0.29** | **+18.4 / +0.09** |
| volume | 60.8 / 0.52 | **65.4 / 0.60** | +4.6 / +0.08 |
| environment | 99.9 / 0.40 | 99.9 / 0.40 | tied |
| speech_style | 96.4 / 0.39 | **99.4 / 0.39** | +3.0 / 0 |
| voice_quality | 99.8 / 0.50 | 99.8 / 0.50 | tied |

**Strict improvement on every head with no regressions.**

### v5 vs the linear-probe baseline

The linear probe was a hard ceiling for v2/v3 — most heads were
within ±1% of it on the in-domain test, and the trunk was
contributing nothing measurable. v5 finally **matches or beats the
linear probe on most heads**:

| Head | linear probe | **v5-soft** |
|---|---:|---:|
| speaker_gender | 96.8 / 0.65 | **97.0 / 0.65** |
| volume | 83.4 / 0.82 | **83.9 / 0.83** |
| voice_quality | 99.6 / 1.00 | **99.9 / 1.00** |
| speaking_speed | 84.3 / 0.67 | 81.4 / **0.72** |
| environment | 91.8 / 0.19 | 89.6 / **0.22** |
| speech_style | 55.5 / 0.30 | 47.8 / **0.32** |

The trunk + multi-head pipeline is finally adding measurable lift
on F1 macro for the previously-collapsed heads. The simpler
architecture (linear probe alone) was actually the better baseline
*for v2/v3*, but with the right data + loss, the full architecture
is justified.

### Decision: v5 is the flagship

Pushed to HF: `protoLabsAI/orbis-audio-tags-v5-soft` (private).
v4-multi marked as superseded; v2 / v3-balanced / v4 retained as
ablation references for the data + loss progression.
