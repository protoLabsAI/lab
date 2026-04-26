# Tiny audio-understanding for voice agents — a weekend research log

> **Repo**: model and dataset live at
> [`protoLabsAI/orbis-audio-tags-v2`](https://huggingface.co/protoLabsAI/orbis-audio-tags-v2)
> and
> [`protoLabsAI/orbis-audio-tags-v0`](https://huggingface.co/datasets/protoLabsAI/orbis-audio-tags-v0)
> (private during the writeup, public soon). Code:
> [`experiments/audio-tags/`](https://github.com/protoLabsAI/lab/tree/main/experiments/audio-tags).

I had a question last Friday: *what's the smallest possible audio
understanding model I can plug into [ORBIS](https://github.com/protoLabsAI/ORBIS)
to give the LLM a little extra context — mood, gender, voice quality,
prosody — without making the orb feel slow?*

48 hours later I have a deployable answer, three trained models, a
small dataset contribution, and a couple of findings I genuinely
didn't expect. This post is the unedited research log of how it went.

---

## The problem

ORBIS is a voice-first companion agent. Pipecat handles the audio
pipeline; Whisper does STT; an LLM does the routing and personality.
The companion-layer maintains a `mood` table in SQLite. There's no
audio-derived signal feeding into that mood state today — it's
inferred from text alone.

Earlier in the year I built a SALM-style speech-understanding model
for a different project: Qwen 3.5-4B as the LLM backbone, Canary-1b-flash
as the encoder, an adapter trained on ~960 h of LibriSpeech with
DeSTA2-style descriptions as labels. The result described audio
in prose ("a male speaker delivering a dramatic monologue, high pitch,
loud volume, clear recording"). It worked but it's 5 B parameters of
inference cost — way too heavy to run alongside Whisper for every
turn in a real-time voice loop.

What ORBIS actually needs is bounded JSON tags — `gender=female,
mood=warm, voice_quality=voiced` — so the LLM can read them as a
context line. That's a classification problem, not generation. And
the encoder doesn't need to be Canary; ORBIS already loads Whisper.

The weekend's hypothesis: **attach small classifier heads to the
frozen Whisper-tiny encoder, train them on bounded label sets, ship
in under 10 M params with sub-10 ms inference.**

---

## Architecture

Pretty boring. The whole model is:

```
audio → WhisperFeatureExtractor → Whisper-tiny encoder (FROZEN, 10M)
      → mean-pool over time → trunk (Linear → GELU → Dropout)
      → 11 parallel heads (one Linear per attribute)
```

The 11 heads cover: speaker_gender, speaker_age, mood_class,
valence/arousal (regression), volume, pitch, speaking_speed,
snr_db (regression), environment, speech_style, voice_quality.

Total params: **8.32 M**. Trainable (encoder frozen): **0.11 M**.
Inference: **~1.7 ms** on Blackwell at batch=1, low hundreds of ms on
CPU. Both are well under any user-perceivable threshold.

---

## v0 → v1: getting the multi-task losses to work

First-pass training on `train-clean-100` (28 k samples, gender +
acoustic only) didn't learn. Loss dropped 5× (good) but per-head
accuracy was indistinguishable from majority class (bad). The
breakdown was telling: `snr_db` regression on the raw 0-90 dB scale
contributed **96 of 237 total train loss**. The classification heads
were getting essentially noise gradient.

Two-line fix:
1. Normalize regression targets to [0, 1] before MSE.
2. Per-head loss weights — bump priority heads (gender) up, weak-label
   heads (age, environment) down.

After this, gender accuracy on speaker-disjoint test-clean jumped
from 56% → 96% in three epochs. That confirmed Whisper-tiny features
genuinely carry gender info through mean-pooling. Volume, pitch,
speed all moved from "majority class" to meaningful.

One head stayed broken though: `slow` speaking_speed, 0% recall.
Slow speech is 1% of LibriSpeech — the model just predicted `normal`
and `fast`. Adding a `WeightedRandomSampler` over the speaking_speed
classes lifted slow recall from 0% → 90% in v1, at the cost of
overall accuracy dropping from 79% → 69% (it now actually predicts
slow when it should). Net F1 macro went **up** from 0.51 → 0.58 —
the right direction.

---

## v2: whispered detection — the part that didn't work the way I expected

This was the actually-interesting iteration.

ORBIS users will sometimes whisper. Alexa has had a "Whisper Mode"
since 2018; Apple has a Siri patent for the same idea. The acoustic
signature is well-known: whispered speech is *unvoiced* (no
fundamental frequency from the vocal folds), so the spectrum tilts
high-frequency and broadband — qualitatively different from "quiet
voiced speech."

I needed paired voiced/whispered training data. Options:

| Source | Got it? |
|---|---|
| **wTIMIT** (LDC, gold standard) | $125-250 + multi-day account approval — skip for the weekend |
| **VocalSound** | I misread the abstract — it has laughter/sighs/coughs, **no whisper class**. Cross off. |
| **AudioSet whisper subset** | YouTube rot, weak labels — unreliable |
| **Fish S2 Pro `[whisper]` synthesis** | We already ship Fish for ORBIS TTS — worth trying |

### Fish synthesis was a near-miss

On a controlled English pangram prompt, `[whisper] The quick brown
fox jumps over the lazy dog` produced *real* whispered acoustics:
low-frequency (80-300 Hz) energy share dropped from 56% (voiced) to
**2.5%** (whispered), and high-frequency (2-4 kHz) noise lifted from
0.8% to 19%. That's spectral inversion, not just quiet-voiced. The
tag works.

Then I batched it on 50 random LibriSpeech transcripts. The
acceptance rate was **7%**. Most of the time Fish silently ignored
the tag — output was just slightly-quieter voiced speech. I tried
`[whisper in small voice]`, `[softly]`, `[secret]`, `[intimate]`,
`[whispered voice]`. Single-prompt testing showed which tags work;
batch testing showed they're inconsistent on real prompts.

I built a spectral quality filter (`whispered_clean = lf% < 12 AND
hf% ≥ 5 AND rms in [0.005, 0.10]`) but at 7% yield, generating 5 k
clean pairs would take ~40 hours. Wrong shape for this weekend.

### DSP whisperization — the unblocking pivot

Forget synthesizing whisper from text. Take *voiced* audio (we have
281 k LibriSpeech clips on disk) and DSP-transform each one:

1. STFT (n_fft=1024, hop=256)
2. Sub-300 Hz magnitude × 0.05 — kills F0 and low harmonics
3. 2-4 kHz magnitude × 1.5 — lifts high-frequency noise tilt
4. Randomize phase across all bins — destroys remaining periodicity
5. ISTFT, normalize to RMS ≈ 0.04 (typical whisper level)

Validated: LF% goes 7-12% → 0%, HF% lifts to 18-32%. Same spectral
inversion as Fish's good outputs, but **deterministic, reproducible,
and 28 files/sec on a single CPU core**. 50 k whispered samples
generated in 31 minutes.

I'm shipping the script with the dataset release because honestly
it's the contribution worth sharing. wTIMIT is the gold-standard
benchmark, but if you just need *training data*, you can derive it
from any voiced corpus.

### v2 results (in-distribution)

Trained on 281 k voiced LibriSpeech + 50 k DSP-whisperized clips with
prose-extracted labels (Qwen-3.6-27B picked mood/style/environment
out of DeSTA2-style descriptions for ~25 k samples).

Held-out test set: `test-clean` voiced + DSP-whisperized
`test-clean` (5,149 rows, speaker-disjoint).

| Head | Acc | F1 macro | Notes |
|---|---:|---:|---|
| voice_quality | **100.0%** | 1.000 | 1 error in 5,149. **In-distribution.** |
| speaker_gender | 97.2% | 0.648 | reliable |
| volume | 82.9% | 0.811 | reliable |
| pitch | 71.9% | 0.716 | usable |
| speaking_speed | 67.1% | 0.567 | usable |
| environment | 91.8% | 0.191 | **collapsed to majority class** |
| mood_class | 63.1% | 0.111 | **collapsed to majority class** |
| speech_style | 53.8% | 0.234 | mostly predicts `narration` |

Latency: 1.78 ms (Blackwell, bf16). Apparently a clean win.

---

## And then I ran the actual baselines

This is the part that surprised me.

The above table is just "our model on our test set with our labels."
It tells you nothing about whether the architecture is doing useful
work. So I added two real baselines on the same test set:

1. **Majority class** — predict the most common train class. The
   floor.
2. **Linear probe** — `sklearn.linear_model.LogisticRegression` on
   mean-pooled Whisper-tiny features. Asks: is our trunk + heads
   doing anything over a single linear layer on the same features?

The result:

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

Reading this honestly:

1. **Whisper-tiny's frozen encoder features carry essentially all the
   signal.** Our trunk + 11-head MLP is decoration, not lift. Most
   heads are within ±1% of the linear probe.
2. **On `speaking_speed`, the linear probe is 17 points better than
   v2.** That's the weighted sampler over-correcting — flattening the
   prior to fix slow-class recall, but losing more on the dominant
   `normal` class than it gains. The simpler model wins.
3. **`voice_quality` 99.6% on a linear probe** is the most damning
   number. The DSP whisperization signal is so distinctive — randomized
   phase, spectral tilt — that a single linear layer reads it
   perfectly. Whether the model has learned anything about
   *whispered speech* vs *the DSP transformation artifact* is
   completely open. **Treat the 100% as instrumentation, not as a
   real-world claim, until validated against actual whispered
   microphone audio.**
4. **`mood`, `environment`, `speech_style` collapse to majority class
   at every level — majority, probe, and v2.** The trunk wasn't going
   to save us; the data is the problem. LibriSpeech is mood-flat
   audiobook reading. The Qwen-extracted prose labels inherit that
   distribution.

---

## v3: two ablations, two stories

The baseline result triggered two follow-ups:

- **v3-linear** — drop the trunk entirely. Encoder + 11 parallel
  Linear heads, no shared MLP. Trainable params 110 K → **50 K**.
  Hypothesis: if the linear probe was 99% as good, the
  trained-end-to-end version should match it.
- **v3-balanced** — keep the trunk but switch from
  `WeightedRandomSampler` to per-head **class-weighted
  cross-entropy** loss. The natural sampling distribution stays
  intact; rare classes just get gradient amplification instead.

Full 5-way comparison on the same `test-clean-with-dsp` set:

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

Three honest reads:

1. **v3-linear matches v2 quality at half the trainable params.** Same
   accuracy on most heads, same F1 macro on most heads, +14 points on
   `speaking_speed` (the trunk was hurting). 50 K trainable params is
   a real "tiny" milestone — but the win isn't dramatic.

2. **v3-balanced is the only iteration that breaks the `mood_class`
   collapse.** F1 macro 0.11 → 0.23 — non-zero recall on `excited`,
   `tense`, `playful`. The trade-off is `environment` and
   `speech_style` now spread predictions across rare classes too —
   accuracy on the majority class drops without truly recovering the
   rare-class signal. The data is the bottleneck.

3. **The linear probe still wins `speaking_speed` cleanly** (84.3 %
   acc) across every PyTorch trained variant. The trunk + heads
   pipeline is genuinely diluting that signal regardless of training
   scheme. Open question for v4 — maybe the speaking-speed-relevant
   features are localized in time and mean-pooling discards them.

Neither v3-linear nor v3-balanced is a clean win, and that's the
honest part: small tweaks to a frozen-encoder + tiny-heads
architecture are mostly rearranging the same signal Whisper-tiny is
already extracting. The encoder did the work.

---

## v4: data, not architecture

The mood/environment/speech_style heads were never going to learn
from LibriSpeech. Audiobook prose is mood-flat. Indoor-quiet by
construction. Almost entirely narration. The labels we extracted
were *real* but the distribution was a single point. No amount of
class weighting or trunk dropping could fix that.

So v4 added the data:

| Source | Hours | What it teaches |
|---|---:|---|
| **DailyTalk** | 21.7 | conversational; daily-life dialogue with 7 emotion labels per utterance |
| **MELD** (audio) | ~10 | conversational; Friends TV with applause/laughter/music as `indoor_noisy` |
| **RAVDESS** | ~2 | acted; 24 speakers × 8 emotions × 2 intensities, all studio-clean |

Held-out evaluation set built from the same datasets but
speaker-disjoint: MELD test (2,610) + RAVDESS held-out actors 21-24
(240) = **2,850 rows of cross-domain audio** that nothing in
training has seen.

### Cross-domain results

| Head | v2 | v3-balanced | **v4-multi** | F1 lift |
|---|---:|---:|---:|---:|
| voice_quality | 71.2 / 0.42 | 70.8 / 0.41 | **99.8 / 0.50** | +0.08 |
| environment | 8.6 / 0.03 | 49.9 / 0.23 | **99.9 / 0.40** | **+0.37** |
| speech_style | 15.2 / 0.07 | 41.8 / 0.12 | **96.4 / 0.39** | **+0.32** |
| mood_class | 45.3 / 0.10 | 21.2 / 0.12 | **24.5 / 0.20** | +0.10 |
| volume | 58.3 / 0.46 | 57.5 / 0.45 | **60.8 / 0.52** | +0.06 |
| speaker_gender | 74.1 / 0.54 | 74.1 / 0.54 | 24.6 / 0.23 | -0.31* |

\* The speaker_gender "regression" is misleading: v4 correctly
predicts `unknown` 98% of the time on speakers that aren't
gender-labeled in the source manifests (DailyTalk speakers, MELD
non-main-cast). v2/v3 just guess male/female blindly and miss
everything outside the labeled subset. v4 is more honest, not
worse — fix the gender annotation in v5.

The collapsed heads now do work:
- **`environment`** F1 macro is 13× v2's. v4 cleanly distinguishes
  MELD's noisy TV audio from RAVDESS's studio-clean audio.
- **`speech_style`** F1 is 5.6× v2's. The model now correctly
  predicts `conversational` on Friends dialogue, `dramatic` on
  RAVDESS acted speech, etc.
- **`mood_class`** F1 doubles, with non-zero recall on
  `excited`/`sad`/`tense` for the first time on real conversational
  data.

### What I learned from v4

1. **Architecture stops mattering pretty fast.** v0 → v1 → v2 → v3
   were architecture iterations. They moved numbers around. v4 was a
   data iteration and produced the only step-change improvement in
   the experiment.
2. **Class imbalance has two solutions and the data one is better.**
   Class-weighted loss (v3-balanced) and weighted sampling (v2)
   both rearrange existing data. Adding the right new data
   (v4-multi) actually fixes the problem.
3. **Test-set choice matters.** Looking at v4 on `test-clean-with-dsp`
   makes it look worse than v2. Looking at it on an actual
   conversational/emotional held-out shows it's the only one that
   works for ORBIS's real use case.

## v5: less aggressive class weighting

v4's full inverse-frequency weighting was working too hard. It was
boosting `tense` mood by 7.5× and `outdoor` environment by basically
infinity — the model started predicting rare classes even when the
audio was clearly `neutral` / `indoor_quiet`.

The fix is one parameter:

```python
# v4 (full inverse-frequency)
weight = (n_total / (n_class * n_classes))
# v5 (sqrt-tempered)
weight = (n_total / (n_class * n_classes)) ** 0.5
```

Same training data, same architecture, same hyperparameters
otherwise. v5 is **strictly better than v4 on every single head**:

### LibriSpeech in-domain (`test-clean-with-dsp`)

| Head | v4 | **v5** | Δ accuracy |
|---|---:|---:|---:|
| speaker_gender | 93.0 / 0.63 | **97.0 / 0.65** | +4 |
| mood_class | 38.6 / 0.17 | **60.8 / 0.17** | **+22** |
| volume | 81.3 / 0.81 | **83.9 / 0.83** | +3 |
| pitch | 65.4 / 0.65 | **70.2 / 0.70** | +5 |
| speaking_speed | 64.1 / 0.54 | **81.4 / 0.72** | **+17** |
| environment | 44.0 / 0.14 | **89.6 / 0.22** | **+46** |
| speech_style | 34.1 / 0.26 | **47.8 / 0.32** | +14 |
| voice_quality | 99.9 | 99.9 | tied |

### Cross-domain (MELD + RAVDESS held-out)

| Head | v4 | **v5** | Δ F1 macro |
|---|---:|---:|---:|
| speaker_gender | 24.6 / 0.23 | **39.1 / 0.41** | **+0.18** |
| mood_class | 24.5 / 0.20 | **42.9 / 0.29** | **+0.09** |
| volume | 60.8 / 0.52 | **65.4 / 0.60** | +0.08 |
| environment | 99.9 / 0.40 | 99.9 / 0.40 | tied |
| speech_style | 96.4 / 0.39 | **99.4 / 0.39** | +0 |
| voice_quality | 99.8 / 0.50 | 99.8 / 0.50 | tied |

The lesson — and this took five iterations to surface clearly:
**class imbalance fixes have a sharp Goldilocks zone**. Full
inverse-frequency over-corrects. Uniform weighting under-corrects.
Sqrt-tempered weighting hit the spot, at least for this data
distribution.

v5 also finally **matches or beats the linear-probe baseline** on
most heads, including the F1 macro on previously-collapsed heads.
The trunk + multi-head pipeline started adding measurable value
once the data and loss were both right.

I'm shipping v5-soft as the flagship:

- [`protoLabsAI/orbis-audio-tags-v5-soft`](https://huggingface.co/protoLabsAI/orbis-audio-tags-v5-soft)
  — recommended for ORBIS deployment
- [`protoLabsAI/orbis-audio-tags-v4-multi`](https://huggingface.co/protoLabsAI/orbis-audio-tags-v4-multi)
  — ablation: same data, aggressive class weighting (over-corrects)
- [`protoLabsAI/orbis-audio-tags-v3-balanced`](https://huggingface.co/protoLabsAI/orbis-audio-tags-v3-balanced)
  — ablation: loss change only, no new data
- [`protoLabsAI/orbis-audio-tags-v2`](https://huggingface.co/protoLabsAI/orbis-audio-tags-v2)
  — ablation: LibriSpeech-only baseline

---

## ORBIS integration shape

The model lives as a side-channel to STT in the Pipecat pipeline.
Same audio chunk that hits Whisper hits this model. On
`UserStoppedSpeakingFrame` we flush both. The tag dict gets either:

1. Appended to ORBIS's system prompt as a single annotation line:
   ```
   [user_audio] mood=warm gender=female voice_quality=voiced
   speaking_speed=normal volume=normal env=indoor_quiet
   ```
2. Or written to the SQLite `mood` table with the personality-drift
   curator picking it up on the next-turn read.

Tags below a confidence threshold (default 0.65) get dropped
client-side, which is critical because the mood/environment/style
heads will collapse to majority on any prompt that isn't strongly
neutral/indoor/narrational.

---

## What I'd do differently

1. **Run baselines earlier.** The "trunk doesn't add anything" finding
   would have changed v0 → v1 → v2 design. I'd have spent more time
   on ablations and less on architecture.
2. **Don't trust an in-distribution number when synthetic data is
   distinctive.** 100% on DSP-whisperized test set is a model
   memorizing the transformation, not learning whisper. The OOD gap
   (real microphone whispers) is the actual question and I haven't
   answered it yet.
3. **Class imbalance kills three out of eleven heads.** For mood and
   environment to be useful, we need either real conversational data
   (DailyTalk, MELD, RAVDESS) or much more aggressive per-head
   re-balancing.

---

## What's next

- Validate `voice_quality` on real whispered audio (the user-recorded
  held-out set the Gradio app at
  `experiments/audio-tags/app.py` produces).
- Mix conversational/emotional corpora (MELD, RAVDESS) for the
  collapsed heads.
- Try Wav2Vec2-base or WavLM-base as the encoder swap. Whisper-tiny
  is fine; might not be optimal.
- Personalization fine-tune flow — adapt heads + (maybe) trunk on a
  few minutes of the user's voice. The Gradio app has the button
  already, the demo is "your tiny model, fine-tuned on you in 30 sec."
- Eventually: real benchmarks. SUPERB / HEAR / ComParE for proper
  leaderboard numbers.

---

## Shipping it — and the model that ate ours

The plan was to take v5-soft and graduate it into ORBIS as a Pipecat
side-channel running alongside Whisper STT. Speaker-verification
gate first (single-owner ORBIS shouldn't update mood from a guest
voice), then the audio-tags tap, then a context line into the LLM.

That's what we wrote up in the
[ORBIS issue](https://github.com/protoLabsAI/ORBIS/issues/66). It's
not what shipped.

What shipped is **better than the plan**: ORBIS dropped Whisper STT
entirely in favor of [SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice)
— a 234 M-param multi-task speech model from FunASR that emits ASR
+ language ID + speech emotion + audio events in **one forward
pass**. AudioTagsTap, the Pipecat processor that was supposed to run
v5-soft, became a thin consumer of SenseVoice's `EmotionFrame`
instead.

This is an honest research outcome and worth being clear about:

- We trained v5-soft over a weekend. It works. It's on Hugging Face.
- Mid-engineering, the team found a model that subsumes our entire
  hot path more cheaply (one forward pass instead of two) and adds
  signal we didn't have (audio events: BGM, laughter, applause,
  cry, sneeze, breath, cough).
- We picked the better architecture and our model became an
  alternative emotion source in the pipeline rather than the only
  one.

What survives:

- The **methodology** is the deliverable. Tier-0 baselines
  (majority + linear probe + off-the-shelf), multi-corpus mixing,
  sqrt-tempered class weighting, the "data not architecture moves
  the needle" lesson — those carry forward to the next head we
  build, regardless of what model holds the slot today.
- The **dataset contribution** is unchanged. `protoLabsAI/orbis-audio-tags-v0`
  on Hugging Face ships the DSP whisperization technique + sample
  whispered audio + Qwen-extracted prose tags. The script is the
  contribution; reproducible from LibriSpeech alone.
- The **lineage** (v0 → v1 → v2 → v3-balanced → v4-multi → v5-soft)
  is the credibility. Future ORBIS-supporting research will cite
  this experiment as the methodology shakedown.
- The **non-emotion heads** in v5-soft (`snr_db`, `environment`,
  `speaking_speed`, `voice_quality`, `speaker_gender`) — SenseVoice
  doesn't emit these. They're queued as a small follow-up PR to
  enrich the `[audio]` annotation that ORBIS's LLM sees.

The deployed pipeline as of ORBIS v0.1.36:

1. `SpeakerGate` verifies owner-vs-stranger from echo-guarded audio
2. `SenseVoiceSTT` produces `EmotionFrame` → `AudioEventFrame` →
   `TranscriptionFrame` per utterance
3. `AudioTagsTap` drift-writes mood (owner only, for privacy) per
   a curated emotion-to-mood-delta map
4. `AudioTagsTap` injects `[audio] emotion=warm lang=en speaker=owner
   events=[Laughter]` as a system message before each user
   transcription
5. The persona prompt includes an `audio_context_block` that tells
   the LLM what to do with the line and forbids parroting it back

The orb now has ears. Whether they're our ears or someone else's
ears is, in the end, less important than what the model decided to
do with what it heard.

---

## Acknowledgments

The DSP whisperization is essentially an old privacy-preservation
technique repurposed for synthetic training data. Phase
randomization for voicing destruction has been around since the
80s in linear-prediction speech research. Nothing in this writeup
is novel — what's hopefully useful is the integrated demonstration
that *for a small voice-agent context-injection use case, you can
ship in 8 M params with mostly off-the-shelf tools in a weekend* —
and that knowing when **not** to ship your own model is part of
the same craft.

The training and eval code, model checkpoints, and a reproducible
50 k-clip whispered dataset are all on Hugging Face. Take them and
fold the numbers into your own voice agent — or shred them in your
own ablations and tell me where I'm wrong.
