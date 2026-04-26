# Design: SenseVoice Audio-Pre Integration

**Date:** 2026-04-26
**Status:** Draft
**Phase:** 1 (perception layer)
**ORBIS issue:** [#66](https://github.com/protoLabsAI/ORBIS/issues/66) — follows #35 (prerequisite: #35 PRs #45/#48 merged)
**Lab experiment:** `experiments/companion-stack/pipes/audio-pre/`

---

## Problem

ORBIS today sends audio straight from EchoGuard → Whisper → LLM. The LLM
receives text and nothing else — no signal about how the user sounded, their
emotional state, or what language they were speaking. A companion that can't
perceive emotion is missing a fundamental input.

The Phase 1 roadmap planned to fix this with two separate models running
in parallel: Whisper (STT) + audio-tags v5 (emotion2vec-based, 4-class).
This design replaces that plan with a single model — SenseVoice-Small —
that produces STT + 7-class emotion + language + audio events in one
70ms forward pass, while keeping the audio-tags v5 non-emotion heads
(SNR, environment, speaking-rate) as a parallel lightweight tap.

---

## Goals

1. Inject a structured `[audio]` annotation block into every LLM turn
   carrying: emotion label, confidence bucket, language, and audio events
2. Write emotion → mood as a per-turn nudge (reconciled against the
   neglect baseline — R15 fix)
3. Replace Whisper with SenseVoice-Small as the default local STT backend
4. Keep the audio-tags v5 non-emotion heads (SNR, env, speaking-rate)
   as a parallel tap — they aren't in SenseVoice

## Non-goals

- Valence/arousal continuous tracking (Phase 3)
- Prosody extraction / distress detection (Phase 2 stretch goal)
- On-device emotion inference (future — ONNX path exists when needed)
- Changing anything about the speaker-gate frame contract (PRs #45/#48)

---

## Architecture

### Pipeline placement

```
transport.input()
    ↓
EchoGuardSuppressor          (existing)
    ↓
RTVIProcessor                (existing)
    ↓
SpeakerGate                  (PR #45/#48 — owner-vs-stranger, ECAPA on-device)
    ↓  emits: OwnerVerifiedFrame / StrangerDetectedFrame (pass-through to STT)
SenseVoiceSTT                (new — replaces LocalWhisperSTT / OpenAISTTService)
    ↓  emits: TranscriptionFrame (unchanged contract for all downstream)
    ↓  also emits: EmotionFrame, AudioEventFrame (new, consumed by taps below)
AudioTagsTap                 (new — subscribes to EmotionFrame + raw audio)
    ↓  writes mood nudge → mem.personality
    ↓  appends [audio] block to per-turn system context
user_agg → ... rest of pipeline unchanged
```

Two shapes coexist as the research doc describes:
- **Gate** — SpeakerGate transforms frames (OwnerVerifiedFrame / StrangerDetectedFrame)
- **Tap** — AudioTagsTap observes EmotionFrame, fires async, pushes original frames unchanged

SenseVoiceSTT sits between them: it receives gated audio (owner-verified
or stranger-tagged) and emits both the standard TranscriptionFrame and
the new side-channel frames.

### SpeakerGate interaction

`AudioTagsTap` reads `OwnerVerifiedFrame` to decide whether to write mood.
Stranger audio still gets emotion-tagged (useful for the LLM context:
"a stranger sounded angry"), but mood writes are owner-only. The tap
suppresses mood writes when the last seen frame was `StrangerDetectedFrame`.

---

## New Components

### 1. `SenseVoiceSTT` (`voice/stt.py`)

A new `SegmentedSTTService` subclass added as a third `STT_BACKEND` option
(`STT_BACKEND=sensevoice`). Follows the exact same pattern as `LocalWhisperSTT`.

**Model:** `FunAudioLLM/SenseVoiceSmall` — 234M params, 936MB, Apache 2.0.
**Loading:** Lazy, same pattern as `_get_local_pipe()`. Prewarm on boot.
**Input:** WAV bytes from pipecat's SegmentedSTTService aggregator (same as Whisper).
**Inference:** FunASR `AutoModel` — one forward pass, non-autoregressive.

**Output frames per utterance:**

| Frame | When | Notes |
|---|---|---|
| `TranscriptionFrame` | always (if non-empty text) | unchanged downstream contract |
| `EmotionFrame` | always | even if emotion is `neutral` or `emo_unk` |
| `AudioEventFrame` | when non-speech events present | `BGM`, `Laughter`, `Applause`, etc. |

`EmotionFrame` and `AudioEventFrame` are emitted **before** `TranscriptionFrame`
so downstream taps can annotate the turn context before the LLM sees the text.

**Emotion parsing:** SenseVoice embeds emotion as a special token in the raw
output text (`<|HAPPY|>transcribed words`). Parse with a regex on the raw
output before calling `rich_transcription_postprocess`. If `ban_emo_unk=True`
(recommended), unknown emotion is suppressed and the frame carries
`emotion="neutral"` as the fallback.

**Language detection:** SenseVoice also outputs a language token (`<|en|>`,
`<|zh|>`, etc.). Include as `lang` field in `EmotionFrame` — free passenger,
no extra cost.

**`STT_BACKEND` selection:**

```
STT_BACKEND=local        → LocalWhisperSTT (unchanged)
STT_BACKEND=openai       → OpenAISTTService (unchanged)
STT_BACKEND=sensevoice   → SenseVoiceSTT (new default for GPU deployments)
```

`make_stt()` gains a third branch. No other changes to `app.py`.

**Prewarm:** `prewarm_stt()` calls `_get_sensevoice_model()` on the
background thread at startup — same pattern as Whisper prewarm.

**Latency target:** <150ms per utterance on GPU (published: 70ms for 10s
audio; expect higher for short 1–2s utterances due to fixed overhead).
Log `[stt.sensevoice]` with duration + emotion + lang, mirroring
`[stt.local]` log format.

### 2. `EmotionFrame` and `AudioEventFrame` (`agent/frames.py` — new file)

```python
@dataclass
class EmotionFrame(Frame):
    """Emitted by SenseVoiceSTT once per utterance.

    emotion: one of neutral | happy | sad | angry | fearful | disgusted |
             surprised | emo_unk
    confidence: high | medium | low  (derived from token probability if
                available; otherwise "medium" as sentinel)
    lang: BCP-47 code detected by SenseVoice (en, zh, ja, ko, yue)
    speaker_verified: True when last SpeakerGate frame was OwnerVerifiedFrame
    """
    emotion: str
    confidence: str        # "high" | "medium" | "low"
    lang: str
    speaker_verified: bool

@dataclass
class AudioEventFrame(Frame):
    """Emitted by SenseVoiceSTT when non-speech audio events are detected.

    events: list of detected events, e.g. ["Laughter", "BGM"]
    """
    events: list[str]
```

Both live in a new `agent/frames.py` — keeps custom ORBIS frame types
separate from the pipecat import chain, same boundary as `agent/prosody.py`.

**Note on confidence:** SenseVoice's standard API doesn't expose token
logits. `confidence` is derived heuristically — if the FunASR model is
run with `return_raw_text=True`, the token score may be extractable from
internal state. If not, always emit `"medium"` and document it clearly.
This is a known limitation; it constrains the distress-detection stretch
goal to Phase 2 when we add prosody features.

### 3. `AudioTagsTap` (`agent/audio_tags.py` — new file)

A `FrameProcessor` that:

1. Subscribes to `EmotionFrame` and `AudioEventFrame`
2. Tracks last `OwnerVerifiedFrame` / `StrangerDetectedFrame` to gate mood writes
3. On `EmotionFrame`:
   - Maps SenseVoice emotion label → mood nudge (valence + arousal delta)
   - Writes nudge to `mem.personality.set_mood()` **only for owner audio**
   - Appends `[audio]` annotation to per-turn context (see below)
4. Passes all frames through unchanged

**Emotion → mood mapping** (initial calibration, tunable via config):

| SenseVoice label | valence delta | arousal delta |
|---|---|---|
| `happy` | +0.10 | +0.05 |
| `surprised` | 0 | +0.10 |
| `neutral` | 0 | 0 |
| `sad` | -0.10 | -0.05 |
| `fearful` | -0.05 | +0.10 |
| `angry` | -0.15 | +0.15 |
| `disgusted` | -0.10 | +0.05 |

Deltas are additive nudges to current mood, clamped to [-1, 1].
NOT overwrites — this is the R15 fix: we nudge from whatever baseline
neglect set, rather than replacing it.

**R15 fix:** `AudioTagsTap` never calls `set_mood()` with absolute values.
It always reads current mood first, applies the delta, and writes the
result. The neglect baseline survives; emotion nudges it per-turn.
`agent/neglect.py` is unchanged.

**Per-turn context injection:** The tap needs to add the `[audio]` block
to the LLM's system context on each turn. Two options:

- **Option A (simpler):** Emit a `SystemFrame` with the annotation just
  before forwarding the `TranscriptionFrame`. Pipecat's `LLMContextAggregator`
  will pick it up as a system-role inject for that turn.
- **Option B (cleaner):** Store the annotation in a shared `UserState`
  field; `_effective_prompt` reads it and includes it in the next
  `_effective_prompt` call.

**Use Option A.** It's turn-scoped (annotation disappears after the turn,
doesn't accumulate in context), doesn't require modifying `_effective_prompt`,
and follows the same pattern as other per-turn system message injections.

**`[audio]` block format:**

```
[audio] emotion=happy lang=en events=[] speaker=owner
```

Single line, machine-readable keys, no filler text. The LLM is told in
the persona system prompt what the `[audio]` block means and how to
use it (adapt tone, acknowledge indirectly if warranted, never parrot it
back literally). The persona prompt addition is one paragraph, added to
the `tool_use_block` or as its own `audio_context_block()` function in
`app.py`.

**Owner-only mood writes, but always annotate:** Even for stranger audio,
the `[audio]` block is injected (the LLM knowing a stranger sounds angry
is useful context). Mood writes are gated on `speaker_verified=True`.

### 4. R15 fix — neglect reconciliation

No new code required beyond `AudioTagsTap`'s delta-write approach above.
The fix is architectural: `set_mood()` is already a partial-update function
(it accepts `None` for unchanged fields). The tap never passes absolute
values — always deltas from `get_mood()` — so neglect's baseline is preserved.

One test is added to `tests/test_neglect.py` to verify that a sequence of
`(apply_soft_neglect → emotion nudge)` produces the expected combined result,
not just the emotion nudge value.

---

## `_effective_prompt` addition

A new `audio_context_block()` function in `app.py` (alongside
`tool_use_block`, `repair_block`, etc.) returns a paragraph that tells
the orb about the `[audio]` annotation:

```python
def audio_context_block() -> str:
    return (
        "## AUDIO CONTEXT\n\n"
        "Each turn may begin with an [audio] line describing how the user "
        "sounded: emotion, language, and background events detected from "
        "their voice before transcription. Use this as additional signal — "
        "if emotion=angry, lean toward de-escalation; if emotion=happy, "
        "match the energy. Never quote or paraphrase the [audio] line back "
        "to the user. It is your perception, not their statement."
    )
```

This block is added to `_effective_prompt()` after `repair_block()`.
It is static — the same text every session. The per-turn emotion value
is in the `SystemFrame` injected by `AudioTagsTap`, not in this block.

---

## audio-tags v5 non-emotion heads

The existing audio-tags v5 model (`protoLabsAI/orbis-audio-tags-v5-soft`)
remains in use for its non-emotion outputs:

| Head | Output | Still needed? |
|---|---|---|
| emotion | 4-class label | ❌ replaced by SenseVoice 7-class |
| snr | float | ✅ |
| environment | indoor/outdoor/noisy/quiet | ✅ |
| speaking_rate | slow/normal/fast | ✅ |

The `AudioTagsTap` runs the v5 model heads (minus emotion) in parallel
with `SenseVoiceSTT` on `UserStoppedSpeakingFrame`. The SNR, environment,
and speaking-rate outputs are appended to the same `[audio]` line:

```
[audio] emotion=happy lang=en events=[] speaker=owner snr=high env=indoor rate=normal
```

The v5 model loads are lazy, identical to the existing audio-tags
experiment pattern. If `AUDIO_TAGS=off` env var is set, the non-emotion
heads are skipped and the `[audio]` line omits those fields.

---

## Config and env vars

| Variable | Default | Meaning |
|---|---|---|
| `STT_BACKEND` | `local` | `local` \| `openai` \| `sensevoice` |
| `SENSEVOICE_MODEL` | `FunAudioLLM/SenseVoiceSmall` | HF model ID |
| `SENSEVOICE_DEVICE` | `cuda` if available | `cuda` \| `cpu` |
| `AUDIO_TAGS` | `on` | `on` \| `off` — enables non-emotion v5 heads |
| `AUDIO_TAGS_MODEL` | `protoLabsAI/orbis-audio-tags-v5-soft` | HF model ID |

`SENSEVOICE_MODEL` is stored under `HF_HOME=/mnt/models/huggingface`
(already set in `~/.bashrc`). No storage config changes needed.

---

## Data flow diagram

```
UserStartedSpeakingFrame ──────────────────────────────────────→ [accumulate in SegmentedSTTService buffer]
InputAudioRawFrame (n)  ──────────────────────────────────────→ [buffer]
UserStoppedSpeakingFrame ─→ SenseVoiceSTT.run_stt(wav_bytes)
                                │
                                ├─→ EmotionFrame(emotion, lang, confidence, speaker_verified)
                                ├─→ AudioEventFrame(events=[...])   (if events detected)
                                └─→ TranscriptionFrame(text, user_id, timestamp)

EmotionFrame ─→ AudioTagsTap
                    │
                    ├─→ [mood nudge] mem.personality.set_mood(Δvalence, Δarousal)
                    │                (owner-only, delta not absolute)
                    ├─→ [run v5 non-emotion heads on same audio segment]
                    └─→ SystemFrame("[audio] emotion=... lang=... snr=... ...")
                         └─→ LLMContextAggregator sees it as per-turn system inject

TranscriptionFrame ─→ user_agg ─→ LLM (with [audio] already in context)
```

---

## Testing

### `tests/test_sensevoice_stt.py`
- Stub model via loader factory injection (same pattern as ECAPA test)
- Verify `EmotionFrame` emitted before `TranscriptionFrame`
- Verify `AudioEventFrame` only emitted when events detected
- Verify empty audio yields no frames (not a crash)
- Verify `speaker_verified` field reflects last seen SpeakerGate frame
- Verify `STT_BACKEND=sensevoice` routes to `SenseVoiceSTT` in `make_stt()`

### `tests/test_audio_tags_tap.py`
- Owner audio: verify mood write fires
- Stranger audio: verify mood write suppressed, `[audio]` block still injected
- R15: verify `(neglect_baseline → emotion_nudge)` produces correct combined mood
- Verify `SystemFrame` emitted with correct `[audio]` line format
- Verify `AUDIO_TAGS=off` suppresses v5 head inference

### `tests/test_neglect.py` (additions)
- New test: `apply_soft_neglect` followed by `AudioTagsTap` nudge
  yields combined result, not overwrite

---

## Sequencing within #35

```
PR 1 (speaker-gate foundation)      — in review, 3 bugs to fix
PR 2 (ECAPA embedder)               — in review, approved with minor notes
PR 3 (SenseVoice + AudioTagsTap)    — this design, starts after PR 1+2 merge
```

PR 3 is self-contained. It touches:
- `voice/stt.py` (add `SenseVoiceSTT` class + `sensevoice` branch in `make_stt`)
- `agent/frames.py` (new file — `EmotionFrame`, `AudioEventFrame`)
- `agent/audio_tags.py` (new file — `AudioTagsTap`)
- `app.py` (wire `AudioTagsTap` into pipeline, add `audio_context_block()`)
- `tests/test_sensevoice_stt.py` (new)
- `tests/test_audio_tags_tap.py` (new)
- `tests/test_neglect.py` (additions)
- `pyproject.toml` or `requirements.txt` (add `funasr`, `torch-audio` deps)

R15 is fixed as part of PR 3 (it's the `AudioTagsTap` delta-write pattern),
not as a separate PR. The fix is two lines of logic in the tap.

---

## Open questions (not blockers)

1. **SenseVoice confidence scores:** The standard FunASR API doesn't expose
   token probabilities. If we want `confidence=high/medium/low` to be
   meaningful (not always `"medium"`), we need to either use the internal
   `return_raw_text=True` path or accept the limitation. Decide before
   implementing the distress-detection stretch goal in Phase 2.

2. **`[audio]` block in context accumulation:** SystemFrame injects per-turn
   but pipecat's rolling summarizer may compress history including these
   lines. Test whether the summarizer strips or preserves `[audio]` lines
   after compression. If it preserves them, they'll accumulate correctly
   in long sessions; if it strips them, that's fine too (they're per-turn
   signal, not long-term memory).

3. **Language routing:** SenseVoice detects zh/en/ja/ko/yue. If the detected
   language differs from the persona's configured language, should ORBIS
   adapt? Out of scope for Phase 1 — just log it. Phase 2 can add language-
   specific persona routing.

4. **audio-tags v5 emotion head deprecation:** The v5 model on HuggingFace
   still has an emotion head. We're not removing it — the model artifact
   stays as-is. We just don't run the emotion head in production anymore.
   Update the model card to note this when the blog post goes out.
