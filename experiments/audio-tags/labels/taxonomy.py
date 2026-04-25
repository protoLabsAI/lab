"""Tag taxonomy v0 — locked for the first training run.

Each entry declares: head name, head type, label set (or output range),
loss type, and the source of training labels. Dropping a head is fine;
adding one mid-run breaks the model file format. Bump to v1 if the
schema changes.

Audit findings (see labels/audit.py output):
  - "low" pitch is 0.2% of LibriSpeech → merged with "medium" → {medium, high}
  - "quiet" volume is 0.04% → dropped → {normal, loud}
  - "slow" speed is 1% → kept but oversampled at training time
  - LibriSpeech is mood-flat audiobook speech → mood/valence/arousal
    targets come from emotion2vec on audio + prose extraction; we
    expect the LibriSpeech-only baseline to underperform on mood until
    we mix in MELD/RAVDESS/DailyTalk
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Head:
    name: str
    type: str  # "classification" | "regression"
    classes: tuple[str, ...] = ()
    range: tuple[float, float] | None = None
    source: str = ""  # how labels are produced
    notes: str = ""


HEADS: tuple[Head, ...] = (
    # ── speaker ─────────────────────────────────────────────────────
    Head(
        name="speaker_gender",
        type="classification",
        classes=("female", "male", "unknown"),
        source="prose-extract from description text + spkrec gender model fallback",
    ),
    Head(
        name="speaker_age",
        type="classification",
        classes=("child", "young_adult", "adult", "older_adult", "unknown"),
        source="prose-extract; weak label, expect noisy",
        notes="Optional v0 head — drop if labels too noisy.",
    ),
    # ── mood / affect ───────────────────────────────────────────────
    Head(
        name="mood_class",
        type="classification",
        classes=(
            "neutral",
            "calm_positive",
            "warm",
            "excited",
            "sad",
            "tense",
            "playful",
        ),
        source="emotion2vec-large on audio (primary) + prose-extract (validation)",
        notes="LibriSpeech is mood-flat — expect MELD/RAVDESS finetune later.",
    ),
    Head(
        name="valence",
        type="regression",
        range=(-1.0, 1.0),
        source="emotion2vec arousal/valence outputs",
    ),
    Head(
        name="arousal",
        type="regression",
        range=(-1.0, 1.0),
        source="emotion2vec arousal/valence outputs",
    ),
    # ── acoustic / prosody ──────────────────────────────────────────
    Head(
        name="volume",
        type="classification",
        classes=("normal", "loud"),
        source="rule-extracted from RMS (existing labels)",
    ),
    Head(
        name="pitch",
        type="classification",
        classes=("medium", "high"),
        source="rule-extracted from F0 (existing labels, 'low' merged into 'medium')",
    ),
    Head(
        name="speaking_speed",
        type="classification",
        classes=("slow", "normal", "fast"),
        source="rule-extracted from words/sec (existing labels)",
    ),
    Head(
        name="snr_db",
        type="regression",
        range=(0.0, 90.0),
        source="rule-extracted (existing labels)",
        notes="Useful as a noise-floor proxy for ORBIS environment hint.",
    ),
    # ── environment / context ──────────────────────────────────────
    Head(
        name="environment",
        type="classification",
        classes=("indoor_quiet", "indoor_noisy", "outdoor", "phone_call", "unknown"),
        source="prose-extract + SNR threshold; LibriSpeech is all 'indoor_quiet'",
        notes="Real signal will only emerge after Common Voice / AMI / DailyTalk mix.",
    ),
    Head(
        name="speech_style",
        type="classification",
        classes=("conversational", "narration", "oratorical", "dramatic", "unknown"),
        source="prose-extract from description text",
        notes="Genre cue for ORBIS persona — 'is the user reading vs talking?'",
    ),
    # ── voice quality (v0.1) ─────────────────────────────────────────
    Head(
        name="voice_quality",
        type="classification",
        classes=("voiced", "whispered"),
        source="LibriSpeech (all voiced) + DSP-whisperized LibriSpeech (whispered) "
               "+ self-recorded held-out",
        notes="Added v0.1 (2026-04-25). DSP whisperization: STFT phase "
              "randomization + sub-300Hz attenuation + HF lift. Validated "
              "on self-recorded held-out for distribution shift.",
    ),
)

HEADS_BY_NAME: dict[str, Head] = {h.name: h for h in HEADS}

# Wire format version — bump if HEADS changes
SCHEMA_VERSION = "v0.1"


def example_output() -> dict:
    """Canonical example of the JSON ORBIS would receive."""
    return {
        "schema": SCHEMA_VERSION,
        "speaker": {"gender": "female", "age": "adult"},
        "mood": {"class": "warm", "valence": 0.42, "arousal": -0.15},
        "acoustic": {
            "volume": "normal",
            "pitch": "medium",
            "speaking_speed": "normal",
            "snr_db": 38.2,
            "environment": "indoor_quiet",
            "voice_quality": "voiced",
        },
        "style": "conversational",
        "confidence": {
            "speaker.gender": 0.94,
            "mood.class": 0.71,
            "environment": 0.88,
            "voice_quality": 0.99,
        },
    }


if __name__ == "__main__":
    import json

    print(f"Audio-tags taxonomy {SCHEMA_VERSION}\n")
    for h in HEADS:
        if h.type == "classification":
            label = f"{len(h.classes)}-way: " + ", ".join(h.classes)
        else:
            label = f"regression in {h.range}"
        print(f"  {h.name:18s}  {h.type:14s}  {label}")
        if h.notes:
            print(f"  {' ' * 18}  ↳ {h.notes}")
        print()
    print("Example output:")
    print(json.dumps(example_output(), indent=2))
