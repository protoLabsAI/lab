# ORBIS-Audio-Tags-v0 — Dataset Plan

For the HuggingFace release alongside the v2 model. Goal: a small,
honest, fully-reproducible dataset that lets others train their own
audio-tag heads without LDC paywalls or YouTube rot.

## What's in vs out

| Source | Status | Hours | Whispered? | License |
|---|:---:|---:|:---:|---|
| LibriSpeech (clean-100/360, other-500) | reuse | ~960 | ❌ | CC-BY-4.0 |
| Synthesized via Fish S2 Pro (paired voiced + whispered) | **make** | ~8 | ✅ | Ours, CC-BY-NC-4.0 (Fish output license) |
| Self-recorded held-out | **make** | ~0.1 | ✅ | Ours, CC-BY-4.0 |
| wTIMIT (LDC) | **skip** | — | — | $125-250, days to acquire |
| VocalSound | **skip for v0** | — | ❌ | Has no whisper class — useful for future "non-speech vocalization" head |
| AudioSet whisper subset | **skip** | — | weak | YouTube rot, weak labels |

## Why Fish synthesis works as ground truth

Empirical check on `/tmp/fish-whisper-test/` (2.5-3s prompt, default voice):

|  | RMS | Unvoiced frame % | Notes |
|---|---:|---:|---|
| Fish voiced | 0.206 | 12% | normal-voiced ratio |
| Fish `[whisper]` | 0.046 | **51%** | stage-whisper acoustics |
| Real soft whisper (typical) | ~0.04 | 80-100% | gold standard |
| Real stage whisper | ~0.05 | 40-70% | what Fish matches |

Fish output is "stage whisper," which is actually closer to what
ORBIS will see at typical microphone distances than intimate whisper.
We accept the gap and document it.

## Generation plan (~8h on Blackwell, runnable Saturday)

**Source prompts**: 5,000 transcripts from LibriSpeech (random
sample of utterances 5-15 words). Reuse the transcripts in
`/mnt/data/salm-duplex/data/*-attributes.jsonl` — already curated.

**Per prompt**, generate two samples:
1. `text` (voiced)
2. `[whisper] {text}` (whispered)

Yields **10,000 audio files** (~5h on disk @ 44.1kHz mono ~ 2.5GB).

**Voice diversity**: cycle Fish voice IDs across the prompt set to
avoid speaker-coupled overfitting. Use `default` + 3-5 cloned voices
already in `/mnt/data/fish-references/`.

**Validation**: random-sample 200 outputs, audit unvoiced ratio,
clip duration, intelligibility. Flag and regenerate any with
unvoiced<30% on whispered or unvoiced>20% on voiced (likely
synthesis failures).

## Self-recorded held-out (~10 min)

User records ~30 short utterances in two passes:
1. Normal speech
2. Whispered (varying intensity)

Saved as 16kHz mono WAV. Tagged in a tiny manifest. Used **only** at
eval time — never seen in training. This is the fence against
Fish-only overfit.

## HuggingFace dataset shape

Repo: `huggingface.co/datasets/protoLabsAI/orbis-audio-tags-v0`

```
orbis-audio-tags-v0/
├── README.md          (dataset card)
├── manifest.parquet   (audio_path, source, voice_quality, all_tags...)
├── synthesized/
│   ├── voiced/        (5000 WAVs from Fish, 16kHz)
│   └── whispered/     (5000 WAVs from Fish, 16kHz)
├── held-out/          (10 min self-recorded, 16kHz)
└── labels/
    ├── prose-train.jsonl       (Qwen-extracted prose tags)
    ├── librispeech-attributes.jsonl   (full 281k attribute join)
    └── speakers.txt    (LibriSpeech-style gender lookup, copied)
```

We do **NOT** redistribute LibriSpeech audio — link to the canonical
download. The labels parquet keys by `audio_path` so users join
locally after fetching LibriSpeech.

## Provenance and limitations (for the dataset card)

- **Whispered samples are synthetic.** Real-whisper acoustics are
  approximated, not exactly reproduced. Held-out self-recorded set
  benchmarks the gap.
- **Mood/style tags are LLM-extracted from descriptions, not human-
  labeled.** They're consistent under that LLM but inherit any of its
  biases.
- **English-only.** Fish supports multilingual but synthesis quality
  for whisper drops in non-English (per Fish docs).
- **Indoor/clean only.** No environment noise variation. ORBIS
  deployments in noisy rooms will need a separate noise-augmentation
  pass.
- **Speaker demographics inherit LibriSpeech.** ~50/50 gender, but
  US/UK English readers, audiobook genre. No children's voices.

## Why the LDC skip is fine

wTIMIT is the academic gold standard, but for this experiment:
- Cost ($125-250) + multi-day account approval doesn't fit weekend
- It's a *benchmark*, not a *training corpus* — it's small (16h)
- Our Fish-synth corpus is 8h, comparable scale, faster to obtain
- We can request wTIMIT post-blog for rigorous benchmark numbers in
  a follow-up
