# prosody-tagger — make the LLM's text speakable

**Pipe**: text-post.
**Status**: planned (Phase 3).

## Problem

The LLM's text output is grammatically rich but **prosodically
flat**. TTS reads it linearly. Fish S2 Pro supports inline tags
(`[softly]`, `[pause:300]`, `[laughing]`, `[whisper]`, etc.) that
dramatically improve listener-perceived naturalness — but the LLM
doesn't insert them by default unless system-prompted to, and even
then inconsistently.

## Why ORBIS needs it specifically

ORBIS *already has* the prosody-tag-stripping pattern in
`protoVoice/agent/prosody.py` (Fish consumes them, Kokoro/OpenAI
strip them). What it doesn't have is a deterministic *inserter*.
The companion layer's emotional resonance lives in this pipe — a
warm response read flat is just a sentence; the same response
with `[softly]` and `[pause:200]` lands as comfort.

## Target behavior

Given:
- LLM text response
- Current mood state (audio-tags + LLM-inferred)
- Persona config (orb's tone preferences)

Output: same text with prosody tags inserted.

Examples:

```
in:  "I'm sorry to hear that."
out: "[softly] I'm sorry [pause:200] to hear that."

in:  "That's fantastic news!"
out: "[excited] That's fantastic news!"

in:  "Let me think about that for a second."
out: "Let me [pause:300] think about that [thinking] for a second."
```

## Candidate architectures

1. **Span-tagger over BERT-style encoder** — sequence labeling
   (BIO scheme) over tokens, decides where to insert which tag.
   ~100 M params.
2. **Small T5 / FLAN-T5** seq2seq — input plain text + mood, output
   tagged text. More expressive but more expensive; ~300 M params.
3. **Rule engine + small classifier** — punctuation-based pause
   insertion + a sentence-level "needs softening" classifier on top.
   The pragmatic v0; ships in days.
4. **LLM as a small distillation target** — generate (text, mood,
   tagged text) triples with Qwen 3.6-27B, distill into a small
   LM.

Start with (3); upgrade to (1) or (4) when we have data.

## Datasets

- **Self-generated synthetic** — Qwen 3.6-27B generates
  (mood, plain_text, tagged_text) triples from a prompt template.
  Hand-review a sample for quality.
- **Real ORBIS responses** — once we have them, cluster by mood +
  manually annotate gold tags.
- **Existing prosody-rich corpora** — VCTK, audiobook recordings
  with manual SSML annotations. Limited but useful for spot-checking.

## Eval plan

Hard problem — prosody is inherently subjective. Plan:

1. **A/B blind listening test** — small panel rates plain TTS vs
   tagged TTS on naturalness, warmth, emotional alignment. ~50
   pairs.
2. **Tag insertion precision/recall** — on synthetic gold data,
   does the tagger insert tags in the right spans?
3. **Latency** — must be < 50 ms (this runs between LLM response
   and TTS start).

## Deliverables

- HF model: `protoLabsAI/orbis-prosody-tagger-v0`.
- ORBIS integration: drop-in upgrade to
  `protoVoice/agent/prosody.py` — currently only strips, needs
  to also insert.
- Blog post: "Why your voice agent sounds robotic, and how to fix
  it cheaply."

## Open questions

- Per-persona prosody — different orbs have different tag profiles
  (warm orb uses `[softly]` more, energetic orb uses `[excited]`)?
- Stream-friendly insertion — TTS might start before LLM response
  is complete; can the tagger run incrementally on text chunks?
- Calibration with audio-tags mood — closes the loop where the
  user's mood drives the orb's prosody.

## Dependencies

- Audio-tags mood signal in production (Phase 1 dependency).
- Fish S2 Pro tag schema documentation (we have this from earlier
  work).
