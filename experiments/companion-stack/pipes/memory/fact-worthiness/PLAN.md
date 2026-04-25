# fact-worthiness — what's worth saving long-term

**Pipe**: memory.
**Status**: planned (Phase 3).

## Problem

ORBIS saves to its `facts` table. The current curator decides what
to save via LLM-driven extraction or hand-rolled rules.

Both approaches have failure modes:

- LLM extraction is expensive (every turn) and inconsistent
  (sometimes saves trivial chitchat, sometimes misses a real
  preference).
- Hand-rolled rules are brittle (regex over names, dates) and miss
  semantic facts ("user mentioned they're a vegetarian for the
  third time").

## Target output

Binary classifier per turn: `is_fact_worthy ∈ {True, False}`.

Optionally with structure:
- `category`: `personal_info | preference | event | task |
  relationship | trivia | none`
- `extracted_fact`: `(subject, relation, object)` triple if
  category != none

For a v0, just the binary gate. Curator runs LLM-extraction only
on positive predictions, cutting curator cost by ~90%.

## Why ORBIS needs it specifically

The whole companion-layer thesis is that the orb *remembers*. Memory
quality is how that promise lives or dies. Today the bottleneck is
that curator overhead is high, so the curator runs sparingly, so
real facts get missed.

A cheap binary gate inverts that — run curator on most turns, but
only do the expensive LLM extraction on the ~10% that score positive.

## Candidate architectures

1. **DistilBERT + binary head** (~66 M params) trained on synthetic
   + hand-labeled (utterance, fact_worthy) pairs.
2. **Sentence-transformer + logistic regression** for the v0.
3. **Few-shot via SetFit** if labeled data is scarce.

## Datasets

- **Synthetic via LLM** — Qwen 3.6-27B generates ORBIS-style
  conversational turns + a fact-worthiness label. Bootstrap 1-2k
  examples.
- **Real ORBIS logs** — manually labeled subset, gold standard.
- **PERSONA-CHAT, MultiWOZ, DailyDialog** — public conversational
  datasets (not directly ORBIS-shape but useful for transfer).

## Eval plan

1. **Recall @ precision = 0.9** — false negatives (missing real
   facts) hurt more than false positives (saving trivia, which the
   half-life decay culls naturally anyway).
2. **Curator-cost reduction** — with the gate in place, what's the
   per-session reduction in LLM-extraction calls vs no gate?
3. **Long-horizon fact recall** — after N sessions, can the orb
   answer "remember when I said X?" tests at the same rate as the
   LLM-curator-on-every-turn baseline?

## Deliverables

- HF model: `protoLabsAI/orbis-fact-worthiness-v0`.
- ORBIS integration: pre-curator gate in the memory writer. The
  expensive LLM-driven `(subject, relation, object)` extractor
  runs only on positive predictions.
- Blog post (with coreference-resolver if shipped together):
  "Memory architectures for tiny voice agents."

## Open questions

- Per-domain calibration — preferences vs events vs personal_info
  may need different thresholds.
- Should this output extracted facts directly, or only gate the
  expensive extraction path?
- Multi-turn context — facts often come from a 2-3 turn exchange,
  not a single utterance. Does the classifier need the prior turn
  as input?

## Dependencies

- ORBIS logs at some scale to evaluate against. v0 can ship on
  synthetic data only.
- Coreference-resolver pairs naturally with this — knowing *which
  entity* a fact is about completes the (subject, relation, object)
  triple.
