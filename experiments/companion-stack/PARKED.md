# PARKED — companion-stack

Parked 2026-05-22. ORBIS retired as a product; the umbrella + 9 unshipped pipes have no downstream consumer. See [project_brand_pivot.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md).

## What survives the pivot

- **audio-tags v0–v5** — already public on HF `protoLabsAI/orbis-audio-tags-*`. Lives at `experiments/audio-tags/` as the standalone brand exemplar. The `pipes/audio-pre/audio-tags/` copy here is the research home; do not delete.
- **Phase 0 + Phase 1 findings** in `ROADMAP.md` and `LEARNING.md` are breakdown material for protolabs.studio. Frozen-encoder + tiny-heads, sqrt class weighting, tier-0 baselines, DSP whisperization — all worth a post.

## What parks (the rest)

`pipes/audio-pre/{speaker-verification,wake-word,prosody-tagger}`, `pipes/text-pre/{intent-classifier,tool-need-predictor,fact-worthiness}`, `pipes/llm-context/{reranker,planner-executor}`, `pipes/visual/mood-to-palette`. Scaffolds, partial runs, no shipped artifacts.

## Where it stood

- Phase 0 ✅ shipped (audio-tags)
- Phase 1 ✅ engineering done (speaker-verification, SenseVoice substitution merged in ORBIS #66); content owed
- Phase 2 (intent-classifier, tool-need-predictor, reranker) was next; never started

## How to resume

A pipe model graduates only if it serves a new consumer. The ORBIS-shaped consumer is gone. Either:
1. New voice product needs a pipe → unpark that one pipe, port to a fresh experiment dir
2. A pipe finding stands alone as a breakdown → write the blog from existing artifacts; no resume needed

Don't reopen the umbrella. The thesis (small models per pipe) survives in audio-tags; reuse that shape if a real consumer appears.
