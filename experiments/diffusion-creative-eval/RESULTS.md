# Diffusion-vs-AR creative-writing eval — RESULTS

**Date:** 2026-06-14
**Question:** For the `protolabs/fast` creative-writing lane, does DiffusionGemma (DG, text-diffusion)
write more human-like / less-sloppy prose than its AR fallback (Gemma 4 26B-A4B FP8)?
**Method:** 50 Reddit-WritingPrompts prompts with human reference stories. Each model drafts a
~350-word story per prompt. Scored with Rosmine-style metrics: MMD (embedding distribution
distance, Qwen3-Embedding-0.6B), Token-L2 (1-gram word distribution), slop-sign rate
(em-dash + stock-phrase + "not X, it's Y" per 1k words), self-BLEU, and a pairwise judge
(local 27B, Qwen — same judge for both candidates, so the head-to-head is unbiased).

## Scorecard

| model  | MMD↓   | TokenL2↓ | JMQ↑ | slop/1k↓ | self-BLEU | tok/s |
|--------|--------|----------|------|----------|-----------|-------|
| dg     | 0.0591 | 0.0662   | 1.92 | 6.10     | 0.0004    | 382   |
| gemma4 | 0.0516 | 0.0630   | 1.96 | 5.64     | 0.0003    | 197   |
| human  | 0.0    | 0.0      | 1.0  | 1.56     | 0.0001    | —     |

**Direct head-to-head:** Gemma 4 wins **64%** of pairwise comparisons (DG 36%).

## Findings

1. **AR Gemma 4 writes modestly but consistently better creative prose than DG.** It wins the
   direct head-to-head 64/36 and is closer to human on MMD (0.052 vs 0.059), Token-L2, and slop
   (5.6 vs 6.1 /1k). The signal is consistent across all three quality axes.
2. **DG's only edge is speed: ~1.9× (382 vs 197 tok/s).** The lane tradeoff is now quantified:
   ~1.9× throughput for a ~64/36 quality deficit on creative writing.
3. **JMQ is not discriminating here.** Both models score ~1.9 (judge prefers polished LLM prose
   over raw Reddit human text — the documented LLM-judge bias; optimal vs-human JMQ is 1.0).
   The reliable signals are MMD, slop, and the *direct* head-to-head (where judge bias cancels).
4. **Both models are ~3.5–4× sloppier than humans** (5.6–6.1 vs 1.56 slop/1k). The Rosmine
   anti-slop gap is present in our prose lane regardless of which model serves it.

## Caveats / next steps
- n=50; MMD uses Qwen3-Embedding-0.6B (not Rosmine's llama-embed-nemotron-8b) so absolute
  magnitudes are internal-only, not comparable to Rosmine's paper. The *ranking* is what holds.
- Judge is the local 27B. For a publishable number, re-run the head-to-head through an
  independent gateway judge (Claude) once `ava:4000` is reachable.
- Scale to ~200 prompts and add the WildBench creative subset (`runners/run_wildbench.py`).
- Human refs are Reddit WritingPrompts (informal register); a literary-fiction reference set
  would test a different "human" target.

## Repro
```
python fetch_refs.py                 # -> data/human_refs.jsonl (80 human prompt+story pairs)
python generate.py --label dg        # DG on :8002
# (swap GPU1 to AR Gemma 4, then:)
python generate.py --label gemma4
python score.py --gen out/gen_dg.jsonl out/gen_gemma4.jsonl
```
