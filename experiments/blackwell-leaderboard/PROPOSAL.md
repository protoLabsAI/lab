# blackwell-leaderboard — PROPOSAL

> Status: draft 2026-05-22. Not started. Comment / push back before this graduates to `PLAN.md`.

## Thesis

A reproducible 2026 snapshot of every model we run on 192 GB of Blackwell VRAM, scored on the same eval bar as the current cloud frontier. The leaderboard in `README.md` is six weeks stale; the snapshot itself is the artifact.

The content angle — *"What 192 GB of VRAM buys you in 2026"* — is the breakdown the audience filter (§3) actively wants: practitioners deciding whether to self-host vs pay per-token. Real numbers, raw JSON outputs published, claw-eval is already open-source.

## What ships

1. **HuggingFace dataset** at `protoLabsAI/blackwell-leaderboard-2026q2` — raw eval outputs + scores per model per suite, JSONL.
2. **Blog post** at protolabs.studio — the narrative around the numbers. *"Self-host vs frontier in 2026 — the honest math."*
3. **Refreshed `README.md` leaderboard** in this repo, with a permanent link to the dataset and an `as-of` date.
4. **One reproduction recipe** — single command per model, every score in the table reproducible from the published configs.

Privacy posture: HF dataset goes up private during the run, public after blog passes voice review.

## Scope

### Models — local

Pulled from `models/vllm-swap.sh`. One config per model unless the quant story matters (then both).

| Model | Quant | Config | Notes |
|---|---|---|---|
| Qwen3.6-4B | INT4 | `qwen-4b-int4` | Edge floor |
| Qwen3.6-4B | bf16 | `qwen-4b` | LoRA base |
| Qwen3.6-9B | FP8 | `qwen-9b-fp8` | On-the-fly FP8 |
| Qwen3.6-27B | INT4 + MTP | `qwen-27b-int4-mtp` | Daily driver chat/creative |
| Qwen3.6-27B | FP8 + MTP | (current `vllm.service`) | Daily driver thinking |
| Qwen3.6-35B MoE | FP8 official | `qwen-35b` | Speed king |
| Qwen3.6-35B MoE | FP8 heretic | (current `vllm-fast.service`) | Uncensored variant |
| Qwen3.6-122B | INT4 TP=2 | `qwen-122b-int4` | Quality ceiling (faster on PCIe) |
| Qwen3.6-122B | FP8 TP=2 | `qwen-122b-fp8` | Quality ceiling (official) |
| Gemma 4 31B | FP8 | dense alt | If MTP config lands by run time |

### Models — cloud reference

Same eval, same prompts. Run via the gateway.

- Claude Sonnet 4.6 + 4.7 (1M)
- Claude Opus 4.6 + 4.7 (1M)
- Claude Haiku 4.5 (the floor)
- GPT-5.4
- Gemini 3.1 Pro
- GLM 5 Turbo (the open-weights frontier reference)

### Suites

Tiered. Run more on the local family, sample on the cloud refs (cost gate).

| Tier | Suite | Tasks | All-models? |
|---|---|---|---|
| 1 | **claw-eval pass^3** | 52 EN | yes |
| 1 | **function-call** | 8 | yes |
| 2 | **custom suites** (coding, reasoning, structured, summarization, instruction-following) | 35 | yes |
| 2 | **safety + refusal** (XSTest 450 + custom safety 5) | 455 | yes |
| 3 | **WildBench sampled** | 100 of 1024 | local-only first; cloud refs only on a 25-task subset |
| 3 | **creative_writing + roleplay + svg + research** | 19 | local only (open question — see below) |

## Tier-0 baselines (audio-tags principle)

Borrowed from the methodology that worked on audio-tags. Three reference points before any model is "ranked":

- **Majority / random**: pass^3 random baseline = 0% for closed-form tasks. Establishes the floor.
- **Linear probe equivalent**: cheapest cloud model (Haiku 4.5). If our 4B INT4 beats Haiku, that's the story. If it doesn't, that's also the story.
- **Off-the-shelf comparable**: pull the most-recent published claw-eval numbers (if any) + a snapshot from LMSys / Artificial Analysis / OpenRouter rankings on at least one overlapping benchmark to ground the rankings against external signal.

If our internal scoring contradicts an external leaderboard by more than ~5 points on the same model, **investigate before publishing.** Either we found a bug in our pipeline or the external board is contaminated; either is publishable, but we have to know which.

## Eval ladder

```
1. Tier 1 (claw + function-call) on local family       — ~2 days wall
2. Tier 2 (custom + safety) on local family            — ~3 days wall
3. Cloud refs on Tier 1 + Tier 2                       — ~$X budget gated
4. WildBench sampled                                   — opt-in based on budget left
5. Cross-domain held-out (refusal at scale)            — ~1 day
6. Sanity checks (Tier-0 baselines comparison)         — pre-publish
```

Order matters: cheapest first, fail fast. If claw-eval pipeline breaks on the first model swap, we want to know inside two hours, not on day five.

## Risks

| Risk | Mitigation |
|---|---|
| `claw-eval` submodule drift mid-run | Pin to a specific commit before kickoff; record it in `RESULTS.md` |
| Cloud model deprecation | Snapshot dates per provider; if a model is sunset mid-run, note it |
| Gateway flakiness skews cloud numbers | Re-run flaky cloud requests with same seed; document retries |
| Submitting numbers worse than already-published ones | Don't publish until tier-0 sanity check passes. Negative result is fine; *uninvestigated* negative result is not |
| Pre-existing WIP in `evals/runners/*.py` not reviewed | Resolve `evals/run.sh` + runner edits + commit before kickoff |

## Content target (drives all earlier decisions)

**Title candidates:**
- *"What 192 GB of VRAM buys you in 2026"*
- *"Self-host vs frontier — the 2026 honest math"*
- *"Every Qwen3.6 we have, scored against every cloud model we use"*

**Lede must be a specific number** (V1 — specific over abstract). Something like *"Our 27B INT4 at 53 tok/s sits 4 points below Sonnet 4.7 on our agentic-tool benchmark. We pay zero per token."*

**Cross-links (P7 — patterns over products):**
- HF dataset → repo → blog → claw-eval submodule
- Cite our prior FP8 quants on `protoLabsAI/`
- Cite `models/RESULTS.md` for the MTP / TP=2 findings that informed config choices

## Open questions

1. **pass^3 or pass^1 for cloud refs?** Pass^3 triples cost. Lean: pass^3 on Tier 1 only, pass^1 on Tier 2+.
2. **WildBench yes/no?** 1,024 tasks ≈ tens of hours per local model. If included, sampled to 100; the full 1,024 becomes a follow-up.
3. **Subjective suites (creative_writing, roleplay) in or out?** They need LLM-judge scoring; judge variance is its own can of worms. Lean: include in the dataset, *don't include in the headline leaderboard*. Footnote-tier.
4. **What's the headline metric?** Pass^3 average across Tier 1, or a composite? Composites hide things; lean toward two-number reporting (pass^3 + tok/s) per model.
5. **Daily downtime budget.** vLLM swaps are ~5 min. Cloud refs hit the gateway and don't disrupt local serving. If runs are scheduled overnight, no downtime on the daily-driver alias. Confirm before kickoff.
6. **Does `proto-bench/` get pulled into this or stay parked?** Its SWE-bench / Terminal Bench task definitions are coding-agent-shaped (parked direction), but the proto_agent harness might be reusable. Decision deferred to PLAN.md.

## Out of scope (explicit)

- New eval suites — use what `evals/` already has
- Fine-tuning anything — pure inference benchmark
- Comparing to specific competitors by name in the blog (V7 mechanics, not feelings)
- Promising future runs on a schedule (snapshots are snapshots)

## Next step

Review this. When the open questions resolve, this graduates to `PLAN.md` with the run order pinned and the cloud budget approved.
