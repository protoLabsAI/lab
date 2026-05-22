# tiny-models-bench — PLAN

> Replaces the cloud-heavy `blackwell-leaderboard` plan that was drafted earlier this session. User redirect 2026-05-22: *"chill on cloud compute. lets get into tiny models up to say 9b."* See [memory](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_tiny_models_direction.md).

## Thesis

A capability-vs-cost map of every tiny LLM (≤9 B) that's worth loading in 2026. Brand piece: *"We tested every tiny LLM we could load. Here's what they're actually good at, and where the cliff lives."*

The heavy rig is the **forge** — used to load, swap, and (in follow-on experiments) fine-tune + quantize + export to LiteRT-LM. The *deploy* target is whatever consumer hardware our audience runs. This experiment is the run-only first pass; v2 adds fine-tune.

## What ships

1. **HuggingFace dataset** `protoLabsAI/tiny-models-bench-2026q2` — JSONL: per (model, suite, task, trial) → score + tokens + duration.
2. **Headline table** in `RESULTS.md` — model × task accuracy + tok/s + VRAM. Lead with one specific number per V1.
3. **Blog post** `BLOG.md` on protolabs.studio — the capability cliff, the surprise winners, the recipe for choosing one.
4. **`models/RESULTS.md`** addition — pin the new tok/s and VRAM numbers per model.

## Models (≤9 B, sorted by params)

| Rank | Model | Params | Quant | On disk? | Notes |
|---|---|---|---|---|---|
| 1 | SmolLM2-135M | 135 M | bf16 | no — pull | Hugging Face SmolLM2 floor |
| 2 | function-gemma-270m | 270 M | bf16 | no — pull | Gemma 3 base, fine-tuned for function calling |
| 3 | SmolLM2-360M | 360 M | bf16 | no — pull | |
| 4 | Qwen3.6-0.8B-Base | 800 M | bf16 | yes | Already inventoried |
| 5 | Gemma 4 E2B | ~2 B | bf16 + on-the-fly FP8 | check | The talk's headline edge model |
| 6 | Qwen3.6-2B-Base | 2 B | bf16 | yes | Training-experiments inventory |
| 7 | Llama-3.2-1B | 1 B | bf16 | no — pull | Standard reference |
| 8 | Llama-3.2-3B | 3 B | bf16 | no — pull | |
| 9 | Phi-3-mini | 3.8 B | bf16 + on-the-fly FP8 | no — pull | Microsoft's reference small |
| 10 | OLMo-1B | 1.2 B | bf16 | no — pull | AI2's open model |
| 11 | Qwen3.6-4B | 4 B | INT4 + bf16 + on-the-fly FP8 | yes | Existing inventory |
| 12 | Gemma 4 E4B | ~4 B | bf16 + on-the-fly FP8 | check | The talk's upper edge model |
| 13 | Qwen3.6-9B | 9 B | FP8 + bf16 | yes | Upper bound of "tiny" per user spec |

Quantization variants run only where they meaningfully change the deploy story (INT4 for serving, on-the-fly FP8 for inference). bf16 always present as the baseline.

## Suites

Local judge throughout (`protolabs/fast` = Gemma 4 26B MoE FP8). Zero cloud spend.

| Tier | Suite | Tasks | Trials | Why it matters for tiny models |
|---|---|---|---|---|
| 1 | **function-call** | 8 | 3 | The headline task per the talk — function-gemma 270 M is fine-tuned to this |
| 1 | **structured_output** | 5 | 3 | Tiny models hate JSON; this is where the cliff usually lives |
| 1 | **instruction_following** | 5 | 3 | Constraint adherence at small scale |
| 1 | **refusal** (XSTest + simple_safety) | 550 | 1 | Does a 270 M model have a moral compass? Real question |
| 2 | **reasoning** | 5 | 1 | Sub-4B will fail this; that's a finding |
| 2 | **summarization** | 5 | 1 | Narrow task, classic small-model use case |
| 2 | **coding** (gen only, 5/10) | 5 | 1 | Sanity floor — not a coding-agent eval, just "can this write a 10-line function" |
| 3 | **claw-eval** (sampled 6 of 52) | 6 | 1 | Diagnostic only — most tiny models will floor at near-zero; we include to *show* the floor |

**Out:** WildBench, full claw-eval pass^3, creative_writing, roleplay, svg_generation, research. Tiny models are not the audience for those.

## Tier-0 baselines

1. **Floor.** Random / majority pass^3 = 0 % on function_call. If anything lands at 0 %, suite is broken — or the model genuinely can't do it (acceptable).
2. **Reference.** Qwen 27B INT4 (current daily-driver smart alias) run on Tier 1 once as the *adult* line. The blog says "here's where tiny crosses big" → that's the line.
3. **Off-the-shelf.** Pull external numbers for at least one model where they exist (Gemma 4 E2B vs Google's published evals; function-gemma vs the published function-gemma blog post). If we're >5 pts off, investigate.

## What we don't promise

- *Won't* score tiny models on hard tasks they can't do and call that a fair comparison. The framing is **capability map + cliff location**, not "loser models scored low."
- *Won't* run an on-device latency claim without on-device hardware to back it up. Tok/s numbers in this experiment are *Blackwell tok/s* — the brand piece can note "Pixel 7 numbers exist in the Google AI Edge talk; we ran the same models on Blackwell to surface the upper-bound throughput."
- *Won't* include fine-tuned variants of any model in this round. That's v2 (`tiny-models-finetune/`).

## Run plan (overnight, single GPU)

GPU 0 cycles through swaps; GPU 1 keeps Gemma 4 26B MoE up as the judge. No TP=2 needed — every model fits in <20 GB.

| Night | Models | Wall estimate |
|---|---|---|
| 1 | SmolLM2 (135M/360M) + function-gemma 270M + OLMo-1B + Llama-3.2-1B | ~3 hrs (pulls + runs) |
| 2 | Qwen 0.8B Base + Qwen 2B Base + Llama-3.2-3B + Phi-3-mini | ~3 hrs |
| 3 | Gemma 4 E2B (bf16 + on-the-fly FP8) + Gemma 4 E4B (bf16 + on-the-fly FP8) | ~3 hrs |
| 4 | Qwen3.6-4B (INT4 + bf16 + on-the-fly FP8) + Qwen3.6-9B (FP8 + bf16) | ~3 hrs |
| 5 | Re-run anything flaky + Tier-0 cross-checks + judge-variance sanity | ~2 hrs |

Total wall: ~14 hours across 5 overnight slots. Daily-driver smart alias down on GPU 0 from ~22:00–05:00 each night. **No daytime disruption.**

## Risks

| Risk | Mitigation |
|---|---|
| Local judge (Gemma 4 26B) variance on tiny-model outputs | Single trial Tier 2+, document variance honestly. Spot-check 5 % of judge calls by hand pre-publish. |
| Models with no on-the-fly FP8 support | Fall back to bf16 only; note in inventory |
| Pulls are large for some of these (Phi-3-mini etc.) | Stage pulls during evening, runs at night |
| LiteRT-LM export not validated for every model | Out of scope here — that's the v2 experiment |

## What follows this (v2 candidates)

- **`tiny-models-finetune/`** — pick one task (function calling on our own 8 tools), apply the function-gemma methodology, publish the recipe end-to-end. Headline: *"How we fine-tuned a 270 M model to 90 % on tool calling, in one evening."*
- **`tiny-models-export/`** — take the v1 winners, run them through `litert-torch` → `LiteRT-LM`, produce `.litertlm` files + a load-and-run script + size/quant comparison. Pairs with avaLab.
- **`tiny-models-quant/`** — INT8 / INT4 / FP8 sweep at <2 B params specifically. Where does quantization break for tiny?

## Kickoff gate

**Resolved by the redirect:**
- ~~Cloud budget~~ — zero. Local judge only.
- ~~TP=2 windows~~ — none needed.
- **Pull list approval.** Need ~30–50 GB of new HF downloads (Phi-3-mini ~7 GB, Llama-3.2 1B+3B ~6 GB, SmolLM2 variants ~2 GB, function-gemma ~0.5 GB, Gemma 4 E2B/E4B if not on disk, OLMo-1B ~2 GB). All to `/mnt/models/huggingface/`, plenty of room. Go-ahead implicit unless flagged.
- **Night 1 kickoff time.** Earliest sensible: tonight after 22:00 local. Drop the daily smart alias for ~3 hrs.

## Reproducibility

Every score reproducible by:
```
bash models/vllm-swap.sh <config_or_new_swap_entry>
cd evals
./run.sh function-call --model local --all-suites
./run.sh custom --suite structured_output --model local --trials 3
./run.sh refusal --model local --dataset xstest,simple_safety
```

New vllm-swap.sh entries needed for: SmolLM2 variants, function-gemma 270 M, OLMo-1B, Llama-3.2-1B/3B, Phi-3-mini, Gemma 4 E2B/E4B. Pattern follows existing `gemma4-e4b-fp8` config in `models/vllm-swap.sh`.
