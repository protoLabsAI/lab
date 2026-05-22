# blackwell-leaderboard — PLAN

> Graduated from `PROPOSAL.md` 2026-05-22. Open questions resolved below.
> **Blocked on user go-ahead for kickoff** — see "Kickoff gate" at the bottom.

## Locked decisions

| Open question | Lock |
|---|---|
| pass^3 or pass^1 for cloud? | **pass^3 on Tier 1 (claw-eval) for everyone; pass^1 on Tier 2+.** Cloud judge cost is now near-zero — Tier 2 LLM judges route through `protolabs/fast` (Gemma 4 26B MoE) by default after `5edc936`. |
| WildBench yes/no? | **Skip for v1.** Scope discipline. WildBench is its own breakdown — sampled 100 → full 1024 — and pulls focus from the leaderboard. |
| Subjective suites in or out? | **In as appendix; not in headline.** creative_writing + roleplay + svg_generation + research land in the dataset and get a footnote table. Headline leaderboard = claw-eval + coding/reasoning/structured/summarization/instruction_following + safety/refusal. |
| Headline metric? | **Two numbers per row: pass^3 (claw-eval) + tok/s.** No composite. Composites hide things. |
| Daily downtime budget? | **Overnight runs only.** Aliases stay intact 09:00–22:00 local; runs kick off after 22:00 and complete before 09:00. Each vLLM swap ~5 min — the daily-driver model stays parked overnight, swapped back at run end. |
| proto-bench harness reuse? | **No.** `BENCHMARKS.md` lifted here as `REFERENCES.md` (related-work for the blog); the rest of proto-bench stays parked. |

## Model inventory (final)

The previous PROPOSAL.md listed heretic; reality is Gemma 4 26B MoE in the fast lane (`e42d5a1`). Heretic is on disk at `/mnt/models/quantized/Qwen3.6-35B-A3B-uncensored-heretic-FP8` but won't be evaluated unless we add an "uncensored prose" footnote later.

### Local

| # | Model | Quant | Config | GPU | Why include |
|---|---|---|---|---|---|
| 1 | Qwen3.6-4B | INT4 | `qwen-4b-int4` | 0 | Edge floor (297 tok/s) |
| 2 | Qwen3.6-4B | bf16 | `qwen-4b` | 0 | LoRA base (155 tok/s) |
| 3 | Qwen3.6-9B | FP8 | `qwen-9b-fp8` | 0 | On-the-fly FP8 sweet spot |
| 4 | Qwen3.6-27B | INT4 + MTP | `qwen-27b-int4-mtp` | 0 | Daily-driver chat/creative |
| 5 | Qwen3.6-27B | FP8 + MTP | `qwen36-27b-fp8-mtp` | 0 | Daily-driver thinking |
| 6 | Qwen3.6-35B MoE | FP8 | `qwen-35b` | 0 | Speed king |
| 7 | Gemma 4 26B-A4B MoE | FP8 | `gemma4-moe-fast` | 1 | Current fast lane (judge model) |
| 8 | Gemma 4 31B | FP8 | `gemma4-31b-fp8` | 0 | Dense alt |
| 9 | Qwen3.6-122B | INT4 TP=2 | `qwen-122b-int4` | 0+1 | Quality ceiling (faster on PCIe) |
| 10 | Qwen3.6-122B | FP8 TP=2 | `qwen-122b-fp8` | 0+1 | Quality ceiling (official) |

### Cloud refs (gateway-routed)

Snapshot dates locked at kickoff (record in `RESULTS.md`).

| Model | Tier-1 (pass^3) | Tier-2 (pass^1) |
|---|---|---|
| Claude Haiku 4.5 | yes | yes — the floor |
| Claude Sonnet 4.7 (1M) | yes | yes |
| Claude Opus 4.7 (1M) | yes | yes |
| GPT-5.4 | yes | yes |
| Gemini 3.1 Pro | yes | yes |
| GLM 5 Turbo | yes | yes — open-weights frontier ref |

Snapshot table is **canonical for the blog**; if a model gets deprecated mid-run, footnote it and don't refresh that row.

### Quants out of scope for v1

Heretic (parked), Qwen 4B FP8 (overlap with INT4 / bf16), Qwen 27B FP8 TP=2 (single-GPU FP8+MTP is the same model, faster), Cydonia, Llama 70B (creative-only, scope discipline).

## Suites — locked

| Tier | Suite | Runner | Tasks | Trials | Models |
|---|---|---|---|---|---|
| 1 | **claw-eval** | `./run.sh claw` | 52 EN | 3 (pass^3) | all (10 local + 6 cloud) |
| 1 | **function-call** | `./run.sh function-call` | 8 | 3 | all |
| 2 | **coding** | `./run.sh custom --suite coding` | 10 | 1 | all |
| 2 | **reasoning** | `./run.sh custom --suite reasoning` | 5 | 1 | all |
| 2 | **structured_output** | `./run.sh custom --suite structured_output` | 5 | 1 | all |
| 2 | **summarization** | `./run.sh custom --suite summarization` | 5 | 1 | all |
| 2 | **instruction_following** | `./run.sh custom --suite instruction_following` | 5 | 1 | all |
| 2 | **safety + refusal** | `./run.sh refusal --dataset xstest,simple_safety` | 550 | 1 | all |
| App | creative_writing | `./run.sh custom --suite creative_writing` | 5 | 1 | local only |
| App | roleplay | `./run.sh custom --suite roleplay` | 5 | 1 | local only |
| App | svg_generation | `./run.sh custom --suite svg_generation` | 5 | 1 | local only |
| App | research | `./run.sh custom --suite research` | 4 | 1 | local only |

**Headline leaderboard column count** = 8 (Tier 1 + Tier 2). Appendix gets 4 more.

## Tier-0 baselines

Sanity-check rails. Each must pass before publishing.

1. **Random / majority floor.** For claw-eval, pass^3 random = 0%. For refusal (XSTest), majority class (always-comply) = 55.5% (250/450), always-refuse = 44.5%. If any model lands at these floors, the pipeline is broken.
2. **Linear-probe equivalent.** Haiku 4.5 is the cheapest cloud model in the run; if our 4B INT4 beats Haiku, that's the headline. If Haiku beats Qwen 122B, the pipeline is broken.
3. **Off-the-shelf comparable.** Pull at least one external snapshot per local model — OpenRouter / Artificial Analysis / LMSys — for any overlapping metric. If our internal score and the external one differ by >5 points on the same model, investigate before publishing.

## Run order (overnight slots)

Each line = one overnight window. Order: smallest first, swap-cost-amortizing second (run all suites on a swapped model before swapping again).

| Night | Models | Wall estimate | Notes |
|---|---|---|---|
| 1 | Qwen 4B INT4 (full Tier 1+2+App) | ~3 hrs | Fail-fast — if claw pipeline breaks on the smallest model, we know before bigger swaps |
| 2 | Qwen 4B bf16 + Qwen 9B FP8 | ~6 hrs | Both single-GPU swaps |
| 3 | Qwen 27B INT4 MTP + Qwen 27B FP8 MTP | ~7 hrs | 27B FP8 MTP = current `vllm.service`, swap to INT4 first |
| 4 | Qwen 35B MoE FP8 + Gemma 4 26B MoE FP8 | ~6 hrs | Gemma is current `vllm-fast.service` — runs without disrupting daytime |
| 5 | Gemma 4 31B FP8 | ~5 hrs | Dense alt — pairs with the 27B/35B comparison |
| 6 | Qwen 122B INT4 TP=2 | ~6 hrs | Both GPUs. Daytime aliases down — coordinate with user. |
| 7 | Qwen 122B FP8 TP=2 | ~6 hrs | Both GPUs. Same constraint. |
| 8 | All 6 cloud refs (Tier 1 + Tier 2) | ~4 hrs | No local-serving impact; gateway-routed |
| 9 | Sanity checks + Tier-0 + cloud Tier 2 finishing | ~3 hrs | Pre-publish gate |

**Nights 6+7 are the only daytime-blocking windows.** Daily-driver aliases unavailable across both nights. Need explicit user confirmation for those.

## Artifact targets

1. **HF dataset** `protoLabsAI/blackwell-leaderboard-2026q2` — JSONL of raw outputs + scores, one file per (model, suite). Private during runs, public on blog.
2. **`RESULTS.md`** in this dir — headline table + per-model breakdown + Tier-0 cross-checks + run dates + claw-eval submodule pin.
3. **`BLOG.md`** in this dir — draft of the protolabs.studio post. Lead with a specific number per V1.
4. **`README.md` refresh** at repo root — new leaderboard table, permanent link to HF dataset, `as-of` date.
5. **`figures/`** in this dir — at least one tok/s vs pass^3 scatter, one cost-per-correct-answer (cloud) bar chart.

## Reproducibility checklist (per model)

Every score in the headline table must be reproducible by:
```
bash models/vllm-swap.sh <config>             # exact config name from inventory
cd evals
./run.sh claw --model local --tasks <list> --trials 3 --port-offset 200
./run.sh function-call --model local --all-suites
./run.sh custom --suite <suite> --model local --trials 1
./run.sh refusal --model local --dataset xstest,simple_safety
```
The config name + claw-eval submodule SHA + run date pinned in `RESULTS.md`.

## Risks (refined from proposal)

| Risk | Mitigation |
|---|---|
| Daily-driver aliases down during nights 6+7 | Explicit user heads-up window — Discord post, calendar block, or just both nights on a weekend |
| Cloud model deprecation mid-run | Snapshot model IDs at kickoff in `RESULTS.md`; if deprecated mid-run, footnote the row |
| Claw-eval submodule drift | Pin SHA at kickoff: record `git -C evals/claw/claw-eval rev-parse HEAD` in `RESULTS.md` |
| LLM-judge variance on custom suites | Single trial × pass^1 — accept the variance, document it. Don't chase it with extra trials for v1. |
| Sanity check fails on Tier-0 cross-check | Hard stop. Do not publish. Investigate (likely either submodule drift or a bad swap) |
| Run interrupts (crash, network blip) | Per-model checkpointing — completed suites land as separate JSONL on disk, restart from gap |
| Power blip mid-run | UPS-resilient until ~10 min; longer outage = re-run that night's model. Acceptable. |

## Out of scope for v1 (explicit)

- WildBench (own breakdown)
- protoCLI / coding-agent evals (out of game)
- Fine-tuning anything
- Per-token cost analysis with full provider price math (rough number only)
- Comparing to specific competitors by name in the blog (V7: state the numbers, name the brand only in the credit table)
- Schedule promises ("we'll re-run quarterly")

## Kickoff gate

Need three things from the user before night 1:

1. **Cloud budget confirmation.** pass^3 on Tier 1 + pass^1 on Tier 2 across 6 cloud models. Estimate: ~$80–150 worst case (Opus dominates), depending on response length. Cap at $200?
2. **Window for nights 6+7 (122B TP=2).** Daily-driver aliases down. Suggest a weekend.
3. **First-night kickoff time.** Night 1 = Qwen 4B INT4. Low-risk, single-GPU, ~3 hrs. Earliest sensible start: tonight after 22:00 local.

Once green-lit, the first concrete action is `bash models/vllm-swap.sh qwen-4b-int4` followed by the claw-eval run.
