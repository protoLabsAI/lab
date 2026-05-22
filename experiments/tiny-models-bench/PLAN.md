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

Expanded 2026-05-22 after due-diligence sweep of the 2026 SLM/TLM landscape. Sources: Hugging Face state-of-OS spring 2026, awesomeagents Edge LLM leaderboard, promptquorum mobile LLM bench, BentoML SLM survey, Google AI Edge talk references.

| Tier | Model | Params | HF repo | Quant | On disk? | Headline why |
|---|---|---|---|---|---|---|
| **Sub-1B** | SmolLM2-135M-Instruct | 135 M | `HuggingFaceTB/SmolLM2-135M-Instruct` | bf16 | no — pull | Reference floor |
| | SmolLM2-360M-Instruct | 360 M | `HuggingFaceTB/SmolLM2-360M-Instruct` | bf16 | no — pull | Step on the ladder |
| | functiongemma-270m-it | 270 M | `google/functiongemma-270m-it` | bf16 | no — pull (gated) | Function-calling base, Gemma 3 lineage. Compare base vs fine-tuned variant below. |
| | functiongemma-270m-ft-mobile-actions | 270 M | `litert-community/functiongemma-270m-ft-mobile-actions` | bf16 | no — pull | The 46 % → 90 % delta from the talk; lets us measure the fine-tune gap directly |
| | Llama-3.2-1B-Instruct | 1.2 B | `meta-llama/Llama-3.2-1B-Instruct` | bf16 | no — pull (gated) | Standard reference, tool-call support |
| | Gemma 3 1B-it | 1 B | `google/gemma-3-1b-it` | bf16 | no — pull (gated) | 2026 mobile tok/s monster (~2,500 tok/s on mobile GPU per HF report) |
| | Qwen3.6-0.8B-Base | 0.8 B | local | bf16 | **yes** | Existing inventory, multimodal at sub-1 B per HF |
| **1–3B** | SmolLM2-1.7B-Instruct | 1.7 B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | bf16 | no — pull | HF flagship small |
| | Qwen3.6-2B-Base | 2 B | local | bf16 | **yes** | Existing inventory |
| | Gemma 3 4B-it | 4 B | `google/gemma-3-4b-it` | bf16 + on-the-fly FP8 | no — pull (gated) | "Best all-around edge" per 2026 bench (MMLU 43.6, IFEval best-in-class) |
| | Gemma 4 E2B-it | 2.3 B eff / 5.1 B loaded | `google/gemma-4-E2B-it` | bf16 + on-the-fly FP8 | check disk | Apache 2.0; AI Core's edge default per the talk |
| | Llama-3.2-3B-Instruct | 3.2 B | `meta-llama/Llama-3.2-3B-Instruct` | bf16 | no — pull (gated) | "Best tool-call support" per 2026 mobile bench |
| | Phi-4-Mini-Instruct | 3.8 B | `microsoft/Phi-4-mini-instruct` | bf16 + on-the-fly FP8 | no — pull | Leading reasoning at this size (GSM8K 88.6 %, ARC-C 83.7 %) + function calling |
| | OLMoE-1B-7B-0125-Instruct | 1.3 B active / 6.9 B total MoE | `allenai/OLMoE-1B-7B-0125-Instruct` | bf16 | no — pull | Tiny-MoE counter-test (does the MTP-MoE penalty story extend?) |
| **3–9B** | Gemma 4 E4B-it | 4.5 B eff / 8 B loaded | `google/gemma-4-E4B-it` | bf16 + on-the-fly FP8 | check disk | Apache 2.0; AI Core's upper edge model |
| | Qwen3.6-4B | 4 B | local | INT4 + bf16 + on-the-fly FP8 | **yes** | Existing inventory, 3 quant variants |
| | IBM Granite 4.1-8B-Instruct | 8 B | `ibm-granite/granite-4.1-8b-instruct` | bf16 + on-the-fly FP8 | no — pull | New 2026 release; IBM claims it matches 32 B MoE on enterprise tasks |
| | Qwen3.6-9B | 9 B | local | FP8 + bf16 | **yes** | Upper bound of "tiny" per user spec |
| **Separate hardware** | BitNet b1.58 2B-4T | 2 B (1.58-bit) | `microsoft/BitNet-b1.58-2B-4T` | 1.58-bit native | no — pull | **CPU-only** via `bitnet.cpp`; doesn't run on GPU. Include as footnote table, not on vLLM. |

Notes:
- **(gated)** means the HF repo requires accepting the model's terms once (`huggingface-cli login` + visit the model page). Need to confirm we have accept on Google/Meta/Microsoft before pulls fire.
- **Quant variants** run only when the deploy story changes (INT4 for serving, on-the-fly FP8 for inference); bf16 always present as baseline.
- **BitNet** is the only model that won't serve via vLLM — it's pure-CPU via `bitnet.cpp`. Worth including for the 2026 capability map but on a different evaluation track.
- **Dropped from earlier draft:** Phi-3-mini (superseded by Phi-4-Mini), OLMo-1B (lighter 2026 signal vs OLMoE-MoE), Gemma 3 12B (>9 B cap), Gemma 3 27B (>9 B cap).

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

## protolabs/fast audit + proposed reduction (2026-05-22)

GPU 1 snapshot at planning time:

| Process | VRAM | Port | Notes |
|---|---|---|---|
| `vllm-fast.service` (Gemma 4 26B MoE FP8) | 56.3 GB | 8002 | `--gpu-memory-utilization 0.72 --max-model-len 131072` |
| Fish S2 Pro TTS (`tools.api_server`, --half --compile) | 19.8 GB | 8092 | up 2 d 9 h |
| Qwen3-Embedding-0.6B | 2.0 GB | 8001 | rag-bench `serve_embed.py` |
| **Used / Free** | **78.1 / 19.1 GB** | | of 96 GB |

protolabs/fast is provisioned for general conversational use (131K ctx, ~26 GB KV pool budget). Judge calls don't need any of that — claw-eval rubrics top out at ~4K tokens, refusal classification <1K, structured-output grading 1–2K. Proposed unit edit, applied when we kick off the bench:

```
--gpu-memory-utilization 0.55   (was 0.72)   — frees ~16 GB
--max-model-len 32768           (was 131072) — judge has 8× headroom over largest grader prompt
```

Net effect: GPU 1 free goes from 19 GB → ~35 GB. Enough headroom to load **any model in this experiment's inventory ≤9 B FP8** alongside Gemma 4 26B without a second GPU touch — meaning runs proceed without disturbing GPU 0's daily-driver smart alias. Revert post-bench.

Restore command sequence:
```
sudo sed -i 's/--gpu-memory-utilization 0.55/--gpu-memory-utilization 0.72/;s/--max-model-len 32768/--max-model-len 131072/' /etc/systemd/system/vllm-fast.service
sudo systemctl daemon-reload && sudo systemctl restart vllm-fast
```

## Run plan (overnight, GPU 1 alongside trimmed fast)

GPU 1 hosts both Gemma 4 26B judge (trimmed) + the tiny model under test. GPU 0 left untouched: daily smart alias stays up around the clock. Most tiny models fit in <10 GB FP8; some 9 B configs need bf16 → ~18 GB. All inside the freed ~35 GB headroom.

| Night | Models | Wall estimate |
|---|---|---|
| 1 | SmolLM2 (135M / 360M / 1.7B) + functiongemma 270M base + ft-mobile-actions | ~3 hrs |
| 2 | Llama-3.2-1B + Llama-3.2-3B + Gemma 3 1B + Qwen 0.8B / 2B Base | ~3 hrs |
| 3 | Gemma 3 4B (bf16 + FP8) + Gemma 4 E2B (bf16 + FP8) + Gemma 4 E4B (bf16 + FP8) | ~4 hrs |
| 4 | Phi-4-Mini (bf16 + FP8) + Qwen3.6-4B (INT4 + bf16 + FP8) | ~3 hrs |
| 5 | IBM Granite 4.1 8B (bf16 + FP8) + Qwen3.6-9B (FP8 + bf16) + OLMoE-1B-7B | ~3 hrs |
| 6 | BitNet b1.58 2B (CPU via bitnet.cpp, separate track) + re-runs + Tier-0 + judge spot-check | ~3 hrs |

Total wall: **~19 hours across 6 overnight slots**. GPU 0 smart alias **never down**. Daytime evals against tiny models possible if we want them (GPU 1 has the room).

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
- **Run-time impact** — none on smart alias (GPU 0 untouched per the audit reduction above).

**Open:**

- **HF gated-model access.** Several entries require accepting model terms once on Hugging Face from `artificial-citizen` (the HF account):
  - `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/functiongemma-270m-it`
  - `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`
  - `ibm-granite/granite-4.1-8b-instruct` (may not require accept)

  Ungated pulls fire immediately: SmolLM2 family, `litert-community/functiongemma-270m-ft-mobile-actions`, Phi-4-Mini, OLMoE, BitNet, `microsoft/Phi-4-mini-instruct`.

- **Apply protolabs/fast trim before or after bench?** Recommendation: before, to free GPU 1 headroom. ~5 min downtime on fast alias during restart.

- **Night 1 kickoff time.** No daytime constraint anymore. Can run during the day if pipeline smokes clean.

## Reproducibility

Every score reproducible by:
```
bash models/vllm-swap.sh <config>      # or a manual co-resident launch on GPU 1
cd evals
./run.sh function-call --model local --all-suites
./run.sh custom --suite structured_output --model local --trials 3
./run.sh refusal --model local --dataset xstest,simple_safety
```

New vllm-swap.sh entries needed for: SmolLM2 (3 sizes), functiongemma 270M (base + ft), Llama-3.2-1B / 3B, Gemma 3 1B / 4B, Gemma 4 E2B / E4B (check on-disk first), Phi-4-Mini, Granite 4.1 8B, OLMoE-1B-7B. Pattern follows existing `gemma4-e4b-fp8` config; CUDA_VISIBLE_DEVICES=1 + low gpu-memory-utilization (0.10–0.20) so the eval target co-resides with the trimmed Gemma 4 26B judge.
