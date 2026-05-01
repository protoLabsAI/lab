# CLAUDE.md — protoLabs AI Lab

Monorepo for model evaluation, training, inference infrastructure, and ML experiments.

## Structure

- `packages/lab-core/` — Shared Pydantic models, GPU utils (strict, tested, publishable)
- `evals/` — LLM eval suite: claw-eval, custom suites, function-call, RAG (strict, tested)
- `models/` — Model inventory, vllm-swap.sh, benchmarks
- `training/` — Fine-tuning workspace (LLaMA-Factory configs, datasets)
- `experiments/` — ML experiments (context-1, ltx-video, flux2, quantize, pixel-gen, stt-whisper, voice-agent)
- `infra/` — Prometheus exporters, vLLM systemd, gateway configs (gateway runs on ava node)

## How we operate

The default work pattern in this lab is a **closed cycle**:

```
experiment  →  report  →  engineering  →  test  →  content  →  repeat
```

Every phase has an exit criterion. Don't move to the next phase until
the current one is done.

| Phase | What it is | Exit when |
|---|---|---|
| **experiment** | research, training, iteration in `experiments/<name>/` | model artifact + Tier-0 baselines (majority + linear probe + one off-the-shelf comparable) + cross-domain held-out eval |
| **report** | internal `RESULTS.md` with honest numbers, confusion matrices, what didn't work | written without softening the failures — the report is for next-session-us |
| **engineering** | wire the artifact into the consuming product (ORBIS, gateway, CI) | a PR landed on the consuming repo's main branch |
| **test** | validate under real conditions, not just held-out splits | one round of real-world signal observed (real users, real traffic, real audio — not benchmark replay) |
| **content** | public artifact: HF model card + dataset card + blog post on protolabs.studio | merged blog + public HF release (private during draft, public after review) |
| **repeat** | next experiment, informed by what `test` and `content` revealed | always |

**Bias toward shipping.** The trap is doing experiment → report → experiment ad infinitum. The cycle is only valuable when it closes. If a piece of work has been in `report` for a week without engineering, that's the signal to either ship or kill it.

**What this means for new experiments:**
- Don't start a new experiment unless the previous one is past `engineering` (or has been explicitly parked with a memo)
- Each experiment plans for its `content` deliverable on day one — what's the blog headline? — and reverses-out the work needed to get there
- Default to publishing publicly via `protoLabsAI/...` on HuggingFace and protolabs.studio for the writeup. Privacy is a temporary state during drafting, not the default.

## Companion-stack research workspace

`experiments/companion-stack/` is the umbrella for [ORBIS](https://github.com/protoLabsAI/ORBIS)-supporting research. New experiments around the conversational AI loop go there, organized by pipe (`audio-pre/`, `text-pre/`, `llm-context/`, `text-post/`, `memory/`, `visual/`).

Read `experiments/companion-stack/README.md`, `ROADMAP.md`, and `LEARNING.md` before starting a new ORBIS-related experiment. The audio-tags experiment (`pipes/audio-pre/audio-tags/`) is the worked exemplar — copy its shape.

**Heuristic for new pipes:** if the LLM would have to guess something a small classifier could measure, you're paying LLM cost for a job a 1.7 ms head could do better. That's the whole game.

## Brand & monetization — protolabs.studio

protolabs.studio is the public face. The work in this lab is only valuable to the brand if it surfaces there.

Discipline:
- **Every shipped experiment produces a blog draft** before the next experiment starts. Drafts live in `experiments/<name>/BLOG.md`. The audio-tags experiment is the template.
- **HuggingFace org `protoLabsAI/`** is the canonical model + dataset publishing target. Models go up private during drafting, flipped public on blog publish.
- **Cross-link aggressively.** Blog → HF model + dataset → ORBIS PR → back to blog. The whole stack should be one click apart.
- **Cite our own prior work.** v5 cites v4 cites v3 cites v2 cites v0. The lineage is the credibility.

When in doubt about whether something should be public, default to publishing. The lab's expertise is only monetizable to the extent it's visible.

## Using uv

```bash
uv sync                                    # sync all workspaces
uv run pytest                              # run tests (lab-core + evals)
uv run proto-eval claw --model local       # run evals CLI
uv run models --gpu single                 # show model inventory
uv run ruff check .                        # lint everything
```

## Daily setup (dual GPU)

Both vLLMs run as systemd services and auto-start on boot:

| GPU | Service | Model | Port | Mode | tok/s |
|-----|---------|-------|------|------|-------|
| 0 | `vllm.service` | Qwen3.6-27B-FP8 | :8000 | thinking, 262K | ~150 |
| 1 | `vllm-fast.service` | Qwen3.6-35B-A3B-uncensored-heretic-FP8 | :8002 | instruct + uncensored, 131K | ~226 |

Gateway aliases: `protolabs/smart` → 27B (thinking + preserve_thinking), `protolabs/fast` → 35B heretic (instruct, uncensored).

Heretic un-retired 2026-05-01: was retired 2026-04-29 after we discovered the `logit_bias: {<think>: -100, </think>: -100}` clamp recommended by the model card corrupts generation (1-token role-marker garbage on thinking-engaging prompts). At the time we needed *some* form of suppression because the gateway didn't strip `<think>...</think>` markup yet; logit_bias was the only knob the model card pointed at. Now that the gateway's `_ThinkStripper` handles `<think>...</think>` post-emit (homelab-iac#26/#32/#35), we don't need to suppress at sample-time at all — let the model emit, gateway strips. Validated with the same 4-probe matrix used for the official 35B (simple, thinking-bait, gateway tool-call, uncensored sanity) — generation pathway clean, no role-marker garbage, `metadata.thinking` populated on tool-call turns same as the official model. Heretic is at `/mnt/models/quantized/Qwen3.6-35B-A3B-uncensored-heretic-FP8` (our FP8 quant, 34 GB). Trade-off: same trained-in always-thinks behavior on tool-call turns (~130 chars/turn through the gateway-strip pipeline) — gain is uncensored prose for creative/red-team/eval work without the corporate-hedge layer. The HF README still recommends the broken `logit_bias` workaround; needs an update before we'd point others at this config.

Tuned MoE kernels live in `models/moe-configs/` — symlinked into vLLM's `fused_moe/configs/` via `bash models/install-moe-configs.sh` (run after fresh vLLM installs / upgrades).

## Running Models

```bash
bash models/vllm-swap.sh qwen-35b           # speed king MoE FP8 official, 262K (180 tok/s)
bash models/vllm-swap.sh qwen-9b-fp8        # on-the-fly FP8 (140 tok/s)
bash models/vllm-swap.sh qwen-4b-fp8        # on-the-fly FP8 edge (140 tok/s)
bash models/vllm-swap.sh qwen-27b-int4      # daily driver, agentic (53 tok/s)
bash models/vllm-swap.sh qwen-27b-int4-mtp  # daily driver + MTP, chat/creative (70 tok/s)
bash models/vllm-swap.sh qwen-4b-int4       # edge deploy, fastest absolute (297 tok/s)
bash models/vllm-swap.sh qwen-4b            # LoRA base bf16 (155 tok/s)
bash models/vllm-swap.sh qwen-122b-fp8      # quality ceiling FP8 official TP=2 (112 tok/s)
bash models/vllm-swap.sh qwen-122b-int4     # quality ceiling INT4 TP=2 (122 tok/s)
bash models/vllm-swap.sh qwen-27b-fp8-tp2   # FP8 official TP=2 (70 tok/s, 131K)
```

## Speed Testing

```bash
bash models/speed-test.sh           # 5 runs on current model (800 tok gen)
bash models/speed-test.sh 10        # 10 runs
bash models/speed-test.sh 3 short   # 3 short runs (200 tokens)

# A/B compare baseline vs optimized config
cd evals && bash run-ab-speed.sh qwen-4b-int4 5
```

Reports decode tok/s (1/TPOT), wall tok/s, TTFT, and TPOT from vLLM's `/metrics` endpoint — not wall-clock estimation.

### Optimization Flags (`-opt` configs)

Suffix any config with `-opt` to enable P1+P2 flags:
- `--async-scheduling` — overlap scheduling with execution
- `--enable-prefix-caching` — reuse KV cache for repeated prefixes
- `--performance-mode interactivity` — auto-tune scheduler for latency
- `--kv-cache-dtype fp8` — halve KV cache memory, double context capacity

**Measured impact (single-request, P1+P2 only):** minimal (+1-3% tok/s). Real wins are under concurrent load and multi-turn (prefix caching). FP8 KV doubles context capacity.

### MTP Speculative Decoding (`-mtp` configs)

Native Qwen3.5 Multi-Token Prediction — big speed gains on dense models:

| Model | Baseline | + MTP | Gain | Tool Calling |
|-------|:--------:|:-----:|:----:|:------------:|
| **27B INT4** | 53 tok/s | **70 tok/s** | **+32%** | Works, but T08 quality regresses |
| **9B** | 92 tok/s | **112 tok/s** | **+22%** | Works, no quality loss |
| **35B MoE** | 171 tok/s | 153 tok/s | -11% | N/A — slower, don't use |

- MTP helps dense models, hurts MoE (routing overhead > speculation savings)
- 9B + MTP is safe for all workloads including tool calling
- 27B + MTP: use for chat/creative (70 tok/s), avoid for complex agentic (T08 regresses)
- MoE FP8 env vars: `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency`

### TP=2 Tuning (122B, 35B-tp2)

NCCL env vars for PCIe (no NVLink): `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=8`

**Tested results:**
- **FIX: `NCCL_P2P_DISABLE=1` enables stable CUDA graphs on TP=2 Blackwell PCIe**
- 122B INT4: 23→**122 tok/s** (5.3x, stable over 10 runs)
- 35B MoE: 22→**205 tok/s** (9.3x, stable over 10 runs)
- Root cause: ACS enabled on PCIe bridges corrupts P2P during CUDA graph replay
- Disabling P2P forces shared memory transport — slight overhead but fully stable
- TTFT: 3077ms→29ms with prefix caching after warmup
- Power draw: only 88-96W per card at 600W limit — MoE is not power-bound
- 35B TP=2: prefix caching fixed 1.8s TTFT → 0.5s (**-70%**), wall tok/s +25%
- `VLLM_USE_FLASHINFER_MOE_FP8` crashes on 122B FP8 (unsupported quant scheme) — don't use
- **Previous finding that TP=2 needs enforce-eager was WRONG** — `NCCL_P2P_DISABLE=1` fixes it

## Running Evals

```bash
cd evals

# Profile runs (recommended)
./run.sh profile --name quick --model local    # ~15 min smoke test, 1 trial
./run.sh profile --name full --model local     # ~60-90 min comprehensive, 3 trials

# Individual runs
./run.sh claw --model local --tasks T02,T04,T06,T08 --port-offset 200
./run.sh custom --suite coding --model local --trials 1
./run.sh custom --suite reasoning --model local --trials 1
./run.sh function-call --model local --all-suites
```

### Eval Suites

| Suite | Tests | What it measures |
|-------|:-----:|-----------------|
| **claw-eval** | 52 EN | Agentic tool use (email, calendar, CRM, ops, finance) |
| **coding** | 10 | Generation (5) + analysis/review/security (5) |
| **instruction_following** | 5 | Constraint adherence, format compliance |
| **reasoning** | 5 | Math, logic puzzles, deduction, pattern recognition |
| **structured_output** | 5 | JSON, YAML, SQL, markdown tables, log parsing |
| **summarization** | 5 | Compression, action extraction, TL;DR |
| **safety** | 5 | Refusal, jailbreak resistance, PII, security review |
| **creative_writing** | 5 | Prose, narrative, character voice |
| **roleplay** | 5 | RPG GM quality, world building |
| **svg_generation** | 5 | SVG validity, accuracy, animation |
| **research** | 4 | Synthesis, conflicting sources, hallucination |
| **function_call** | 8 | Basic (5) + edge cases (3) |

### Eval Profiles

- **quick** — 6 claw tasks + 6 custom suites + FC, 1 trial (~15 min)
- **full** — 20 claw tasks + 10 custom suites + FC, 3 trials pass^3 (~60-90 min)

## Model Inventory (`/mnt/models`)

| Model | Size | tok/s | +MTP | Quick Score | Role |
|-------|------|:-----:|:----:|:-----------:|------|
| **Qwen 35B MoE FP8** | 35GB | **180** | — | — | **Speed king, 262K ctx, single GPU (Qwen official)** |
| **Qwen 9B FP8** | 19GB* | **141** | — | — | On-the-fly FP8 (+53% vs bf16) |
| **Qwen 4B FP8** | 8.8GB* | 141 | — | — | On-the-fly FP8 edge |
| **Qwen 27B INT4** | 29GB | 53 | **70** | **86/103** | Daily driver (MTP for chat, baseline for agents) |
| **Qwen 122B FP8** | 119GB | **112** | — | — | Quality ceiling FP8 TP=2 (Qwen official) |
| **Qwen 122B INT4** | 74GB | **122** | — | **89/103** | Quality ceiling INT4 TP=2 (faster on PCIe) |
| **Qwen 27B FP8 TP=2** | 29GB | **70** | — | — | 131K ctx, TP=2 (Qwen official) |
| **Qwen 9B BF16** | 19GB | 92 | **112** | 72/103 | Fine-tune base (cold storage) |
| **Qwen 4B INT4** | 3.8GB | 297 | — | 56/103 | Edge deploy (fastest absolute) |

\* On-the-fly FP8 (`--quantization fp8`) loads bf16 weights from disk, quantizes during model load. No separate quant files needed.
| Cydonia 24B | 44GB | — | — | — | Creative/roleplay (holding) |
| Llama 70B AWQ | 38GB | 38 | — | — | Creative/roleplay (holding) |
| Llama 8B AWQ | 5GB | — | — | 50/102 | Eval baseline (poor) |
| Qwen 2B BF16 | 4GB | 307 | — | — | Training (5/5 reasoning!) |
| Qwen 0.8B BF16 | 1.5GB | 547 | — | — | Training |

Base models (0.8B, 2B, 4B) also downloaded for pretraining. FP8 quants in `/mnt/models/quantized/` and on [HuggingFace protoLabsAI](https://huggingface.co/protoLabsAI).

Cold storage (`/mnt/data/models-cold/`): FLUX.2-klein 9B+base (100GB), Z-Image+Turbo (51GB), Voxtral-Mini-4B (17GB), OCR models (11.4GB).

## Blackwell GPU Constraints

- CUDA graphs work on single GPU — don't use `--enforce-eager` (37-470% speedup)
- TP=2 needs `--enforce-eager` (memory corruption under sustained load)
- `--disable-custom-all-reduce` always needed for TP=2 (PCIe, not NVLink)
- No xformers / Flash Attention — use PyTorch native SDPA
- FlashInfer backend crashes — don't use `--attention-backend flashinfer`
- `--kv-cache-dtype fp8` forces FlashInfer attention internally → same crash. Don't use until upstream fixes Blackwell sm120 support.
- `VLLM_USE_FLASHINFER_MOE_FP8=1` rejects Qwen's block-wise [128,128] FP8 quant scheme. Don't use with Qwen3 FP8 models (122B, 35B, etc.).
- `-O3` (torch compile level 3) regresses MoE inference by ~25% — MoE routing is too dynamic for the compiler. Safe on dense models (27B uses it), avoid on MoE.
- INT4 safe on dense models, unstable on MoE (use BF16 for MoE)
- Capability cliff at 4B→2B: sub-4B models can't do agentic tool use

## Thinking models — vLLM reasoning-parser gotchas

The `--reasoning-parser qwen3` flag is greedy: any output before `</think>` is classified as `reasoning_content`. If the model fails to emit a closing `</think>` (common in long agent loops, amplified by `preserve_thinking=true`), the **entire answer** lands in `reasoning_content` and `content` is empty. Downstream consumers reading only `content` see a blank response. Confirmed upstream: [vllm-project/vllm#40528](https://github.com/vllm-project/vllm/issues/40528) — no fix yet.

Mitigations in this stack:
- **Gateway-side:** LiteLLM custom callback `thinking_normalizer.py` salvages `reasoning_content → content` when content is empty, strips inline `<think>...</think>` blocks, and exposes the raw trace as `reasoning` (OpenRouter convention). Lives in `homelab-iac` `stacks/ai/config/litellm/callbacks/`.
- **Eval-side:** `claw-eval`'s `Message.text` accessor falls back to `reasoning_content` and rsplits on `</think>` for the primed-think case (defense-in-depth for direct-to-vLLM consumers).

Heretic history note (resolved 2026-05-01): heretic was originally retired 2026-04-29 after we found the `logit_bias: {248068:-100, 248069:-100}` clamp recommended by the model card causes 1-token role-marker garbage (`Action`, `assistant`, `Human`) on prompts that engage the thinking pathway. Re-validated 2026-05-01 once the gateway-strip pipeline removed the need for any sample-time suppression: without `logit_bias`, heretic generates cleanly and behaves equivalently to the official 35B-A3B-FP8 (same `<think>...</think>` markup tendency, same gateway-strip outcome) — with the bonus that "uncensored" actually means uncensored. Now serves as `protolabs/fast`. The official `Qwen/Qwen3.6-35B-A3B-FP8` has the same model-side thinking behavior; the only difference is heretic's training removes the corporate-hedge layer.

Tool-call thinking observability (2026-05-01): the `qwen3.5-tool-calling.jinja` template originally had a "Fix 19" block that auto-disabled `enable_thinking` whenever the request carried tools. Side effect: every agentic turn through `protolabs/smart` skipped thinking entirely, and `metadata.thinking` came up empty on tool-call traces — we initially wrote this off as an upstream vLLM gap. It wasn't. Removing Fix 19 lets the model emit `<think>thoughts</think><tool_call>...</tool_call>`; the `qwen3_xml` tool-call parser still extracts the tool call cleanly, and the gateway's `_ThinkStripper` (primed-think branch — model output starts after the `<think>\n` priming, so the stream contains only the closing `</think>`) captures the thinking body to `metadata.thinking` and re-emits via `delta.reasoning_content`. Verified end-to-end: 4-probe matrix (direct vLLM, gateway streaming, non-tool regression, multi-tool selection) all clean, no `<tool_call>` markup leaks into reasoning (vLLM #39056 isn't a risk for Qwen3.6-27B-FP8 with this template). Cost: ~1s extra per tool-call turn (200-400 tokens of thinking before the tool call). For latency-critical agent loops where thinking isn't worth it, route through `protolabs/fast` (35B-A3B-FP8 + the `-nothink` template, which keeps Fix 19 equivalent via hardcoded `enable_thinking=false`).

## Secrets

All secrets in Infisical at `secrets.proto-labs.ai`. Never commit secrets. Gateway `start.sh` injects at runtime via Machine Identity.

## Storage

- `/mnt/models` — frequently-accessed model weights only (1TB NVMe, 420GB free)
- `/mnt/data` — datasets, checkpoints, outputs, cold model storage (2TB NVMe)
- `/mnt/data/models-cold/` — FLUX, Z-Image, Voxtral, OCR models (moved off fast drive)
- `/mnt/scratch` — logs, caches, docker volumes (disposable)
