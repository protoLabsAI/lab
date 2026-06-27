# CLAUDE.md — protoLab (the heavy rig)

This repo is **substrate #2 — Models** in the [protoLabs.studio](https://protolabs.studio) portfolio. Heavy rig: 2× RTX PRO 6000 Blackwell, 192 GB VRAM total. FP8 / vLLM at scale. The model gateway every other studio service hits.

Sibling: [`avaLab`](https://github.com/protoLabsAI/avaLab) on the A6000 prosumer rig (GGUF / llama.cpp / Ollama / ComfyUI). Cross-cutting work lives here.

Studio-wide brand contract: [`protoLabsAI/studio-brand`](https://github.com/protoLabsAI/studio-brand). Read [`docs/explanation/portfolio.md`](https://github.com/protoLabsAI/studio-brand/blob/main/docs/explanation/portfolio.md) before reshaping anything in this repo's structure.

## Audience filter

The §3 filter from `studio-brand/docs/reference/foundation.md` applies to everything we publish from here:

- If we have to explain context or tokens — not our audience.
- If they need more than quickstart + example code + docs — not our audience.
- If they can't read the experience via CLI output and raw JSON — not our audience.

Open-source code itself has no audience filter — anyone can fork. The breakdowns, README copy, model cards, and dataset cards we publish *do*. No intro-to content. No hand-holding. **Assume competence.**

## What this repo ships to the brand

- **Findings.** CUDA graphs on Blackwell (+37–470%), INT4 routing corruption on MoE, FP8 KV cache broken on sm120, NCCL_P2P_DISABLE=1 fixing TP=2 PCIe — patterns to study and steal.
- **Models on HF.** FP8 quants of Qwen3.6 family published to [`protoLabsAI`](https://huggingface.co/protoLabsAI) via `experiments/quantize/`.
- **Eval suite.** `evals/` is the lab's open-source pattern. claw-eval is already a public substrate.
- **The gateway.** `infra/gateway/` is operational infrastructure for every other studio service.

## What was parked 2026-05-22

ORBIS retired as a product. Coding-agent and prompt-product work out of scope. Side bets parked under `experiments/*/PARKED.md`: companion-stack, voice-agent, salm-duplex, stt-whisper, tts-compare, agent-lightning, rlm, qwen3-omni, image-gen-eval, flux2, pixel-gen, ltx-video. Audio-tags survives as the standalone brand exemplar.

The thesis "small specialized models per pipe of the conversational loop" survives in audio-tags. If a new consumer surface appears, unpark one pipe at a time as a fresh experiment.

## Structure

- `packages/lab-core/` — Pydantic models, GPU utils, path constants. Publishable.
- `evals/` — claw-eval (submodule), custom suites, function-call, RAG, refusal. Strict + tested.
- `models/` — vLLM configs (`vllm-swap.sh`), MoE kernels (`moe-configs/`), benchmark scripts.
- `training/` — fine-tuning workspace (LLaMA-Factory configs, TRL).
- `experiments/` — active: `audio-tags/`, `context-1/`, `quantize/`, `embedding-bench/`, `gemma4-eval/`, `proto-bench/`, `rag-bench/`, `vllm-dashboard/`. Everything else parked.
- `infra/` — gateway (LiteLLM + Langfuse), vLLM systemd, Prometheus exporters.

## How we operate

The cycle:

```
experiment → report → engineering → test → content → repeat
```

Each phase has an exit criterion; don't move on until current phase is done.

| Phase | What | Exit when |
|---|---|---|
| **experiment** | research in `experiments/<name>/` | model artifact + tier-0 baselines + cross-domain held-out eval |
| **report** | internal `RESULTS.md` with honest numbers and what didn't work | written without softening — for next-session-us |
| **engineering** | wire the artifact into a consuming surface (gateway, eval, training) | PR landed on consuming repo's main |
| **test** | validate under real conditions | one round of real-world signal (real users, real traffic) |
| **content** | public artifact: HF model + dataset cards + protolabs.studio blog post | merged blog + public HF release |
| **repeat** | next experiment, informed by what shipped | always |

Default to publishing publicly via `protoLabsAI/` on HuggingFace and protolabs.studio for the writeup. Privacy is a drafting state, not a target. **Every shipped experiment produces a blog draft in `experiments/<name>/BLOG.md` before the next experiment starts.** audio-tags is the template.

## Daily setup (dual GPU) — Ornith-35B-FP8 replicas (2026-06-27)

All services run as systemd and auto-start on boot. **Daily driver = 2× our `Ornith-1.0-35B-FP8` replicas** (one per GPU, gateway round-robin). North star in [`FOCUS.md`](FOCUS.md); model published at [`protoLabsAI/Ornith-1.0-35B-FP8`](https://huggingface.co/protoLabsAI/Ornith-1.0-35B-FP8) (our quant — upstream FP8 was broken; SSM kept bf16, 92.9% truly-fp8, parity-verified).

| GPU | Service | Model | Port | Notes |
|-----|---------|-------|------|-------|
| 0 | `vllm.service` | Ornith-1.0-35B-FP8 | :8000 | replica A, util 0.90, 256K, vision, `local` |
| 1 | `vllm-replica-b.service` | Ornith-1.0-35B-FP8 | :8003 | replica B, util 0.72 (Fish+embed co-resident), 256K, vision, `local` |
| 0 | `embed-b.service` | Qwen3-Embedding-0.6B | :8004 | embed B (doubled) |
| 1 | `embed-server.service` | Qwen3-Embedding-0.6B | :8001 | embed A |
| 1 | `protovoice-stack.service` | Fish S2-Pro TTS | :8092 | ~20GB, lazy-load |

`vllm-fast.service` (retired Gemma fast lane) is **disabled**. Gateway: `protolabs/smart` round-robins (`least-busy`) across :8000+:8003, embedding alias balances :8001+:8004 ([homelab-iac#174](https://github.com/protoLabsAI/homelab-iac/issues/174)). ~207 tok/s/request, ~6500 tok/s aggregate. **Replicas beat DP+EP/TP=2 on PCIe** — never shard a model that fits one card (benchmarked C32: DP+EP 3830 · single 3990 · 2 replicas ~6500). Eval judges target a replica (`local` :8000/:8003) now the Gemma lane is gone. WAN upload is ~100 Mbps (a 100BASE-TX choke, [homelab-iac#176](https://github.com/protoLabsAI/homelab-iac/issues/176)) — publish big models via per-file commits or cloud-quantize.

> **Everything below this line is HISTORICAL** — dFlash, DiffusionGemma, AR-Gemma fast lane, 27B-MTP smart lane — all **superseded by the Ornith replica setup above**. Kept for breakdown material + the GPU/util/Fish-TTS budget math.

**2026-06-25: smart lane tried dFlash spec-decode, then REVERTED to MTP (see tradeoff note below).** For ~2.5h ran z-lab's block-diffusion **dFlash** draft (`z-lab/Qwen3.6-27B-DFlash`, 2B bf16, drafts for Qwen3.6-27B). Serves on **stock vLLM 0.22.1 — no bump, no PR, no `speculators` pip** (0.22.1 already ships `qwen3_dflash.py`/`DFlashQwen3ForCausalLM` + `v1/spec_decode/dflash.py`; the model card's "needs PR #40898 for interleaved SWA" note is stale for our build). Measured **74.6 → 106.9 decode tok/s (+43%)** single-stream in the live prod config (`-O3`, 225K, 512 seqs); tool-calling (`qwen3_xml`) + thinking verified clean, lossless (target verifies every accepted token). Only the `--speculative-config` line changed (MTP → `{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}`); `num_speculative_tokens=10` is the tuned peak (sweep 3/6/10/16 → 96.8/114.3/116.9/106.2 tok/s on a leaner GPU1 config). Backup unit at `~/dev/.vllm-bump-review/unit-backups/vllm.service.pre-dflash-*` — rollback to MTP = restore + `daemon-reload` + restart. Full work in `experiments/dflash/` (README/RESULTS/sweep.sh/run-dflash.sh/bench.sh/conc_bench.py/conc-driver.sh).

**⚠️ CONCURRENCY TRADEOFF → REVERTED to MTP same day (2026-06-25 21:57).** dFlash wins **single-stream only**. It does **not scale** — aggregate throughput plateaus ~540 tok/s regardless of load, while MTP scales near-linearly. Crossover ~C=3–4; by C=32 **MTP is ~3× faster**. Aggregate tok/s (dFlash N=10 vs MTP): C1 110.7/73.2 · C4 219.7/269.9 · C8 458.5/545.4 · C16 543.6/984.0 · C32 503.7/**1471.2**. Cause: dFlash drafts 10 tok/step @ ~16% accept (~11× verify-batch inflation, saturates compute under load); MTP drafts 1 @ ~79% (~2× inflation, stays efficient). `-O3` A/B = wash (543.6 vs 546.5 @ C16; earlier 116.9 was leaner ctx/seqs, not -O3). Eval quality held (custom 84% / FC 91% ≥ MTP baseline; lossless as expected).

**Why reverted:** initially kept dFlash on the assumption the smart lane is single-stream/interactive. Then **live Prometheus + LiteLLM-gateway traffic analysis** showed the opposite: when the lane is *actually busy* it's **parallel-agent fan-out at C=4–8 with a deep wait queue (18–24)** — real agents (protoAgent, Jon, QwenCode, langchainjs apps spawning parallel sub-agents), not single-stream. dFlash's single-stream win serves <5% of active samples; the C=4–8 regime that defines real heavy usage is exactly where MTP wins. So MTP is the correct default. Data source: Prometheus on ava (`100.101.189.45:9090`) — `vllm:num_requests_running{model="local"}` + `litellm_*` per-deployment/user-agent attribution. **dFlash stays a one-command swap-in for any genuinely single-stream lane** (draft cached at `models--z-lab--Qwen3.6-27B-DFlash`; set `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}'`). Full work + numbers in `experiments/dflash/` (README/RESULTS/sweep.sh/run-dflash.sh/bench.sh/conc_bench.py/conc-driver.sh).

Gemma fast lane has its own dFlash draft (`z-lab/gemma-4-26B-A4B-it-DFlash`) but needs unmerged vLLM **PR #41703** (gemma4 dflash not in 0.22.1) + MoE-spec is our historical −11% risk case.

**2026-06-14: fast lane REVERTED to AR Gemma 4 26B-A4B (DiffusionGemma demoted to on-demand) — now the shipped daily-driver fast lane.** A creative-writing head-to-head (`experiments/diffusion-creative-eval/`, 50 prompts vs human refs, MMD/Token-L2/slop + pairwise judge) found **AR Gemma 4 beats DiffusionGemma 64/36** on creative quality, lower MMD, less slop; DG's only edge was ~1.9× raw speed, which doesn't justify the quality loss + ops cost (DG can't guided-decode, no token streaming, OOM-on-restart, short-output collapse). Then two free wins on the AR lane: **(1)** official RedHat **FP8-Dynamic** quant (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`, per-tensor, pre-quantized — no `--dtype/--quantization`, loads 28 GB directly) = +4%; **(2)** **auto-tuned fused-MoE Triton config** for our exact shape (E=128,N=704, sm120; `models/moe-configs/` + `install-moe-configs.sh`) = +2%. Net **199 → 211 tok/s**, streams incrementally, guided-decode works. Unit: `/etc/systemd/system/vllm-fast.service` (serves the RedHat repo, util 0.72, NO `-O3` — regresses MoE ~25%, gemma4 parsers, `VLLM_USE_FLASHINFER_SAMPLER=0` + `TRITON_ATTN`). DG remains a one-command swap-in via `models/diffusion-fast-lane.sh` (+ backup unit in `~/dev/.vllm-bump-review/`). Also fixed an embed-server memory bloat found during the swap (6.4→1.7 GB; `experiments/rag-bench/serve_embed.py` MAX_BATCH + empty_cache + expandable_segments). **Everything below this line is the historical DiffusionGemma deployment, kept for the on-demand-DG runbook + the GPU1 util/Fish-TTS budget math.**

**2026-06-13: fast lane switched to DiffusionGemma 26B-A4B (text-diffusion), replacing Gemma 4.** Runs Google's day-zero diffusion runner via the `vllm/vllm-openai:gemma` **Docker container** (not in pip vLLM 0.22.1), launched by `models/diffusion-fast-lane.sh`: GPU1 `device=1`, :8002, `local-fast`, `--quantization fp8`, `--max-model-len 262144`, `--diffusion-config '{"canvas_length":256}'`, `--gpu-memory-utilization 0.68`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, gemma4 tool+reasoning parsers. FP8 (~26 GB) co-resides with Fish TTS + embed; **util 0.68 (not 0.72) + expandable_segments are required** — at 0.72 it OOMs during graph capture (the diffusion runner uses more transient memory than AR Gemma). Rollback: old AR unit at `~/dev/.vllm-bump-review/unit-backups/vllm-fast.service.pre-diffusion-*`. **Streaming caveat: it does NOT stream incrementally** — `stream:true` returns the whole canvas as one chunk (block-sized bursts beyond 256 tok), no token-by-token. It's a latency engine for long-form/low-concurrency content (~2–4× faster, prose quality on par with the Gemma-4 it's built on), NOT for short outputs / structured / guided / tools — full eval in `experiments/diffusiongemma/RESULTS.md`. The Gemma-4 notes below are historical, kept for the GPU1 util/Fish-TTS budget math (which still applies to the shared card).

**DiffusionGemma fast-lane runbook (gotchas hit + fixes, all baked into `models/diffusion-fast-lane.sh`):**
- **256-token truncation** — DG's `generation_config.json` defaults `max_new_tokens: 256`, which vLLM applies to any request that doesn't set `max_tokens` → outputs cut off mid-sentence. Fixed with **`--override-generation-config '{"max_new_tokens": 4096}'`** (merges in, keeps the diffusion params — do NOT use `--generation-config vllm`, that discards `max_denoising_steps`/`entropy_bound`/sampler config the runner needs).
- **Thinking leak / empty content** — DG's `chat_template.jinja` auto-enables thinking whenever there's a **system/developer message, tools, or `enable_thinking`** (an agent harness trips this). Thinking then either leaks `<|channel>thought…<channel|>` into `content` or eats the whole token budget (empty content). Fixed with a **no-think chat template** `models/diffusion-fast-nothink.jinja` (prepends `{%- set enable_thinking = false -%}`, forced via `--chat-template`) — thinking is now off regardless of client params. Plain user-only requests never tripped it; only system-message/agent traffic did.
- **Intermittent multi-second stalls** — first-hit Triton JIT compiles for shapes the startup graph-capture missed (`jit_monitor`). Fixed by `models/diffusion-fast-warmup.sh` wired as `ExecStartPost` (sweeps shapes on every start). Genuinely-novel shapes still pay a one-time compile.
- **Latency variance** — denoising steps per request vary wildly (observed 7 → 211); long/hard outputs cost multiples. `max_denoising_steps` (default 48/canvas, in `generation_config`) is the bounding lever — lower it via `--override-generation-config` to cap worst-case latency at a quality cost (not currently capped).
- **Residual `<|channel>` leak** still belongs to the gateway scrub (`_ThinkStripper`, homelab-iac#147) — route clients through `api.proto-labs.ai`/`ava:4000`, not direct to `:8002`, for belt-and-suspenders.

vllm-fast lives alone on GPU 1 as of 2026-05-03 — ComfyUI was disabled when image-gen work moved to avaLab. Gemma 4 26B MoE FP8 at `--gpu-memory-utilization 0.72 --max-model-len 262144` (256K). **util dropped 0.80 → 0.72 on 2026-05-30 because Fish TTS is now active** (S2-Pro server on `:8092`, ~19.8 GB on GPU 1) — the earlier "Fish TTS inactive" budget no longer holds. With Fish TTS (~19.8 GB) + qwen3-embed (~2 GB) co-resident, util 0.80 (76 GB) OOMs against ~73 GB free; 0.72 (68 GB) fits with headroom and still loads full 256K. KV math on this card: roughly `pool = (96 × util) − model_size − overhead`. If Fish TTS is ever stopped, util can go back to 0.80.

`--reasoning-parser gemma4` keeps Gemma's thinking off the `content` channel — verified clean across single-token-streaming reasoning/tool-call/constrained prompts (2026-05-30). **Caveat (2026-06-02): one client-side leak was reported via `protolabs/fast`** — reasoning landing on `content` as bare `thought…`/`<|channel>` text, the known upstream gemma4-parser class ([vllm#39885](https://github.com/vllm-project/vllm/issues/39885); related #38855/#39392), triggered by batched-delta streaming around the tool-result/multi-turn boundary. **Could not reproduce on the current build (0.20.1)** — clean across ~68 probes at default `--stream-interval 1` *and* under the documented trigger: `--stream-interval 20` + forced tool calls (`tool_choice: required`, `<|tool_call>` landing in a batched delta) + 16-way concurrent, raw-SSE inspected. The primary #39885 case (closed upstream) appears already handled here; the still-open residual PRs (#42875/#39898) target a narrower case we can't trigger, so a vLLM bump isn't load-bearing. Mitigation is a belt-and-suspenders gateway scrub ([homelab-iac#147](https://github.com/protoLabsAI/homelab-iac/issues/147), live + reversible via env flags) that strips Gemma's `<|channel>…<channel|>`/bare `thought` the way the Qwen `_ThinkStripper` handles `<think>` — assumed working, treated as observational since there's no synthetic repro to confirm against. Model + parser config on the box are correct and current — this is purely the upstream parser's streaming path.

**2026-05-31: LFM2.5-8B-A1B ("nano") evaluation — FAILED on Blackwell, Fish TTS retained.** Tried to drop in **LiquidAI/LFM2.5-8B-A1B** (hybrid conv+attn MoE, `Lfm2MoeForCausalLM`, 8.3B/1.5B-active, BF16 ~16.9 GB) on GPU 1 (port 8003) in place of the Fish/protovoice stack. Model downloaded fine and vLLM 0.20.1 registers `lfm2_moe`, but it **will not serve on sm120**: the unquantized MoE expert path routes to `flashinfer_cutlass_fused_moe`, which JIT-compiles an sm120 CUTLASS module and dies — `RuntimeError: No supported CUDA architectures found for major versions [12]` (flashinfer/compilation_context.py). Same FlashInfer-broken-on-Blackwell class as our attention findings, now in the fused-MoE path. **Not yet retried** with the flashinfer MoE backend disabled (candidate fix: force the Triton/native fused-MoE path instead of flashinfer-cutlass — investigate `VLLM_USE_FLASHINFER_MOE_*` / fused-moe backend selection). Weights kept in cache at `models--LiquidAI--LFM2.5-8B-A1B`. Fish/protovoice stack was stopped during the attempt and has been **re-enabled** (`protovoice-stack.service` active+enabled, :8092/:8093). Gemma stays at util 0.72.

Fast-lane history (preserved for context): originally Qwen 35B MoE FP8 official → heretic-FP8 (un-retired for uncensored prose, see [memory](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md)) → Gemma 4 26B MoE. Heretic was briefly reverted to and back on 2026-05-30 — it works (179 tok/s, clean with `--reasoning-parser qwen3`) but needs 32K output headroom for its trained-in think to close, so Gemma stays the default fast lane. The heretic HF card recommended a `logit_bias: {<think>:-100, </think>:-100}` clamp that corrupts generation — don't use, and ignore the HF README's workaround if you ever load that model. Gateway's `_ThinkStripper` (homelab-iac #26/#32/#35) handles `<think>...</think>` post-emit for any thinking-tagged model.

Tuned MoE kernels in `models/moe-configs/` symlink into vLLM's `fused_moe/configs/` via `bash models/install-moe-configs.sh` (run after fresh vLLM installs / upgrades).

## Running models

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

## Speed testing

```bash
bash models/speed-test.sh           # 5 runs on current model (800 tok gen)
bash models/speed-test.sh 10        # 10 runs
bash models/speed-test.sh 3 short   # 3 short runs (200 tokens)

cd evals && bash run-ab-speed.sh qwen-4b-int4 5
```

Reports decode tok/s (1/TPOT), wall tok/s, TTFT, and TPOT from vLLM's `/metrics` endpoint — not wall-clock estimation.

### Optimization flags (`-opt` configs)

Suffix any config with `-opt` to enable P1+P2:
- `--async-scheduling` — overlap scheduling with execution
- `--enable-prefix-caching` — reuse KV cache for repeated prefixes
- `--performance-mode interactivity` — auto-tune scheduler for latency
- `--kv-cache-dtype fp8` — halve KV cache memory, double context capacity

Measured impact (single-request, P1+P2 only): minimal (+1–3% tok/s). Real wins are under concurrent load and multi-turn (prefix caching). FP8 KV doubles context capacity.

### MTP speculative decoding (`-mtp` configs)

Native Qwen3.5 Multi-Token Prediction:

| Model | Baseline | + MTP | Gain | Tool calling |
|-------|:--------:|:-----:|:----:|:------------:|
| **27B FP8** | 50 tok/s | **73.6 tok/s** | **+47%** | TBD |
| **27B INT4** | 53 tok/s | **70 tok/s** | **+32%** | Works, T08 quality regresses |
| **9B** | 92 tok/s | **112 tok/s** | **+22%** | Works, no quality loss |
| **35B MoE** | 171 tok/s | 153 tok/s | -11% | N/A — slower |

MTP helps dense, hurts MoE (routing overhead > speculation savings). 9B + MTP safe for tool calling. 27B FP8 + MTP daily driver on GPU 0. 27B INT4 + MTP: chat/creative only, avoid for complex agentic. MoE FP8 env vars: `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency`.

### TP=2 tuning (122B, 35B-tp2)

NCCL env vars for PCIe (no NVLink): `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=8`.

- **`NCCL_P2P_DISABLE=1` enables stable CUDA graphs on TP=2 Blackwell PCIe**
- 122B INT4: 23 → **122 tok/s** (5.3×, stable over 10 runs)
- 35B MoE: 22 → **205 tok/s** (9.3×, stable over 10 runs)
- Root cause: ACS enabled on PCIe bridges corrupts P2P during CUDA graph replay
- Disabling P2P forces shared memory transport — slight overhead, fully stable
- TTFT: 3077 ms → 29 ms with prefix caching after warmup
- Power draw: 88–96 W per card at 600 W limit — MoE is not power-bound
- 35B TP=2: prefix caching fixed 1.8 s TTFT → 0.5 s (**−70%**), wall tok/s +25%
- `VLLM_USE_FLASHINFER_MOE_FP8` crashes on 122B FP8 (unsupported quant scheme) — don't use
- Previous finding that TP=2 needs enforce-eager was wrong — `NCCL_P2P_DISABLE=1` is the fix

## Running evals

```bash
cd evals

./run.sh profile --name quick --model local    # ~15 min smoke test, 1 trial
./run.sh profile --name full --model local     # ~60-90 min comprehensive, 3 trials

./run.sh claw --model local --tasks T02,T04,T06,T08 --port-offset 200
./run.sh custom --suite coding --model local --trials 1
./run.sh custom --suite reasoning --model local --trials 1
./run.sh function-call --model local --all-suites
```

### Suites

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

Profiles: **quick** — 6 claw + 6 custom + FC, 1 trial (~15 min). **full** — 20 claw + 10 custom + FC, 3 trials pass^3 (~60–90 min).

## Model inventory (`/mnt/models`)

| Model | Size | tok/s | +MTP | Quick Score | Role |
|-------|------|:-----:|:----:|:-----------:|------|
| **Qwen 35B MoE FP8** | 35GB | **180** | — | — | Speed king, 262K ctx, single GPU (Qwen official) |
| **Qwen 9B FP8** | 19GB* | **141** | — | — | On-the-fly FP8 (+53% vs bf16) |
| **Qwen 4B FP8** | 8.8GB* | 141 | — | — | On-the-fly FP8 edge |
| **Qwen 27B FP8** | 29GB | 50 | **73.6** | — | Daily driver on GPU 0 (MTP on by default in vllm.service, 225K ctx; verified on hybrid-Mamba 3.6 / 0.22.1, 79% accept) |
| **Qwen 27B INT4** | 29GB | 53 | **70** | **86/103** | Alt daily driver |
| **Gemma 4 31B FP8** | 62GB | 42.9 | TBD | — | Dense alt; MTP config ready |
| **Qwen 122B FP8** | 119GB | **112** | — | — | Quality ceiling FP8 TP=2 |
| **Qwen 122B INT4** | 74GB | **122** | — | **89/103** | Quality ceiling INT4 TP=2 |
| **Qwen 27B FP8 TP=2** | 29GB | **70** | — | — | 131K ctx, TP=2 (Qwen official) |
| **Qwen 9B BF16** | 19GB | 92 | **112** | 72/103 | Fine-tune base (cold storage) |
| **Qwen 4B INT4** | 3.8GB | 297 | — | 56/103 | Edge deploy (fastest absolute) |

\* On-the-fly FP8 (`--quantization fp8`) loads bf16 from disk, quantizes during load.

Base models (0.8B, 2B, 4B) downloaded for pretraining. FP8 quants in `/mnt/models/quantized/` and on [HuggingFace protoLabsAI](https://huggingface.co/protoLabsAI).

Cold storage (`/mnt/data/models-cold/`): FLUX.2-klein 9B+base (100GB), Z-Image+Turbo (51GB), Voxtral-Mini-4B (17GB), OCR models (11.4GB). Image-gen models live here pending eventual reclaim — they belong on avaLab now.

## Blackwell constraints

**2026-06-13: prod migrated to vLLM 0.22.1 / CUDA 13** (was 0.20.1 / CUDA 12.8). torch 2.11+cu130, transformers 5.12. Supply-chain-reviewed hash-locked install. Two env vars are now REQUIRED on sm120 that 0.20.1 set automatically — without them you hit the FlashInfer sm75 crash:
- **`VLLM_USE_FLASHINFER_SAMPLER=0`** — 0.22.1 routes top-k logit sampling through FlashInfer's JIT, which rejects sm120. Now set in both units (`vllm.service`, `vllm-fast.service`) + `vllm-swap.sh`. Universal; needed for every model.
- **`VLLM_ATTENTION_BACKEND=TRITON_ATTN`** on the Gemma fast lane (belt-and-suspenders; 0.22.1 still auto-forces it for gemma4's heterogeneous head dims). Qwen auto-selects fine without it.
- Cutover was a venv directory-swap (`vllm-env` ⇄ `vllm-env-0.20.1-bak`) — this breaks pip console-script shebangs (`vllm`, `hf`); rewrite `vllm-env-OLD → vllm-env` across `bin/` after. `huggingface-cli` is gone in the new env → use `hf`. Rollback = swap dirs back + revert units; backups in `~/dev/.vllm-bump-review/unit-backups/`.

- CUDA graphs work on single GPU — don't use `--enforce-eager` (37–470% speedup).
- TP=2 needs `NCCL_P2P_DISABLE=1` (NOT `--enforce-eager` as previously thought). `--disable-custom-all-reduce` always needed for TP=2 (PCIe, not NVLink).
- No xformers / Flash Attention — use PyTorch native SDPA.
- FlashInfer attention backend still auto-selects on some paths and crashes — Gemma4 auto-forces `TRITON_ATTN`; for others rely on auto-select + `VLLM_USE_FLASHINFER_SAMPLER=0`. Don't force `--attention-backend flashinfer`.
- **FlashInfer-on-sm120 is now CRACKED for the JIT/cutlass paths (2026-06-13, flashinfer 0.6.11 / CUDA 13).** The old "FlashInfer requires sm75 or higher" wall was a chain of arch-flag + nvcc-version + cccl-guard + OOM + linker issues, all fixable. The cutlass fused-MoE compiles+loads on sm120 with: `FLASHINFER_CUDA_ARCH_LIST=12.0f`, `CUDA_HOME=…/nvidia/cu13` (CUDA-13.3 nvcc, not system 12.8), `NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`, `MAX_JOBS=4`, plus `cu13/lib64→lib` + bare `.so` symlinks. **Full recipe: `experiments/quantize/FLASHINFER-SM120-RECIPE.md`.** This is the door to FP8-KV cache (2× ctx), bf16 MoE, nano-LFM2, Mistral-128B — each still needs end-to-end serve validation.
- Unquantized **bf16 MoE** (e.g. 35B-A3B bf16) on 0.22.1 routes the fused-MoE to flashinfer-cutlass and hits the above wall — was fine on 0.20.1. Either apply the sm120 recipe, or use FP8/INT4 MoE (Triton path, works). `VLLM_USE_FLASHINFER_MOE_FP8=1` rejects Qwen's block-wise [128,128] FP8 quant scheme — don't use with Qwen3 FP8.
- `VLLM_USE_FLASHINFER_MOE_FP8=1` rejects Qwen's block-wise [128,128] FP8 quant scheme. Don't use with Qwen3 FP8 (122B, 35B, etc.).
- `-O3` (torch compile level 3) regresses MoE inference by ~25% — MoE routing is too dynamic for the compiler. Safe on dense (27B uses it), avoid on MoE.
- INT4 safe on dense, unstable on MoE (use BF16 for MoE).
- Capability cliff at 4B → 2B: sub-4B can't do agentic tool use.

## Thinking models — vLLM reasoning-parser gotcha

`--reasoning-parser qwen3` is greedy: any output before `</think>` is classified as `reasoning_content`. If the model fails to emit a closing `</think>` (common in long agent loops, amplified by `preserve_thinking=true`), the **entire answer** lands in `reasoning_content` and `content` is empty. Downstream consumers reading only `content` see a blank response. Upstream: [vllm-project/vllm#40528](https://github.com/vllm-project/vllm/issues/40528) — no fix yet.

Mitigations:
- **Gateway-side:** LiteLLM custom callback `thinking_normalizer.py` salvages `reasoning_content → content` when content is empty, strips inline `<think>...</think>` blocks, and exposes the raw trace as `reasoning` (OpenRouter convention). In `homelab-iac` `stacks/ai/config/litellm/callbacks/`.
- **Eval-side:** `claw-eval`'s `Message.text` accessor falls back to `reasoning_content` and rsplits on `</think>` for the primed-think case.

**vllm-fast must have `--reasoning-parser qwen3`** (2026-05-02 fix). Without it, heretic's trained-in always-thinks behavior emits `<think>` blocks straight into `content` for non-tool-call turns. The downstream cost: the protoCLI release-notes generator once published a leaked thinking trace as real release notes.

**Tool-call thinking observability** (2026-05-01): the `qwen3.5-tool-calling.jinja` template originally had "Fix 19" disabling `enable_thinking` when tools were present. Removing it lets the model emit `<think>thoughts</think><tool_call>...</tool_call>`; `qwen3_xml` parser extracts the tool call cleanly, gateway `_ThinkStripper` (primed-think branch) captures the thinking body to `metadata.thinking`. ~1 s extra per tool-call turn (200–400 tokens of thinking before the tool call). For latency-critical agent loops, route through `protolabs/fast` (the `-nothink` template).

## Secrets

All secrets in Infisical at `secrets.proto-labs.ai`. Never commit secrets. Gateway `start.sh` injects at runtime via Machine Identity.

## Storage

- `/mnt/models` — frequently-accessed model weights only (1TB NVMe, 420GB free)
- `/mnt/data` — datasets, checkpoints, outputs, cold model storage (2TB NVMe)
- `/mnt/data/models-cold/` — FLUX, Z-Image, Voxtral, OCR
- `/mnt/scratch` — logs, caches, docker volumes (disposable)
- `/mnt/pool` — **REMOVED 2026-05-28**: the 37TB mergerfs HDD pool (2x 20TB IronWolf Pro) was pulled and relocated to external housing on another machine. No bulk HDD storage on this node anymore; training corpora that lived here (incl. the 6.4T salm-duplex set) are now on the external box.

Full rules in [`/home/ava/dev/CLAUDE.md`](../CLAUDE.md) (the node-level CLAUDE.md). The hardlink trap from `huggingface-cli download --local-dir` is documented there — read it before cleaning HF caches.
