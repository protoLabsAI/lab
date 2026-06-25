# dflash — block-diffusion speculative decoding eval

**Status:** active (started 2026-06-25)
**Question:** Does dFlash (block-diffusion drafter) beat our current spec-decode (MTP on the smart lane) on Blackwell sm120, and is it worth shipping?

## Background

dFlash ([NVIDIA blog](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/),
[z-lab/dflash](https://github.com/z-lab/dflash), [arXiv 2602.06036](https://arxiv.org/pdf/2602.06036))
is a speculative-decoding **drafter** that replaces EAGLE-3 / MTP. Instead of an autoregressive
draft emitting one token per forward pass, dFlash is a tiny **block-diffusion** model that predicts
a whole block of masked future tokens in a *single* forward pass, conditioned on the target model's
intermediate hidden states (`target_layer_ids`). Single-pass drafting → low spec overhead.
Claimed **2.3–2.8× over EAGLE-3**; up to 15× throughput on gpt-oss-120b at high interactivity.

Relevant to us because it targets our exact hardware (sm120) and beats EAGLE-3, which our
[gemma4 speed-levers research](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/reference_gemma4_moe_speed_levers.md)
flagged as the top fast-lane speed lever.

## What's on our box (discovery 2026-06-25)

- **vLLM 0.22.1** ships dFlash infra natively: `vllm/v1/spec_decode/dflash.py`,
  `vllm/model_executor/models/qwen3_dflash.py`, registry `DFlashDraftModel → DFlashQwen3ForCausalLM`,
  and the speculators config transform (`configs/speculators/algos.py @register_speculator("dflash")`).
  **Qwen3-family dFlash is supported; Gemma4 dFlash is NOT** (no gemma dflash model file —
  gated on unmerged vLLM PR #41703).
- `speculators` pip package is **not installed** and **not needed for serving** — vLLM loads the
  speculators-format draft checkpoint directly. (The pip lib is for *training/converting* drafts.)

### Checkpoint ↔ our models

| Our lane / model | On disk | Official z-lab dFlash draft | vLLM support |
|---|---|---|---|
| **Smart lane — Qwen3.6-27B-FP8** (dense) | ✅ | ✅ `z-lab/Qwen3.6-27B-DFlash` (2B bf16) | qwen3_dflash ✅ — card says needs PR #40898 for interleaved SWA (verify; may be stale) |
| Qwen3.5-9B (dense) | ✅ (+FP8) | ✅ `z-lab/Qwen3.5-9B-DFlash` (sliding-window) | qwen3_dflash ✅ |
| Qwen3.5-4B (dense) | ✅ | ✅ `z-lab/Qwen3.5-4B-DFlash` | qwen3_dflash ✅ |
| **Fast lane — Gemma-4-26B-A4B** (MoE) | ✅ (+RedHat FP8-Dyn) | ✅ `z-lab/gemma-4-26B-A4B-it-DFlash` (0.4B) | ❌ needs vLLM **PR #41703** (gemma4 dflash unmerged) |
| Qwen3.6-35B-A3B (MoE) | ✅ (+FP8) | ✅ `z-lab/Qwen3.6-35B-A3B-DFlash` | qwen3_dflash ✅ (MoE — spec historically regresses, see below) |

Draft configs use `architectures: ["DFlashDraftModel"]`, `block_size` 16, `target_layer_ids`
spread across the target's depth, `use_sliding_window: true`. Both 27B and 9B drafts carry
`sliding_attention` layer types → the interleaved-SWA path the card flags.

## Baselines to beat (from CLAUDE.md / MTP table)

- **Smart lane today:** Qwen 27B-FP8 **+ MTP** = ~74 tok/s (+48% over 50 baseline). dFlash must beat *MTP*, not baseline.
- 9B + MTP = 112 tok/s (+22%). 9B is the cheap apples-to-apples sanity check.
- **MoE caution:** spec decode has regressed on our MoE before — 35B MoE + MTP = **−11%**
  (routing overhead > speculation). dFlash is a different mechanism but unproven on our MoE;
  the Gemma fast lane (MoE) is the risky case.

## Plan

1. **Tier 0 — does it serve at all on stock 0.22.1?** Try `Qwen3.6-27B-FP8` (smart-lane target) +
   `z-lab/Qwen3.6-27B-DFlash` on a spare port. If the SWA path errors → bump vLLM to PR #40898
   in an **isolated venv** (`~/dev/vllm-dflash-env`), leave prod `vllm-env` untouched.
2. **Tier 1 — measure.** A/B decode tok/s (`models/speed-test.sh`) + spec acceptance length
   (`vllm:spec_decode_*` from `/metrics`) vs current 27B+MTP. Long + short prompts.
3. **Tier 2 — the lane that matters most.** If the smart-lane win is real, evaluate the Gemma MoE
   fast lane via PR #41703 (separate build), accepting the MoE risk.
4. Report honest numbers in `RESULTS.md` (incl. acceptance length, TTFT, concurrency behavior).

## Constraints (sm120, from CLAUDE.md)

- `VLLM_USE_FLASHINFER_SAMPLER=0` required on every model (0.22.1 routes sampling through
  FlashInfer JIT which rejects sm120).
- **No flash_attn on sm120** — the card's `--attention-backend flash_attn` won't work; we
  auto-select / force `TRITON_ATTN`. The dflash draft's internal attn backend may need overriding.
- Don't force `--attention-backend flashinfer` (crashes). Don't use `-O3` on MoE (−25%).

## Files

- `run-dflash.sh` — serve a `TARGET` + `DRAFT` on a given `GPU`/`PORT` with dflash speculative-config.
- `bench.sh` — A/B harness: warm up, hit `models/speed-test.sh`, scrape spec-accept metrics.
- `RESULTS.md` — findings (written without softening).
