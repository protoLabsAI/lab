# Qwen3.6-35B-A3B-FP8 — Benchmark Results

**Date:** 2026-04-25
**Model:** `Qwen/Qwen3.6-35B-A3B-FP8` (official FP8 weights, 35GB)
**Hardware:** NVIDIA RTX PRO 6000 Blackwell, GPU 1 (96 GB VRAM)
**vLLM config:** TRITON_ATTN backend, FP8 KV cache, 256K context, no-thinking template
**Served as:** `local-voice` on port 8002
**Role:** Executor / voice / agentic (no CoT — thinking disabled at vLLM level)

---

## Speed

| Metric | Value |
|--------|-------|
| Decode throughput | **231.7 tok/s** |
| TTFT | **23ms** |
| Context | 256K |
| GPU utilization | 0.73 (shares GPU 1 with Fish Audio S2-Pro, embed server) |

Baseline for Qwen3.5-35B-A3B at 170 tok/s (+36% improvement).
MoE kernel tuned for Blackwell SM 12.0 via `benchmark_moe.py` (2026-04-25, ~2.5hrs, 640 configs × 18 batch sizes).

---

## Eval Results

Full profile, 3 trials, pass^3 threshold.

### claw-eval (Agentic tool use)

| Tasks | Passed | Score |
|-------|--------|-------|
| 20/20 | ✅ 100% | All T02–T99 categories |

Clean sweep across email, calendar, CRM, ops, finance tool-use categories.
No thinking tokens — tool calls are immediate, no CoT preamble.

### Custom Suites

| Suite | Result | Pass^3 |
|-------|--------|--------|
| **creative_writing** | 23/25 | 92% |
| **roleplay** | 5/5 | 100% |
| **svg_generation** | 4/5 | 80% |
| **research** | 4/4 | 100% |
| **function_call** | 8/8 | 100% |

From the earlier quick profile (1 trial):

| Suite | Result |
|-------|--------|
| coding | 10/10 |
| reasoning | 5/5 |
| structured_output | 5/5 |
| instruction_following | 5/5 |
| summarization | 5/5 |
| safety | 5/5 |

**Total: 99/102 tasks passed** across all suites.

---

## Analysis

**Strengths:**
- Tool calling is flawless (100% function_call, 100% claw-eval) — the no-thinking mode does not degrade tool accuracy at all
- Speed advantage over dense models is significant: 231 tok/s vs 53 tok/s for 27B INT4, ~4.4x faster
- KV cache efficiency: MoE has only 10 full-attention layers → ~4.4M token KV pool at 256K → 17 concurrent 256K sessions
- Reasoning, structured output, safety all perfect

**Weaknesses:**
- `svg_generation` 4/5 (80%) — one failure expected; visual/spatial reasoning without CoT is harder
- `creative_writing` 23/25 (92%) — two pass^3 failures; narrative craft at the very high bar benefits from thinking

**Verdict:** Excellent executor model. No-thinking is the right mode for this role — tool calling, instruction following, structured output, and speed are all top tier. CoT is not needed for execution tasks and would add 50-200ms latency per turn in a voice context.

---

## Infrastructure

- Service: `/etc/systemd/system/vllm-voice.service`
- No-thinking template: `models/templates/qwen3_nonthinking.jinja`
- MoE kernel config: `models/moe-configs/E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json`
- TRITON_ATTN required — FlashInfer JIT fails on Blackwell SM 12.0
- GPU memory util 0.73 to coexist with Fish Audio (~23GB resident on GPU 1)

## Comparison

| Model | Role | tok/s | Claw | FC | Notes |
|-------|------|------:|-----:|---:|-------|
| Qwen3.6-35B-A3B-FP8 | Executor | **231** | 20/20 | 8/8 | No thinking, MoE tuned |
| Qwen3.5-27B-INT4 | Daily driver | 53 | — | — | MTP: 70 tok/s |
| Qwen3.6-27B-FP8 | Planner | ~70* | — | — | Thinking enabled, 256K |

\* Estimated; not yet benchmarked on this machine at 256K.
