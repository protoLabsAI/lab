---
license: apache-2.0
base_model: InternScience/Agents-A1
base_model_relation: quantized
tags:
  - nvfp4
  - vllm
  - compressed-tensors
  - moe
  - agentic
  - qwen3.5
  - blackwell
pipeline_tag: text-generation
---

# Agents-A1 — NVFP4 (W4A4, calibrated MoE)

Calibrated **NVFP4** quantization of [`InternScience/Agents-A1`](https://huggingface.co/InternScience/Agents-A1)
(agentic Qwen3.5-35B-A3B hybrid MoE) for vLLM. **21.8 GB — and it matches or beats the official
FP8 on every quality axis we measured, at 60% of its size.**

Quantized and gate-verified by [protoLabs](https://protolabs.studio) on 2× RTX PRO 6000
Blackwell (sm120), vLLM 0.22.1.

## Quality — paired gate vs `InternScience/Agents-A1-FP8` (official)

Same suites, same judge, same harness, thinking-on, **8192-token budget both sides**
(pairing across budget methodologies is how quant evals lie — we re-baselined the FP8
ourselves rather than quote its unbounded-budget numbers):

    axis                                FP8 official   NVFP4 (this)   delta
    ----------------------------------  ------------   ------------   ------
    reasoning-v2 (24, solver-graded)    0.840          0.868          +0.028
    function-call (54, deterministic)   88.9%          90.7%          +1.8
    claw agentic (paired-85, judged)    0.640          0.648          +0.008
    code spec-delta (8, exec, x3)       0.476 ±0.098   0.453 ±0.072   −0.023

**Long-context coherence:** adversarially probed at 4K/16K/32K/60K — needle recall perfect,
zero degeneration flags, hostile-judge clean. Note for near-max-context use: this model
reasons at length — at 60K depth, leave >5K tokens of headroom or disable thinking
(`chat_template_kwargs: {"enable_thinking": false}`), or generation hits the context ceiling
mid-think.

## Speed — single RTX PRO 6000, client-side seeded benchmark

    regime (ISL/OSL)   C    output tok/s   TTFT p50   goodput (TTFT≤2s, TPOT≤50ms)
    ----------------  ---   ------------   --------   ----------------------------
    chat 1k/1k          1        215           63ms    —
    chat 1k/1k          8       1028          300ms    1.00
    context 8k/1k       1        200          313ms    —
    context 8k/1k       8        751         1630ms    0.63

A 35B-class agentic model at 215 tok/s single-stream on one workstation card.

## Serving (the config that works on sm120 — not optional)

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve protoLabsAI/Agents-A1-NVFP4 \
  --moe-backend marlin --max-num-seqs 256 --language-model-only \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
```

- **`--moe-backend marlin` is required on sm120** (RTX 50xx / PRO 6000): the default
  flashinfer/trtllm fused-MoE FP4 kernel segfaults in its `Sm120_SafeFP4` path
  (upstream report with native trace forthcoming; cross-ref flashinfer#3119).
- `--max-num-seqs 256`: the hybrid DeltaNet state cache scales with max sequences.
- `--language-model-only`: text-only serving; the vision-tower profile pass crashes on
  this stack. (Vision weights ship in the checkpoint, unquantized, for stacks that
  handle them.)
- MTP tensors are included (grafted from the Qwen3.5-35B-A3B base via
  `save_mtp_tensors_to_checkpoint` — Agents-A1 shipped without them), **but vLLM
  spec-decode is currently incompatible with the marlin MoE backend** (the global
  backend must also serve the bf16 draft MoE). For MTP speedups use the
  [GGUF build](https://huggingface.co/protoLabsAI/Agents-A1-MTP-GGUF).

## Recipe (provenance — reproduce it)

- llm-compressor `main` @0.12.1.dev41 (post-#2848 — **release 0.12.0 lacks qwen3.5-MoE
  support entirely; builds before 2026-06-22 save a broken expert layout**),
  transformers==5.12.2 (exactly — 5.10 and ≥5.13 both break this path differently),
  compressed-tensors 0.17.1.
- `NVFP4` W4A4, 128 × ultrachat @2048, `moe_calibrate_all_experts=True`.
- **Kept bf16:** MoE router gates + shared-expert gate (low-bit routers corrupt MoE),
  all `linear_attn.*` (DeltaNet), vision tower, `lm_head`, embeddings, MTP tensors.
- Pipeline: `protoLabsAI/protoLab` → `experiments/quantize/a1_requant.py`.

## Need a different quant?

Open a Community discussion — requests usually ship within 48h.
All benchmark rows behind this card (both sides of the pairing, including where we lose):
[`protoLabsAI/lab-benchmarks`](https://huggingface.co/datasets/protoLabsAI/lab-benchmarks) ·
charts at [protolabs.studio/lab](https://protolabs.studio/lab).
