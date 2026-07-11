---
license: apache-2.0
base_model: deepreinforce-ai/Ornith-1.0-35B
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

# Ornith-1.0-35B — NVFP4 (W4A4, calibrated MoE)

Calibrated **NVFP4** quantization of [`deepreinforce-ai/Ornith-1.0-35B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B)
(Qwen3.5-35B-A3B hybrid MoE) for vLLM. **24 GB, and it decodes at the same speed as FP8
while every quality axis lands at parity or better.** This is the model we run as our own
daily driver — the gate below is the quant measured against the exact checkpoint serving
our production traffic.

Quantized and gate-verified by [protoLabs](https://protolabs.studio) on 2× RTX PRO 6000
Blackwell (sm120), vLLM 0.22.1.

## Quality — paired gate vs `protoLabsAI/Ornith-1.0-35B-FP8` (our parity-verified daily driver)

Same suites, same judge, same harness, thinking-on, 8192-token budget both sides.
Outliers re-trialed ×3 on **both** sides before verdicts — one apparent regression
(0.88 → 0.00 on a single task) turned out to be baseline drift: the FP8 scores 0.00
on it today too. Quant evals need paired, same-day baselines or they lie.

    axis                                FP8 (prod)     NVFP4 (this)   delta
    ----------------------------------  ------------   ------------   ------
    claw agentic (paired-35, judged)    0.672          0.677          +0.005
    function-call (54, deterministic)   93.0%          94.4%          +1.4
    reasoning-v2 (24, solver, x3)       0.882 ±0.021   0.889 ±0.042   +0.007
    code spec-delta (8, exec, x3)       0.616          0.580 ±0.062   −0.036

**Long-context coherence:** adversarially probed at 4K/16K/32K/60K — needle recall perfect,
zero degeneration flags (char-runs, n-gram loops, compression collapse), hostile-judge clean
at every rung. Long-context agentic runs verified out to >131K served context.

This is **measurement-resolution parity, not a lossless format**: deltas straddle zero and
sit inside our suites' noise (code-exec single-trial std alone is ±0.06). On dense models
NVFP4 costs real quality (see our 9B card: −0.028 claw, one reproducible failure mode);
on A3B MoE we cannot distinguish it from the FP8 base.

## Speed (speed-test-v2, client-side, random dataset, fixed seed)

    regime          C   ttft p50   tpot p50   agg tok/s   goodput
    ------------   --   --------   --------   ---------   -------
    chat 1k/1k      1       62ms      4.8ms       207.8      0.20
    chat 1k/1k      8      297ms      7.8ms       950.6      0.81
    context 8k/1k   1      305ms      4.9ms       194.2      0.19
    context 8k/1k   8     1520ms      9.3ms       738.4      0.64

FP8 reference on the same card: ~207 tok/s at C1 — **NVFP4 costs nothing in speed and
saves 11 GB of VRAM** (more KV headroom, or denser replica packing).

## Serving (vLLM ≥ 0.22.1, sm120)

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve protoLabsAI/Ornith-1.0-35B-NVFP4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --language-model-only \
  --moe-backend marlin \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

- **`--moe-backend marlin` is required** on sm120 — the trtllm auto-selected NVFP4 MoE
  path (`Sm120_SafeFP4`) segfaults even on clean checkpoints.
- The checkpoint **preserves the MTP speculative-decoding tensors** (bf16, excluded from
  quantization). vLLM's marlin MoE backend can't currently run an unquantized MTP draft
  over quantized experts, so serve MTP-less; the heads are there for the GGUF pipeline
  (llama.cpp `--spec-type draft-mtp`) and for when the backend gap closes.
- DeltaNet linear-attention layers, embeddings, router gates, and `lm_head` kept bf16
  (quantizing any of them breaks the hybrid arch — recipe in repo).

## Recipe

llm-compressor (main, post-#2848), NVFP4 (16-elem blocks, E4M3 scales), 128 × 2048-token
ultrachat calibration with `moe_calibrate_all_experts=True`, data-collator pinned for the
4-D DeltaNet inputs, MTP tensors re-attached from the base checkpoint post-save.
transformers==5.12.2. Full script in this repo.

---

Need a different size/format? Open a Community discussion — we usually ship within 48h.
Numbers on this card trace to rows in
[`protoLabsAI/lab-benchmarks`](https://huggingface.co/datasets/protoLabsAI/lab-benchmarks);
the board lives at [protolabs.studio/lab](https://protolabs.studio/lab).
