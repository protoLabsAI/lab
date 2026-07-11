---
license: apache-2.0
base_model: deepreinforce-ai/Ornith-1.0-9B
base_model_relation: quantized
tags:
  - nvfp4
  - vllm
  - compressed-tensors
  - speculative-decoding
  - mtp
  - qwen3.5
  - blackwell
pipeline_tag: text-generation
---

# Ornith-1.0-9B — NVFP4 (W4A4, calibrated) + MTP

Calibrated **NVFP4** quantization of [`deepreinforce-ai/Ornith-1.0-9B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) for vLLM — with the
[MTP draft head](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP) included as a bf16 sidecar
for lossless speculative decoding out of the box.

**10.4 GB (from 19 GB bf16) · ~1.5× faster than bf16+MTP under identical load · full release-gate
parity vs bf16 · coherence-verified to 60K context.** Quantized and gate-verified by
[protoLabs](https://protolabs.studio) on RTX PRO 6000 Blackwell (sm120), vLLM 0.22.1.

## Serve

```bash
vllm serve protoLabsAI/Ornith-1.0-9B-NVFP4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

sm120 (workstation Blackwell) notes: `VLLM_USE_FLASHINFER_SAMPLER=0` required; if the FlashInfer
NVFP4 JIT fights your CUDA install, `VLLM_NVFP4_GEMM_BACKEND=cutlass` (or `marlin`) — quality
verified identical on the cutlass path. Drop `--speculative-config` to serve without MTP.

## Quality — release gate vs bf16 (same suites, same judge, thinking-on, 8192-token budget)

    axis                              bf16     NVFP4    delta
    --------------------------------  -------  -------  ------
    function-call (54, deterministic)  93%      96%     +3
    reasoning-v2 (24, solver-graded)   0.726    0.684   -0.042
    code-exec-v2 (8, exec-graded, x3)  0.391    0.405   +0.014
    claw agentic (paired-29, judged)   0.819    0.791   -0.028

Claw outliers re-trialed ×3 on **both** sides before the verdict; bf16 numbers are our own
published baselines, same harness.

**Known regression (the honest caveat):** `T12_expense_report` — the quant reproducibly (0/3)
misses a duplicate-transaction-detection requirement that bf16 partially satisfies (0.70 ×3).
One agentic judgment, categorical under quant. Everything else is parity or better.

**Long-context coherence:** adversarially probed at 4K/16K/32K/60K — needle recall perfect at
every depth, zero degeneration flags (char-runs, n-gram loops, compression ratio, template
leakage), hostile-judge clean. We don't publish tok/s at depths where a model babbles.

## Speed — RTX PRO 6000 Blackwell, vLLM 0.22.1, client-side seeded benchmark

Same client, same seeds, both models with MTP (single-stream-only numbers tell you nothing
about load — full methodology in the protoLabs benchmark notes):

    regime (ISL/OSL)   C    bf16+MTP   NVFP4+MTP   speedup
    ----------------  ---   --------   ---------   -------
    chat 1k/1k          1     82.7       132.2      +60%
    chat 1k/1k          8    620.8       907.1      +46%
    context 8k/1k       1     73.1       108.1      +48%
    context 8k/1k       8    385.9       598.0      +55%

Decode-at-depth (C=1): 105 tok/s @4K → 75 @16K → 56 @32K → **37 @64K** — a 2.8× fade where
dense transformers typically fade ~10× by 50K (the hybrid DeltaNet trunk keeps constant-size
state on 24 of 32 layers). At 64K the 4.1 s TTFT is prefill physics — long context here is a
decode story, not a first-token one.

MTP acceptance on the NVFP4 target: **0.76** on real text (vs 0.762 on bf16 — the quant costs
the draft head nothing). Benchmark-table numbers above use random-data prompts where acceptance
drops to ~0.31 for both sides equally; real-text throughput runs higher.

## Recipe (provenance — reproduce in ~30 min)

- llm-compressor 0.10.1, `NVFP4` preset (E2M1 weights, 16-elem blocks, E4M3 scales + FP32
  tensor scale; W4A4, dynamic-local activations), 512 × ultrachat calibration @2048.
- **Kept bf16:** all `linear_attn.*` (DeltaNet — low-precision activations corrupt the hybrid
  SSM), vision tower, `lm_head`, embeddings. 128 attention/MLP linears quantized.
- MTP sidecar: `model-mtp.safetensors` (15 tensors, bf16) — verified against the base model:
  spec decode verifies every drafted token, output distribution unchanged.
- Pipeline + verification scripts: `protoLabsAI/protoLab` → `experiments/quantize/`.

## Need a different quant?

Open a Community discussion — size/format requests usually ship within 48h.
GGUF (llama.cpp) builds of this exact quant: [`Ornith-1.0-9B-MTP-GGUF`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF)
(6.6 GB NVFP4+MTP mixed — smaller than Q8_0, tensor-core accelerated on Blackwell).
All benchmark rows: [`protoLabsAI/lab-benchmarks`](https://huggingface.co/datasets/protoLabsAI/lab-benchmarks) · charts at [protolabs.studio/lab](https://protolabs.studio/lab).
