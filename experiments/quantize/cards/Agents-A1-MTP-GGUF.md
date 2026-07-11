---
license: apache-2.0
base_model: InternScience/Agents-A1
base_model_relation: quantized
tags:
  - gguf
  - llama.cpp
  - nvfp4
  - mtp
  - speculative-decoding
  - moe
  - agentic
pipeline_tag: text-generation
---

# Agents-A1 — GGUF with MTP speculative decoding

llama.cpp builds of [`InternScience/Agents-A1`](https://huggingface.co/InternScience/Agents-A1)
**with an MTP draft head grafted in** — A1 shipped without the `mtp.*` tensors its
Qwen3.5-35B-A3B base carries, so no other A1 GGUF can do `--spec-type draft-mtp`.
This one can: **+46% measured, no separate draft model.**

## Files

    file                        size     notes
    --------------------------  -------  ------------------------------------------
    Agents-A1-MTP-NVFP4.gguf    20.8 GB  NVFP4 experts/attention, Q8_0 trunk, MTP head
    Agents-A1-MTP-Q8_0.gguf     37.8 GB  reference quality, MTP head

## Measured (RTX A6000 48 GB, `-n 150` greedy)

    config                       gen tok/s
    ---------------------------  ---------
    NVFP4, no spec               127.7
    NVFP4, --spec-type draft-mtp 187.0   (+46%, draft acceptance 0.62)

A 35B-class agentic MoE at 187 tok/s on a prosumer card in ~21 GB. The MTP head was
grafted from the base Qwen3.5-35B-A3B — never re-trained on A1 — and acceptance holds
anyway.

**Blackwell (RTX PRO 6000 / RTX 50xx-class), MTP on, mean of 6 diverse prompts:**

    NVFP4 + draft-mtp:  305 tok/s  (287–336)

A 35B-A3B at the same speed our 9B runs — the NVFP4×MTP multiplication (verify-step
batching feeds the FP4 tensor cores) reproduces on MoE. Finding writeup on the
protoLabs Ornith cards.

## Runtime compatibility

llama.cpp (spring-2026+), LM Studio, and recent Ollama (~0.31+): ✅. Older Ollama fails
with "layer N missing attn_qkv" — update and re-pull. (Verified on the 9B sibling; if
this 35B-MoE build misbehaves on your runtime, open a discussion with the error.)

## Usage

```bash
llama-server -hf protoLabsAI/Agents-A1-MTP-GGUF:NVFP4 --spec-type draft-mtp -ngl 99
```

Requires llama.cpp with NVFP4 (type 40) + MTP support (both merged spring 2026).
The quant tag resolves (`:NVFP4` / `:Q8_0`) — filenames follow the standard convention.

## Provenance & honesty notes

- Converted from bf16 + grafted MTP tensors; NVFP4 layers quantized by `llama-quantize`
  (llama.cpp b9829) with the trunk (DeltaNet/router/norms) held at Q8_0.
- The vLLM sibling [`Agents-A1-NVFP4`](https://huggingface.co/protoLabsAI/Agents-A1-NVFP4)
  carries the full paired quality gate vs the official FP8 (matches or beats it on all
  four axes). **This GGUF's own gate (paired vs official FP8, same judge/harness):**
  FC 88.9% (identical to FP8) · coherence clean to 28K · reasoning 0.87 vs 0.84 at a
  16K thinking budget · claw agentic paired-84: 0.60 vs 0.63.
- **Two honest caveats.** (1) This build reasons noticeably longer than the vLLM
  artifact (different quant scales → different reasoning paths) — at an 8K generation
  budget it exhausts thinking before answering on some tasks; **give it ≥16K for
  reasoning-heavy work** or disable thinking for quick lookups. (2) One solver-graded
  sequence task regresses at any budget, and Chinese-language agentic tasks run ~0.03
  softer — if your workload is zh-heavy agentic, prefer the vLLM build.
- Not a vision build — text-only.

## Need a different quant?

Open a Community discussion — requests usually ship within 48h.
Rows: [`protoLabsAI/lab-benchmarks`](https://huggingface.co/datasets/protoLabsAI/lab-benchmarks) ·
[protolabs.studio/lab](https://protolabs.studio/lab).
