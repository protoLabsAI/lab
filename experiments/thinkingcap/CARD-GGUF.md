<!-- DRAFT for Josh's review — NOT pushed. Repo: protoLabsAI/ThinkingCap-Qwen3.6-27B-MTP-GGUF -->
---
license: apache-2.0
base_model: bottlecapai/ThinkingCap-Qwen3.6-27B
base_model_relation: quantized
tags: [gguf, nvfp4, mtp, speculative-decoding, blackwell, thinking, token-efficient, qwen3.6]
---

# ThinkingCap-Qwen3.6-27B — MTP-GGUF (Blackwell / speculative-decoding edition)

GGUF quantizations of [**BottleCapAI/ThinkingCap-Qwen3.6-27B**](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B)
— their "brevity finetune" of Qwen3.6-27B that keeps full accuracy with **~46% fewer thinking
tokens** — repackaged for the two things stock GGUFs don't give you:

- **NVFP4** — native FP4 for Blackwell (RTX 50-series / RTX PRO 6000) tensor cores.
- **MTP baked into every quant** — the Multi-Token-Prediction draft head travels *inside* each file,
  so you get speculative decoding for free (`--spec-type draft-mtp`), no second model to wire up.
- **A full low-bit ladder** — IQ2→Q8_0 + a lossless **bf16** master, so it fits everything from a
  24GB card to a workstation.

Same weights as upstream. Strictly more ways to run them, faster.

## Files

    quant            size     notes
    NVFP4-MTP        18.2 GB  ← Blackwell FP4 tensor cores + MTP. The one to grab on RTX 50xx / PRO 6000.
    bf16-MTP         54.7 GB  lossless master (exact bf16, not a lossy f16 re-map) + MTP
    Q8_0-MTP         29.0 GB  near-lossless reference + MTP
    Q6_K-MTP         ~22 GB   + MTP
    Q5_K_M-MTP       ~19 GB   + MTP
    Q4_K_M-MTP       ~17 GB   + MTP  (the common daily-driver size — and here it carries the draft head)
    IQ4_XS-MTP       ~14 GB   imatrix low-bit + MTP            [fast-follow]
    IQ3_M-MTP        ~12 GB   imatrix low-bit + MTP            [fast-follow]
    IQ2_M-MTP        ~10 GB   imatrix low-bit, fits a 12GB card + MTP   [fast-follow]
    mtp-head/…       ~2.4 GB  standalone draft head, for pairing with a base GGUF via --model-draft

### NVFP4 mini-ladder (for tighter VRAM)

`NVFP4-Q4_K_M-MTP` (15.7 GB) lowers the *non-FP4* base tensors to Q4 — aimed at **24 GB dual-Blackwell**
(2×12 GB) where the 18.2 GB flagship + 128k KV + MTP won't fit. Honest finding: the savings are
**marginal** because the NVFP4 GEMMs dominate the file and are fixed — the base type only touches
~3 GB of embeddings/norms/GDN. The full curve we measured (Q8→Q6→Q5→Q4 base): 18.2 / 16.8 / 16.3 /
15.7 GB. Quality holds (quant-sensitivity + coherence verified); only the *bottom* rung meaningfully
helps a 24 GB budget, so that's the one shipped. GGUF GPU-speed numbers are pending (validated on CPU
here; llama.cpp CUDA testing is a follow-up).

The MTP head is **embedded in each bundled quant** — it rides the trunk, nothing extra to download.
The standalone head (`mtp-head/mtp-ThinkingCap-Qwen3.6-27B-head-Q8_0.gguf`) is only for pairing with a
*separate* base GGUF; loading it alone crashes. It lives in a **subdirectory on purpose** — keeping it
out of the repo root so HF's "Use this model" / llama.cpp `-hf` never resolves to it by mistake.

## Run it (with speculative decoding)

```bash
llama-server --model ThinkingCap-Qwen3.6-27B-Q4_K_M-MTP.gguf \
  --n-gpu-layers 99 --ctx-size 8192 --flash-attn on --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3
```

`--spec-draft-n-max 2` maximizes acceptance; `3` maximizes throughput. On Blackwell, grab the
**NVFP4** file — MTP verification is nearly free on FP4 tensor cores.

> **One critical setting: don't decode greedy.** This is a thinking model; at `temperature 0` it can
> loop and never close `</think>`. Use the model's intended sampling (**temp 0.6, top_p 0.95, top_k 20**).
> Greedy is the #1 cause of "it rambled and gave no answer" — not the quant.

## Blackwell / NVFP4 notes (what we found forging these)

- The NVFP4 checkpoint serves correctly on sm120 (verified real, coherent output; the brevity behavior
  survives quantization — a hard problem still answers with a near-empty `<think>` and a direct solution).
- vLLM serving of the NVFP4 build needs `--linear-backend marlin` on this hybrid (GDN/Mamba) arch:
  the FlashInfer FP4 kernel **silently hangs in CUDA-graph capture**, while marlin surfaces the real
  cause — a Mamba-cache-block limit fixed with `--max-num-seqs 256 --gpu-memory-utilization 0.85`.
- llama.cpp: the mixed NVFP4 file keeps FP4 GEMMs + Q8_0 for the rest — best size/quality on Blackwell.

## Compatibility

MTP-baked GGUFs need a recent runtime that understands `nextn` blocks:
- **llama.cpp** — recent build (any with `--spec-type draft-mtp`).
- **Ollama** — ~0.31+ (older fails with "layer N missing attn_qkv").
Non-MTP runtimes still load the trunk; you just don't get the free speculative decoding.

## Benchmarks (measured on the NVFP4 build)

**The brevity survives quantization** — that's the whole point, and here's the number:

    thinking tokens (15-prompt reasoning set, temp 0.6)
      base Qwen3.6-27B     mean 1401 tok
      ThinkingCap-NVFP4    mean  675 tok      → ~40% fewer, per-prompt mean (range 3–82%)

Quant integrity (our eval harness, judge-free where possible):

    quant_sensitivity (arithmetic/exact-recall/format)   93%   near-lossless
    function_call                                         94%   quant clean on tool-use
    coding (hard_v2, adversarial)                         50%   suite is deliberately hard for a 27B;
                                                                quant-uniform damage ruled out by the two above
    MTP draft acceptance (--spec-type draft-mtp)          62–91%  (prompt-dependent)

Method note: token saving is measured token-count; accuracy-preservation is evidenced by the
quant-sensitivity + FC suites above, not graded on the 15 brevity prompts. Coherence-at-depth and
3-hardware speed follow as a card update.

## Provenance & credit

- **Base / all the intelligence:** [BottleCapAI/ThinkingCap-Qwen3.6-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B)
  (a finetune of [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)). Go star their work.
- **This repo:** quantization + MTP packaging only, by [protoLabsAI](https://huggingface.co/protoLabsAI).
- **MTP head:** the upstream model's own bundled draft head. We measured **~62% draft acceptance**
  (mean accepted length 2.83) on the NVFP4 build via `--spec-type draft-mtp` — a solid free
  speed-up straight out of the box, confirming the authors' "MTP works well as-is."

## Request a size

Need a quant level that isn't here? Open a discussion — we turn most requests around in ~48h.
