# QUEUED — NVFP4×MTP acceptance sweep: does the multiplication decay at low acceptance?

Follow-up to the Ornith-1.0-9B "306 tok/s" post. dipankarsarkar (2026-07-06) nailed the
mechanism — MTP splits decode into a memory-bound single-token half and a compute-bound
*verify* half; NVFP4 wins both, but only while the verify batch stays wide (acceptance
high). His question: **does the +52% hold at low acceptance, or decay to additive when the
draft head misses?**

## What we can already say (endpoint data, from the 9B MTP card)

    Blackwell sm120, acceptance-controlled, 6 prompts:
      file     no-MTP   +MTP            MTP lift
      Q4_K_M   205.1  → 239 (216–252)    +17%
      NVFP4    201.5  → 306 (287–330)    +52%

The `no-MTP` column is the answer's skeleton: **NVFP4 (201.5) ≈ Q4_K_M (205.1)** at batch=1.
Plain decode is memory-bound and there FP4 and K-quant are tied — so NVFP4's *entire* lead
over Q4 lives in the verify batch. At accept→0 / batch→1 it doesn't decay to an additive
floor above K-quant; it **converges** with K-quant (the ~201/205 tie). We've measured the
0.57–0.77 band (code→prose spread + n-max sweep, speedup flat ~1.7×); the deep OOD tail
(0.1–0.3) is unmeasured. This experiment fills the curve between the measured band and the
batch=1 tie.

**Hypothesis:** gap = (NVFP4+MTP)/(Q4+MTP) tok/s starts ~1.28× at high accept (306/239) and
decays monotonically toward ~1.0 (201/205) as acceptance→0. Multiplicative → convergence.

## Ready to run (artifacts ON DISK, no download)

- NVFP4:  `/mnt/data/hf-staging/gguf-rename/Ornith-1.0-9B-MTP-NVFP4.gguf`
- Q4_K_M: `/mnt/data/hf-staging/gguf-rename/Ornith-1.0-9B-MTP-Q4_K_M.gguf`
- Harness: `experiments/mtp/validate.sh` (already reports acceptance + decode tok/s)
- Serve:  `~/dev/llama.cpp/build-cuda/bin/llama-server -m <gguf> --spec-type draft-mtp -ngl 99`
- **Needs a free GPU** (currently Qwythos :8010 + the Qwythos eval hold both cards; run after).

## Method

1. Serve each GGUF (NVFP4, Q4_K_M) on a free card, `-n 200` greedy, flash-attn.
2. **Drive acceptance across the full range** (three levers, combine to reach the tail):
   - HIGH (~0.77): code / structured prompts (predictable → draft hits).
   - MID  (~0.55–0.65): general prose; raise `n-max` 3→4.
   - LOW  (~0.1–0.3): OOD/adversarial — random-token strings, code-switching, high-entropy
     digits, `--temp 1.5`. Force the draft to miss so the verify batch collapses toward 1.
3. Per request read llama-server's **draft acceptance** + decode tok/s; bin by measured accept.
4. Per bin, record: NVFP4+MTP tok/s, Q4+MTP tok/s, gap=NVFP4/Q4, and the no-MTP floor.
5. Plot **gap vs acceptance**. Confirm/refute convergence toward the 201/205 tie.

## Deliverable

An acceptance-swept curve (gap-over-accept) → a row/plot in `protoLabsAI/lab-benchmarks`
(CC-BY-4.0) + a follow-up reply/blog addendum answering the public question with data.
~30–45 min once a card is free.
