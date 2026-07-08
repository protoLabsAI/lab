# Card amendments — existing Ornith repos (draft for Josh's read, 2026-07-03)

Push together with the new NVFP4 repo so links resolve. Full new-repo card:
`Ornith-1.0-9B-NVFP4.md` in this dir.

## 1. Ornith-1.0-9B-MTP-GGUF  (14K dl — the audience repo)

**New file added to repo:** `Ornith-1.0-9B-NVFP4-MTP.gguf` (6.6 GB)

**New card section (insert after the intro):**

> ### NEW: NVFP4 build — 6.6 GB, smaller than Q8_0, Blackwell tensor-core path
>
> `Ornith-1.0-9B-NVFP4-MTP.gguf` — calibrated NVFP4 (`GGML_TYPE_NVFP4`) on all
> attention/MLP GEMMs, Q8_0 on the DeltaNet trunk/embeddings, MTP `nextn` head included.
> Converted directly from our gate-verified
> [vLLM NVFP4 quant](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-NVFP4) — same calibrated
> scales, full quality/coherence receipts on that card.
>
>     Measured on both hardware classes, `-n 200` greedy, MTP = `--spec-type draft-mtp`.
>     Blackwell MTP numbers are means over 6 diverse prompts (prose/code/creative/factual/
>     runbook/technical); ranges shown. Ampere numbers are single-run indicative.
>
>                        Ampere A6000       Blackwell (sm120)
>     file       size    no-MTP   +MTP      no-MTP   +MTP
>     ---------  ------  ------   ------    ------   ----------------
>     Q4_K_M     5.8 GB  104.6    153.4     205.1    239  (216–252)
>     NVFP4-MTP  6.6 GB   70.7     84.8     201.5    **306  (287–330)**
>
> **Read this honestly:** on Ampere and older, Q4_K_M is smaller AND faster — use it. On
> Blackwell (RTX 50xx / PRO 6000), NVFP4+MTP is the fastest rung in this repo by ~28%
> (worst NVFP4 prompt beats best Q4_K_M prompt). **Why — measured, acceptance-controlled:**
> draft acceptance is near-equal on both files (0.52 vs 0.49, same prompt/box), so the
> differential is per-step verify cost: a 2-token MTP verify step costs ~0% extra on
> NVFP4's tensor-core GEMMs vs ~+28% on the K-quant dequant path. MTP's speedup is
> effectively multiplicative with NVFP4 and only partial with K-quants — a differential
> we haven't seen measured elsewhere (spec-decode × FP8 compounding is documented by
> TensorRT-LLM; the FP4-vs-K-quant verify-cost split appears to be new data).
> Workload note: code prompts run hottest (330), creative prose lowest (287) — MTP
> acceptance tracks predictability.
>     Q8_0 (+nextn)    9.8 GB   parity reference, MTP-capable (71.5 t/s Ampere)
>
> Measured: **97.9 tok/s** generation on RTX A6000 (Ampere — no FP4 tensor cores, still fast),
> MTP draft acceptance 0.62 via `--spec-type draft-mtp`. Requires llama.cpp ≥ b-spring-2026
> (NVFP4 type 40 + MTP merge).

**Add the CTA (bottom of card):**

> **Want a different size/format?** Open a Community discussion — requests usually ship
> within 48h. That's how most of the quants in this repo got here.

**Also:** label all existing single-stream speed claims as single-stream; link
`protoLabsAI/lab-benchmarks` + protolabs.studio/lab.

## 2. Ornith-1.0-9B-MTP  (sidecar repo)

Add after the results section:

> ### Works on quantized bases too
>
> Verified on [`Ornith-1.0-9B-NVFP4`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-NVFP4)
> (W4A4): acceptance **0.76** vs 0.762 on bf16 — quantizing the target costs the draft head
> nothing, and NVFP4+MTP measures ~1.5× bf16+MTP under identical load. Same one-command merge;
> or just use the NVFP4 repo, which ships this sidecar in-box.

+ CTA + benchmarks links.

## 3. Ornith-1.0-35B-FP8  (daily-driver quant)

- Family paragraph: link 9B-NVFP4 (vLLM), 9B GGUF repo, sidecar repo; note 35B-A3B NVFP4 is
  next in the pipeline (it'll be the MoE best-case).
- CTA + benchmarks links; label single-stream numbers.
- No number changes — its recipe/verification content already meets the card contract.

## Checklist before push (steelman)

- [ ] Q8_0 smoke on ava (running)
- [ ] GGUF-side coherence mini-probe vs llama-server
- [ ] all links resolve (new repos must exist first — push order: NVFP4 repo → GGUF file+card → amendments)
- [ ] lab-benchmarks dataset created + first rows (gate + speed-v2 + depth JSONs)
- [ ] Josh sign-off on every card
