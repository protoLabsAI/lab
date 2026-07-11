# ThinkingCap-Qwen3.6-27B — quant-release (marketing play)

Ride the freshness (published 2026-07-07, hours old). Run BottleCapAI's brevity-finetune of
Qwen3.6-27B through the proven `/quant-release` pipeline (Ornith-9B ladder): NVFP4 + MTP + GGUF,
publish to `protoLabsAI`. **Nothing more** — not a technique-replication project.

## Rubric (§0)
- arch ✓ `qwen3_5`/`qwen3_6` tags → `Qwen3_5ForConditionalGeneration` (VL-hybrid, recipe proven)
- ≤35B ✓ (27B dense-hybrid + vision tower)
- NVFP4 gap ✓ for *ThinkingCap* (hours old). `nvidia/Qwen3.6-27B-NVFP4` exists for the BASE →
  differentiator = **brevity-finetune × NVFP4 × MTP** (unoccupied); base-NVFP4 = a control baseline.
- license ✗ **UNSPECIFIED** — see note. NOT gating local work (Josh's call 2026-07-07).

## Structure (from index.json, pre-full-download)
- shard1: 784 tensors · shard2: 400 · **model-base-aux.safetensors: 15 = the MTP head**
  (`mtp.fc.weight`, `mtp.layers.0.*`) — **bundled, graft is FREE** (no base download / extract_mtp).
- **333 visual tensors** → VL → `fix_vl_keys` mandatory, keep `re:.*visual.*` bf16.
- Cached siblings useful as controls: `Qwen3.6-27B-FP8`, `nvidia/Qwen3.6-27B-NVFP4`, `z-lab/…-DFlash`.

## Watch-items
1. **License UNSPECIFIED** (publish blocker, NOT local blocker). Base = Qwen/Qwen3.6-27B. Unlicensed
   finetune legally = all-rights-reserved. Josh: don't gate — they build on open source, blocking a
   derivative would be hypocritical; proceed local, resolve before/at push (which needs approval anyway).
2. **MTP accept% risk** — bundled head is the *base's* (verbose-trained) drafter on a *terse* target →
   accept% may drop. Graft is free, but MEASURE accept%; distill head only if it craters.
3. **THE FINDING: does brevity survive quantization?** −46% thinking tokens is the whole point; quant
   rot shows as verbosity/repetition. Gate MUST measure token-count delta pre/post-NVFP4, not just
   accuracy. "Brevity survives NVFP4" is the signature result that makes this a real release.

## Pipeline
    [running] download ~55GB -> /mnt/models cache
    1. NVFP4 forge  quant-env · quantize.py --method nvfp4 · fix_vl_keys · GPU1 (pause replica-b, restore after)
    2. MTP sidecar  copy model-base-aux (mtp.*) into artifact · ensure re:.*mtp.* ignored · measure accept%
    3. serve+verify serve-nvfp4.sh :8011 · emulation-backend correctness oracle · real-output curl
    4. GATE         reasoning-v2 · code-exec-v2 ×3 · FC · claw paired · coherence@depth (hard) · speed-v2
                    + TOKEN-COUNT DELTA (watch-item 3) · Qwen3.6-27B base as control · judge=protolabs/reasoning
    5. GGUF ladder  convert_hf_to_gguf · llama-quantize mixed (NVFP4 GEMM + Q8_0) · standalone mtp-head · smoke every file
    6. steelman -> stage -> JOSH per-artifact approval -> publish (vLLM repo + GGUF repo + lab-benchmarks rows)

## Serving findings (card material)
- **Forge clean:** 256 linears NVFP4-packed, vision/GDN/mtp/embed/lm_head bf16, MTP grafted from
  its own bundled `model-base-aux`, zero key-mangling (no fix_vl_keys needed — loaded via
  `Qwen3_5ForConditionalGeneration`). 27GB (48% off 52GB bf16).
- **CUDA-graph capture needs the Mamba-block fix:** hybrid GDN model → `max_num_seqs` default 1024
  exceeds available Mamba cache blocks (321 at util 0.5) → capture fails. FlashInfer NVFP4 kernel
  **silently HANGS** on this; **marlin surfaces it as a clean ValueError.** Fix = `--linear-backend
  marlin --gpu-memory-utilization 0.85 --max-num-seqs 256` → graphs capture in 6s, **54 tok/s**.
- **WORKING SERVE CONFIG:**
  `serve … --gpu-memory-utilization 0.85 --max-num-seqs 256 --reasoning-parser qwen3
   --linear-backend marlin` + sm120 recipe env. Gate serve adds `--max-model-len 65536
   --tool-call-parser qwen3_xml --enable-auto-tool-choice`.
- **CRITICAL EVAL CONFIG — never greedy.** temp=0 makes this thinking model loop and never close
  `</think>` → qwen3 parser returns empty content (looks like "runaway/corruption" but isn't).
  Proper sampling (temp 0.6 / top_p 0.95) → closes cleanly, natural stop.

## Watch-item 3 — PRELIMINARY: brevity SURVIVES NVFP4 ✓
With correct sampling: easy problems answered directly (no think); hard problem (f(f(x))=0) →
near-empty `<think>` (~2 tok) + a clean structured direct solution, correct-track (trig-sub
2cos(3θ)=-1), natural stop at ~1801 tok. Output coherent & sophisticated. The apparent "runaway"
was a greedy-decoding artifact, NOT quant rot. Formal token-count-vs-base measurement still to run
at the gate.

## Staging checklist (before push — each gated on Josh's per-artifact approval)
- [ ] **Move standalone head → `mtp-head/` subdir** (Ornith footgun: top-level head breaks HF "Use
      this model" / `-hf` resolution — it resolves to the head, which crashes loaded alone). Trunk
      files keep `<Model>-<QUANT>-MTP.gguf` naming (matches Ornith-9B precedent).
- [ ] Smoke-test EVERY published file (loads + coherent output; MTP A/B via `--spec-type draft-mtp`).
- [ ] Two repos: `ThinkingCap-Qwen3.6-27B-NVFP4` (vLLM artifact + sidecar) · `-MTP-GGUF` (ladder + card).
- [ ] Scan artifacts for secrets; confirm all card links resolve.
- [ ] Card gets Josh's read first; push each artifact only on explicit go.

## Status log
- 2026-07-07: rubric done, download done, forge done+verified, serve solved (marlin+mamba fix),
  correctness+brevity confirmed at temp 0.6. NEXT: gate (re-serve gate-ready → eval suite +
  token-delta vs base + coherence@depth + speed), then GGUF ladder. Restore replica-b + Fish after.
