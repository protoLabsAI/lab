# REAP-70B — a ~70B MoE for the missing mid-size slot

**Thesis:** the ~70B-total / ~10–12B-active MoE slot is a real market gap (labs ship small-efficient ≤35B MoE or ≥120B flagships; nothing in between). It's the ideal shape for this rig: NVFP4 → ~35 GB → one card, decode speed of a 12B-active model, quality between our 35B and a 122B. We can *make* one by expert-pruning a larger MoE with **REAP** (Router-weighted Expert Activation Pruning, Cerebras, arXiv:2510.13999) — one-shot, **no retraining**.

## Base decision: GLM-4.5-Air (106B/12B, MIT)

Chosen over Qwen3.5-122B (fallback) and Nemotron-3-Super-120B (hybrid-Mamba, no REAP support):

| source | total/active | prune → 70B | arch | REAP support | license |
|---|---|---|---|---|---|
| **GLM-4.5-Air** ✅ | 106B / 12B | **~37% experts** (gentlest) | clean MoE (128e, top-8 +1 shared, 45 MoE layers) | **proven** — `cerebras/GLM-4.5-Air-REAP-82B-A12B` | MIT |
| Qwen3.5-122B-A10B | 122B / 10B | ~47% experts (near cliff) | clean MoE (256e, top-8) | arch supported | Apache-2.0 |
| Nemotron-3-Super-120B | 120B / 12B | ~40% | hybrid Mamba2-MoE | **none** (write adapter) + sm120 risk | Nvidia OML |

Why GLM-4.5-Air: newest clean-MoE in band, MIT, **smallest cut** (best retention), and the only one where **REAP is already validated on the identical arch** with a reference checkpoint to regression-test against.

## Feasibility on this node — GO, no cloud

- **Prune the FP8 checkpoint via `device_map="auto"`** → ~106 GB loads into 192 GB VRAM; calibration runs there. **The 61 GB RAM wall is bypassed** (it's the trap that blocks the bf16 path and the repo's `layerwise_prune.py` CPU-RAM path — avoid both).
- One-shot, **no heal pass** (Cerebras ships un-healed). Prior art: REAP'd a 397B on 2× RTX 6000 Pro (our cards).
- Effort ~1 day + one full-GPU calibration window.

## Known risks (what our gate must catch)

REAP holds ~95%+ on generative/coding/tool at ≤50% prune, but: **knowledge/multiple-choice dips**, and **termination/looping roughly doubles** (3.6%→7.2% on a community GLM REAP) — a behavioral cost accuracy hides. Our claw agentic suite + an explicit **loop/termination-rate metric** are the gate.

## Pipeline

1. Download `zai-org/GLM-4.5-Air` **FP8** (~106 GB → /mnt/scratch).
2. `CerebrasResearch/reap` (Apache-2.0, stock HF Transformers); GLM-4.5 arch supported. Crib calibration mix from the Cerebras REAP-82B card.
3. Calibration sweep, ratio ~0.37, ~12k code/reasoning/tool samples @ 16k tok, `--enforce-eager`, **both cards** → out: ~70B / ~12B-active.
4. Gate vs 122B + our 35B: claw / FC / reasoning / coherence + **loop-rate**. Regression-check against `cerebras/GLM-4.5-Air-REAP-82B-A12B`.
5. NVFP4 the 70B → ~35 GB, one card. Publishable ("own-niche": no ~70B REAP of GLM-4.5-Air exists; +NVFP4+MTP is ours).

## The 397B question — settled

Can't REAP a 397B here, two independent reasons: (a) 397 GB FP8 ≫ 192 GB VRAM (no calibration fit); (b) 397B→70B ≈ 82% cut, far past REAP's ~50% quality cliff → gutted, even on cloud. Source must be ~110–140B so the cut stays ≤50%.

## Fork (open)

- **A — forge our own 70B** (deeper cut of GLM-4.5-Air): our artifact, most differentiated, needs the calibration window.
- **B — adopt Cerebras's REAP-82B** + our NVFP4+MTP: fastest, proven, but 82B not 70B and it's their prune.
- **C — both**: grab the 82B now as the sanity reference, forge our 70B next.
