# Ornith-1.0-9B MTP — proof run (2026-06-27)

First instance of the `experiments/mtp/` toolkit. Quick proof run (Josh-approved short
window): does grafting + distilling a Qwen3.5 MTP head onto Ornith-1.0-9B recover a usable
speculative-decode acceptance rate?

## Setup

- **Base**: `deepreinforce-ai/Ornith-1.0-9B` (dense Qwen3.5-9B fine-tune, ships 0 MTP tensors).
- **Donor head**: `Qwen/Qwen3.5-9B` (15 `mtp.*` tensors).
- **Corpus**: 3942 of Ornith-9B's **own** no-think generations (~1.8M tokens) on
  WildBench + ToolACE seed prompts. No external/proprietary data (self-distillation).
- **Distill**: froze base, trained only the 487 MB head, 1 epoch, lr 2e-4, 492 steps.
  Train loss 2.0 → ~1.0–1.1.
- **Measurement**: served off-gateway on :8005 with
  `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, probed acceptance
  over ~9k draft tokens, no-think (matched to training). Plain-9B baseline ~75 tok/s
  single-stream (challenger bench).

## Results

Acceptance measured two ways (sampled, T=0.7): **coding** probe prompts and **corpus**-style
(WildBench/ToolACE) prompts. All lossless by construction.

| Variant | Accept (coding) | Accept (corpus) | tok/s | Notes |
|---|:---:|:---:|:---:|---|
| Plain Ornith-9B (no MTP) | — | — | ~75 | challenger baseline |
| **Graft (Qwen head, untrained)** | 0.763 | 0.742 | ~117 | +49–57%, zero training/data |
| Distilled v1 (hard CE) | 0.721 | 0.691 | ~98 | **regressed** — wrong objective |
| **Distilled v2 (KL distrib-match)** | **0.765** | **0.762** | **~121** | **beats graft; best head** |

**Bottom line:** the graft is free and excellent; naive (hard-CE) distillation *hurts*; the
correct objective — KL to the target's own next-token distribution — recovers and edges past
the graft. v2 is the shippable head; the *objective lesson* is the finding.

## Findings (honest)

1. **The graft alone is already excellent.** Qwen3.5-9B's MTP head transfers to Ornith-1.0-9B
   nearly intact — **0.74–0.76 acceptance**, ~**117 tok/s (+49–57%)**, **lossless by
   construction** (the base verifies every drafted token; output distribution is provably
   unchanged, so quality == plain Ornith regardless of head quality). This **overturns the
   prior assumption** that a fine-tune's moved hidden states would collapse a donor head's
   acceptance — Ornith-1.0 is a light enough fine-tune that base-Qwen's residual-stream
   distribution is preserved. The graft needs no training and no data; it's the floor.

2. **Distillation v1 did NOT beat the graft — it consistently regressed it** (~0.05 below
   graft on every distribution). And we ran the diagnostics to root-cause it:

   | Measurement | Graft | Distilled v1 |
   |---|:---:|:---:|
   | vLLM acceptance, coding prompts (sampled) | 0.763 | 0.721 |
   | vLLM acceptance, corpus prompts (sampled) | 0.742 | 0.691 |
   | `eval_head` greedy proxy, fp32 | 0.715 | **0.868** |
   | `eval_head` greedy proxy, bf16 | 0.714 | **0.868** |
   | `eval_head` greedy proxy, **pre**-norm hidden | 0.697 | — |

   **Ruled out** (each tested, not assumed):
   - *Rope position offset* — a uniform position shift cancels in relative rope.
   - *Hidden state pre/post-norm* — graft proxy is **higher** on post-norm (0.715) than
     pre-norm (0.697), so post-norm (what we trained on) is correct.
   - *Precision* — bf16 proxy == fp32 proxy (0.868), so fp32-train/bf16-serve isn't it.
   - *Prompt distribution* — distilled regresses even on in-distribution corpus prompts.

   **Root cause = wrong training objective for the metric.** Distillation used hard-label CE
   on Ornith's sampled tokens. That sharpens the head's **argmax** (greedy proxy 0.71→0.87)
   but **degrades distribution calibration**. vLLM's MTP acceptance is *rejection sampling*
   of a drafted token against the target's distribution — it rewards a draft distribution
   that **matches the target**, not a peaky argmax. The graft head (jointly trained with the
   base) is well-calibrated; hard-CE over-sharpened ours. Hence greedy↑ but sampled-accept↓.

3. **v2 (KL distribution-matching) fixed it and beats the graft.** Changing only the
   objective — KL to the target's own next-token distribution at t+1 (soft targets =
   `lm_head(base_hidden[t+1])`, exactly what the verifier compares against) — flipped the
   result: **0.762 corpus / 0.765 coding, ~121 tok/s**, edging past the graft on both
   distributions and crushing v1 (0.691). Same data, same lr, same steps; only the loss
   changed. This is the cleanest possible confirmation that the objective, not the data or
   the forward, was the issue. KD temperature 1.0 (the teacher's natural distribution) is
   correct here — lowering it toward the sampling temp sharpens targets back toward the
   hard-CE failure.

## Conclusion

The graft is a free, lossless +49–57% (ship it as the floor). The *right* distillation
objective (KL, not CE) squeezes a bit more (~0.74→0.76, near the native ~0.79 ceiling) and,
just as importantly, the *wrong* objective actively hurts — the reusable lesson for any MTP
head re-alignment. **v2 (KL) is the shippable head.**

Remaining headroom (optional, small): thinking-on + larger corpus, multi-epoch with
early-stop, push toward native ~0.79.

## Artifacts

- **Distilled v2 (KL) — best head**: `/mnt/data/checkpoints/ornith-9b-mtp-kl` (0.762/0.765, ~121 tok/s).
- Graft (free floor): `/mnt/data/checkpoints/ornith-9b-mtp-graft` (0.742/0.763, ~117 tok/s).
- Distilled v1 (hard CE, superseded): `/mnt/data/checkpoints/ornith-9b-mtp` (0.691).
- Corpus: `/mnt/data/datasets/ornith-9b-mtp/corpus.jsonl` (3942 self-generated samples).
- Toolkit: `experiments/mtp/` (graft.py, gen_corpus.py, distill.py, eval_head.py, validate.sh).
