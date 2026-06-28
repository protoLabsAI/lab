# EAGLE-3 on Ornith-1.0-9B

Sibling to `experiments/mtp/`. Where MTP is a 1-token in-checkpoint head, **EAGLE-3** is a
separate trained draft model (feature-conditioned on the target's hidden states at multiple
layers) that drafts a *tree* the target verifies in one pass → longer accepted runs → higher
ceiling. Both are lossless (the target verifies every token).

## Graft baseline (2026-06-28) — zero training

Same move as the MTP graft: take an existing **Qwen3.5-9B** EAGLE-3 draft and run it against
the Ornith-9B fine-tune. Draft: `BLR2/Eagle3-Qwen3.5-9B` (`LlamaForCausalLMEagle3`, 1 layer,
763 MB, reads target aux hidden states at layers [1,15,28]). Served Ornith-9B + draft via
`--speculative-config '{"method":"eagle3","model":"BLR2/Eagle3-Qwen3.5-9B","num_speculative_tokens":N}'`
on :8005 (Blackwell). Acceptance/tok-s probe (no-think single-stream):

| Variant | accepted/step | single-stream tok/s | vs plain |
|---|:---:|:---:|:---:|
| plain Ornith-9B (no spec) | — | ~75 | — |
| MTP (distilled, in-checkpoint) | 0.76 | ~121 | +60% |
| **EAGLE-3 graft, n=3** | 1.21 | **138.8** | **+85%** |
| EAGLE-3 graft, n=5 | 1.37 | 136.3 | +82% |

**Finding: EAGLE-3 beats MTP on Ornith-9B even *untuned*** (graft 138.8 vs MTP 121). The
Qwen3.5-9B draft transfers to the fine-tune (Ornith is light enough — same as the MTP-graft
result). `n=5` accepts slightly more per step but the extra verify cost cancels it — the
graft's acceptance (~1.2–1.4/step) is too low to exploit longer drafts, so **n=3 ≈ ceiling**.

## Concurrency: EAGLE-3 vs MTP (the decisive test)

Single-stream wins lie — the daily-driver regime is concurrent. Aggregate output tok/s,
`conc_bench.py` (fixed-length, no-think), same base, MTP(n=1) vs EAGLE-3 graft(n=3):

| C | MTP (n=1) | EAGLE-3 graft (n=3) | winner |
|---|:---:|:---:|---|
| 1 | 118.8 | **135.6** | EAGLE +14% |
| 4 | 347.9 | **435.9** | EAGLE +25% |
| 8 | 866.5 | **956.6** | EAGLE +10% |
| 16 | **1662.7** | 1644.6 | ~tie |
| 32 | **2915.0** | 2541.2 | MTP +15% |

**dFlash pattern, milder.** EAGLE-3's tree wins the **interactive regime (C=1–8, +10–25%)**,
crosses over ~**C=16**, and **MTP wins heavy batch (C=32)** — the 3-token tree inflates the
verify batch and competes with concurrent requests for compute. Acceptance holds for EAGLE
(~0.9 vs MTP ~0.7/step); it's the verify cost that erodes the aggregate win, not acceptance.

## Verdict: nothing to ship — it's a use-it recipe

- The interactive-regime win (C=4–8, +10–25% over our shipped MTP) comes from an **existing
  public draft (`BLR2/Eagle3-Qwen3.5-9B`) + vLLM's built-in `eagle3`** — *download + point
  vLLM at it*. No artifact for us to ship; our contribution is the **finding** (it transfers
  to the fine-tune, beats MTP at C≤8, lossless). Unlike MTP, where DeepReinforce dropped the
  head and we had to graft+distill+publish.
- **MTP stays the in-checkpoint option** for heavy-batch (C≥16) and simplicity.

## Training an Ornith draft — ruled out (no transfer tax)

We checked whether the graft's modest 1.2 accept/step was a *transfer tax* (recoverable by
training on Ornith) or just the draft's ceiling. **Diagnostic: run the BLR2 draft on its
*native* target, base Qwen3.5-9B, and compare.**

| draft + target | accepted/step | tok/s |
|---|:---:|:---:|
| BLR2 on **base Qwen3.5-9B** (native) | 1.26 | 141.4 |
| BLR2 on **Ornith-9B** (graft) | 1.21 | 138.8 |

**Identical (~4%, noise) → there is no transfer tax.** The draft works the same on Ornith as
on the model it was trained for (Ornith is a light fine-tune — same as the MTP graft 0.74→0.76).
So training an Ornith-specific draft would recover ~nothing.

The 1.2/step is the draft's actual ceiling **on our workload** — likely because our probe is
coding/agentic, which is far less predictable than the ShareGPT chat that EAGLE-3's headline
2–4 accept/step numbers come from (out-of-distribution → drafts accept less). The only lever
left is training a *better/bigger* draft on our own distribution — about draft quality, not
Ornith-specificity, and a speculative +0.2–0.5/step for a full SpecForge run. **Not worth it.**

**Conclusion: don't train.** The free graft *is* the EAGLE-3 win. Ship MTP (in-checkpoint,
heavy-batch); use the EAGLE-3 graft as a free interactive-lane recipe; train nothing.
(If a higher-quality draft is ever wanted: SpecForge/`speculators`, train on
`/mnt/data/datasets/ornith-9b-mtp/corpus.jsonl` or a bigger our-distribution set — but expect
modest, distribution-bound returns, not the headline 2–4×.)

## Spec-decode ladder (Ornith-9B, dense)

plain ~75 → MTP (in-checkpoint, shipped) 121 → **EAGLE-3 graft 138.8** (the practical top —
training ruled out, no transfer tax). All lossless. EAGLE-3 is the higher single-stream/
interactive ceiling; MTP is the cheaper in-checkpoint option that wins heavy batch. (On the
MoE 35B daily driver, neither is a win — spec-decode is a dense-model play; MTP was −11% there.)

## Spec-decode family — when to use which

All of these are **lossless** (verified spec-decode: the target verifies every drafted token
via rejection sampling, so the draft only moves speed, never quality — "lossless" =
distribution-lossless, not bitwise). vLLM 0.22.1 supports: `ngram`, `medusa`, `mtp`,
`eagle`/`eagle3`, `dflash`. They differ in cost-to-build and where on the workload×concurrency
surface they pay off:

| Method | What it drafts | Build cost | Wins on | Concurrency |
|---|---|---|---|---|
| **ngram** | copies recent n-grams from the context (string lookup, no model) | **free** | copy-heavy: RAG, code-edit, summarization, structured/JSON, long-ctx quoting, tool-echo | same batch-inflation penalty; long copies inflate hard |
| **Medusa** | K *parallel, independent* heads off the final hidden state | trained heads | general (legacy) — superseded by EAGLE | heads non-autoregressive → low accept |
| **MTP** | 1 autoregressive head (hidden + next-tok emb) | 1 layer (graft+distill) | general, cheap, in-checkpoint | scales near-linearly (low inflation) |
| **EAGLE-3** | feature-conditioned *tree* drafter | separate trained draft | general, highest accept, interactive | wins C≤8, inverts ~C≥16 (tree inflation) |
| **dFlash** | 2B block-diffusion draft | separate model | single-stream latency only | plateaus, loses by C≈4 (`experiments/dflash/`) |

**Mental model — it's a curve over workload × concurrency, not a number:**
- **ngram** = free, *spiky*: huge on copy-heavy, ~0 (and slightly negative) on novel gen.
- **MTP** = cheap, general, concurrency-friendly (our shipped in-checkpoint option, heavy-batch).
- **Medusa** = trained, general, low ceiling (legacy; MTP/EAGLE cover its niche better).
- **EAGLE-3** = most machinery, highest ceiling, best *interactive* (C≤8).

**ngram is NOT "free with no downside."** It's free to *enable* (no model/training) but not to
*run*: every drafted token still costs verify compute, so (1) spurious short-n-gram matches draft
wrong continuations → wasted verify; (2) same concurrency batch-inflation as the rest (a long
copied span is a big inflation) → can go **net-negative** at high C on non-copy traffic; (3)
wildly workload-dependent, so bad to blanket-enable on mixed traffic. **Route it to copy-heavy,
low-concurrency lanes** (RAG, code-edit, structured) — don't turn it on globally.

**Production setups route by lane** — e.g. ngram on RAG/structured, EAGLE/MTP on chat — rather
than picking one globally. The only question per method is whether your lane sits where its
curve pays.

Run logs: `results/{local→ornith-9b-eagle3}_…`. Related: `experiments/mtp/`, `experiments/dflash/`, `project_ornith_9b_mtp`, `project_eagle3_ornith9b`.
