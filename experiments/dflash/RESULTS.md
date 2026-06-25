# dflash — RESULTS

**Date:** 2026-06-25 · **Stack:** vLLM 0.22.1 / CUDA 13, sm120 (RTX PRO 6000 Blackwell), single GPU (GPU1)

## TL;DR

dFlash **serves out-of-the-box on our stock vLLM 0.22.1 — no bump, no PR build, no `speculators` pip
install** — for the **Qwen3-family** smart lane. Target `Qwen/Qwen3.6-27B-FP8` (hybrid-Mamba) +
`z-lab/Qwen3.6-27B-DFlash` (2B bf16 draft) loaded cleanly, produced coherent output, and beat our
shipped **27B+MTP** baseline by **+21% to +38%** single-stream depending on content. The card's
"needs PR #40898 for interleaved SWA" note is **stale for our build** — it just works.

## Numbers (same model, same hardware, single-stream)

`models/speed-test.sh` reports **decode tok/s = 1/TPOT** (pure decode, excludes TTFT).
The think-off pass is **end-to-end wall tok/s** (sequential client). Both shown.

| Config | think-ON decode¹ | think-ON wall | think-OFF wall | accept_len/draft | accept rate |
|---|:--:|:--:|:--:|:--:|:--:|
| 27B + **MTP** (prod, :8000) | 74.6 | 74.3 | 73.4 | 0.76–0.90 | 76–90% |
| 27B + **dFlash** (:8003) | **103.2** | **102.6** | **88.5** | 1.47–1.88 | 10–13% |
| **dFlash advantage** | **+38%** | **+38%** | **+21%** | — | — |

¹ decode tok/s from `/metrics` (8 runs, 800-tok essay). TTFT ~43 ms both. dFlash TPOT 9.7 ms vs MTP 13.4 ms.

## Observations

- **Acceptance is modest** — dFlash proposes `num_speculative_tokens=15` (block_size 16) but only
  ~10–13% of drafted tokens are accepted (mean **1.47–1.88 accepted per draft**). Still wins on
  throughput because block-diffusion drafting is a **single cheap forward pass** — lots of wasted
  draft tokens cost little. The paper's 3–4+ accept-lengths weren't reproduced here (draft is
  marked "still under training" on HF; `num_speculative_tokens` untuned).
- **Counterintuitive: acceptance is HIGHER with thinking on** (1.88 vs 1.47 think-off). Reasoning
  boilerplate is more predictable/draftable than varied creative prose → the dFlash win is **larger
  on reasoning-heavy traffic** (+38%) than on creative prose (+21%). Good fit for the *thinking*
  smart lane specifically.
- **MTP** at `num_speculative_tokens=1` accepts ~0.76–0.90 extra tokens/step at high rate (76–90%) —
  efficient but capped at +1 token. dFlash trades acceptance rate for cheaper, longer proposals.
- vLLM wired the draft EAGLE-style: shares target embed/lm_head, extracts hidden states from target
  aux layers `(2, 17, 32, 47, 62)` (= draft `target_layer_ids` + 1). Model load 31.9 GiB, draft +3.2 GiB.
- Non-fatal `SM 12.x requires CUDA >= 12.9` probe line appeared (our `run-dflash.sh` doesn't pin
  CUDA_HOME to cu13 like the prod units); load proceeded past it, FLASH_ATTN backend selected, graphs
  captured fine. Pin cu13 env if any JIT path later turns fatal.

## Verdict

**Real, shippable single-stream win on the smart lane with zero stack risk.** +38% on reasoning
traffic over MTP is meaningful for the `protolabs/smart` thinking lane. Before shipping:

1. **Tune `num_speculative_tokens`** (15 → try 4/8/12) — 10% acceptance suggests over-drafting; a
   shorter block may lift net efficiency or cut wasted verify.
2. **Concurrency test** — dFlash's headline paper wins are at high interactivity/batch; we only
   measured single-stream. Verify it doesn't regress under concurrent load (spec decode often helps
   less when the GPU is already batch-saturated).
3. **Quality regression check** — run the eval suite through the dFlash lane; spec decode is supposed
   to be lossless (verified by target) but confirm vs the MTP lane on claw/custom.
4. **Tool-calling / structured** — only tested free-form generation.

## NOT yet tested

- **Gemma fast lane (26B-A4B MoE)** — `z-lab/gemma-4-26B-A4B-it-DFlash` exists but **needs unmerged
  vLLM PR #41703** (gemma4 dflash not in 0.22.1). MoE + spec is our historical risk case
  (35B MoE + MTP = −11%). Build PR into isolated `~/dev/vllm-dflash-env` if the smart-lane win
  justifies chasing the fast lane.
- Other Qwen targets with official drafts on disk: 35B-A3B (MoE), 122B-A10B, 9B, 4B.

## Tuning sweep — `num_speculative_tokens` (2026-06-25, GPU1, think-on, decode tok/s)

| N | decode tok/s | accept_len/draft | rate |
|:--:|:--:|:--:|:--:|
| 3 | 96.8 | 1.61 | 53.5% |
| 6 | 114.3 | 2.04 | 34.1% |
| **10** | **116.9** | **2.17** | 21.7% |
| 15 | 103.2 | 1.88 | 12.5% |
| 16 | 106.2 | 1.99 | 12.4% |

Peak at **N=10**. Classic spec-decode arc: too few proposals underuse the cheap single-pass block
draft; too many waste verify compute. (`sweep.sh "3 6 10 16"`.)

## PROMOTED to prod smart lane (2026-06-25)

Swapped `vllm.service` MTP → `{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}`
(only that line changed; `-O3`, 225K, 512 seqs, mamba cache, `qwen3_xml` tool parser all kept).
Backup: `~/dev/.vllm-bump-review/unit-backups/vllm.service.pre-dflash-20260625-192202`.

**Live prod bench (`:8000`, -O3, 225K):** decode **106.9 tok/s** (was 74.6 MTP = **+43%**),
TPOT 9.35 ms, accept_len 1.94. Smoke: coherence ✅, tool-call (`get_weather` think-on + think-off) ✅,
finish_reason `tool_calls` ✅. Rollback = restore backup unit + `daemon-reload` + restart.

Note: prod (106.9) trails the leaner GPU1 sweep N=10 (116.9) — diff is `-O3` + 225K ctx + 512 seqs.
**Open lever:** A/B `-O3` on/off on the prod config (sweep was no-`-O3`); could recover ~9%.

## Concurrency + `-O3` A/B (2026-06-25, GPU1, prod-faithful config, ignore_eos 400-tok reqs)

Aggregate output tok/s at concurrency C (`conc-driver.sh` → `conc_bench.py`):

| C | dFlash N=10 (-O3) | dFlash N=10 (no -O3) | MTP (-O3) | winner |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 110.7 | 110.5 | 73.2 | **dFlash +51%** |
| 4 | 219.7 | 265.2 | 269.9 | MTP ≈ dFlash(no-O3) |
| 8 | 458.5 | 450.2 | 545.4 | MTP +19% |
| 16 | 543.6 | 546.5 | 984.0 | MTP +81% |
| 32 | 503.7 | 487.6 | **1471.2** | **MTP +192%** |

**Headline: dFlash wins single-stream (+51%) but DOES NOT SCALE.** Aggregate plateaus ~540 tok/s
regardless of load; MTP scales near-linearly to 1471 @ C=32. Crossover ~C=3–4.

- **Why:** dFlash drafts 10 tok/step at ~16% acceptance → ~11× verify-batch inflation → compute
  saturates as soon as requests stack. MTP drafts 1 tok at ~79% acceptance → ~2× inflation → stays
  efficient under batch. dFlash acceptance holds (~1.6 accept_len) under load — it doesn't *break*,
  it just can't amortize its wasted draft compute once the GPU is busy.
- **`-O3` is a wash** for this lane (543.6 vs 546.5 @ C=16; tie across the curve). The earlier
  GPU1 sweep's 116.9 was the leaner 32K-ctx / smaller-seqs config, **not** `-O3`. No prod change.

**Decision (user, 2026-06-25): KEEP dFlash N=10 in prod.** Bet is the smart lane skews
single-stream/interactive. If it starts taking concurrent gateway/eval batch load, revert to MTP
(backup unit, one command). dFlash is a **single-stream latency engine**, not a throughput engine.

## Quality regression check (eval suite, 2026-06-25, dFlash lane direct on :8000)

Quick profile, claw skipped (claw harness hung on T02 judging — separate issue, not a model fault;
the T02 agent itself completed correctly in 2.9s). Custom + function-call:

| | dFlash N=10 | MTP baseline (06-14) |
|---|:--:|:--:|
| custom | **91/108 = 84.3%** | 78/108 = 72.2% |
| function-call | **49/54 = 91%** | 47/54 = 87.0% |
| suites passed | 13/13 | 14/14 |

dFlash ≥ baseline on both → **no quality regression** (expected: spec decode is lossless, target
verifies every accepted token). Caveat: baseline predates this branch's tool-call grader fix, so the
gap is partly grader, not model — conservative claim that holds: dFlash didn't cost quality. To fully
separate, re-run MTP on the current eval branch.

**Eval harness gotchas hit:** (1) `run.sh` needs Infisical; run fully-local by invoking
`python -m runners.run_profile ... --gateway-url http://localhost:8000/v1` with
`GATEWAY_API_KEY=not-needed JUDGE_GATEWAY_URL=http://localhost:8000/v1`. (2) `run_profile` only passes
`--local` to its sub-`run.sh` calls when `GATEWAY_API_KEY`/`LITELLM_API_KEY` is set
(`have_key`, run_profile.py:197) — without it every suite "fails" in 0s via Infisical. Worth a 1-line
fix. (3) claw stage hung on judging — use `--skip-claw` for a fast custom+FC quality check.

## Blog

Draft captured as **[protoContent#332](https://github.com/protoLabsAI/protoContent/issues/332)**
("A +51% speed win that vanished under load") — `blog` label, status draft.

## Repro

```bash
# draft already at /mnt/models/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash
MODE=dflash   bash experiments/dflash/run-dflash.sh   # GPU1:8003, target+draft
MODE=mtp      bash experiments/dflash/run-dflash.sh   # baseline (or just bench prod :8000)
bash experiments/dflash/bench.sh 8003 8 long          # decode tok/s + acceptance
```
