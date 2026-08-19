# review-eval running results

## 2026-07-24 — SWE-PRBench eval_100 baseline: protolabs/fast (Ornith-1.0-35B-NVFP4)

Full 100-PR paper split, config A, agent budget 16384, `SWEPR_HTTP_TIMEOUT=900`,
judge=protolabs/cloud (DeepSeek V4). Fast lane was otherwise idle; 100/100 records.

| metric | fast | paper reference (config A) |
|---|---|---|
| overall | **0.100** (0.092 weighted) | Haiku 4.5 0.153 · GPT-4o 0.113 · Llama-3.3-70B 0.079 |
| detection rate | 0.136 | best ~0.31 |
| precision | 0.106 | SWR-Bench best-config precision 0.167 for scale |
| hallucination | **0.099** | frontier range 0.193–0.417 |
| false-positive rate | 0.099 | frontier 0.193–0.417 |
| attempt rate | 0.69 (31 no-attempts) | — |

By type: Type1_Direct 0.143 / Type2_Contextual 0.161 / Type3_Latent 0.074 detection.

**Read:** Ornith lands between Llama-70B and GPT-4o overall — respectable for a local 35B-A3B
NVFP4 — but the *shape* is the story: hallucination 0.099 is **half the best frontier rate**,
paired with low detection and 24/31 no-attempts being deliberate explicit `[]` verdicts
(7 were parse failures, 6 judge parse-fallbacks). Static-mode Ornith is a conservative,
low-recall/low-noise reviewer — which converges with the #26 production anecdote ("2 findings
in ~10 reviews, both true, both minor"). The thin-review behavior is at least partly the
model's disposition, not only budget truncation. The system-layer A/B will show whether the
panel harness compensates (it exists to force looking).

**Caveats (honest-numbers):** our judge is DeepSeek V4, not the paper's GPT-5.2 — absolute
numbers are not leaderboard-submittable; the paper claims ranking stability across judges,
so cross-model comparisons *within our own runs* (same judge) are the valid use. Smart-side
comparator run pending Josh's model pick + a quiet lane.

## 2026-07-24 — SWE-PRBench smoke (3 PRs, config A, judge=protolabs/cloud, agent budget 16384)

**Purpose:** validate the model-layer pipeline end-to-end, not to produce publishable numbers
(n=3, easy PRs only).

| agent | overall | detection | FPR | hallucination | notes |
|---|---|---|---|---|---|
| protolabs_fast (Ornith-35B) | 0.398 | 0.50 | 0.00 | 0.00 | pipeline clean end-to-end; Type1 0.75 / Type2 0.00 detection |
| :8041 "smart" | — | — | — | — | **INVALID — all zeros were client-timeout artifacts, not scores** |

Paper frontier reference: overall 0.079–0.153, hallucination 0.193–0.417 (100-PR split — not
comparable to an n=3 smoke, but the fast numbers are not obviously broken).

**What broke, and the fixes now in place:**

1. **Harness hard-codes a 120s HTTP read-timeout** (`model_clients.py _post_with_retries`).
   Local lanes generating into a 16k budget exceed it → `agent_parse_failed` → the report
   renders as all-zero scores with `no_attempt_count=3`. This is the time-domain version of
   the token-starvation false-zero rule ([[feedback_eval_prod_token_budget]]). Patched
   locally: timeout now reads `SWEPR_HTTP_TIMEOUT` (default 120). Set `SWEPR_HTTP_TIMEOUT=900`
   for local lanes.
2. **":8041 = smart" is not a stable identity.** During the smoke it was serving
   ThinkingCap-Qwen3.6-27B-heretic-NVFP4 (manual serve, 07-23, aliases `smart`+`protopen`;
   `vllm-smart.service` is in failed state) — not Laguna, not Gemma-31B. Any scoring run must
   record `/v1/models` + the serve cmdline in the run record, or the A/B is against an
   unknown comparator.
3. **Lane contention invalidates eval numbers both ways.** At smoke time :8041 had a growing
   ~78-deep wait queue of prod traffic. Check `vllm:num_requests_waiting ≈ 0` before scoring,
   or serve a dedicated eval instance.

**Open before the real model-layer baseline (eval_100):** pick the smart-side comparator
(Gemma-31B per the 2-lane fleet? Laguna? ThinkingCap-heretic?) — Josh's call, it's whatever
Vera would actually run on — and run when the lane is quiet.
