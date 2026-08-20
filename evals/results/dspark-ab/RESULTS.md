# DSpark A/B — does spec-decode earn its keep at real concurrency?

**Date:** 2026-08-11 · **Lane:** DSV4-Flash / jasl fork :8041 · **Harness:** `models/bench_coherent.py`

The jasl cutover was justified on **+78% C1 decode** from DSpark (K=5). The open question — never
tested — was whether that inverts under the **C=4–8 parallel-agent fan-out that actually is our
traffic**. This is the [[dflash]] risk case: a single-stream win that flipped to 3× *slower* by C=32.

Both arms: identical prompts (seed 42), coherent English prose, 1024-token input / 256-token output,
`--max-num-seqs 16` unchanged. Arm B disabled spec-decode via a temporary systemd drop-in
(`Environment=SPEC=`), verified by `speculative_config=None` and zero SpecDecoding metric lines.

## Result — it inverts between C=4 and C=8

```
   C |  ON tok/s  OFF tok/s    delta |  ON tpot  OFF tpot |  ON ttft  OFF ttft
-----+-------------------------------+--------------------+-------------------
   1 |     199.1       97.4    +104% |    4.8ms    9.5ms  |    101ms    179ms
   4 |     299.5      273.0     +10% |   12.5ms   13.3ms  |    225ms    307ms
   8 |     369.2      432.1     -15% |   19.8ms   16.4ms  |    247ms    367ms
  16 |     494.4      608.4     -19% |   29.0ms   21.9ms  |    407ms    615ms
```

**Crossover is ~C=5–6.** At C=1 DSpark more than doubles throughput; by C=8 it costs 15%, and by
C=16 it costs 19%. Same shape as dFlash, but a later crossover (dFlash flipped at C=3–4) because
DSpark's acceptance is far healthier.

## But it is NOT a clean loss — TTFT is better with DSpark ON at every concurrency

101 vs 179 · 225 vs 307 · 247 vs 367 · 407 vs 615 ms (p50). DSpark improves time-to-first-token
**even where it costs aggregate throughput**. For interactive and agentic work that is the
user-visible number, and it partly offsets the throughput loss. This is a genuine tradeoff, not a
straight regression:

- **DSpark ON** — better TTFT everywhere; better throughput at C≤4; worse throughput at C≥8.
- **DSpark OFF** — worse TTFT everywhere; better throughput at C≥8.

## ⚠️ This measurement UNDERSTATES DSpark — do not flip the lane on it alone

Acceptance during these runs was **15.7–30.3%**. Live production traffic on the same lane measures
**39–48%** (mean acceptance length ~3.0–3.4). The corpus here is Shakespeare (`sonnet.txt`), which
is coherent but harder to predict than the technical/code/agentic text this lane actually serves.

Acceptance is the whole mechanism: higher acceptance means fewer wasted draft positions, which
**pushes the crossover to the right**. At production acceptance the inversion point is plausibly
C=8–12 rather than C=5–6 — which would place it above, not inside, our normal operating band.

**Next step before any config change:** re-run with a production-representative corpus (technical
prose / code / agent transcripts) that reproduces 39–48% acceptance, and confirm where the
crossover actually lands. Only then decide.

## Structural note: K cannot be tuned down

Per-position acceptance on real traffic: **0.82, 0.57, 0.32–0.43, 0.14–0.30, 0.06–0.13**. Positions
4–5 land ~10–14% of the time — mostly wasted compute. Normally you would trim K to 3 and recover it,
but `serve-dsv4-flash-jasl.sh` documents that **DSpark's block size forces K≥5** (smaller produces
incorrect output; the fork validates this). So that waste is structural and untunable on this fork.

## Interaction with the MAXSEQS work

Raising `--max-num-seqs` above 16 pushes the lane toward higher concurrency — exactly where DSpark
costs throughput. The two changes are not independent and should be evaluated together, not
sequentially.

---

## ⚠️ CONTAMINATION FOUND AFTER THE FACT (2026-08-11, same session)

While running a follow-up MAXSEQS A/B on the same harness, run-to-run spread on **identical
configs** measured **31–60%**. Root cause: the lane was serving **production traffic throughout** —
8 completions in a 45 s "idle" window, 1–3 concurrent, from `100.101.189.45` (the ava gateway).

**This A/B was run under the same uncontrolled load.** Treat the numbers above as directional, not
final:

- The **direction** is credible — the effect is large (+104% → −19%), monotonic across four
  concurrencies, matches the documented dFlash precedent, and has a clear mechanism (verify-batch
  inflation competing for saturated compute).
- The **magnitudes are not trustworthy**, and the C=1 arm is the most suspect: background load of
  1–3 concurrent requests perturbs a nominally single-stream measurement far more than a C=16 one,
  which would inflate the apparent ON-vs-OFF gap at C=1 specifically.
- The **crossover point (C≈5–6) is therefore soft** — it could sit meaningfully higher, and the
  low-acceptance corpus (15–30% vs 39–48% live) already biases it low for a second, independent
  reason.

**Do not act on this without a re-run under a traffic-quiesced lane** (see the MAXSEQS task for the
protocol: quiesce the gateway, verify flat `request_success_total`, n≥5, interleave arms A/B/A/B).

Lesson, third instance in one day: **instrument the load variable in any serving experiment and
hold it constant across arms.** A benchmark that shares a lane with prod traffic is measuring both.
