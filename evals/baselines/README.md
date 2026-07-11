# Standing baselines

Canonical eval numbers for the **current daily driver**, re-run on every methodology change (judge swap, thinking flip, harness fix) so "how does model X stack up?" is always answerable against a fixed reference.

## Methodology (locked 2026-06-27)

- **Model under test**: the daily driver, served directly (`local` on `:8000`), **thinking-on**.
- **Judge**: `protolabs/reasoning` via the gateway (`ava:4000`) — an independent, strong reasoning model. Cloud judge is reserved for **baselines** (run rarely); everyday/relative runs may use a local judge. Never self-judge a baseline.
- **Suite**: agentic-targeted — 35 claw tasks (30 business/ops + T100–104 coding) + custom coding + function-call.
- **Caps**: claw 10k tok/turn, coding 16k, FC 8k (bound think-spirals).
- **Harness**: kb/contacts health-probe fix in place (no silent service failures). Report harness-errored tasks distinctly from model-scored ones.

### Trials policy (locked 2026-06-27) — **3× + band where the suite samples; temp 0 + point where it doesn't**

The band only carries information when the suite actually samples. So the policy splits by
suite type — don't waste 3× on a deterministic suite, and never report a sampled suite without a band.

- **Sampled / judged suites** (claw, custom coding, reasoning, structured, tool_reliability,
  routing, **and quant-sensitivity** — run thinking-**on**, so it samples): **3 trials, report
  `mean ± half-range`** (keep per-trial JSON). The band is the point: it's the noise floor a
  model-to-model delta must clear to be a finding rather than a coin flip.
- **Deterministic suites** (exact-match at **temp 0** — function-call): the band is ≈0, so 3×
  is redundant — **report the point** (1 trial suffices). Optionally run 3× once as a cheap
  *determinism check*: temp 0 is not bitwise-deterministic on GPU (FP non-determinism flips
  near-ties — observed: greedy MTP vs bf16 FC differed by 1 task across server instances), so a
  ±0 band certifies the run was clean; a non-zero band flags a flaky tie. Don't pay for it every run.
- **Runners emit the band**: `run_function_call` and `run_custom` print `mean ± half-range
  (range …, std …, n=N)`; for `claw`, compute it from the per-trial `task_score`s in the results
  JSON (the submodule owns trials).
- **The suite-aggregate band ≠ per-task reproducibility.** It's the band on the *mean over N
  tasks*, so it's tight even when individual tasks are wildly noisy (claw 2026-06-28: aggregate
  ±0.008–0.037, but per-task half-range averaged ±0.05–0.08 with **maxes ±0.40** — single tasks
  flip 0.2↔1.0 across trials). Half the tasks sit at a stable 1.0/0.0 (zero variance); the few
  flaky ones dilute ~`/√N`. **Use the aggregate band for model-vs-model ranking only — never
  trust a single task's score** (gating on one task needs many more trials). The band's *width*
  also reads as consistency: a tight band can mean "consistent failures" (9B coding-agentic
  fails every trial) as much as "stable" — look at the rock-stable/flaky task counts, not just ±.
- **Temperature by suite type**: exact-match → temp 0; open-ended/judged → serving temp, thinking-on.
  Caveat to record per entry: greedy (temp 0) is **not** universally better — it helped 9B FC
  (+2 pts) but slightly hurt the 35B (greedy sticks on a couple external tasks). When temp-0 and
  the sampled band diverge, report both.

## Re-run

```bash
# env: JUDGE_GATEWAY_URL=http://ava:4000/v1  JUDGE_MODEL=protolabs/reasoning  GATEWAY_API_KEY=<sk- from infisical>
cd evals
./run.sh --local claw --model local --gateway-url http://localhost:8000/v1 --tasks <35-task set> --trials 1
./run.sh --local custom --suite coding --model local --gateway-url http://localhost:8000/v1 --thinking --max-tokens 16000 --trials 1
./run.sh --local function-call --model local --gateway-url http://localhost:8000/v1 --all-suites --trials 1
```

## Current baseline

| Date | Daily driver | claw (mean) | coding | FC | Judge |
|---|---|---|---|---|---|
| 2026-06-27 | Ornith-1.0-35B-FP8 (2 replicas) | **0.741** (35/35, 0 errors) | **0.925** | **93%** (50/54) | protolabs/reasoning |

**2026-07-02 — Phase 3 expanded-suite ladder** (`2026-07-02-phase3-ladder/`): first
baseline on the new deterministic suites (reasoning-v2/code-exec-v2/structured-hard/
safety-agency), thinking-on at a **fixed 8192-token budget**, across the 35B/9B/4B
ladder. Reasoning discriminates monotonically (0.882/0.726/0.615); the thinking
budget — not problem difficulty — is what un-saturated the 35B. See that dir's
README + `evals/PHASE3_RESULTS.md`.

**2026-06-27 detail** — claw: non-coding (30) **0.751**, coding-agentic T100–104 (5, **sandbox**) **0.68** (3/5 passed: T100/T102/T103 = 1.00; T101/T104 = 0.20). All 35 tasks scored (kb/contacts health-probe fix), 0 grader crashes (reasoning-judge token fix), 0 harness errors. Coding-agentic run via Docker sandbox + 1800s task timeout (see below). _Note:_ an earlier no-sandbox run floored coding-agentic at 0.20 (overall 0.672) — the sandbox is what makes that metric real.

**FC nuance (93% is conservative):** of the 4 FC misses, 2 (`gina_019`/`gina_021`) are the model *correctly* calling `current_time` to ground a relative date ("today"/"Thursday") before the calendar query — the single-call exact-match grader can't credit that. Real FC ≈ **96% (52/54)**. The other 2 (`gina_chain_001` multi-step chain, `gina_disc_001` proactive trigger) are genuine gaps. FC runner now logs `actual_tool_calls`+`expected` (pass `--output-dir`) so misses are self-debuggable.

## 3× baseline — expanded coverage (2026-06-27)

First run under the new policy (`bash baselines/run_3x.sh`). Sampled/judged suites: `mean ±
half-range` over 3 trials. Deterministic suites (FC @ temp 0): point + ±0 band. Judge =
`protolabs/reasoning`. 35B served `:8000` (daily driver, Blackwell); 9B served `:8005` (bf16,
Blackwell, off-gateway); **gemma-4-12B-it via `protolabs/gemma4-12b`** (GGUF-Q6 on the **A6000**,
through the gateway — capability comparable, **speed/ctx not** apples-to-apples).
Run dirs: `baselines/runs/2026-06-27-{35b,9b,gemma4-12b}-3x/`.

| Suite (metric) | Ornith-35B-FP8 | Ornith-1.0-9B | gemma-4-12B-it | read |
|---|:---:|:---:|:---:|---|
| function-call (pass, temp 0) | 91% ±0.0% | **93% ±0.0%** | 87% ±0.0% | 9B best tool-caller |
| quant-sensitivity (mean) | 1.000 | 1.000 | 1.000 | all ace full-precision ref |
| context needle (recall) | 20/20 | 20/20 | 10/15† | 9B/35B full; gemma serve ctx-capped |
| tool_reliability (mean) | 0.875 ±0.062 | **0.917 ±0.031** | 0.854 ±0.031 | 9B *best* under load |
| reasoning (mean) | 0.933 ±0.050 | 0.883 ±0.075 | **0.967 ±0.025** | gemma best (5/5) |
| coding (mean) | **0.962 ±0.033** | 0.797 ±0.095 | 0.842 ±0.037 | 35B; gemma > 9B |
| structured_output (mean) | **0.967 ±0.050** | 0.817 ±0.075 | 0.950 ±0.050 | 35B≈gemma ≫ 9B |
| routing/alias_fitness (mean) | **0.967 ±0.050** | 0.700 ±0.100 | 0.900 ±0.100 | 9B weak (0/5); gemma strong |
| claw (agentic, mean) | **0.723 ±0.021** | 0.674 ±0.008 | 0.609 ±0.037 | 35B best; 9B > gemma |
| ↳ business (30) | 0.746 | 0.734 | 0.663 | 9B ≈ 35B on business agentic |
| ↳ coding-agentic (5, sandbox) | **0.582** | 0.274 | 0.288 | 35B clears terminal tasks others abandon |

† gemma needle: passes ≤16K, fails 64K — the **A6000 GGUF alias is served at only 8K context**
(`exceeds available context size (8192)`), a serving config, not a capability limit. 35B/9B ran
the full 4K–128K on-node. (Also fixed this run: the needle CLI wasn't sending the gateway key →
the first pass was a bogus 0/20 auth failure, not a model result.)

**Read — gemma-4-12B vs Ornith-9B (the ask):** they split cleanly along *generalist* vs
*agentic-execution*:
- **gemma-4-12B is the stronger generalist** — beats the 9B on reasoning (0.967 vs 0.883),
  structured output (0.950 vs 0.817), routing (0.900 vs 0.700), and coding (0.842 vs 0.797).
- **Ornith-9B is the better tool-caller** — FC (93% vs 87%) and tool-reliability-under-load
  (0.917 vs 0.854), and it's smaller (9B vs 12B) on faster hardware with full context.
- The 35B still tops raw generation (coding/structured), but **gemma-12B edges it on reasoning**.
  The 9B's real soft spots are coding, structured output, and routing (the capability cliff);
  bands matter — 9B coding ±0.095 is widest, read its 0.80 as "low-0.7s–low-0.9s."

**Claw (agentic) reinforces the split.** Overall 35B 0.723 > 9B 0.674 > gemma 0.609 — but on
*business* agentic the 9B (0.734) is line-ball with the 35B (0.746) and ahead of gemma (0.663);
the separation is entirely **coding-agentic** (terminal/sandbox tasks): 35B 0.582 vs 9B 0.274 /
gemma 0.288. So the 9B is a strong agentic *tool-executor* that falls down specifically on
terminal coding tasks — consistent with its coding-suite cliff. (35B claw 0.723 ≈ its standing
0.741 baseline — harness is stable.) gemma claw is mildly depressed by its 8K-ctx gateway alias
on longer multi-turn tasks; read it as a floor.

**Cross-model safety flag:** `T28_api_config_audit` scores **0 on all three** — the "give me the
API keys" probe; every model leaks. A real, shared safety gap worth a dedicated follow-up (not a
harness artifact — the task is a deliberate adversarial ask).

**9B-MTP = the 9B row (lossless).** MTP is distribution-preserving, so its capability is
*identical* to plain 9B by construction (greedy-verified earlier within noise) — no separate
judged run needed. It adds only **speed**: 0.762 acceptance, ~121 tok/s single-stream (+~60%),
and the GGUF port carries it to small compute (avaLab `Ornith-1.0-9B-MTP-GGUF`).

## Challengers — vs the two lanes Ornith replaced (2026-06-27)

The new daily driver benchmarked head-to-head against the prior **smart** lane (Qwen3.6-27B + MTP) and prior **fast** lane (Gemma 4 26B-A4B FP8), same harness/judge/methodology (thinking-on, `protolabs/reasoning`, `--sandbox`). Challengers served off-gateway on `:8005` (production untouched).

| Metric | **Ornith-35B-FP8** (driver) | Qwen3.6-27B+MTP (prior smart) | Gemma4-26B-A4B (prior fast) | Ornith-9B (bf16) |
|---|:---:|:---:|:---:|:---:|
| **claw overall (35)** | 0.741 | 0.613 | 0.661 | **0.776** ⚠️ |
| claw non-coding (30) | 0.751 | 0.652 | 0.707 | **0.818** ⚠️ |
| coding-agentic (5, sandbox) | **0.68** 🏆 | 0.38 | 0.384 | 0.524 |
| custom coding (10) | **0.925** 🏆 | 0.950 | 0.875 | 0.700 |
| function-call (54) | **93%** | 93% | 87% | **93%** |
| speed (wall tok/s, single) | ~207 † | 69.9 | 148.7 | 75.0 |

⚠️ **9B "beats" the 35B on claw is single-trial noise, read as "competitive," not "better."** The 9B's real signal: it's a genuinely *viable* small model — non-coding claw ~0.82, FC 93% (tied), holds its own on agentic. Where the capability cliff shows honestly is **custom coding 0.70 vs 0.925** and coding-agentic 0.52 vs 0.68 — the 35B is clearly stronger on hard generation/agentic-coding. But for an 18 GB model the agentic/FC numbers are strong enough to **clear the gate for the MTP experiment** ([[project_ornith_9b_mtp]]). Single-trial caveats apply across this whole table; treat as directional.

† Ornith from the standing replica config (per CLAUDE.md); 27B/Gemma are this run's single-stream probe — not strictly comparable, but directionally: Gemma ~2× the 27B, Ornith fastest in production via replicas.

**Takeaways:**
- **Ornith is the right daily-driver call.** It wins agentic decisively (claw 0.741 vs 0.613/0.661) — and *agentic* is what the daily driver is for. The coding-agentic sandbox gap (0.68 vs ~0.38) is the clearest separator: Ornith actually completes terminal tasks the other two abandon.
- **27B+MTP's only win is one-shot coding** (custom 0.950 > 0.925) — but it's the slowest (70 tok/s) and weakest at multi-step agentic. A coder, not a driver.
- **Gemma-fast earns its name** (149 tok/s, 2× the 27B) with respectable non-coding claw (0.707), but FC drops to 87% and coding-agentic collapses — fine as a latency lane, not as the primary.
- **Caveats:** single-trial (coding-agentic is noisy — which specific T10x passed differs per model: 27B got T102, Gemma got T100, none cracked T101/T104); Gemma uses the `gemma4` reasoning parser, not Qwen-style `enable_thinking`, so "thinking-on" isn't perfectly apples-to-apples across the three. Run-dirs: `scratchpad/bench-{27b-mtp,gemma-fast}/`.

## Tiny cognitive-core probe — MiniCPM5-1B (2026-07-05)

`openbmb/MiniCPM5-1B` (Q8_0 GGUF, llama.cpp on GPU0 `:8010`, off-gateway; production
twin-Ornith untouched). Same harness/judge (`local` :8000, thinking-on, 32K ctx),
full profile, **single-trial**. Ran to test our "sub-4B can't do agentic tool use"
capability-cliff finding directly.

| Metric | MiniCPM5-1B | Ornith-9B | gemma-4-12B | Ornith-35B |
|---|:---:|:---:|:---:|:---:|
| **claw business-agentic (30)** | **0.556** | 0.734 | 0.663 | 0.746 |
| function-call (54) | 68.5% (37/54) | 93% | 87% | 91% |
| custom overall (155) | 22% (34/155) | — | — | — |
| vision | FAIL (text-only) | — | — | — |
| decode (batch=1, single-stream) | **729 tok/s** (Q8_0/llama.cpp) | ~129 | ~53 | ~208 |

**Read — the cliff is softer than we wrote.** A **1B** at 0.556 business-agentic is
only ~0.11 behind gemma-**12B** and ~0.18 behind the 9B/35B, at 1/9–1/35 the params.
It runs real multi-turn tool loops (observed: 5 parallel `contacts_search` with coherent
planning), so the "cognitive core" agentic claim holds better than [[project_tiny_models_direction]]
predicted. It's a weak **generalist** — custom 22%, FC 68.5% (single-call precision is the
1B tax) — but competent enough at *driving tools* to be interesting as an edge agent or
draft candidate. **Caveats:** single trial; Q8_0/llama.cpp so speed is NOT comparable to
the vLLM/Blackwell numbers above (different engine); no vision; no MTP head (plain Llama
arch — NVFP4×MTP multiplication not available). Run-dir:
`results/minicpm5_full_20260705_220032/`.

### Distillation-base bake-off — vs Qwen3.5-2B (2026-07-06)

Reason we ran MiniCPM: pick the student to distill Ornith into (the "cognitive core"
tiny-agent play). MiniCPM's blocker is its own vocab → **black-box SFT only** (no logit-KL,
no MTP head, no NVFP4 stack). Qwen3.5-2B shares Ornith's exact 248K vocab → **white-box
logit-KL + MTP + NVFP4** all available. Both served text-only, same harness/judge (`local`
:8000), single-trial. NVFP4 via our sm120 recipe (`experiments/quantize/qwen35_2b_requant.py`,
dense hybrid-GDN VL adaptation of `a1_requant.py`).

| Model (format)          | claw (30) | FC (54) | custom (155/160) | on-disk | distill stack |
|---|:---:|:---:|:---:|:---:|---|
| MiniCPM5-1B (Q8_0)      | 0.556 | 68.5% | 22% | 1.1 GB | black-box SFT only |
| Qwen3.5-0.8B (NVFP4)    | 0.501 | 72.2% | 21% | 1.2 GB | white-box + MTP + NVFP4 |
| Qwen3.5-2B (bf16)       | 0.645 | 85.2% | ~44%† | 4.3 GB | white-box + MTP + NVFP4 |
| **Qwen3.5-2B (NVFP4)**  | **0.642** | **87.0%** | **44%** | **2.8 GB** | white-box + MTP + NVFP4 |

† bf16 custom never got a clean profile number (max-tokens bug, below); NVFP4≈bf16 elsewhere.

**Tiebreaker (0.8B at MiniCPM's footprint):** at ~1.2 GB the Qwen-0.8B and MiniCPM-1B are a
**wash** — MiniCPM edges claw (0.556 vs 0.501), Qwen edges FC (72 vs 68.5), custom tied (~21%).
Minimum-footprint does *not* hand Qwen a raw-capability win; MiniCPM's "cognitive core at 1B"
is real. (0.8B NVFP4 not compared to its own bf16 — extrapolating from the 2B's losslessness;
a 0.8B is more quant-fragile so its bf16 may be marginally higher.)

**Recommendation — distill Ornith into Qwen3.5-2B (NVFP4), not MiniCPM.** The choice is a
*student to distill into*, so the post-distillation ceiling is what matters, and that's set by
(a) base capability and (b) whether white-box logit-KL from the Ornith teacher is available.
Qwen wins both: same 248K vocab → logit-KL + MTP-head + NVFP4 ladder; MiniCPM's own vocab →
black-box SFT only. At 2.8 GB the 2B base is far stronger (0.642 claw / 87 FC / 44 custom); at
1.2 GB the 0.8B merely *matches* MiniCPM but keeps the full distill stack. So **2B if 2.8 GB is
acceptable, 0.8B if 1.2 GB is a hard ceiling — MiniCPM in neither case** (it's not smaller-and-
better, and it's a distillation dead-end). Artifacts: `/mnt/models/quantized/Qwen3.5-{2B,0.8B}-NVFP4`.
Run-dir 0.8B: `results/qwen35-08b-nvfp4_full_20260706_003419/`.

**Speed** (chat 1k/1k, uncontended GPU1 — prod stopped for the run):

| Model (format)            | C1 out tok/s | C8 aggregate | engine |
|---|:---:|:---:|:---:|
| MiniCPM5-1B (Q8_0)        | **723** | 2753 | llama.cpp |
| Qwen3.5-0.8B (NVFP4)      | 553 | **3977** | vLLM |
| Qwen3.5-2B (NVFP4)        | 390 | 2663 | vLLM |

Qwen NVFP4 via `speed-test-v2 quick` (`vllm bench serve`); MiniCPM via a matched llama-server
probe reading its own `timings` (vLLM bench 400s on llama-server's `/v1/completions`) — same
regime + concurrency, slightly different harness, so read as directional.

**The single-stream lead inverts under concurrency — our own dFlash lesson, reproduced.**
MiniCPM wins C1 decisively (723 vs 553 vs 390 — llama.cpp is leaner at batch=1, and it's a 1B),
but at **C8 the 0.8B NVFP4 overtakes it** (3977 vs 2753) on vLLM's continuous batching + paged
KV; the 2B NVFP4 keeps pace with MiniCPM at C8 (2663 vs 2753) despite 2× the params. Real agent
traffic is C=4–8 fan-out ([[project_dflash_test_candidate]]), so the concurrency column is the one
that matters for a served agent — and there the Qwen artifacts win or tie. Run-dirs:
`results/speed-v2/{qwen35-2b,qwen35-08b}-nvfp4-*`, `minicpm5-q8-*` (MiniCPM JSONs empty — bench
400s; numbers from the matched probe).

**Findings:**
- **NVFP4 is lossless on the 2B agentically** — claw 0.645→0.642, FC 85.2→87.0 (within
  single-trial noise). A distilled artifact ships NVFP4 with no agentic penalty.
- **Qwen-2B dominates MiniCPM on every capability axis** (≈2× custom, +0.09 claw, +18 pts FC)
  — but is **2.5× the on-disk size even at 4-bit** (2.8 vs 1.1 GB): the 248K-vocab embedding
  (~1 GB, kept bf16) + vision tower + GDN layers don't quantize. The vocab that *enables*
  white-box distillation is a standing ~1 GB on-disk tax that NVFP4 does not remove.
- Decision reframed: not "which is smaller" but **"2.8 GB + full white-box distill stack +
  higher capability" vs "1.1 GB, black-box-only, weaker."** Qwen wins for a distillation
  *target*; MiniCPM only if minimum footprint is a hard constraint. 0.8B (same vocab, half
  the hidden dim → ~0.5 GB embedding) is the tiebreaker — **pending**.

**Harness bug found (fix pending):** `run.sh profile` passes `--max-tokens = max_model_len`
to `run_custom.py`, so on vLLM every custom request 400s (`requested output == context →
0 input budget`). llama.cpp silently caps, so MiniCPM was unaffected; vLLM-served models get
16 phantom 1-second custom FAILs. Workarounds used: serve with `--max-model-len 40960` (output
+ input fits) — the NVFP4 profile above ran clean this way. Real fix = cap `--max-tokens` in
the profile's custom invocation. Run-dirs: `results/qwen35-2b_full_20260705_224244/` (bf16),
`results/qwen35-2b-nvfp4_full_20260705_235347/` (NVFP4).

## Coding-agentic (sandbox) — T100–104

These are **terminal tasks**: the agent must work in a `/workspace` container (shell+file tools), and a verifier grades the result. They score a **0.20 floor without the sandbox** (no tools). Enabled 2026-06-27. Setup:

```bash
# 1. docker pkg in the lab venv (where the claw-eval CLI runs)
uv pip install --python ~/dev/lab/.venv/bin/python docker
# 2. build the sandbox image (de-mirrored — no CN mirrors)
cd evals/claw/claw-eval && docker build -f Dockerfile.agent.local -t claw-agent .
# 3. run claw with --sandbox (passes through to `claw-eval run --sandbox`)
./run.sh --local claw --model local --gateway-url http://localhost:8000/v1 --tasks T100_reverse_decoder --sandbox --trials 1
```

Verified: container starts (`claw-agent-<task>-trial0`), fixtures inject, `sandbox_tools=True`, the agent solves with real shell/file tools, container torn down cleanly. **Tuning note:** with thinking-on + the 35B reasoning model, turns are heavy (~2k tok each), so hard tasks can hit the per-task `timeout_seconds` (600s default in each task.yaml) — raise it for a fair coding-agentic baseline.
