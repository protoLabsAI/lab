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
