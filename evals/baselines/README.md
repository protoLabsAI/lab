# Standing baselines

Canonical eval numbers for the **current daily driver**, re-run on every methodology change (judge swap, thinking flip, harness fix) so "how does model X stack up?" is always answerable against a fixed reference.

## Methodology (locked 2026-06-27)

- **Model under test**: the daily driver, served directly (`local` on `:8000`), **thinking-on**.
- **Judge**: `protolabs/reasoning` via the gateway (`ava:4000`) — an independent, strong reasoning model. Cloud judge is reserved for **baselines** (run rarely); everyday/relative runs may use a local judge. Never self-judge a baseline.
- **Suite**: agentic-targeted — 35 claw tasks (30 business/ops + T100–104 coding) + custom coding + function-call.
- **Caps**: claw 10k tok/turn, coding 16k, FC 8k (bound think-spirals).
- **Harness**: kb/contacts health-probe fix in place (no silent service failures). Report harness-errored tasks distinctly from model-scored ones.

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
