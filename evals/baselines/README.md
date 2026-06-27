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

| Date | Daily driver | claw (mean task_score) | coding | FC | Judge |
|---|---|---|---|---|---|
| _2026-06-27_ | Ornith-1.0-35B-FP8 (2 replicas) | _(populating)_ | _(populating)_ | _(populating)_ | protolabs/reasoning |
