# long-horizon agentic suite (T200-series) — design

Phase 3 P0 (FOCUS.md). The blind spot the Agents-A1 challenger run exposed: our
agentic tasks resolve in ≤10-30 turns with a single fixed objective. Models
trained for horizon (45K-token trajectories) tie models that aren't on our
suite because nothing here *requires* horizon. This suite makes 20–50 tool
calls structurally necessary and keeps grading deterministic.

## What makes a task "long-horizon" (design invariants)

1. **Sequential phase structure** — later phases consume earlier phases'
   artifacts, so the work cannot be compressed into a few big turns.
2. **Mid-task requirement change** — a fixture instructs the agent to read an
   update file only after phase 1 is complete (`UPDATE.md` mechanic). Tests
   goal persistence under changed constraints — the "sustained goal
   consistency" weakness Agents-A1's own limitations section names.
3. **Deterministic end-state verification** — `env_snapshot_commands` assert
   file/DB/git state; per-task `grader.py` scores component-wise (partial
   credit per phase). No LLM judge anywhere.
4. **Struggle-zone calibration** — target Ornith-35B task_score 0.3–0.7
   (T104-class, not T103-class which it aces).
5. **Trajectory instrumentation from day one** — the sandbox dispatcher's
   `tool_dispatch` records + per-phase verifier outcomes are exported with
   every run; these are shard candidates for the verified-trajectory dataset
   (`experiments/agentic-data/RESEARCH.md` play #1).

## Placement

Tasks live in `claw-eval/tasks/T2xx_*` (the submodule — our repo, public;
"evals as the open-source pattern"). Runner support is unchanged: same schema
as T100–104, `run_claw --tasks T200,...` + `--sandbox`.

## The five tasks

| id | phases | tool-call floor | verifiable end state |
|---|---|---|---|
| **T200_regression_hunt** | git-bisect a seeded regression → fix → add regression test → write timeline report → tag commit | ~20 (log/diff/run-test cycles) | hidden tests pass; tag exists; report names the guilty commit hash + root cause string |
| **T201_pipeline_backfill** | parse 3 log formats → load SQLite → reconcile vs ledger CSV → UPDATE.md arrives → re-reconcile with new rules → emit summary JSON | ~25 | DB row/aggregate assertions; JSON exact-match vs hidden expected; UPDATE rules applied (old rules fail a canary row) |
| **T202_api_migration** | vendored lib v2 has breaking API (renames + semantic change in return type) → migrate 8 call sites → keep suite green → changelog | ~20 | hidden test suite; grep-forbid legacy API symbols; changelog lists every touched file |
| **T203_config_drift** | write an audit tool per spec → run it over 12 configs → fix drift → UPDATE.md tightens one rule mid-task → re-audit | ~20 | independent recompute finds 0 drift; audit report lists exactly the seeded drifted keys (no more, no less) |
| **T204_protocol_server** | implement a line-protocol TCP server from SPEC.md → pass client conformance script → SPEC-v2 arrives (frame format + auth change) → migrate without breaking old clients (version negotiation) | ~30 | conformance script v1+v2 both pass against the running server at snapshot time |

Fixture discipline (learned from T101/T103 postmortems): no probing of
container internals required; all needed state visible under /workspace; every
grader assertion derivable from fixtures committed in-repo; graders re-derive
expected values from fixtures at grade time (no hand-copied constants that rot).

## Rollout order

T201 first (pure-Python fixtures, easiest to make deterministic), then T200
(git fixture scripted), T203, T202, T204 (server lifecycle at snapshot time is
the fiddliest). Calibrate each on Ornith-35B (5 trials — claw variance is
bimodal ±0.4) before building the next; adjust phase counts to hold the
0.3–0.7 band.

## Status

- **T201_pipeline_backfill — BUILT + fixture/verify/reference loop validated
  (2026-07-02).** `make_fixtures.py` (seeded), `verify_backfill.py` (re-derives
  ground truth from raw fixtures at grade time — no hardcoded constants),
  `grader.py` (phase-weighted + gates), `ref_solution.py` (maintainer tool,
  scores 1.0 on every component → the task is solvable and the verifier is
  correct). **Not yet run against a model** — needs the claw Docker sandbox path
  (`run_claw --tasks T201 --sandbox`), which is the calibration step under task #9.
  This is the suite's highest-value axis: it's the one place a 35B is expected to
  leave headroom (mid-task `UPDATE.md` rule change tests goal persistence).
- T200/T202/T203/T204 — designed above, not built.
