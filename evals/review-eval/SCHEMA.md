# Replay-run output contract

Reconciled 2026-07-24 against the **shipped runner** (pr-reviewer#39, v0.18.0/v0.19.0
`replay.py`) — where this document and the implementation disagreed, the implementation won.
`score_ab.py` consumes this shape natively (and still accepts the older batched
`{"run", "reviews": [...]}` shape).

## Run JSON — one review per file, as the runner emits

```json
{
  "run": {
    "repo": "protoLabsAI/protoAgent",
    "pr": 2208,
    "head": "<reviewed sha — pinned, from replay_manifest.jsonl, verbatim>",
    "round": 1,
    "recipe": "code-review-structural",
    "model": "protolabs/fast",
    "trial": 1,
    "stamp": "<runner timestamp>"
  },
  "verdict": "FAIL",
  "findings": [
    {
      "file": "operator_api/config_routes.py",
      "line": 271,
      "severity": "major",
      "claim": "sync _apply_settings_changes blocks the event loop",
      "evidence": "…"
    }
  ],
  "telemetry": {
    "failed_steps": 0,
    "truncated": false,
    "confined": 0,
    "grounding_checked": 3,
    "grounding_downgraded": 0,
    "converge_reason": "stable",
    "converge_notes": 0,
    "dispositions": 2,
    "unaccounted_priors": 0,
    "step_seconds": {},
    "token_usage": {}
  }
}
```

Scorer field usage: findings `file`/`line`/`severity` (matching + severity split);
`telemetry.truncated` (`findings=[] && truncated` scores as a harness failure, not a clean
pass — the `fast` 6k-reasoning-tokens incident); `run.model`/`run.trial` (grouping). `claim`/
`evidence` are carried for human review, not scored on.

## ⚠️ Open ask on the runner (the one gap found in reconciliation)

`telemetry.dispositions` is a **count**; the disposition objects
(`{"prior": "file.py:271", "disposition": "fixed|refuted|…", "why": "…"}`) are dropped.
The honesty axis — false `fixed`/`refuted` on a still-present defect, the
pr-reviewer-plugin#37/#38 class that caused the original incident — **cannot be scored from
a count**. Runner should emit the objects as a top-level `"dispositions": [...]` array;
`score_ab.py` already consumes that field.

`token_usage` is `{}` unless the host runner surfaces usage — fine, but the truncation flag
must remain accurate without it.

## Ground truth JSONL

One row per labeled finding, protoLab#24 schema:

```
{"repo": "...", "pr": 2208, "head": "...", "round": 1, "severity": "major",
 "file": "...", "line": 142, "ground_truth": "true",
 "grounding_method": "blob+executed_test", "disregarded_evidence": false, "note": "..."}
```

`ground_truth` ∈ `true` / `false` / `false_negative` / `true_unverified` / `unverified`.
`false_negative` rows are defects the panel is *expected* to find — they drive recall.
Rows with `grounding_method` ∈ `assertion_only`/`not_grounded` are excluded from precision.
