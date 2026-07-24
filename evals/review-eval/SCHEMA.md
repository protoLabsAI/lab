# Replay-run output contract

What the plugin-side replay runner (qaEngineer / pr-reviewer-plugin) must emit per run, and
what `score_ab.py` consumes. One JSON file per (model, trial).

## Run JSON

```json
{
  "run": {
    "model": "protolabs/fast",
    "panel_version": "qaEngineer v0.8.0",
    "trial": 1,
    "started": "2026-07-24T05:00:00Z"
  },
  "reviews": [
    {
      "repo": "protoLabsAI/protoAgent",
      "pr": 2208,
      "head": "<reviewed sha — pinned, not PR tip>",
      "round": 1,
      "findings": [
        {
          "severity": "major",
          "file": "src/agent/loop.py",
          "line": 142,
          "title": "event loop blocked by sync call",
          "body": "…",
          "disposition": "confirmed"
        }
      ],
      "telemetry": {
        "grounding_checked": 3,
        "unaccounted": 0,
        "converge_reason": "stable",
        "output_tokens": 1234,
        "reasoning_tokens": 5678,
        "truncated": false
      }
    }
  ]
}
```

Notes:
- `head` must be the SHA from `replay_manifest.jsonl`, verbatim — ground-truth labels are
  only valid against the blob they were graded on (protoLab#24 rule).
- `telemetry` mirrors pr-reviewer-plugin#34 fields plus token accounting. `truncated` is
  true when the finder hit its output budget before emitting a final answer — the scorer
  treats `findings=[] && truncated` as a harness failure, not a clean pass.
- `disposition` ∈ `confirmed` / `refuted` / `fixed` / `unverified` — as the panel posted it,
  so false `fixed`/`refuted` can be scored against ground truth (the honesty axis).

## Ground truth JSONL

One row per labeled finding, protoLab#24 schema:

```
{"repo": "...", "pr": 2208, "head": "...", "round": 1, "severity": "major",
 "file": "...", "line": 142, "ground_truth": "true",
 "grounding_method": "blob+executed_test", "disregarded_evidence": false, "note": "..."}
```

`ground_truth` ∈ `true` / `false` / `false_negative` / `true_unverified` / `unverified`.
`false_negative` rows are defects the panel is *expected* to find — they drive recall.
