# review-eval — Vera (QA panel) model A/B harness

Lab-side half of [protoLab#26](https://github.com/protoLabsAI/protoLab/issues/26).
Scores **Vera-on-model-X vs Vera-on-model-Y** — the deployed panel with the model as the
variable — against ground truth. Not a raw-model eval; see the #26 comment thread for why
(budget truncation, verify-pass honesty, and guard interactions only exist inside the harness).

## Two layers

| layer | runner | input | what it answers |
|---|---|---|---|
| **system** (the swap gate) | Vera panel replay mode (lands in qaEngineer / pr-reviewer-plugin) | `replay_manifest.jsonl` — fixed PR set at pinned SHAs, incl. planted defects | did switching Vera's model improve/degrade the seat |
| **model** (cheap regression signal) | SWE-PRBench static harness at `/mnt/data/review-eval/swe-prbench` | 350 public PRs, frozen context | finder-judgment recall vs published frontier numbers |

The system layer is what "should we switch models?" gates on. The model layer stays raw and
static on purpose — comparability with published numbers is the only thing static mode buys.

## Files

- `replay_manifest.jsonl` — the replay PR set. Seeded with the discriminating cases from the
  2026-07-22 dogfood; grow toward ~20 PRs. `known_defects` counts come from the protoLab#24
  ground-truth dataset (`qaEngineer:data/review-eval/`, moving to this node as it grows).
- `truth.seed.jsonl` — provisional ground-truth rows derivable from the defect issues alone
  (protoAgent#2210/#2143 evidence). The authoritative set is the #24 dataset; these rows get
  reconciled against it when it lands on this node — do not grow this file by hand-labeling.
- `score_ab.py` — scores one or more replay-run JSONs against ground truth. Emits per-model:
  precision, recall, honesty, signal-to-noise (severity split), truncation rate.
  `--self-test` runs an embedded fixture. No deps beyond stdlib.
- `SCHEMA.md` — the run-output contract replay mode must emit (the interface between the
  plugin-side runner and this scorer).

## Scoring rules (inherited from protoLab#24)

- Rows labeled `assertion_only` / `not_grounded` are **excluded from precision** — an
  assertion is not a verification.
- A finding matches a ground-truth row on `(repo, pr, file)` + line within ±10; ambiguous
  matches are reported, not silently resolved.
- `disregarded_evidence` rows count against **honesty**, not just precision — the failure is
  confirming despite refuting evidence in view, which a precision number flattens.
- Truncation is a first-class outcome: a review with `findings=[]` and reasoning tokens at
  budget is a *harness failure*, distinct from a clean pass. (The `fast` incident: 6000
  output tokens of reasoning, no final answer.)

## Variance

Panel runs are nondeterministic. A 1-finding delta on 20 PRs is noise — the gate detects
gross regressions. For close calls, re-trial only the discriminating PRs (`--trials` on the
plugin-side runner), not the whole set. No pass^3 (house rule: breadth over repetition).

## Refs

protoLab#24 (ground-truth dataset + schema) · protoLab#25 (SWE-PRBench install) ·
protoLab#26 (this harness) · pr-reviewer-plugin#34 (guard telemetry the scorer consumes) ·
protoAgent#2208 / #2210 / #2143 (planted-defect + shipped-miss cases)
