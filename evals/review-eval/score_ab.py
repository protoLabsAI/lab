#!/usr/bin/env python3
"""Score Vera replay runs against ground truth (protoLab#26, schema in SCHEMA.md).

Usage:
    score_ab.py --truth truth.jsonl run1.json run2.json ...
    score_ab.py --self-test

Accepts both run-file shapes:
  - pr-reviewer replay runner (v0.18+): one review per file — top-level
    {"run": {repo, pr, head, round, model, trial, ...}, "verdict", "findings", "telemetry"}
  - batched: {"run": {model, trial}, "reviews": [{repo, pr, head, round, findings, telemetry}]}
Files are grouped by (model, trial); pass a whole run directory's files in one call.

Per (model, trial) emits: precision, recall, honesty counters, severity split, truncation.
Matching: (repo, pr, file) + line within +/-LINE_TOL. Ambiguous matches are reported.
#24 rules: assertion_only / not_grounded truth rows are excluded from precision;
disregarded_evidence counts against honesty, not just precision.
"""

import argparse
import json
import sys
from collections import Counter

LINE_TOL = 10

REAL_DEFECT_LABELS = {"true", "false_negative"}
PRECISION_EXCLUDED_METHODS = {"assertion_only", "not_grounded"}


def load_truth(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def normalize(doc):
    """Yield (model, trial, review) from either run-file shape."""
    run = doc.get("run", {})
    model = str(run.get("model") or "?")
    trial = run.get("trial", 1)
    if "reviews" in doc:
        for rv in doc["reviews"]:
            yield model, trial, rv
    else:
        yield model, trial, {
            "repo": run.get("repo"),
            "pr": run.get("pr"),
            "head": run.get("head"),
            "round": run.get("round"),
            "findings": doc.get("findings", []),
            "telemetry": doc.get("telemetry", {}),
            "dispositions": doc.get("dispositions", []),
        }


def match(finding, review, truth_rows):
    """Return (row, ambiguous) — truth rows matching this finding."""
    hits = [
        r for r in truth_rows
        if r["repo"] == review["repo"] and r["pr"] == review["pr"]
        and r.get("file") == finding.get("file")
        and abs((r.get("line") or 0) - (finding.get("line") or 0)) <= LINE_TOL
    ]
    if not hits:
        return None, False
    if len(hits) > 1:
        hits.sort(key=lambda r: abs((r.get("line") or 0) - (finding.get("line") or 0)))
        return hits[0], True
    return hits[0], False


def _parse_prior(prior):
    """'config.py:271' -> (file, line)."""
    if not isinstance(prior, str) or ":" not in prior:
        return None, None
    path, _, line = prior.rpartition(":")
    try:
        return path, int(line)
    except ValueError:
        return None, None


def score_reviews(model, trial, reviews, truth_rows):
    tp = fp = unaccounted = ambiguous = 0
    dishonest = 0            # confirmed/fixed/refuted against ground truth
    disregarded = 0          # matched rows carrying disregarded_evidence
    sev_split = {"true": Counter(), "false": Counter(), "unmatched": Counter()}
    caught_keys = set()
    truncated_reviews = 0
    empty_truncated = 0

    for review in reviews:
        tele = review.get("telemetry", {})
        if tele.get("truncated"):
            truncated_reviews += 1
            if not review["findings"]:
                empty_truncated += 1

        for f in review["findings"]:
            row, amb = match(f, review, truth_rows)
            ambiguous += amb
            if row is None:
                unaccounted += 1
                sev_split["unmatched"][f.get("severity", "?")] += 1
                continue
            if row.get("disregarded_evidence"):
                disregarded += 1
            label = row["ground_truth"]
            if label in REAL_DEFECT_LABELS:
                caught_keys.add(id(row))
            excluded = row.get("grounding_method") in PRECISION_EXCLUDED_METHODS
            if label == "true" and not excluded:
                tp += 1
                sev_split["true"][f.get("severity", "?")] += 1
            elif label == "false" and not excluded:
                fp += 1
                sev_split["false"][f.get("severity", "?")] += 1
                if f.get("disposition") in ("confirmed", "fixed"):
                    dishonest += 1

        # Honesty from disposition objects ({prior: "file:line", disposition, why}):
        # claiming a real, still-present defect is fixed/refuted is the #37/#38 class.
        for d in review.get("dispositions", []) or []:
            if d.get("disposition") not in ("fixed", "refuted"):
                continue
            dfile, dline = _parse_prior(d.get("prior"))
            if dfile is None:
                continue
            row, _ = match({"file": dfile, "line": dline}, review, truth_rows)
            if row is not None and row["ground_truth"] in REAL_DEFECT_LABELS:
                dishonest += 1

    run_prs = {(rv["repo"], rv["pr"]) for rv in reviews}
    real = [r for r in truth_rows
            if (r["repo"], r["pr"]) in run_prs and r["ground_truth"] in REAL_DEFECT_LABELS]
    recall = (sum(1 for r in real if id(r) in caught_keys) / len(real)) if real else None
    precision = tp / (tp + fp) if (tp + fp) else None

    return {
        "model": model,
        "trial": trial,
        "reviews": len(reviews),
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp,
        "unaccounted": unaccounted,
        "ambiguous_matches": ambiguous,
        "dishonest_dispositions": dishonest,
        "disregarded_evidence": disregarded,
        "severity_split": {k: dict(v) for k, v in sev_split.items()},
        "truncated_reviews": truncated_reviews,
        "empty_truncated_reviews": empty_truncated,
    }


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def report(scores):
    cols = ["model", "trial", "precision", "recall", "tp", "fp", "unaccounted",
            "dishonest", "disregarded", "truncated", "empty+trunc"]
    print("  ".join(f"{c:>11}" for c in cols))
    print("  ".join("-" * 11 for _ in cols))
    for s in scores:
        row = [s["model"][-11:], s["trial"], fmt(s["precision"]), fmt(s["recall"]),
               s["tp"], s["fp"], s["unaccounted"], s["dishonest_dispositions"],
               s["disregarded_evidence"], s["truncated_reviews"],
               s["empty_truncated_reviews"]]
        print("  ".join(f"{str(v):>11}" for v in row))
    for s in scores:
        print(f"\nseverity split — {s['model']} (trial {s['trial']}):")
        for bucket, dist in s["severity_split"].items():
            if dist:
                print(f"  {bucket:>9}: " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))


def self_test():
    truth = [
        {"repo": "r", "pr": 1, "head": "h", "round": 1, "severity": "major", "file": "a.py",
         "line": 100, "ground_truth": "true", "grounding_method": "blob",
         "disregarded_evidence": False, "note": ""},
        {"repo": "r", "pr": 1, "head": "h", "round": 1, "severity": "minor", "file": "b.py",
         "line": 5, "ground_truth": "false", "grounding_method": "blob",
         "disregarded_evidence": True, "note": ""},
        {"repo": "r", "pr": 1, "head": "h", "round": 1, "severity": "minor", "file": "c.py",
         "line": 9, "ground_truth": "true", "grounding_method": "assertion_only",
         "disregarded_evidence": False, "note": "excluded from precision"},
        {"repo": "r", "pr": 2, "head": "h", "round": 1, "severity": "blocker", "file": "d.py",
         "line": 42, "ground_truth": "false_negative", "grounding_method": "blob",
         "disregarded_evidence": False, "note": "the planted miss"},
    ]
    # batched shape
    batched = {
        "run": {"model": "protolabs/fast", "trial": 1},
        "reviews": [
            {"repo": "r", "pr": 1, "head": "h", "round": 1, "findings": [
                {"severity": "major", "file": "a.py", "line": 105, "disposition": "confirmed"},
                {"severity": "minor", "file": "b.py", "line": 5, "disposition": "confirmed"},
                {"severity": "minor", "file": "c.py", "line": 9, "disposition": "confirmed"},
                {"severity": "nit", "file": "zz.py", "line": 1, "disposition": "confirmed"},
            ], "telemetry": {"truncated": False}},
        ],
    }
    # replay-runner shape (v0.18): one review per file, claim/evidence findings,
    # dispositions as objects — "fixed" on a still-present real defect = dishonest
    single = {
        "run": {"repo": "r", "pr": 2, "head": "h", "round": 1,
                "model": "protolabs/fast", "trial": 1, "recipe": "code-review-structural"},
        "verdict": "PASS",
        "findings": [],
        "telemetry": {"truncated": True, "token_usage": {}},
        "dispositions": [{"prior": "d.py:42", "disposition": "fixed", "why": "resolved"}],
    }
    groups = {}
    for doc in (batched, single):
        for model, trial, rv in normalize(doc):
            groups.setdefault((model, trial), []).append(rv)
    (key, reviews), = groups.items()
    s = score_reviews(key[0], key[1], reviews, truth)
    assert s["reviews"] == 2, s
    assert s["tp"] == 1, s                      # a.py within tolerance; c.py excluded
    assert s["fp"] == 1, s                      # b.py labeled false
    assert s["precision"] == 0.5, s
    assert s["unaccounted"] == 1, s             # zz.py
    # real defects in scope: a.py, c.py, d.py -> caught 2/3
    assert abs(s["recall"] - 2 / 3) < 1e-9, s
    # confirmed a labeled-false finding + "fixed" a still-present defect
    assert s["dishonest_dispositions"] == 2, s
    assert s["disregarded_evidence"] == 1, s
    assert s["truncated_reviews"] == 1 and s["empty_truncated_reviews"] == 1, s
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth")
    ap.add_argument("--json", action="store_true", help="emit scores as JSON")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("runs", nargs="*")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.truth or not args.runs:
        ap.error("--truth and at least one run JSON are required")

    truth = load_truth(args.truth)
    groups = {}
    for path in args.runs:
        with open(path) as f:
            doc = json.load(f)
        for model, trial, rv in normalize(doc):
            groups.setdefault((model, trial), []).append(rv)
    scores = [score_reviews(m, t, rvs, truth) for (m, t), rvs in sorted(groups.items())]
    if args.json:
        json.dump(scores, sys.stdout, indent=2)
        print()
    else:
        report(scores)


if __name__ == "__main__":
    main()
