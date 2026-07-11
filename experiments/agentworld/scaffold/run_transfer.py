#!/usr/bin/env python3
"""Phase 2 of the scaffold-transfer probe: measure transfer in the REAL sandbox.

Runs a claw task twice through the real Docker sandbox + grader:
  - baseline arm   : no scaffold (system_prompt_prefix = null)
  - treatment arm  : the sim-matured scaffold injected as system_prompt_prefix

Both arms use the same policy (gateway `protolabs/smart` = Ornith) and the immutable real sandbox as
verifier — AgentWorld touches nothing here. Reports per-trial + mean task_score and pass-rate so we
can see whether a scaffold learned against a hallucinated environment raises real performance.

Per-task claw variance is high (~±0.4), so use several trials per arm before believing a delta.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE_CONFIG = HERE / "config_gateway.yaml"


def make_config(scaffold: str | None, tag: str) -> Path:
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg["model"]["system_prompt_prefix"] = scaffold
    out = HERE / f"_config_{tag}.yaml"
    out.write_text(yaml.safe_dump(cfg, default_flow_style=False))
    return out


def run_arm(task_dir: Path, config: Path, trials: int, port_offset: int, trace_dir: Path,
            image: str) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "claw_eval.cli", "run",
        "--task", str(task_dir),
        "--config", str(config),
        "--trials", str(trials),
        "--trace-dir", str(trace_dir),
        "--sandbox", "--sandbox-image", image,
        "--port-offset", str(port_offset),
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)


def collect_scores(trace_dir: Path, task_id: str) -> list[dict]:
    out = []
    for f in sorted(trace_dir.rglob(f"{task_id}*.jsonl")):
        gr = None
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "grading_result":
                gr = r
        if gr:
            out.append({"trace": f.name, "task_score": gr["task_score"], "passed": gr["passed"],
                        "completion": gr["scores"].get("completion")})
    return out


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows) or 1
    s = {
        "arm": name, "trials": len(rows),
        "mean_task_score": round(sum(r["task_score"] for r in rows) / n, 3),
        "mean_completion": round(sum((r["completion"] or 0) for r in rows) / n, 3),
        "pass_rate": round(sum(r["passed"] for r in rows) / n, 3),
        "scores": [r["task_score"] for r in rows],
    }
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", required=True, type=Path)
    ap.add_argument("--sim-scaffold", required=True, type=Path, help="sim-practiced scaffold .md")
    ap.add_argument("--cold-scaffold", type=Path, default=None,
                    help="placebo: model-authored scaffold w/o practice (the control arm)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--image", default="claw-agent")
    ap.add_argument("--port-offset", type=int, default=400)
    ap.add_argument("--out", type=Path, default=HERE / "results_transfer.json")
    args = ap.parse_args()

    task_id = yaml.safe_load((args.task_dir / "task.yaml").read_text())["task_id"]

    # arms: (name, scaffold_text_or_None). cold inserted between baseline and sim as the placebo.
    arm_specs = [("baseline", None)]
    if args.cold_scaffold:
        arm_specs.append(("cold", args.cold_scaffold.read_text().strip()))
    arm_specs.append(("sim", args.sim_scaffold.read_text().strip()))

    results = {}
    for i, (name, scaffold) in enumerate(arm_specs):
        print(f"\n=== {name} arm: {args.trials} trial(s) in real sandbox ===", flush=True)
        cfg = make_config(scaffold, f"{task_id}_{name}")
        tdir = HERE / "traces" / task_id / name
        run_arm(args.task_dir, cfg, args.trials, args.port_offset + i * 40, tdir, args.image)
        results[name] = summarize(name, collect_scores(tdir, task_id))

    print(f"\n=== SCAFFOLD-TRANSFER RESULT [{task_id}] ===")
    for name, _ in arm_specs:
        s = results[name]
        print(f"  {name:<9} mean_task_score={s['mean_task_score']} "
              f"completion={s['mean_completion']} pass_rate={s['pass_rate']} "
              f"trials={s['trials']} scores={s['scores']}")
    base = results["baseline"]["mean_task_score"]
    out = {"task_id": task_id, "results": results,
           "delta_sim_vs_baseline": round(results["sim"]["mean_task_score"] - base, 3)}
    if "cold" in results:
        out["delta_sim_vs_cold"] = round(results["sim"]["mean_task_score"] - results["cold"]["mean_task_score"], 3)
        out["delta_cold_vs_baseline"] = round(results["cold"]["mean_task_score"] - base, 3)
        print(f"  Δ sim−baseline={out['delta_sim_vs_baseline']:+.3f}  "
              f"Δ sim−cold={out['delta_sim_vs_cold']:+.3f}  Δ cold−baseline={out['delta_cold_vs_baseline']:+.3f}")
    else:
        print(f"  Δ sim−baseline={out['delta_sim_vs_baseline']:+.3f}")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
