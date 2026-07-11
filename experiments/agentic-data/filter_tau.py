"""Reward-filter tau-bench results into canonical verified trajectories.

Reads tau-bench's results/*.json (each entry has `traj` = agent messages + `reward`
from the deterministic DB-state check), keeps only reward==1.0 (verified success),
converts to canonical Trajectory rows -> _raw/ornith_tau.jsonl for build.py to ingest.

  python filter_tau.py <results_dir_or_glob> <env_domain> [--append]
"""
import glob
import json
import sys

OUT = "/mnt/data/datasets/agentic-distill/_raw/ornith_tau.jsonl"


def to_canonical(entry: dict, domain: str) -> dict | None:
    traj = entry.get("traj") or []
    if not traj:
        return None
    msgs = []
    for m in traj:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        tcs = None
        if m.get("tool_calls"):
            tcs = [{"id": t.get("id"), "name": t["function"]["name"],
                    "arguments": t["function"]["arguments"]} for t in m["tool_calls"]]
        msgs.append({"role": role, "content": m.get("content"),
                     "tool_calls": tcs, "tool_call_id": m.get("tool_call_id")})
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    if len(msgs) < 2:
        return None
    tid = entry.get("task_id", "?")
    return {"id": f"ornith_tau__{domain}_{tid}", "source": "ornith_tau",
            "teacher": "ornith-35b", "domain": domain, "messages": msgs,
            "tools": [], "verified": True, "reward": 1.0,
            "license_note": "ours (clean)", "split": "train",
            "meta": {"env": domain, "task_id": tid}}


def main():
    pattern, domain = sys.argv[1], sys.argv[2]
    append = "--append" in sys.argv
    files = glob.glob(pattern) if "*" in pattern else glob.glob(pattern + "/*.json")
    kept = seen = 0
    mode = "a" if append else "w"
    with open(OUT, mode) as fout:
        for f in files:
            data = json.load(open(f))
            for entry in data:
                seen += 1
                if float(entry.get("reward", 0)) >= 1.0:
                    row = to_canonical(entry, domain)
                    if row:
                        fout.write(json.dumps(row) + "\n")
                        kept += 1
    print(f"{domain}: {seen} trajectories seen, {kept} verified (reward=1.0) -> {OUT}")


if __name__ == "__main__":
    main()
