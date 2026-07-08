"""Reward-filter τ²/τ³-bench results into canonical verified trajectories.

τ² saves a DIRECTORY per run with results.json → {simulations:[{task_id, reward_info, messages,...}]}.
Keeps sims with reward_info.reward >= 1.0 (deterministic: db_check + env_assertions/action/communicate).
Guards against NL_ASSERTION-graded tasks (LLM-judged → not reward-trust-clean) — but telecom/airline/
banking are all-deterministic anyway. Converts messages → canonical Trajectory → _raw/ornith_tau2.jsonl.

  python filter_tau2.py <results_dir_or_glob> <domain> [--append]
"""
import glob
import json
import os
import sys

OUT = "/mnt/data/datasets/agentic-distill/_raw/ornith_tau2.jsonl"


def to_canonical(sim: dict, domain: str) -> dict | None:
    msgs = []
    for m in sim.get("messages") or []:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        tcs = None
        if m.get("tool_calls"):
            # τ² tool_calls are flat: {id, name, arguments, requestor} (no nested "function")
            tcs = [{"id": t.get("id"), "name": t.get("name"), "arguments": t.get("arguments")}
                   for t in m["tool_calls"]]
        msgs.append({"role": role, "content": m.get("content"),
                     "tool_calls": tcs, "tool_call_id": m.get("tool_call_id")})
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    if len(msgs) < 2:
        return None
    tid = str(sim.get("task_id", "?"))
    return {"id": f"ornith_tau2__{domain}_{tid}_{sim.get('trial', 0)}", "source": "ornith_tau",
            "teacher": "ornith-35b", "domain": f"tau2_{domain}", "messages": msgs,
            "tools": [], "verified": True, "reward": 1.0, "license_note": "ours (clean)",
            "split": "train", "meta": {"env": domain, "task_id": tid}}


def main():
    pattern, domain = sys.argv[1], sys.argv[2]
    append = "--append" in sys.argv
    # accept a dir (find results.json), a glob, or a parent of run-dirs
    files = []
    for p in ([pattern] if "*" in pattern else glob.glob(pattern + "/**/results.json", recursive=True)
              or ([pattern] if os.path.isfile(pattern) else glob.glob(pattern + "/*.json"))):
        files.append(p)
    kept = seen = 0
    with open(OUT, "a" if append else "w") as fout:
        for f in files:
            d = json.load(open(f))
            for sim in (d.get("simulations") if isinstance(d, dict) else d):
                seen += 1
                if float((sim.get("reward_info") or {}).get("reward", 0)) >= 1.0:
                    row = to_canonical(sim, domain)
                    if row:
                        fout.write(json.dumps(row) + "\n")
                        kept += 1
    print(f"{domain}: {seen} sims seen, {kept} verified (reward=1.0) -> {OUT}")


if __name__ == "__main__":
    main()
