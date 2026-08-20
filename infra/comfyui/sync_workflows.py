#!/usr/bin/env python3
"""Sync ComfyUI UI workflows into the repo as runnable API-format JSON.

The contract: **the UI graph is the source, the API JSON is generated.** You author and
refine in the ComfyUI editor; this pulls the flattened, API-format version of that graph
out of the server and commits it next to the code that drives it.

It exports from the server's run **history** rather than converting the UI file locally.
That is deliberate:

  * The server has already done the flattening. A UI graph is not the graph that runs —
    subgraphs expand (the LTX-2.5 templates go 7 UI nodes -> 42 API nodes), muted and
    bypassed nodes drop out, primitives fold into their consumers. Re-implementing that
    is a second, silently-drifting copy of the frontend's logic.
  * Only a graph that ran can be exported, so an exported API JSON is always one that
    executed. A converter would happily emit graphs that have never worked.

The cost: **history lives in memory and is lost when ComfyUI restarts.** Export after a
run, not next week. `list` shows what is currently exportable.

Going the other way needs no tool — the ComfyUI frontend detects API-format JSON on load
(`isApiJson`/`loadApiJson`), so any file in workflows/ can be dragged straight into the
editor, arranged, and saved as a named UI workflow.

Usage:
  python sync_workflows.py list
  python sync_workflows.py export "MiniMax H3 - Director"     # by workflow name
  python sync_workflows.py export --all
  python sync_workflows.py run workflows/h3-director.api.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("COMFY_HOST", "http://127.0.0.1:8188")
UI_DIR = os.path.expanduser("~/dev/ComfyUI/user/default/workflows")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")


def _get(path: str, timeout: int = 30):
    with urllib.request.urlopen(HOST + path, timeout=timeout) as r:
        return json.loads(r.read())


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "workflow"


def signature(ui_graph: dict) -> frozenset:
    """Identity of a UI graph: its (node id, node type) pairs.

    Workflow `id` alone is not unique — ComfyUI's shipped templates hand the same id to
    every variant derived from them (all three MiniMax H3 templates share one), so a
    T2V export would happily overwrite the I2V one.
    """
    return frozenset((str(n.get("id")), n.get("type")) for n in ui_graph.get("nodes", []))


def ui_workflows() -> dict[str, dict]:
    out = {}
    if not os.path.isdir(UI_DIR):
        return out
    for fn in sorted(os.listdir(UI_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(UI_DIR, fn)) as f:
                out[fn[:-5]] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping {fn}: {exc}", file=sys.stderr)
    return out


def runs_by_signature() -> dict[frozenset, list[dict]]:
    """Successful history entries that carry their UI graph, newest first per signature."""
    hist = _get("/history?max_items=500")
    found: dict[frozenset, list[dict]] = {}
    for pid, entry in hist.items():
        if entry.get("status", {}).get("status_str") != "success":
            continue
        prompt = entry.get("prompt") or []
        if len(prompt) < 4 or not isinstance(prompt[3], dict):
            continue
        ui = prompt[3].get("extra_pnginfo", {}).get("workflow")
        if not ui:  # queued through the API, not the editor — nothing to attribute it to
            continue
        found.setdefault(signature(ui), []).append(
            {"prompt_id": pid, "api": prompt[2], "created": prompt[3].get("create_time", 0)}
        )
    for runs in found.values():
        runs.sort(key=lambda r: r["created"], reverse=True)
    return found


def cmd_list(_args):
    wfs, runs = ui_workflows(), runs_by_signature()
    print(f"{'workflow':44s} {'nodes':>5}  {'runs':>4}  exported")
    print("-" * 78)
    for name, graph in wfs.items():
        matched = runs.get(signature(graph), [])
        dest = os.path.join(OUT_DIR, slug(name) + ".api.json")
        mark = "yes" if os.path.exists(dest) else ("ready" if matched else "-")
        print(f"{name[:44]:44s} {len(graph.get('nodes', [])):5d}  {len(matched):4d}  {mark}")
    print(
        "\nruns = successful editor runs still in the server's in-memory history."
        "\n0 runs means: run it once from the editor, then export (history clears on restart)."
    )


def cmd_export(args):
    wfs, runs = ui_workflows(), runs_by_signature()
    names = list(wfs) if args.all else args.name
    if not names:
        print("nothing selected — pass a workflow name or --all", file=sys.stderr)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    rc = 0
    for name in names:
        if name not in wfs:
            print(f"  ! no such workflow: {name}", file=sys.stderr)
            rc = 1
            continue
        matched = runs.get(signature(wfs[name]), [])
        if not matched:
            print(f"  - {name}: no successful editor run in history — run it once, then export")
            if not args.all:
                rc = 1
            continue
        best = matched[0]
        dest = os.path.join(OUT_DIR, slug(name) + ".api.json")
        with open(dest, "w") as f:
            json.dump(best["api"], f, indent=1)
            f.write("\n")
        with open(dest[: -len(".api.json")] + ".meta.json", "w") as f:
            json.dump(
                {
                    "workflow": name,
                    "prompt_id": best["prompt_id"],
                    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "api_nodes": len(best["api"]),
                    "ui_nodes": len(wfs[name].get("nodes", [])),
                },
                f,
                indent=1,
            )
            f.write("\n")
        print(f"  + {name} -> {os.path.relpath(dest)}  ({len(best['api'])} api nodes)")
    return rc


def submit_and_wait(graph: dict, timeout: int = 3600, poll: float = 5.0):
    body = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=body, headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read()).get("prompt_id")
    except urllib.error.HTTPError as e:
        return None, f"rejected {e.code}: {e.read().decode()[:600]}"
    started = time.time()
    while time.time() - started < timeout:
        time.sleep(poll)
        hist = _get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            files = [
                f.get("filename")
                for out in entry.get("outputs", {}).values()
                for key in ("images", "videos", "audio", "gifs")
                for f in out.get(key, [])
            ]
            status = entry.get("status", {})
            if status.get("status_str") != "success":
                msgs = json.dumps(status.get("messages", []))[:600]
                return None, f"run failed: {msgs}"
            return {"prompt_id": pid, "elapsed": round(time.time() - started, 1), "files": files}, None
    return None, f"timed out after {timeout}s (prompt {pid})"


def free_models(verbose: bool = True) -> None:
    """Unload models and release memory before a big run.

    ComfyUI stages checkpoints in *pinned host RAM* for DynamicVRAM and does not give it
    back between runs — measured 2026-08-19: 41.3 GB RSS still held after one H3 render on
    a 61 GB box, which is how a later multi-checkpoint graph got OOM-killed by the kernel
    while GPU0 still had ~35 GB of VRAM free. This POST drops it back to ~4.5 GB.
    """
    body = json.dumps({"unload_models": True, "free_memory": True}).encode()
    req = urllib.request.Request(HOST + "/free", data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
        if verbose:
            print("freed cached models before run")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"  ! /free failed ({exc}) — continuing", file=sys.stderr)


def cmd_run(args):
    with open(args.path) as f:
        graph = json.load(f)
    if not args.no_free:
        free_models()
    print(f"submitting {args.path} ({len(graph)} nodes) -> {HOST}")
    res, err = submit_and_wait(graph, timeout=args.timeout)
    if err:
        print("FAILED:", err, file=sys.stderr)
        return 1
    print(f"ok in {res['elapsed']}s")
    for f in res["files"]:
        print("  ", f)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="UI workflows and whether each is exportable").set_defaults(fn=cmd_list)
    e = sub.add_parser("export", help="write the API-format graph of a workflow's last good run")
    e.add_argument("name", nargs="*")
    e.add_argument("--all", action="store_true")
    e.set_defaults(fn=cmd_export)
    r = sub.add_parser("run", help="submit an API-format graph and wait")
    r.add_argument("path")
    r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--no-free", action="store_true",
                   help="skip the POST /free that unloads cached models first (keeps a warm model loaded)")
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
