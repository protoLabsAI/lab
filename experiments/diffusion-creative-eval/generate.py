#!/usr/bin/env python3
"""Generate model completions for the creative prompts. Run once per model
(swap GPU 1 between DG and AR Gemma 4). Outputs out/gen_<label>.jsonl."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import chat, normalize_prompt

HERE = os.path.dirname(__file__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8002/v1/chat/completions")
    ap.add_argument("--model", default="local-fast")
    ap.add_argument("--label", required=True, help="e.g. dg or gemma4")
    ap.add_argument("--refs", default=os.path.join(HERE, "data", "human_refs.jsonl"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.9)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.refs)][:a.n]
    out_path = os.path.join(HERE, "out", f"gen_{a.label}.jsonl")
    n_ok = tps_sum = 0
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            prompt = normalize_prompt(r["prompt"])
            instr = (f"Write a short story (~350 words) responding to this writing prompt. "
                     f"Prose only, no preamble.\n\nPrompt: {prompt}")
            txt, ctok, dt = chat(a.endpoint, a.model, instr,
                                 max_tokens=a.max_tokens, temperature=a.temperature)
            if len(txt.split()) < 30:   # DG short-collapse guard: one retry
                txt, ctok, dt = chat(a.endpoint, a.model, instr,
                                     max_tokens=a.max_tokens, temperature=a.temperature)
            tps = ctok / dt if dt else 0
            n_ok += len(txt.split()) >= 30; tps_sum += tps
            f.write(json.dumps({"id": r["id"], "prompt": prompt, "output": txt,
                                "ctok": ctok, "tok_s": round(tps, 1)}) + "\n")
            print(f"  [{i+1}/{len(rows)}] {r['id']} {len(txt.split())}w {tps:.0f}t/s", flush=True)
    print(f"\n{a.label}: {n_ok}/{len(rows)} usable, mean {tps_sum/max(len(rows),1):.0f} tok/s -> {out_path}")

if __name__ == "__main__":
    main()
