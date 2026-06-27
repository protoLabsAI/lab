#!/usr/bin/env python3
"""Fetch human creative references (prompt + human-written story) for the
distance-from-human metrics. Source: Reddit WritingPrompts (human completions).
Writes data/human_refs.jsonl with {id, prompt, human}. Best-effort / resumable-ish.
"""
import os, json, re, sys
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")
from datasets import load_dataset

N = int(os.environ.get("N_REFS", "80"))
OUT = os.path.join(os.path.dirname(__file__), "data", "human_refs.jsonl")

CANDIDATES = [
    ("euclaise/writingprompts", "train", "prompt", "story"),
    ("nothingiisreal/Reddit-Dirty-And-WritingPrompts", "train", "prompt", "completion"),
]

def clean(t):
    t = re.sub(r"\[\s*(WP|EU|CW|TT|IP|RF|PI|PM|MP|OT|SP|CC)\s*\]", "", t, flags=re.I)
    t = t.replace("<newline>", "\n").replace(" 's", "'s")
    return re.sub(r"\s+\n", "\n", t).strip()

rows = []
for name, split, pf, cf in CANDIDATES:
    try:
        ds = load_dataset(name, split=split, streaming=True)
        for ex in ds:
            p, h = clean(str(ex.get(pf, ""))), clean(str(ex.get(cf, "")))
            wc = len(h.split())
            if len(p) > 15 and 150 <= wc <= 700:   # usable-length human stories
                rows.append({"id": f"wp{len(rows):03d}", "prompt": p, "human": h})
            if len(rows) >= N:
                break
        if rows:
            print(f"source={name} collected={len(rows)}")
            break
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:140]}", file=sys.stderr)

if rows:
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {OUT}")
else:
    print("NO REFS FETCHED — model-vs-model harness still works without them.", file=sys.stderr)
    sys.exit(1)
