# agentic-distill — our personal, reusable agentic corpus

One canonical trajectory schema; every source normalizes into it. Student-agnostic and
trainer-agnostic — reuse it for Qwen-2B, the 0.8B, future 9B distills, anything.

## Layout
    schema.py      canonical Trajectory / Message / ToolCall (pydantic, reusable)
    sources.yaml   source manifest — license is metadata, `teacher` drives ablations
    adapters.py    per-source raw->Trajectory normalizers (register + verify columns)
    build.py       stream sources -> normalize -> dedup -> contamination-filter -> versioned JSONL
    ../PILOT.md    the mixing-ratio pilot spec
    ../DATASETS.md the grounded census this manifest is drawn from

## Use
    # build a blended corpus (v0.1) to the datasets drive
    python build.py --version v0.1 --out /mnt/data/datasets/agentic-distill --mix blend
    # pilot arms
    python build.py --mix public --cap-per-source 2000 --version pilot-A
    python build.py --mix ornith --version pilot-B     # needs _raw/ornith_tau.jsonl first

Output: `<out>/<version>/train.jsonl` (canonical rows) + `manifest.json` (revisions + counts).

## Extend
Add a source: block in `sources.yaml` + adapter in `adapters.py`. **Verify the real card's
columns before enabling** — unverified adapters raise by design (grounding rule). ShareGPT/OpenAI
shapes reuse the shared normalizers; bespoke shapes get their own function.

## Storage
Primary: `/mnt/data/datasets/agentic-distill/` (datasets drive, per CLAUDE.md). Raw locally-generated
shards (Ornith rollouts): `_raw/<source>.jsonl`. Optional private HF mirror:
`hf upload protoLabsAI/agentic-distill <dir> --repo-type dataset --private`.

## Not a gate
We train on any source (license is metadata). The only real line: don't re-host a non-commercial
dataset's rows verbatim under our name — train a model on it and publish the *model*. See ../DATASETS.md.
