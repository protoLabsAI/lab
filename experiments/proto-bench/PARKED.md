# PARKED — proto-bench

Parked 2026-05-22. Coding-agent benchmark infrastructure; studio out of the coding-agent game. See [project_brand_pivot.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md).

## What this was

Harness for evaluating + optimizing protoCLI on SWE-bench / Terminal Bench / Aider Polyglot. APO loop (critique → edit → evaluate). Modeled after Microsoft Agent Lightning's `examples/claude_code/`. Lived alongside `experiments/agent-lightning/` and shared its APO algorithm.

## What survives the pivot

`BENCHMARKS.md` — the research digest covering SWE-bench Pro vs LiveCodeBench vs Aider Polyglot vs RustEvo² vs RepoMaster + the ~53-point public-to-private contamination story. **Lift this into the leaderboard experiment's `REFERENCES.md` before deleting.** It's reusable as related-work for any benchmark publication.

## How to resume

Don't, here. If a non-coding benchmark experiment ever needs the APO harness, port the algorithm (not this dir). The `proto_agent.py` SWE-bench/Terminal-Bench bindings are coding-specific and dead-weight under the new direction.
