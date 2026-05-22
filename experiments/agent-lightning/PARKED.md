# PARKED — agent-lightning

Parked 2026-05-22. APO target (protoResearcher SOUL.md) is not on the beach head; studio is out of the coding-agent and prompt-product game. See [project_brand_pivot.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md).

## What this was

~200-line direct re-implementation of Microsoft Agent Lightning's APO loop (beam search + textual-gradient prompt edits) for protoResearcher's system prompt. Framework was unusable as-shipped (DummyTracer / OtelTracer / SharedMemory all broken for our case); we kept the algorithm, dropped the plumbing.

## Where it stood

- `run_apo.py`, `run_multi_apo.py`, `search_combos.py` working
- `train_dpo.py`, `train_grpo.py`, `eval_dpo.py` scaffolded
- See [project_agent_lightning_plan.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_agent_lightning_plan.md): plan was APO → DPO → online GRPO, gated on Harbor mutation-ops import (see [project_harbor_investigation.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_harbor_investigation.md))

## Why parked

P7 (patterns over products) and the §3 audience filter cut against shipping a prompt-optimizer-as-product. The audience already knows how to write APO. *The finding* — that Agent Lightning's plumbing fights its own algorithm and 200 lines wins — is the brand piece, not the codebase.

## How to resume

Don't resume the optimizer. Write the breakdown: *"Why our 200-line APO beat Microsoft's framework."* The diff against `microsoft/agent-lightning` is the post.
