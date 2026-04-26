# Delta Review — 2026-04-26

Changes from original PLAN.md based on current ORBIS state.

## Taxonomy update (7 → 5 classes for v0)

`command_system` removed — ORBIS has no system commands (no timer,
no weather). The actual tool surface is orb-visual controls +
`delegate_to` + personality. Merged `delegate_code` and
`delegate_search` into a single `delegate` class since both route
to `delegate_to()` with different targets; the classifier doesn't
need to pick the delegate — the LLM does that after routing.

| Class | Route to | Examples |
|---|---|---|
| `chat` | LLM (no tools) | "I'm tired", "tell me a joke", "how was your day" |
| `command` | direct tool dispatch | "be warmer", "set palette to ocean", "save this as cozy" |
| `delegate` | delegate_to + LLM | "debug this Python", "search for X", "research Y" |
| `memory` | retrieval + LLM | "remember when I said X", "what did we talk about" |
| `meta` | LLM (fallback) | ambiguous, multi-intent, greetings, meta-conversation |

5 classes. Lower confidence threshold (>0.85) for bypass since
fewer classes means less ambiguity. `meta` is the catch-all — when
max softmax < 0.85, route to `meta`.

## Architecture choice: sentence-transformers + linear head

`all-MiniLM-L6-v2` is already downloaded at
`/mnt/models/huggingface/hub/`. 22M params, 384-dim, ~10ms CPU.
Linear head adds negligible params. This is v0 — if it underperforms,
DistilBERT fine-tune is v1.

## Data strategy

No ORBIS conversation logs exist anywhere on disk. No Langfuse
traces verified. 100% synthetic bootstrap:

1. Generate 300 examples/class (1500 total) via Qwen 3.6-27B on
   localhost:8000 (or via ava:4000 gateway).
2. 80/20 train/test split.
3. Manual spot-check of 50 random examples before training.

## Tier-0 baselines

Per lab rules: majority class + random + linear probe on
sentence-transformer embeddings (no fine-tuning). These run before
any model training.

## Tool-need-predictor collapses into cascade

If intent is `command` or `delegate` or `memory` → needs_tool=True.
If intent is `chat` or `meta` → needs_tool=False.
No separate model needed for v0. Revisit if edge cases warrant it.
