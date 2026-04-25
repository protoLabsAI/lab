# tool-need-predictor — gate the function-calling round-trip

**Pipe**: llm-context.
**Status**: planned (Phase 2).

## Problem

ORBIS's LLM serves dual roles: chat and tool-orchestration. The
function-calling format (Qwen XML, OpenAI tools) requires either:

1. A second prompt with the tool schema attached (extra tokens,
   extra latency), or
2. Always-on tool schema in the system prompt (tokens always
   present, even on "I'm tired today" turns)

Either way, ~70% of conversation turns don't need any tools. We
pay for tool-calling overhead on every turn.

## Why ORBIS needs it specifically

- Pipecat's STT-LLM-TTS loop is latency-critical. Saving 200-500 ms
  on the chat-turn majority is a perceptible UX win.
- ORBIS's tool surface is small (~6 tools); the binary-tool-need
  decision is highly structured.
- Cleanly composable with the intent-classifier — when intent is
  `chat` or `meta_or_unsure`, tool-need is almost certainly false.

## Target output

Binary classifier: `needs_tool ∈ {True, False}`.

If `False`: route LLM call without tool schema in the prompt.
If `True`: route LLM call with full tool schema.

Bonus: predict *which* tool is needed (multi-class), letting us
attach only the relevant tool schema instead of all of them.

## Candidate architectures

1. **Cascade after intent-classifier** — if intent is in
   `command_*` / `delegate_*` / `memory_query`, set
   `needs_tool=True`; else False. Zero new model.
2. **Standalone binary classifier** — sentence-transformer +
   logistic regression, or DistilBERT binary head.
3. **Multi-label classifier** — outputs probability per tool.

Cascade approach is essentially free if intent-classifier ships
first; standalone is more flexible.

## Datasets

- **ORBIS function-calling traces** — once collected, the gold
  source.
- **Synthetic from LLM** — generate paired (utterance,
  expected_tool_call_or_none) data via Qwen 3.6-27B. Bootstrap
  ~1000 examples per class.
- **BFCL v4** (we already have a runner from prior eval work) —
  function-calling benchmarks for sanity.

## Eval plan

1. **Precision @ recall=0.95** — false negatives (missing a needed
   tool) hurt much more than false positives.
2. **Wall-clock latency improvement** on a real ORBIS-trace replay:
   how many tokens / how much time does the gating save?
3. **Confidence calibration** — predicted probability vs realized
   "needed tool" rate.

## Deliverables

- HF model: `protoLabsAI/orbis-tool-need-predictor-v0` (or just a
  rule-set in code if cascade wins).
- ORBIS integration: pre-LLM hook in
  `voice/agent/llm_router.py` (or wherever ORBIS shapes the LLM
  call) that strips the tool schema when `needs_tool=False`.
- Blog post: same as intent-classifier, share the writeup.

## Open questions

- Cascade vs standalone — start with cascade, ablate later?
- What's the "miss tool" cost? If ORBIS gracefully falls back to
  re-prompting with tools, the cost is low and we can be aggressive.
- Should this run before or after the intent-classifier in the
  pipeline?

## Dependencies

- Best-paired with intent-classifier. Probably ship together.
