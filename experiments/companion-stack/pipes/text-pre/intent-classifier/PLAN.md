# intent-classifier — text-pre routing

**Pipe**: text-pre.
**Status**: planned (Phase 2).

## Problem

ORBIS today lets the LLM figure out what the user wants on every
turn, including for trivial routings:

- "Set a timer for 15 minutes" — pure command, no reasoning needed
- "What time is it" — pure command, delegate target obvious
- "I'm thinking about my mom" — pure chat, no tools
- "Can you help me debug this Python error" — clear delegate

Doing all that via the LLM means a function-calling round-trip on
every turn, plus the LLM sometimes guesses wrong (calls a tool
when the user is just chatting).

## Why ORBIS needs it specifically

- The LLM is the slowest piece of the loop. Skipping it for ~30%
  of turns is a step-change in perceived responsiveness.
- ORBIS already has a curated tool surface (`delegate_to`,
  `set_variant`, `apply_palette`, `adjust_param`,
  `save_preset`/`recall_preset`). A bounded classifier maps
  cleanly to it.
- Companion-layer integrity benefits: guard against "the LLM made up
  a tool call that touched personality drift."

## Target taxonomy (v0)

| Class | Route to | Examples |
|---|---|---|
| `chat` | LLM (no tools) | "I'm tired today", "tell me a joke" |
| `command_orb` | direct tool, no LLM | "be warmer", "set palette to ocean" |
| `command_system` | direct tool, no LLM | "set a timer", "what's the weather" |
| `delegate_code` | delegate_to coding agent | "debug this Python", "write a function" |
| `delegate_search` | delegate_to search/RAG | "what's the latest on X" |
| `memory_query` | retrieval + LLM | "remember when I said X" |
| `meta_or_unsure` | LLM (graceful fallback) | ambiguous, multi-intent, etc. |

7-class classifier. Confidence threshold for `meta_or_unsure` —
when no class scores >0.7, fall back to LLM routing.

## Candidate architectures

1. **Sentence-transformer + linear probe** (ship in a week)
   - `sentence-transformers/all-MiniLM-L6-v2` (22 M params)
   - 384-dim embedding → Linear(384, 7)
   - Inference: ~10 ms CPU, <2 ms GPU.
2. **DistilBERT fine-tune** (better, more data)
   - 66 M params
   - Standard sequence-classification head
   - Inference: ~30 ms CPU, ~5 ms GPU.
3. **SetFit** — for low-data efficiency
   - Few-shot from ~50 examples per class via contrastive fine-
     tune of a sentence-transformer.

Start with (1) for a Tier-0 baseline, then (2) if needed.

## Datasets

- **Synthetic generation via LLM** — bootstrap 200-500 utterances
  per class with Qwen 3.6-27B; manually review.
- **ORBIS conversation logs** (when we have them) — real labels by
  hand-curation.
- **MASSIVE / SLURP** — public NLU benchmark datasets for sanity-
  check baselines.

## Eval plan

1. **Macro F1** across all 7 classes on a held-out manually-labeled
   set of ~200 utterances.
2. **Confidence calibration** — when classifier says >0.9, it
   should be right >90% of the time.
3. **Comparison**: majority class, MiniLM linear probe, DistilBERT,
   off-the-shelf MASSIVE-trained classifier.
4. **Latency budget**: < 30 ms CPU end-to-end (including
   tokenization).

## Deliverables

- HF model: `protoLabsAI/orbis-intent-classifier-v0`.
- Pipecat frame processor: `voice/agent/intent_router.py`.
- ORBIS integration: pre-LLM hook that, on high-confidence
  prediction, dispatches directly to a tool / delegate, skipping
  the LLM entirely.
- Blog post: "Stop asking the LLM what it should be doing."

## Open questions

- How aggressive should the bypass-LLM path be? Conservative default:
  bypass only on confidence >0.95 + class in `command_*` set.
- Multi-intent utterances ("set a timer and play music") — punt to
  LLM, or handle in classifier?
- Synthetic-data quality vs real-data quantity tradeoff.

## Dependencies

- None blocking; can start anytime.
- Eventually needs ORBIS conversation log data for real fine-tuning.
