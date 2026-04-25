# companion-stack

Research workspace for **building the best conversational AI we can**
by composing small specialized models around an LLM.

> Reference application: [ORBIS](https://github.com/protoLabsAI/ORBIS).
> Everything we build here should land in ORBIS or graduate to its
> own repo.

## Thesis

LLMs are excellent reasoning engines. They are bad perception
systems, bad signal detectors, bad routers, and prohibitively
expensive on the hot path of a real-time voice loop.

The whole win for a *companion* AI — sub-second responsiveness,
character, memory, presence — comes from a stack of **small
specialized models** sitting at every pipe of the conversational
loop, doing what they're cheap and predictable at, while the LLM
focuses on reasoning and language.

This workspace is the program of work to build that stack.

## The conversational loop, mapped to model slots

```
[mic] ──▶ audio-pre ──▶ STT ──▶ text-pre ──▶ LLM ◀── llm-context ──▶ text-post ──▶ TTS ──▶ audio-post ──▶ [speaker]
                                                │
                                                ├──▶ memory writers (async)
                                                ├──▶ mood/personality drift (async)
                                                └──▶ visual driver (orb expression)
```

Each pipe is a slot for a small specialized model. Most of them
are empty in default voice-agent stacks. Filling them is the
research program.

| Pipe | What lives here | ORBIS today | Best-case |
|---|---|---|---|
| **audio-pre** | tags / VAD / speaker ID / events | Whisper VAD only | full perceptual stack: audio-tags, speaker verification, sound-event detection |
| **text-pre** | intent / topic / NER / sentiment | none — LLM does it all | classification-led routing, deterministic ms-scale gates |
| **llm-context** | embeddings / rerank / tool-need | embedding retrieval | reranker, tool-need predictor, persona conditioner |
| **text-post** | prosody / safety / style | none | prosody tagger, PII redactor, style validator |
| **memory** | fact-worthy / coreference / decay | LLM-driven inserts | fact classifier, importance scorer, daily curator |
| **visual** | mood→palette / animation / expression | manual `apply_palette` calls | continuous mood-driven expression |

## Status

| Pipe | Experiment | Status | Artifact |
|---|---|---|---|
| audio-pre | [audio-tags](./pipes/audio-pre/audio-tags) (v0→v5) | ✅ shipped | [HF: orbis-audio-tags-v5-soft](https://huggingface.co/protoLabsAI/orbis-audio-tags-v5-soft) |
| audio-pre | [speaker-verification](./pipes/audio-pre/speaker-verification) | 📋 planned | — |
| audio-pre | [sound-event-detection](./pipes/audio-pre/sound-event-detection) | 📋 backlog | — |
| text-pre | [intent-classifier](./pipes/text-pre/intent-classifier) | 📋 planned | — |
| text-pre | [topic-router](./pipes/text-pre/topic-router) | 📋 backlog | — |
| llm-context | [tool-need-predictor](./pipes/llm-context/tool-need-predictor) | 📋 planned | — |
| llm-context | [reranker](./pipes/llm-context/reranker) | 📋 planned | — |
| text-post | [prosody-tagger](./pipes/text-post/prosody-tagger) | 📋 planned | — |
| memory | [fact-worthiness](./pipes/memory/fact-worthiness) | 📋 planned | — |
| memory | [coreference-resolver](./pipes/memory/coreference-resolver) | 📋 backlog | — |
| visual | [mood-to-palette](./pipes/visual/mood-to-palette) | 📋 planned | — |

See [ROADMAP.md](./ROADMAP.md) for phasing + priorities.

## Method (consistent across all experiments)

Every experiment in this workspace produces three things:

1. **A model artifact** — small (sub-100 M params), inference-budget
   constrained (sub-200 ms on CPU, sub-20 ms on GPU), released to
   HuggingFace under `protoLabsAI/...`.
2. **Honest evaluation** — at minimum: majority-class baseline,
   linear-probe baseline, our model, plus one off-the-shelf
   comparable model. Held-out sets that match ORBIS's actual
   conditions, not just the training distribution.
3. **A learning** — captured in `RESULTS.md` and ideally a blog
   post in [`content/blog-posts/`](./content/blog-posts).

Each experiment dir follows the same shape:

```
pipes/<pipe>/<experiment>/
├── PLAN.md           ← problem framing, why-ORBIS-needs-this, plan
├── RESULTS.md        ← what we tried, what worked, what didn't
├── BLOG.md           ← optional: writeup
├── labels/           ← extractors, taxonomy, manifests
├── training/         ← model + train scripts
├── eval/             ← eval harness + baselines
├── serve/            ← inference server (if applicable)
└── ...
```

The audio-tags experiment is the canonical example — copy its
shape for new experiments.

## Tie-back to ORBIS

ORBIS lives at [github.com/protoLabsAI/ORBIS](https://github.com/protoLabsAI/ORBIS).
Each finished experiment lands as one of:

- A **Pipecat frame processor** (audio-pre / text-pre / text-post)
- An **HTTP service** the agent calls (heavy classifiers, batch jobs)
- A **memory-writer** in the SQLite curator path
- A **visual driver** that issues `apply_palette` / `adjust_param`
  calls

The "graduation" criterion for an experiment: a PR into ORBIS that
wires the model into the actual conversation loop. Until then it's
research.

## Selection criteria — when to add a small model vs ask the LLM

| Question | If yes, small model. If no, LLM. |
|---|---|
| Output is bounded (yes/no, top-K labels, scalar)? | small |
| Latency budget is sub-100 ms? | small |
| Need it to run on every turn / every audio frame? | small |
| Need same answer twice for same input? | small |
| Requires reasoning across multiple facts? | LLM |
| Output is open-ended language? | LLM |
| Calls run <1× per session? | LLM is fine |

The whole game: **if the LLM would have to guess something a small
classifier could measure, you're paying LLM cost for a job a 1.7 ms
head could do better.**

## See also

- [ROADMAP.md](./ROADMAP.md) — phased priorities
- [LEARNING.md](./LEARNING.md) — study path indexed by pipe
- [pipes/](./pipes/) — one dir per loop stage
- [shared/](./shared/) — common eval / inference infra
