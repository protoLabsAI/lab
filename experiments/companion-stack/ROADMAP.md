# ROADMAP — companion-stack

Phased research priorities, with concrete deliverables per phase.

The principle: **each phase has to land in ORBIS** before the next
one starts in earnest. We're not building a research lab in a
vacuum; we're filling slots in a deployed voice loop. A successful
phase ends with a PR into the ORBIS repo plus a blog post.

---

## Phase 0 — foundation (✅ done)

The audio-tags experiment served as the methodology shakedown: how
do we structure experiments, what does the eval ladder look like,
what's the publishing cadence.

**Outputs**:
- 5 model variants on HuggingFace (`protoLabsAI/orbis-audio-tags-{v2,v3-balanced,v4-multi,v5-soft}` + `orbis-audio-tags-v0` dataset)
- DSP whisperization technique (reproducible synthetic whisper data)
- Honest baseline comparison methodology (majority + linear probe + ablations)
- Blog post draft: [`pipes/audio-pre/audio-tags/BLOG.md`](./pipes/audio-pre/audio-tags/BLOG.md)

**Learnings carried forward**:
- Frozen-encoder + tiny-heads is the right architectural shape for
  the bounded-classification pipes
- Multi-corpus training + sqrt class weighting was the key
- "Tier-0 baselines" (majority + linear probe) prevent self-deception
- Ship privately to HF first; make public after a clean blog draft

---

## Phase 1 — perception layer (next, ~1-2 weeks)

The pipe most LLM-only voice agents skip entirely: **understanding
what just hit the microphone before STT does its thing.**

### Experiments

1. **`audio-pre/audio-tags`** — graduate v5 from research to ORBIS
   integration. Pipecat frame processor that runs in parallel with
   Whisper, writes to ORBIS's `mood` table + injects context line
   into the LLM call. Already-trained, just needs wiring.

2. **`audio-pre/speaker-verification`** — owner-vs-stranger
   classifier. ORBIS is single-owner; the orb should know when
   someone *else* is talking. Candidate models:
   `speechbrain/spkrec-ecapa-voxceleb`, `pyannote/embedding`.
   ~10 sec of owner-enrollment audio, then cosine-similarity gating
   against incoming utterances.

3. **`audio-pre/sound-event-detection`** *(backlog)* — YAMNet or
   PANNs. Tags background events (doorbell, music, baby, dog) so
   the orb can react to the room, not just the words.

### Phase exit criteria

- PR into ORBIS that wires audio-tags + speaker-verification into
  the Pipecat pipeline.
- Blog post: "Adding ears to a voice agent."

---

## Phase 2 — routing + retrieval (~2-3 weeks)

What fires *between* STT and the LLM. Today this is "let the LLM
figure it out"; we want bounded classifiers + better retrieval.

### Experiments

1. **`text-pre/intent-classifier`** — small (~50 M param) classifier
   that routes incoming user text to one of: `chat`, `command`,
   `delegate_to`, `memory_query`, `orb_self_modify`. Saves an LLM
   call when the intent is mechanical.

2. **`llm-context/tool-need-predictor`** — binary classifier: "does
   this turn need any tools at all?" If no, skip the function-calling
   round-trip entirely. Trained on synthetic + real ORBIS traces.

3. **`llm-context/reranker`** — cross-encoder on top of the existing
   Qwen3-Embedding retrieval. Re-ranks top-50 → top-5. Candidate:
   `cross-encoder/ms-marco-MiniLM-L-6-v2` baseline + fine-tune on
   ORBIS memory pairs.

4. **`text-pre/topic-router`** *(backlog)* — categorizes the topic
   (code / calendar / casual / search) and pre-selects the right
   delegate, bypassing the LLM's routing decision.

### Phase exit criteria

- ORBIS LLM calls drop measurably (target: 30%+ of turns skip the
  LLM entirely via intent classification).
- Memory recall quality improves (measured via held-out fact-recall
  set).
- Blog post: "Stop asking the LLM what it should be doing."

---

## Phase 3 — companionship (~3-4 weeks)

The slow stuff. Memory, personality drift, mood. The actual
"companion" layer.

### Experiments

1. **`memory/fact-worthiness`** — classifier that decides whether a
   conversation turn produced a fact worth saving long-term. Most
   turns don't. Currently ORBIS's curator is rule-based or
   LLM-driven; replacing with a trained classifier is cheaper and
   more consistent.

2. **`memory/coreference-resolver`** — resolves "she said yesterday"
   to specific facts row. Uses ORBIS's existing entity registry as
   the link target. Could be off-the-shelf (Spacy / fastcoref) or a
   small fine-tune on ORBIS-specific entities.

3. **`text-post/prosody-tagger`** — inserts Fish-style `[softly]` /
   `[pause:300]` / `[laughing]` tags into the LLM's text response
   before TTS, conditioned on current mood + response semantics.
   Companion piece to your existing Fish work.

4. **`memory/mood-summarizer`** *(backlog)* — daily aggregation of
   audio-tag moods into a personality-drift signal that updates the
   `personality_axes` table.

### Phase exit criteria

- ORBIS demonstrably "remembers" across sessions in a way users can
  point to (test set of "remember when I said X" questions, hit rate).
- Orb tone shifts measurably with user mood (audio-tags drives
  prosody tagger, drives Fish output).
- Blog post: "What makes an AI feel like a companion, not an assistant."

---

## Phase 4 — identity + presence (~2 weeks)

The pieces that make the orb feel *present* — anchored in time,
aware of who it's talking to, reactive in real-time.

### Experiments

1. **`audio-pre/wake-word`** — porcupine / openWakeWord. Lower-
   friction conversation entry than push-to-talk for ambient
   deployments.

2. **`visual/mood-to-palette`** — small classifier or rule-engine
   that drives `apply_palette` / `adjust_param` from
   audio-tag mood + LLM-inferred mood. Continuous ambient
   expression, not user-issued commands.

3. **`visual/speaking-state`** — orb animation drivers tied to TTS
   state (preparing / speaking-quiet / speaking-loud / listening /
   thinking).

4. **`audio-pre/noise-adaptive-prompt`** *(backlog)* — feeds SNR +
   environment from audio-tags into a system-prompt switcher that
   tells the LLM "respond shorter, the room is noisy."

### Phase exit criteria

- Orb's visual state continuously reflects the conversation's
  emotional + activity state without any user prompting.
- Demo video. People can tell the orb is *alive* in a way other
  voice agents aren't.

---

## Phase 5 — research polish (~ongoing)

Once the loop is reasonably full, the work shifts to making it
robust + benchmarkable.

- **SUPERB-style benchmark** — standardized eval across all our
  audio classifiers vs published baselines.
- **End-to-end latency profiler** — what does the full loop cost?
  Where are the bottlenecks?
- **Personalization fine-tune flow** — Gradio app already prototyped
  for audio-tags; generalize to other heads.
- **Cross-language scaling** — currently English-only.

---

## Cross-cutting infrastructure

Some things are shared across all experiments and live in
[`shared/`](./shared/):

- **Eval harness** — Tier-0 baseline runner, comparison table
  generator (already built for audio-tags, generalize).
- **Dataset registry** — central JSONL/parquet manifests with HF
  attribution, licensing, sample counts.
- **Inference patterns** — `Predictor` class, FastAPI service
  template, Pipecat frame-processor template.

---

## What "done" looks like for the program

A working ORBIS where every pipe in the conversational loop is
filled with the right tool for the job:
- Perception: < 200 ms from mic to mood/gender/event tag
- Routing: 30%+ of turns skip the LLM entirely
- Retrieval: rerank-improved fact recall
- Response: prosody + safety + style validators ship
- Memory: trained curator, not hand-rolled rules
- Visual: continuous ambient expression

And every model on the way is publicly released, benchmarked, and
documented so the next person building a voice companion has a
reference architecture, not just a paper.
