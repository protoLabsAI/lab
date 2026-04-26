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

## Phase 1 — perception layer (✅ engineering done, content owed)

The pipe most LLM-only voice agents skip entirely: **understanding
what just hit the microphone before STT does its thing.**

### Outcomes

| Roadmap deliverable | Plan | What shipped |
|---|---|---|
| `audio-pre/speaker-verification` | ECAPA-TDNN + cosine gate | ✅ Shipped via [ORBIS #35 stack](https://github.com/protoLabsAI/ORBIS/issues/35) — PRs #45 / #56 / #58 / #62 / #64 / #67. Includes wizard enrollment, voiceprint endpoint, corrupt-voiceprint failure-mode contract. |
| `audio-pre/audio-tags` (v5 graduation) | Pipecat tap alongside Whisper, writes mood + injects context line | 🔀 Shipped via [ORBIS #66 stack](https://github.com/protoLabsAI/ORBIS/issues/66) — but **architecturally substituted**: SenseVoice-Small (234 M, single forward pass) replaced Whisper-STT + v5 emotion head. AudioTagsTap consumes `EmotionFrame` from SenseVoice instead of running v5 inference. PRs #70 / #73 / #75 / #77 / #81 / #83 / #85. Released as v0.1.29 → v0.1.36. |
| `audio-pre/sound-event-detection` (backlog) | YAMNet/PANNs | 🟡 Half-shipped: SenseVoice's `AudioEventFrame` covers BGM, Laughter, Applause, Cry, Sneeze, Breath, Cough — wired through to the `[audio]` annotation. Doorbell / baby specifically NOT covered (SenseVoice's event taxonomy is voice-adjacent). |

### The SenseVoice substitution (the actual story of Phase 1)

The plan was to graduate v5-soft alongside Whisper. The implementation found a strictly better architecture: **SenseVoice-Small subsumes both transcription and emotion in one forward pass.** A 234 M-param multi-task model from FunASR that emits ASR + language ID + speech emotion + audio events together. ORBIS dropped Whisper STT entirely in favor of SenseVoice-as-STT, and AudioTagsTap became a thin consumer of the `EmotionFrame` it produces.

What this means for v5-soft and the audio-tags research:
- **The methodology survives.** Tier-0 baselines, multi-corpus mixing, sqrt class weighting, the DSP whisperization technique — all carry forward to future heads we add.
- **v5-soft remains valuable as an alternative emotion source** if SenseVoice quality regresses, plus the ablation lineage (v2 → v3-balanced → v4-multi → v5-soft) is the credibility chain for everything we publish next.
- **v5's non-emotion heads (`snr_db`, `environment`, `speaking_speed`)** are NOT in production yet. SenseVoice doesn't cover them; v5 does. Wiring them into the `[audio]` annotation is a small follow-up PR — explicitly deferred from Phase 1.
- **The Hugging Face artifacts stay published.** v5-soft + ablations + dataset are the public reference for the methodology; ORBIS using a different model for production doesn't invalidate them.

### Other things that shipped under the Phase 1 banner

- **R-series engineering debt** — 9 production risks closed across ~10 PRs (R1, R2, R5, R6, R7+R8, R9, R10+R11, R14, R15). Several were Major (R5 silent data corruption, R7+R8 atomic stash/drain, R15 mood-collision). Not on the research roadmap; ate real cycles; was the cost of shipping cleanly.
- **Three-writer mood pattern** (`set_mood` / `drift_mood_toward` / `drift_mood`) — closed R15 and is also Phase-3-ready scaffolding for the eventual `memory/mood-summarizer`.
- **`audio_context_block` in the persona prompt** — a small text-post foothold (tells the LLM what to do with the `[audio] …` annotation, forbids parroting). Not a full text-post experiment but demonstrates the shape.
- **Lifecycle audit docs** in ORBIS (`voice-lifecycle.md` / `voice-lifecycle-risks.md` / `voice-lifecycle-research.md`) — research integration plan that maps the roadmap onto the actual ORBIS pipe slots.

### Phase 1 exit status

- ✅ **Engineering**: PR(s) into ORBIS that wire audio-tags + speaker-verification into the Pipecat pipeline. Live on `STT_BACKEND=sensevoice` + `[sensevoice]` extra. v0.1.36 release queued.
- 🟡 **Test**: starting now — owner-driven ORBIS use to surface confidence calibration drift, mood-injection quality, edge cases the v4-holdout didn't catch.
- ⬜ **Content**: blog post "Adding ears to a voice agent" still owed. Should foreground the SenseVoice substitution as the punchline (we built the model, learned the methodology, found a better fit, applied the methodology to picking it).
- ⬜ **HF flip**: model + dataset repos still private. Flip to public on blog publish.

Phase 2 should not start in earnest until `Content` and the HF flip close.

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
