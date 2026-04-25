# LEARNING — study path indexed by pipe

Each experiment in `pipes/` has its own technical depth. This doc is
a study path that maps **what to learn** to **which pipe it
unlocks**. Read top-down for general intuition, jump to a section
when starting a specific experiment.

---

## The shared mental model (read once, applies everywhere)

Modern AI systems decompose into:

```
raw input → tokenize/encode → representation vectors → task-specific output
```

Every pipe in the companion-stack is some variation. Differences:

- **Input modality** (audio, text, mixed)
- **Encoder** (Whisper, BERT, sentence-transformer, raw embedding)
- **Output style** (classification head, regression head, retrieval
  similarity, generation)
- **Inference loop** (single-pass, autoregressive, iterative)

When a new pipe is unfamiliar, decompose it into these four
questions and the rest falls out.

### Ranked starter resources

1. **Karpathy's "Neural Networks: Zero to Hero"** (YouTube) — for
   the math foundation. If you only do one thing, do this.
2. **Lilian Weng's blog** (`lilianweng.github.io`) — for diffusion,
   attention, RLHF write-ups.
3. **Jay Alammar's "Illustrated Transformer"** — for visual
   intuition.
4. **HuggingFace Audio Course** — directly relevant to half this
   workspace.

These cover the foundation that every pipe assumes.

---

## audio-pre

**What this pipe does**: sit between the microphone and STT, derive
structured information from the raw audio that the LLM otherwise
would have to guess from the transcript alone.

**Concepts to study**:
- Mel-spectrograms + filterbanks (audio's "tokenization")
- CNN-based audio classifiers (PANNs, YAMNet)
- Self-supervised speech representations (wav2vec 2.0, HuBERT,
  Whisper encoder)
- Speaker embeddings (x-vectors, ECAPA-TDNN, contrastive speaker
  models)
- VAD (voice activity detection) — Silero, webrtcvad
- Multi-task learning over shared encoders

**Papers (in order)**:
1. wav2vec 2.0 (Baevski 2020) — frozen-encoder + downstream tasks
2. HuBERT (Hsu 2021) — masked-prediction self-supervised audio
3. Whisper (Radford 2022) — encoder-decoder ASR architecture
4. ECAPA-TDNN (Desplanques 2020) — speaker embedding architecture
5. AST (Audio Spectrogram Transformer, Gong 2021) — vision-style
   transformer on mel

**Hands-on exercises**:
- Re-implement audio-tags' linear probe baseline from scratch on a
  random other dataset (VocalSound, ESC-50)
- Run YAMNet on your own home audio for a day, log what it detects
- Fine-tune ECAPA-TDNN on your own voice for speaker verification

---

## text-pre

**What this pipe does**: between STT output and the LLM call, extract
classifiable signal so the LLM doesn't have to derive it itself.

**Concepts to study**:
- Sentence-level encoders (BERT, sentence-transformers)
- Intent classification + slot filling (NLU)
- NER + entity linking
- Distillation (DistilBERT, TinyBERT) — for the latency budget
- Cross-encoders vs bi-encoders (matters for the rerank pipe too)
- Active learning + pseudo-labeling (because real intent data is
  always scarce)

**Papers (in order)**:
1. BERT (Devlin 2019) — the masked-LM encoder pattern
2. Sentence-BERT (Reimers 2019) — bi-encoder for embeddings
3. DistilBERT (Sanh 2019) — distillation method
4. SetFit (Tunstall 2022) — few-shot classification on
   sentence-transformer features

**Hands-on exercises**:
- Build a 5-class intent classifier with `sentence-transformers` +
  a logistic-regression head on synthetic data
- Generate synthetic ORBIS-shape utterances with an LLM, manually
  label 200, train + eval

---

## llm-context

**What this pipe does**: shape what the LLM sees on its prompt
side. Retrieval, reranking, conditioning.

**Concepts to study**:
- Vector retrieval (cosine, dot product, ANN indexes — FAISS, hnswlib)
- Cross-encoder rerankers (the bi-encoder/cross-encoder split)
- Hypothetical Document Embeddings (HyDE)
- Memory architectures for LLMs (Letta/MemGPT, Graphiti, "poor-
  man's Graphiti on SQLite" which ORBIS uses)
- Tool-use prompting + function-calling correctness
- KV-cache + prefix caching (relevant for prompt-design ergonomics)

**Papers (in order)**:
1. Sentence-BERT (Reimers 2019) — already in text-pre, also here
2. ColBERT (Khattab 2020) — late-interaction retrieval
3. MS MARCO (Bajaj 2018) — the benchmark cross-encoders are
   trained on
4. Self-RAG (Asai 2023) — adaptive retrieval
5. Toolformer (Schick 2023) — tool-need teaching

**Hands-on exercises**:
- Index ORBIS's SQLite `facts` table into FAISS, query with
  Qwen3-Embedding, eyeball top-5 quality
- Add a cross-encoder rerank, measure F1 on a synthetic
  fact-recall set
- Write a binary classifier on top of LLM-traced "this turn used
  tools" data — does it predict correctly on held-out?

---

## text-post

**What this pipe does**: between the LLM's text response and the
TTS. Tag for prosody, validate style, scrub for safety.

**Concepts to study**:
- Text generation conditioning (prefix tuning, prompt scaffolding)
- Sequence labeling (BIO tagging, span classification) — for prosody
  tag insertion
- PII detection (Presidio, regex+model combos)
- Style classifiers (formality, register, sentiment in output)

**Papers (in order)**:
1. SSML basics (W3C spec — not a paper, but the prosody language
   ancestor)
2. PromptTTS / NaturalSpeech 3 — for "controlled prosody" research
   context
3. Presidio (Microsoft) — PII detection toolkit reference
4. CTRL (Keskar 2019) — controlled generation via control codes

**Hands-on exercises**:
- Write a span-tagger that inserts `[pause:300]` after sentence
  boundaries based on syntactic features
- Train a binary "needs softening" classifier from labeled mood +
  response pairs

---

## memory

**What this pipe does**: decides what to save, how to retrieve, how
to decay. The companion-layer's spine.

**Concepts to study**:
- Bi-temporal databases (valid time vs transaction time)
- Entity linking + coreference resolution
- Knowledge graph construction from text
- Forgetting curves + spaced-repetition decay
- LLM-driven knowledge extraction (ORBIS's current default)

**Papers (in order)**:
1. Graphiti / Zep (Cohere/Zep memory papers) — memory-graph
   architecture
2. MemGPT / Letta (Packer 2023) — hierarchical agent memory
3. fastcoref / spanBERT-coref — production coreference
4. "Beyond Goldfish Memory" (Xu 2022) — long-term dialogue memory

**Hands-on exercises**:
- Build a fact-worthiness classifier from synthetic conversation
  data; eyeball precision/recall on held-out turns
- Implement a 90-day half-life decay curator over ORBIS's `facts`
  table, measure how table size stabilizes

---

## visual

**What this pipe does**: drives the orb's visible expression from
internal state (mood, speaking, listening, drift).

**Concepts to study**:
- State machines + soft-state animation (Spring physics, easings)
- Color theory + palette interpolation
- Audio-driven animation (visualizers, lipsync — relevant if orb
  ever gets a "mouth")
- Embedding-to-style mapping (CLIP-driven art direction)

**Papers / refs**:
- The orb's existing shader / palette config in ORBIS
- D3.js / WebGL visualization references
- "Bringing Characters to Life" (Disney's 12 principles applied to UI)

**Hands-on exercises**:
- Wire audio-tags' mood output to `apply_palette` via a small
  rule engine; eyeball whether the orb feels right

---

## Cross-cutting (read once, applies to every experiment)

### Tier-0 baselines (mandatory before claiming any result)

For every classifier:
1. **Majority class** baseline
2. **Linear probe** on a frozen pre-trained encoder (whichever is
   relevant: BERT for text, Whisper for audio, etc.)
3. **One published model** evaluated on the same test set

If your model isn't beating all three, it isn't ready.

The audio-tags experiment exhibits this fully — see
[`pipes/audio-pre/audio-tags/RESULTS.md`](./pipes/audio-pre/audio-tags/RESULTS.md).

### Class imbalance + multi-task loss

- Inverse-frequency CE weighting → too aggressive, over-corrects
- Sqrt-tempered weighting → Goldilocks zone for most tasks
- Weighted random sampler → flattens prior, hurts in-domain
- Loss-weight scheduling (curriculum) → underexplored

### Latency budgeting

For real-time voice loops:
- Audio-pre: < 50 ms per frame
- Text-pre: < 30 ms per turn
- LLM-context (rerank): < 100 ms per turn
- Text-post: < 30 ms per turn
- Memory writers (async): no budget, run between turns

If a new model can't hit its slot's budget, either it doesn't ship
or it needs to be quantized / distilled until it does.

### Multi-corpus training (the v4 lesson)

Single-corpus training will collapse rare classes regardless of
loss weighting. The data is the bottleneck. Default to mixing 3+
corpora that span the actual production distribution before
declaring a head "doesn't work."

---

## How to use this doc

1. Pick a pipe to work on
2. Read its section here
3. Skim the audio-tags experiment as a worked example
4. Open `pipes/<pipe>/<experiment>/PLAN.md`
5. Build, eval, blog
