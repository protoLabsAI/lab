# SALM-Duplex Research Notes

Building a full-duplex voice agent that preserves frozen LLM tool calling.
Lab notebook style — captures decisions, results, and findings as we go.

**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (96GB each, 192GB total VRAM), Ubuntu 24.04
**Author:** protoLabs
**Started:** 2026-04-08

---

## Motivation

protoVoice (our existing modular voice agent) achieves 165ms TTFA but is fundamentally half-duplex — it cannot listen and speak simultaneously, generate backchannels ("mhm", "yeah"), or handle natural conversational overlap. End-to-end models like Moshi/PersonaPlex achieve true full-duplex but have monolithic architectures that cannot leverage frozen reasoning LLMs (and lose tool calling).

**Goal:** Build a full-duplex voice agent that:
1. Has true simultaneous listen/speak capability (Moshi-style)
2. Preserves Qwen 3.5's tool calling ability (frozen backbone)
3. Can be a ReAct agent — call tools mid-conversation
4. Runs on our 2× Blackwell hardware

## Architectural Decisions

### Why SALM-Duplex (NVIDIA, arxiv 2505.15670)

After researching alternatives:

| Approach | True Duplex? | Frozen LLM? | Tool Calling | Verdict |
|----------|:-----------:|:-----------:|:-----------:|---------|
| **SALM-Duplex** | Yes | Yes (or LoRA) | Preserved | **Chosen** — bolts speech I/O onto any causal LLM via channel fusion |
| Moshi/PersonaPlex | Yes | No (Helium baked in) | Hard | Rejected — can't swap LLM |
| Freeze-Omni | Pseudo-duplex | Yes | Preserved | Rejected — locked to Qwen2-7B, no training code |
| Ultravox | No | Yes | **Proven** | Comp only — we want true duplex |
| Pipeline (protoVoice) | No | Yes | Preserved | Status quo |

The **channel fusion** approach is brilliantly simple: at every 80ms frame, element-wise SUM the user's speech embedding + the agent's previous codec embedding, feed into a standard causal LLM, predict text + 4 audio codec tokens in parallel. No special architecture, no separate streams, just addition. SALM-Duplex with TinyLlama-1.1B beat Moshi (7B Helium) on reasoning by **4x** (7.8 vs 1.9 on ASR-QA), proving you don't need speech pretraining.

### Phased approach

| Phase | Goal | Status |
|-------|------|--------|
| **0** | Data prep — extract speech attributes + descriptions for training targets | **Complete** |
| **1** | Speech understanding — train modality adapter to map audio → LLM embedding space | **In progress (v3)** |
| **2** | Speech generation — train NanoCodec decoder for LLM hidden states → audio tokens | Pending |
| **3** | Conversational SFT — joint dialogue training | Pending |
| **4** | Duplex fine-tuning — turn-taking, interruption, backchannels | Pending |
| **5** | Tool calling integration — ReAct agent over voice | Pending |

---

## Phase 0: Data Preparation

### Source datasets

| Dataset | Hours | Samples | Why |
|---------|:-----:|:-------:|-----|
| LibriSpeech test-clean | 5h | 2,620 | Smoke test |
| LibriSpeech train-clean-100 | 100h | 28,539 | Phase 1 baseline |
| LibriSpeech train-clean-360 | 360h | 104,014 | Scale-up |
| LibriSpeech train-other-500 | 500h | 148,688 | Robustness (noisier audio) |
| **Total** | **965h** | **283,861** | |

### Speech attribute extraction (DeSTA2 method)

For each audio sample we extract 7 attributes via Whisper + signal processing:
- **Transcript** (Whisper large-v3-turbo)
- **Duration**, **volume** (RMS bins: quiet/normal/loud)
- **Pitch** (autocorrelation, bins: low/medium/high)
- **SNR** (energy-based, dB)
- **Speaking speed** (words/sec, bins: slow/normal/fast)
- **Sample rate**

These get formatted into a "seed transcript":
```
[00:00:02-00:00:03] Frankly, I cannot always say. (Duration: 2.4s, Volume: loud, Pitch: high, SNR: 39dB, Speaking speed: normal)
```

**Pipeline performance:** 5.4 files/s on a single Blackwell at 375W. 283K samples in ~14.5 hours.

### LLM-generated descriptions

Following DeSTA2 — feed seed transcripts to an LLM with the prompt "What can you hear from this audio?" The LLM generates a rich natural language description that becomes the training target. We used Qwen 4B MoE (Gemma 4 26B-A4B) serving via vLLM at 375W.

**Pipeline performance:** ~5 samples/s with concurrency 8, both GPUs running in parallel. 283K samples in ~14 hours wall-clock.

Example seed → description:
```
SEED: [00:00:02-00:00:03] Frankly, I cannot always say. (Duration: 2.4s, Volume: loud, Pitch: high...)

DESCRIPTION: The audio features a single speaker delivering a brief, declarative phrase: "Frankly, I cannot always say." The voice is characterized by a high pitch and is delivered at a normal speaking volume and pace. The audio is clear with minimal background interference.
```

### Why descriptions instead of raw transcripts

**Critical insight from v1 → v2:** Initially we used raw transcripts as the supervision target. The model learned ASR but couldn't differentiate "transcribe" from "describe" — it just transcribed everything. Switching to LLM-generated descriptions as targets taught the model to:
1. Transcribe speech (because the description includes the transcript)
2. Analyze acoustic properties (pitch, volume, SNR)
3. Combine both into rich natural language

This is essentially distilling Qwen's text reasoning into the speech encoder via the description targets.

### Data format conversion

Lhotse `CutSet` manifests with description as the supervision text and original transcript preserved in custom fields. Random prompt sampled from a pool of 5 paraphrased instructions per sample.

**Output:** `/mnt/data/salm-duplex/manifests/full-960h-described/{train,val}.jsonl.gz`
- Train: 269,667 cuts (95%)
- Val: 14,194 cuts (5%)

---

## Phase 1: Speech Understanding

### Architecture

```
User Audio (16kHz)
    ↓
Canary-1b-flash Encoder (frozen)
    ↓ [1024-dim, 12.5 fps]
Modality Adapter (trainable) ← THE LEARNED PART
    ↓ [2560-dim — Qwen 4B hidden size]
[Inserted at <|audioplaceholder|> in prompt]
    ↓
Qwen 3.5-4B (frozen)
    ↓
Generated text (description, transcript, etc.)
```

### Iteration log

#### v1: IdentityConnector + transcripts as target (FAILED)

- **Adapter:** `IdentityConnector` (just a linear projection, ~22M params)
- **Target:** Raw transcripts
- **LR:** 5e-4, warmup 500
- **Data:** 269K LibriSpeech, batch 4
- **Result:** Trained but model only learned ASR, couldn't describe audio. The "transcribe vs describe" instruction didn't differentiate behavior.

#### v2: IdentityConnector + descriptions as target (PARTIAL SUCCESS)

- **Same architecture as v1**
- **Target:** LLM-generated descriptions (DeSTA2 method)
- **LR:** 5e-4, warmup 500
- **Steps:** 67,000 across multiple resume cycles
- **Final train loss:** 1.484, eval loss 0.2326 (steady decrease, no overfitting)

**Eval results on the trained checkpoint:**

| Test | Result |
|------|--------|
| **ASR** | Working — clean transcriptions of unseen speech |
| **Description** | Working — produces rich descriptions including transcript + acoustic attributes |
| **Text regression** | Qwen LLM intact — answers "What is 2+2?" correctly with thinking |

Example output:
> "The audio clip features a single speaker delivering the phrase, 'frankly, I cannot always say.' The voice is characterized by a high pitch and is delivered at a normal speaking volume and pace. The audio is clear and easy to understand, with a high signal-to-noise ratio (32dB), indicating minimal background interference."

The model successfully bridges audio → Qwen's embedding space and Qwen does what it always does: generates intelligent text. Qwen never learned anything new — we only taught the adapter how to project audio embeddings into Qwen's input space.

**Limitation observed:** The IdentityConnector is too simple. It's a single projection — every sequence position from Canary becomes one Qwen token position. No learned compression, no attention over temporal features, no multi-layer aggregation. Description quality is acceptable but variable across prompts.

#### v3: Q-Former adapter (IN PROGRESS)

Based on research findings from DeSTA2/DeSTA2.5 and the broader speech-LLM literature, key improvements:

| Change | v2 → v3 | Source |
|--------|---------|--------|
| **Adapter** | IdentityConnector → Q-Former (64 query vectors, 6 layers, multi-layer aggregation) | DeSTA2 paper |
| **Encoder layer aggregation** | Last layer only → Weighted sum across layers [0, 5, 11, 17, 23] | DeSTA2.5 |
| **LR** | 5e-4 → **1e-4** | DeSTA2 default |
| **Warmup** | 500 → **2000** | DeSTA2 default |
| **DDP** | Single GPU → **2-GPU** | Throughput 2x |

The Q-Former architecture: 64 learnable query vectors attend (cross-attention) over Canary encoder hidden states, then 6 transformer layers process the queries. This is much richer than a single projection — the model can learn what to extract from the audio and how to compress it temporally.

**Multi-layer aggregation:** Instead of using only Canary's final encoder layer, we extract from layers [0, 5, 11, 17, 23] and learn a weighted sum. Earlier layers carry phonetic/acoustic detail, later layers carry semantic content. The adapter picks what it needs.

**Status:** Currently training. Expected ~6-7 hours on 2-GPU DDP at 375W per card.

---

## Open Questions & Future Work

### Phase 1 improvements not yet tried

These are high-priority for v4 if v3 isn't tight enough:

1. **Unfreeze Canary encoder** — Canary-Qwen-2.5B's official release notes report **+6% WER improvement** from unfreezing. Train frozen first (warmup), then unfreeze at lr=1e-5 for another epoch.

2. **LoRA on Qwen** — Interspeech 2025 shows **21-24% WER improvement** with LoRA on q_proj/v_proj. Critical: gate it OFF at text-only inference time to preserve tool calling. We could use Ultravox's "two-pass" trick.

3. **KL distillation** — Loss = CE(target) + 0.5·KL(speech-conditioned logits || text-only-conditioned logits). Forces speech path to match what Qwen would say given text. **Catastrophic-forgetting-proof by construction.** This is how Ultravox preserves tool calling.

4. **Data diversity** — LibriSpeech is read audiobooks only. Adding GigaSpeech (spontaneous) + Common Voice (accented) would broaden coverage. DeSTA2 → DeSTA2.5 jumped from 154h → 7000h and saw massive gains.

### Tool calling integration (Phase 5)

Key insight from research: **tool calls are ALWAYS text-mediated, even in end-to-end models.** GPT-4o, Gemini Live, Ultravox — all switch to structured text for tool calls, then resume audio generation. Nobody generates tool call audio.

Our planned approach:
- Frozen Qwen text head produces `<tool_call>` tokens naturally
- Audio decoder is a separate trained module that doesn't interfere with text generation
- Intercept tool call tokens from text stream, route to executor
- Model can say "Let me check that" (audio) while tool executes (text)
- Tool results injected back into context, model speaks the answer

Validated by Ultravox v0.7 on AIEWF eval — frozen Llama 3.1 + audio projector preserves function calling.

### Voice quality

Phase 2 will use NanoCodec (1.1 kbps) for speech generation. Quality is lower than Kokoro (which protoVoice currently uses) but the duplex capability is the trade. Potential future work: replace NanoCodec with Fish Audio S2 Pro for better quality, but that requires retraining the speech decoder.

---

## Key Findings (Tier 1)

1. **DeSTA2's description generation method works on our hardware.** We replicated it with Qwen 4B MoE as the description generator instead of Llama3-8B-Instruct. Quality is good and the pipeline scales (5 samples/s per GPU).

2. **IdentityConnector is too simple for serious speech understanding.** It works but loses information. Q-Former with multi-layer aggregation is the next obvious step.

3. **Description targets >> transcript targets.** Training on rich descriptions teaches the model to do BOTH transcription and acoustic analysis, in one unified output style. This was the v1 → v2 unlock.

4. **Frozen LLM approach preserves text capabilities.** The Qwen 3.5-4B backbone in our v2 checkpoint still answers text-only questions correctly. The adapter is purely additive — Qwen never had to forget anything to learn the new audio modality.

5. **Channel fusion (SALM-Duplex's core trick) is the right architectural bet.** Element-wise sum of audio embeddings + LLM input is brilliant in its simplicity. The LLM learns to disentangle the two signals because they occupy different subspaces (continuous vs discrete) after training.

---

## Reproducibility

### Code & data
- **Repo:** `~/dev/lab/experiments/salm-duplex/`
- **Data manifests:** `/mnt/data/salm-duplex/manifests/full-960h-described/`
- **Checkpoints:** `/mnt/data/training/salm-qwen4b-v3/`
- **NeMo SpeechLM2:** `~/dev/nemo/examples/speechlm2/` (cloned from main, April 2026)
- **Training script:** `~/dev/nemo/examples/speechlm2/salm_train.py`

### Pretrained components
- LLM: `Qwen/Qwen3.5-4B` (HuggingFace)
- ASR: `nvidia/canary-1b-flash` (HuggingFace, NeMo format)
- NanoCodec (Phase 2): `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`

### Training environment
- Python 3.12, NeMo 2.8.0rc0, PyTorch 2.11, CUDA 12.8
- Custom env at `~/dev/nemo-env/`

---

## References

- **SALM-Duplex** (NVIDIA, 2025): [arxiv 2505.15670](https://arxiv.org/abs/2505.15670)
- **DeSTA2**: [arxiv 2409.20007](https://arxiv.org/abs/2409.20007)
- **DeSTA2.5-Audio**: [arxiv 2507.02768](https://arxiv.org/abs/2507.02768)
- **Moshi**: [arxiv 2410.00037](https://arxiv.org/abs/2410.00037)
- **Ultravox v0.7**: [HuggingFace](https://huggingface.co/fixie-ai/ultravox-v0_7-glm-4_6)
- **Freeze-Omni**: [arxiv 2411.00774](https://arxiv.org/abs/2411.00774)
- **NeMo SpeechLM2**: [docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/intro.html)
- **VoiceBench**: [arxiv 2410.17196](https://arxiv.org/abs/2410.17196)
- **Cross-Modal KD for Speech LLMs**: [arxiv 2509.14930](https://arxiv.org/abs/2509.14930)
