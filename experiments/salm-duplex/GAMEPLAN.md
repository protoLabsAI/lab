# SALM-Duplex Gameplan — protoVoice v2

**Goal:** Full-duplex voice agent with native tool calling. Speak, think, call tools, speak results — all in real-time.

**Status as of 2026-04-15:** Phase 2 complete. Moving into Phase 3.

---

## Status Summary

| Phase | Status | Finished |
|-------|:------:|----------|
| Phase 0: Data prep | ✅ Done | 2026-04-02 |
| Phase 1: Speech encoder + adapter (SALM) | ✅ Done | 2026-04-09 |
| Phase 2: Conversational speech decoder | ✅ Done | 2026-04-15 |
| Phase 3: Conversational SFT (scale + diversity) | 🟡 Next | — |
| Phase 4: Duplex fine-tuning | ⬜ Pending | — |
| Phase 5: Eval + tool-calling integration | ⬜ Pending | — |

> **Note on phase numbering:** The original plan had Phase 2 = "TTS-only decoder training" and Phase 3 = "Conversational SFT." We compressed these — training the speech decoder directly on real conversational audio (DailyTalk) rather than passing through a TTS-only stage. Phase 3 now expands the conversational corpus rather than introducing conversation for the first time.

---

## Phase 0: Data Prep ✅

### Speech attributes + descriptions (LibriSpeech)
- Pipeline: Whisper large-v3-turbo transcription + 7 acoustic attributes (duration, volume, pitch, SNR, RMS, speaking speed, sample rate)
- **283,861 samples** extracted across test-clean + train-clean-100 + train-clean-360 + train-other-500 (~965h)
- DeSTA2-style LLM descriptions generated via Qwen 4B MoE vLLM, async httpx, 8-way concurrent × 2 GPUs
- Lhotse CutSet v2 manifests at `/mnt/data/salm-duplex/manifests/full-960h-described/` (269,667 train / 14,194 val)
- 5-way PROMPT_POOL variation for instruction diversity

### Conversational datasets (on /mnt/pool, 37TB bulk pool)
- **DailyTalk**: 2,541 dialogues, 21.7h — used for Phase 2 training
- **Emilia-YODAS EN**: 1.4TB, 1,362 tars — `amphion/Emilia-Dataset`
- **Switchboard**: 28GB parquet — `hhoangphuoc/switchboard`
- **AMI**: 28GB parquet (ihm + sdm) — `edinburghcstr/ami`

---

## Phase 1: Speech Understanding ✅

Trained the modality adapter that bridges audio embeddings → Qwen input space.

### Iteration results

| Version | Adapter | Target | Outcome |
|---------|---------|--------|---------|
| v1 | IdentityConnector | Raw transcripts | Failed — only learned ASR, couldn't describe |
| **v2** ← **WINNER** | IdentityConnector | DeSTA2 descriptions | ASR + description both working; Qwen text intact |
| v3 | Q-Former (64 queries, 6 layers, multi-layer agg [0,5,11,17,23]) | Descriptions | Lost ASR ability — over-compressed 12.5fps audio into 64 query slots |

**v2 final metrics:** 67K steps, train loss 1.484, eval loss 0.2326. Qwen tool-calling intact on text-only probes.

### Phase 1 improvements deferred to future v4
1. Unfreeze Canary encoder (per Canary-Qwen-2.5B: +6% WER)
2. LoRA on Qwen q_proj/v_proj with Ultravox's two-pass gating to protect tool calling
3. KL distillation (speech-conditioned vs text-only logits)
4. Data diversity — GigaSpeech spontaneous, Common Voice accents

---

## Phase 2: Conversational Speech Decoder ✅

Trained the NanoCodec decoder head directly on real conversational audio.

### Architecture
- **Frozen** Qwen 3.5-4B backbone (preserves tool calling)
- **Frozen** Canary-1b-flash ASR encoder
- **Frozen** NanoCodec 22kHz 0.6kbps 12.5fps
- **Trained:** 12-layer causal transformer speech decoder (d_model=768, ~130M params)

### Training
- Config: `conf/s2s-decoder-qwen4b-dailytalk.yaml`
- Data: DailyTalk only (2,413 train / 128 val dialogues, Lhotse SHAR)
- 50,000 steps / 400 epochs
- batch_size=1, accumulate_grad_batches=4, DDP 2×Blackwell
- Final checkpoint: `/mnt/data/training/s2s-decoder-qwen4b-dailytalk/checkpoints/step=50000-last.ckpt`
- Top-k retained: step=10000, 20000, 50000

### Upstream patches (carry into Phase 3)
- `nemo/collections/speechlm2/models/duplex_s2s_speech_decoder_model.py` — BOS override (`<|im_start|>` for Qwen 3.5), DynamicCache config fallback
- `nemo/collections/speechlm2/models/salm.py` — encoder_multilayer attribute handling (Q-Former mode)
- `transformers/models/qwen3_5/modeling_qwen3_5.py` — defensive `has_previous_state` guard for hybrid cache

### Run stats (telemetry from `/mnt/scratch/logs/crash-watch/`)
- 31.6h continuous crash-free final run
- Peak GPU0 400W / 72°C, GPU1 411W / 64°C, CPU 77°C, VRM 45°C
- Zero thermal throttles, zero NCCL errors, zero OOMs

---

## Phase 3: Conversational SFT 🟡 NEXT

**Goal:** Scale conversational quality using the larger corpus on `/mnt/pool`.

### Immediate tasks
1. **Evaluate Phase 2 checkpoint** before touching it — ASR quality, speech naturalness, conversation coherence, text-only regression. Establishes a baseline.
2. **Build Lhotse converters** for the new data:
   - Emilia → group utterances by video ID into multi-turn dialogues
   - Switchboard (parquet) → MonoCut with speaker-role supervisions
   - AMI (parquet, ihm variant) → meeting dialogues with 4-5 speaker roles
3. **Merge SHAR manifests**: DailyTalk + Emilia + Switchboard + AMI → combined train/val
4. **Resume training** from `step=50000-last.ckpt` on the expanded corpus

### Expected scale
- DailyTalk: 21.7h (have)
- Switchboard: ~260h
- AMI: ~100h
- Emilia EN: ~46,800h (massive — likely bottleneck will be streaming, not compute)
- **Total:** ~47,000h theoretical; practically 500-1000h subset for initial SFT

### Estimated training time
3-5 days on 2×Blackwell at current 16 steps/min pace. Max_steps tbd based on how much data we actually ingest.

---

## Phase 4: Duplex Fine-tuning ⬜

**Goal:** Learn when to speak/listen, handle interruptions, backchannels.

**Data:**
- AMI SDM (overlap-rich single-distant-mic) — real overlapping speech
- Synthetic duplex from conversational corpus (simulate overlaps/backchannels)
- ICSI Meeting Corpus if we pull it (72h, ~40GB)

**Estimated:** 1-2 days training.

---

## Phase 5: Eval + Tool-Calling Integration ⬜

### Evaluation
- **Full-Duplex-Bench** (if available)
- Latency vs protoVoice baseline (target <200ms TTFA)
- Speech quality: UTMOS, subjective MOS on held-out val set
- Reasoning quality: GPT-4 judge on spoken Q&A
- **Tool calling**: BFCL v4 adapted for audio input (use protoCLI flow as harness)

### Integration
- Streaming inference server (WebSocket/WebRTC)
- Gradio frontend integration
- A/B test vs current protoVoice pipeline

---

## Tool Calling Architecture (Frozen Backbone)

**Key insight:** Tool calls are ALWAYS text-mediated, even in end-to-end models (GPT-Realtime, Gemini Live, Ultravox). The text token stream carries structured tool calls; the audio decoder handles spoken output. Nobody generates tool-call audio.

**Validated by Ultravox v0.7**: Frozen Llama 3.1 + audio projector, function calling benchmarked on AIEWF eval.

### Flow
```
User speaks → Canary Encoder → Adapter → [SUM] ──→ Frozen Qwen 3.5-4B
                                            ↑               │
Agent prev audio → NanoCodec Embed ─────────┘               │
                                                     ┌──────┴──────┐
                                                 Text Head     Audio Head
                                                     │             │
                                              ┌──────┴──────┐      │
                                          Regular      <tool_call> │
                                          text           tokens    │
                                              │             │      │
                                              ▼             ▼      ▼
                                          Audio         Tool    NanoCodec
                                          Decoder       Router   Decoder
                                              │             │      │
                                              ▼             ▼      ▼
                                          Speaker      Execute   Speaker
                                                        tool     (after result)
                                                         │
                                                         ▼
                                                  Inject result
                                                  into context
```

### Why frozen backbone works
- Qwen's tool-calling weights are untouched → no catastrophic forgetting
- Text head produces `<tool_call>` tokens naturally (from Qwen's pretraining)
- Audio head is a separate trained decoder → doesn't interfere with text generation
- Intercept tool-call tokens from the text stream, route to executor
- Model can say filler ("Let me check that") via audio while tool runs via text
- Tool results injected back into context, model speaks the answer

### Open questions
- Can the model learn to emit filler audio concurrently with tool-call tokens?
- How to handle tool latency without breaking duplex flow?
- Multi-step tool chains mid-conversation — does context injection break audio coherence?

---

## Infra / Parallel Work Done

| Item | Status | Location |
|------|:------:|----------|
| Crash-watch telemetry logger (1Hz, fsync per row) | ✅ | `infra/crash-watch/crash-watch.py` |
| Persistent journald | ✅ | `/var/log/journal/` |
| mergerfs HDD pool (2× 20TB IronWolf Pro) | ✅ | `/mnt/pool` (37TB) |
| lm-sensors + smartmontools | ✅ | global |
| vllm.service crash-loop fixed | ✅ | `systemctl disable vllm` during training |
| UPS monitoring (NUT, Goldenmate 1600W) | ⏳ | Pending USB cable plug-in |
| Power-crash root cause | Unconfirmed but resolved in practice | 31.6h crash-free at 400W peak |

---

## Key References

- **SALM-Duplex** (NVIDIA, 2025): [arxiv 2505.15670](https://arxiv.org/abs/2505.15670)
- **DeSTA2 / DeSTA2.5**: [arxiv 2409.20007](https://arxiv.org/abs/2409.20007) / [arxiv 2507.02768](https://arxiv.org/abs/2507.02768)
- **Ultravox v0.7**: [HuggingFace](https://huggingface.co/fixie-ai/ultravox-v0_7-glm-4_6) — our validation that frozen LLM + tool calling works
- **Moshi**: [arxiv 2410.00037](https://arxiv.org/abs/2410.00037) — comp only, rejected (can't swap LLM)
- **NeMo SpeechLM2**: [docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/intro.html)
