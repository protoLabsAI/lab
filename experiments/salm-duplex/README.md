# SALM-Duplex: Full-Duplex Voice Agent

Build a full-duplex voice agent by attaching streaming audio I/O to Qwen3.5-4B
using NVIDIA's SALM-Duplex architecture (channel fusion).

## Architecture

```
User Audio (streaming) → FastConformer Encoder → Modality Adapter ─┐
                                                                     ├─ SUM → Causal LLM (Qwen3.5-4B) → Text + 4 Codec Tokens
Agent Audio (prev frame) → NanoCodec Embeddings ───────────────────┘                                       │
                                                                                                           ▼
                                                                                              NanoCodec Decoder → Speaker
```

At every 80ms frame: encode user audio + embed previous agent audio → sum → LLM predicts text + audio → decode to speaker.
No VAD, no turn detection — the model learns when to speak/listen from data.

## Data Pipeline

| Dataset | Hours | Purpose | Status |
|---------|:-----:|---------|--------|
| LibriSpeech | 960h | ASR baseline | Downloading |
| GigaSpeech/People's Speech | 2,500h+ | ASR + conversational | Pending access |
| AMI Meeting Corpus | 100h | Duplex/overlap | Downloading |
| ICSI Meeting Corpus | 72h | Duplex/overlap | Downloading |
| DailyTalk | 20h | Structured dialogue | Downloading |
| UltraChat Speech (synthesized) | ~2Kh | Instruction following | TODO: synthesize |

Data stored at `/mnt/data/salm-duplex/data/`

## Training Phases

1. **Speech Understanding**: Train encoder + adapter on ASR data (5-10K hours)
2. **Speech Generation**: Train NanoCodec decoder on TTS-style data
3. **Conversational SFT**: Joint training with dialogue data
4. **Duplex Fine-tuning**: Train on overlapping speech data (AMI, ICSI, Fisher)

## Pretrained Components

- NanoCodec: `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`
- FastConformer: `nvidia/stt_en_fastconformer_hybrid_large_streaming_80ms`
- LLM Backbone: `Qwen/Qwen3.5-4B` (frozen or LoRA)

## References

- [SALM-Duplex Paper](https://arxiv.org/abs/2505.15670)
- [NeMo SpeechLM2 Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/intro.html)
- [NanoCodec](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps)
