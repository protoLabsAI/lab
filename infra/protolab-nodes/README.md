# protoLabs.nodes

ComfyUI custom nodes that wire graph workflows into the protoLabs ecosystem: the LiteLLM
gateway (`http://ava:4000/v1`), any OpenAI-compatible agent endpoint (protoAgent `/v1`,
raw vLLM lanes), Fish S2-Pro TTS, and Whisper STT.

## Install

```bash
ln -s ~/dev/lab/infra/protolab-nodes ~/dev/ComfyUI/custom_nodes/protolab-nodes
sudo systemctl restart comfyui
```

Runs entirely in ComfyUI's venv — deps are `requests`, `torchaudio`, `PIL`, all already
present. No model weights load in-process; everything is a thin HTTP client, so GPU1
stays free for the render.

## Auth

Key resolution order: `PROTO_GATEWAY` node input → `PROTOLAB_GATEWAY_KEY` →
`GATEWAY_API_KEY` → `~/dev/lab/evals/.env`. Gateway URL: `PROTOLAB_GATEWAY_URL`
(default `http://ava:4000/v1`). No secrets in this repo; don't type keys into the
ProtoGateway node on graphs you'll share — widget values persist in the workflow JSON.

## Nodes

```
node                       in -> out                 notes
protoLab/LLM
  Gateway                  cfg -> PROTO_GATEWAY      optional; env resolution otherwise
  LLM Chat                 prompt[,IMAGE] -> text,reasoning   model combo = live /v1/models (chat lanes)
  LLM Structured JSON      prompt,schema -> json     vLLM guided decode via response_format
  Agent Chat               prompt -> text,reasoning  ANY OpenAI-compat base_url (protoAgent /v1, :8040...)
protoLab/Prompt
  LTX-2.3 Prompt Enhancer  idea[,IMAGE] -> prompt    uses the canonical Gemma enhancer system
                                                     prompts from the LTX-2 checkout; i2v REQUIRES
                                                     the first frame wired in
  Text Template            {a}{b}{c}{d} -> text
  Show Text                text -> text              display + passthrough
protoLab/Audio
  TTS Fish S2-Pro          text -> AUDIO             gateway /audio/speech; 500s while
                                                     protovoice-stack is parked
  STT Whisper              AUDIO -> text             gateway /audio/transcriptions
```

## Notes

* **Model picker** is fetched from the live gateway with a 3 s timeout and 5 min cache;
  falls back to the static lane list so ComfyUI boots with the gateway down. Non-chat
  lanes (image/TTS/STT/embed) are filtered out. `model_override` (STRING) beats the combo.
* **Think-salvage**: vLLM's greedy qwen3 parser can land the whole answer in
  `reasoning_content` on an unterminated `</think>` (vllm#40528). Every text output here
  runs the same salvage claw-eval uses, so `text` is never silently empty.
* **Vision**: wire IMAGE into LLM Chat / Agent Chat — batch becomes multiple image parts.
  `protolabs/fast` (Ornith) keeps bf16 vision on the quant.
* **Non-goal — image generation.** The gateway's `qwen-image`/`krea2` lanes are protoBanana
  shims that themselves call ComfyUI on this box; calling them from inside a ComfyUI graph
  would be circular. Generate images natively in the graph.

## Example workflow

`workflows/ltx2-t2v-enhanced.json` (API format — POST to `/prompt`, or import in the UI):
raw idea → LTX-2.3 Prompt Enhancer → Show Text → the proven distilled-fp4 t2v graph from
`infra/video-bridge` (8-step ManualSigmas, joint AV, 1280x704@24). Verified end-to-end:
~50 s wall to a finished clip with audio on GPU1.

## Smoke

```bash
~/dev/ComfyUI/venv/bin/python -m pytest smoke.py  # or: venv python smoke.py (live gateway)
```
