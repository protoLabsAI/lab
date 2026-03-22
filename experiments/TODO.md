# Experiments TODO

## Image Generation
- [ ] **Anima** (Cosmos 2B) — anime/illustration gen via ComfyUI (symlinked, ready to go)
  - `anima-preview.safetensors` + `anima-preview2.safetensors` variants
  - 1024x1024, 30-50 steps, CFG 4-5, euler_a / dpmpp_2m_sde_gpu
- [ ] **Z-Image** (6B) — foundation model, rich aesthetics, negative prompting
  - `ZImagePipeline` via Diffusers
- [ ] **Z-Image-Turbo** (6B distilled) — 8-step fast photorealistic gen, sub-second on H800
  - `ZImagePipeline`, guidance_scale=0.0, num_inference_steps=9
  - Bilingual text rendering (EN/CN)
- [ ] **Z-Image-Omni-Base** — gen + editing base (not released yet, watch Tongyi-MAI org)
- [ ] **Z-Image-Edit** — instruction-based image editing (not released yet)

## OCR / Document Understanding
- [ ] **Qianfan-OCR** (5B, Baidu) — image-to-Markdown, tables, charts, Layout-as-Thought
  - transformers (`trust_remote_code=True`) or `vllm serve baidu/Qianfan-OCR --trust-remote-code`
- [ ] **GLM-OCR** (0.9B, ZAI) — lightweight OCR, #1 OmniDocBench, MIT license
  - transformers / vLLM / SGLang / Ollama

## Speech
- [ ] **Voxtral Mini 4B Realtime** (Mistral) — real-time STT, 13 languages, sub-500ms latency
  - `vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602`
  - Pair with Fish Audio S2 Pro for full STT→TTS pipeline
