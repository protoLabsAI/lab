# LTX-2.3 LoRA training — proven pipeline (Blackwell / sm120 / cu128)

End-to-end LoRA fine-tuning of **LTX-2.3-22B** on the RTX PRO 6000, verified 2026-07-13. LoRAs train
in bf16/int8 and **apply to our NVFP4 22B at inference**.

## One command (orchestrated)

```bash
~/dev/LTX-2/.venv/bin/python make_video_lora.py <clips_dir> <style_name> \
    [--rank 32] [--steps 1000] [--bucket 768x512x49] [--with-audio] [--gpu 1] [--no-caption] [--no-wire]
```

`make_video_lora.py` runs the whole chain — **caption → preprocess → fix-conditions → config → train →
wire into ComfyUI** — and is **resumable** (each stage skips if its output exists). Point it at a folder
of clips; get a `<style>.safetensors` LoRA symlinked into ComfyUI `loras/` plus a ready
`LTX-2.3_T2V_<style>_LoRA` workflow (fp4 base + `LoraLoaderModelOnly`). If `<clips_dir>/dataset.json`
(list of `{video_path, caption}`) exists or `--no-caption` is set, captioning is skipped and it's used.

Work dir defaults to `/mnt/data/video-lora/<style>/`. The manual stages below are what it automates.

---

## Manual pipeline (what the orchestrator runs)

## Verified on Blackwell
- Trainer: `~/dev/LTX-2/packages/ltx-trainer` (`.venv`, torch 2.9.1+cu128). Arch-agnostic loader reads
  the arch from the checkpoint's `config` metadata → the **22B (AVTransformer3DModel, 48 layers) trains
  with no special-casing**.
- `int8-quanto` quantization **works on sm120** (block-by-block; skips patchify/proj_out like our fp4 policy).
- Base model to train on: `ltx-2.3-22b-distilled-1.1.safetensors` (bf16, 43 GB). Text encoder:
  `/mnt/models/gemma-3-12b`.
- Footprint (int8, rank 16): **~22 GB** → fits GPU1 alongside an idle ComfyUI. ~1 s/step.
- LoRA output = standard PEFT format (`diffusion_model.…lora_A/B`, rank 16), 205 MB. Loads via
  ComfyUI `LoraLoaderModelOnly` onto the **fp4 base** and measurably changes output (strength 0 vs 1
  = different bytes at fixed seed).

## Pipeline

```bash
cd ~/dev/LTX-2/packages/ltx-trainer
PY=~/dev/LTX-2/.venv/bin/python

# 1. dataset.json — list of {video_path, caption}. LTX likes long captions w/ audio description.

# 2. preprocess: VAE-encode videos → latents + Gemma text embeddings
CUDA_VISIBLE_DEVICES=1 $PY scripts/process_dataset.py dataset.json \
  --resolution-buckets "768x448x25" \
  --model-path /mnt/data/models-cold/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
  --text-encoder-path /mnt/models/gemma-3-12b \
  --video-column video_path \
  --output-dir <preprocessed>
#  ⚠️ GOTCHA: text embeddings land NEXT TO the source clips as <clip>.pt, NOT in
#     <preprocessed>/conditions/. The trainer's PrecomputedDataset expects
#     <preprocessed>/conditions/<relpath>/<name>.pt mirroring <preprocessed>/latents/<relpath>/.
#     Move them: cp <src_dir>/*.pt <preprocessed>/conditions/<relpath>/

# 3. train (config = toy_smoke.yaml here; edit paths/steps/rank/target for real runs)
CUDA_VISIBLE_DEVICES=1 $PY scripts/train.py <config>.yaml --disable-progress-bars
#  → checkpoints/lora_weights_step_NNNNN.safetensors

# 4. use in ComfyUI: symlink into models/loras/, add LoraLoaderModelOnly between the
#    CheckpointLoaderSimple MODEL output and the guider, on the NVFP4 base.
```

## Config notes (`toy_smoke.yaml`)
- `with_audio: false` unless you also preprocess `--with-audio` (needs audio_latents/). Toy run was video-only.
- `quantization: int8-quanto` + `optimizer_type: adamw8bit` to co-reside with ComfyUI. Drop to bf16 +
  free GPU1 for max fidelity.
- `validation.interval: null` for smoke (validation gens are slow + reload Gemma). Enable for real runs.
- rank 16 (low-vram) / 32 (standard). target_modules `to_k/to_q/to_v/to_out.0` match video+audio attn.

## Next (real LoRA)
Pick a concept → gather 20-50 style-consistent clips → caption (`caption_videos.py`, qwen_omni) →
preprocess → train ~1-2k steps → validate. The smoke proved every stage runs on this box.
