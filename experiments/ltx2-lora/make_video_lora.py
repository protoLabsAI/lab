#!/usr/bin/env python3
"""One-command LTX-2.3-22B video LoRA orchestrator: clips folder -> usable LoRA on the fp4 base.

Stages (each skipped if its output already exists — resumable):
  1. caption   clips -> dataset.json   (caption_videos.py / qwen_omni; auto speech+scene)
  2. preprocess dataset -> latents + conditions (process_dataset.py: 22B audio/vae + Gemma)
  3. fix       relocate text embeds into conditions/ mirroring latents/  (the known gotcha)
  4. config    emit training YAML from toy_smoke.yaml, overriding paths/rank/steps
  5. train     train.py -> LoRA .safetensors
  6. wire      symlink LoRA into ComfyUI loras/ + emit a T2V workflow with LoraLoaderModelOnly

Run with the LTX-2 trainer venv:
  ~/dev/LTX-2/.venv/bin/python make_video_lora.py <clips_dir> <style_name> \
      [--rank 32] [--steps 1000] [--bucket 768x512x49] [--with-audio] [--gpu 1] \
      [--no-caption] [--no-wire]

If <clips_dir>/dataset.json exists (or --no-caption), captioning is skipped and that file is used.
"""
import argparse, json, os, subprocess, sys, shutil, glob
from pathlib import Path

TRAINER = Path.home() / "dev/LTX-2/packages/ltx-trainer"
PY = str(Path.home() / "dev/LTX-2/.venv/bin/python")
MODEL = "/mnt/data/models-cold/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
GEMMA = "/mnt/models/gemma-3-12b"
COMFY = Path.home() / "dev/ComfyUI"
TEMPLATE = Path(__file__).parent / "toy_smoke.yaml"
BASE_API = "/mnt/data/ltx-out/base_api.json"   # ComfyUI's own serialized fp4 T2V graph


def sh(cmd, env=None, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env, cwd=cwd)


def stage_caption(clips_dir, ds_json, do_caption):
    if ds_json.exists():
        print(f"[caption] using existing {ds_json}"); return
    if not do_caption:
        sys.exit(f"[caption] no {ds_json} and --no-caption set — provide a dataset.json "
                 '(list of {"video_path","caption"}) or drop --no-caption.')
    sh([PY, TRAINER / "scripts/caption_videos.py", clips_dir, "--output", ds_json], cwd=TRAINER)


def stage_preprocess(ds_json, pre, bucket, with_audio, gpu):
    if (pre / "latents").exists() and any((pre / "latents").rglob("*.pt")):
        print(f"[preprocess] latents exist in {pre}, skipping"); return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [PY, TRAINER / "scripts/process_dataset.py", ds_json,
           "--resolution-buckets", bucket, "--model-path", MODEL,
           "--text-encoder-path", GEMMA, "--video-column", "video_path", "--output-dir", pre]
    if with_audio:
        cmd.append("--with-audio")
    sh(cmd, env=env, cwd=TRAINER)


def stage_fix_conditions(clips_dir, pre):
    """process_dataset writes text embeds next to the source clip as <stem>.pt, NOT into
    conditions/. Mirror the latents/ tree into conditions/ by matching basename."""
    cond = pre / "conditions"
    latents = list((pre / "latents").rglob("*.pt"))
    made = 0
    for lat in latents:
        rel = lat.relative_to(pre / "latents")           # e.g. clips/foo.pt
        dst = cond / rel
        if dst.exists():
            continue
        # find the embed: <clips_dir>/**/<stem>.pt (saved next to source video)
        matches = list(Path(clips_dir).rglob(lat.name))
        embed = next((m for m in matches if m.parent != (pre / "latents" / rel.parent)), None)
        if embed is None:
            print(f"[fix] WARN no embed found for {rel} (looked for {lat.name} under {clips_dir})")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(embed, dst); made += 1
    print(f"[fix] conditions/: {made} embeds relocated, {len(latents)} latents total")
    n_cond = len(list(cond.rglob('*.pt'))) if cond.exists() else 0
    if n_cond < len(latents):
        sys.exit(f"[fix] conditions ({n_cond}) < latents ({len(latents)}) — training will fail to pair. "
                 "Check that captions produced embeds for every clip.")


def stage_config(pre, style, rank, steps, with_audio, out_dir, cfg_path):
    import yaml
    cfg = yaml.safe_load(open(TEMPLATE))
    cfg["model"]["model_path"] = MODEL
    cfg["model"]["text_encoder_path"] = GEMMA
    cfg["lora"]["rank"] = rank; cfg["lora"]["alpha"] = rank
    cfg["training_strategy"]["with_audio"] = with_audio
    cfg["optimization"]["steps"] = steps
    cfg["acceleration"]["quantization"] = "int8-quanto"     # fit alongside ComfyUI
    cfg["data"]["preprocessed_data_root"] = str(pre)
    cfg["validation"]["interval"] = max(steps // 4, 100)    # watch it learn on real runs
    cfg["validation"]["skip_initial_validation"] = True
    cfg["checkpoints"]["interval"] = max(steps // 4, 100)
    cfg["output_dir"] = str(out_dir)
    cfg["hub"]["push_to_hub"] = False; cfg["wandb"]["enabled"] = False
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    print(f"[config] wrote {cfg_path} (rank {rank}, {steps} steps, audio={with_audio})")


def stage_train(cfg_path, out_dir, gpu):
    ckpts = sorted(glob.glob(str(out_dir / "checkpoints" / "lora_weights_step_*.safetensors")))
    if ckpts:
        print(f"[train] checkpoint exists ({Path(ckpts[-1]).name}), skipping train"); return ckpts[-1]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    sh([PY, TRAINER / "scripts/train.py", cfg_path, "--disable-progress-bars"], env=env, cwd=TRAINER)
    ckpts = sorted(glob.glob(str(out_dir / "checkpoints" / "lora_weights_step_*.safetensors")))
    if not ckpts:
        sys.exit("[train] no LoRA checkpoint produced")
    return ckpts[-1]


def stage_wire(lora_path, style):
    loras = COMFY / "models/loras"; loras.mkdir(parents=True, exist_ok=True)
    link = loras / f"{style}.safetensors"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(lora_path)
    print(f"[wire] symlinked LoRA -> {link}")
    # emit a T2V workflow (API format) with LoraLoaderModelOnly on the fp4 base, if base_api exists
    if not os.path.exists(BASE_API):
        print(f"[wire] {BASE_API} not found — skip workflow gen (run one fp4 T2V in the UI first to seed it). "
              f"Manually: add LoraLoaderModelOnly(lora_name={style}.safetensors) between the checkpoint MODEL and the guider.")
        return
    api = json.load(open(BASE_API)); api.pop("4823", None)   # distilled-only
    # find the guider that consumes the checkpoint MODEL (slot 0) on the distilled path
    ckpt_id = next((n for n, v in api.items() if v["class_type"] == "CheckpointLoaderSimple"), None)
    guiders = [(n, inp) for n, v in api.items() for inp, val in v["inputs"].items()
               if isinstance(val, list) and val[0] == ckpt_id and val[1] == 0 and "Guider" in v["class_type"]]
    api["9001"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": [ckpt_id, 0], "lora_name": f"{style}.safetensors", "strength_model": 1.0}}
    for gid, inp in guiders:
        api[gid]["inputs"][inp] = ["9001", 0]
    out = COMFY / "user/default/workflows" / f"LTX-2.3_T2V_{style}_LoRA.json"
    json.dump(api, open(out, "w"), indent=1)
    print(f"[wire] wrote workflow (API) {out} — load via 'Open' or POST to /prompt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips_dir"); ap.add_argument("style")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bucket", default="768x512x49", help="WxHxFrames (H,W div 32; frames%%8==1)")
    ap.add_argument("--with-audio", action="store_true")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--no-wire", action="store_true")
    ap.add_argument("--work", default=None, help="work dir (default /mnt/data/video-lora/<style>)")
    a = ap.parse_args()

    work = Path(a.work or f"/mnt/data/video-lora/{a.style}")
    work.mkdir(parents=True, exist_ok=True)
    ds_json = work / "dataset.json"
    pre = work / "preprocessed"
    cfg_path = work / "config.yaml"
    out_dir = work / "lora-out"
    print(f"=== make_video_lora: style={a.style} | clips={a.clips_dir} | work={work} ===")

    stage_caption(a.clips_dir, ds_json, not a.no_caption)
    stage_preprocess(ds_json, pre, a.bucket, a.with_audio, a.gpu)
    stage_fix_conditions(a.clips_dir, pre)
    stage_config(pre, a.style, a.rank, a.steps, a.with_audio, out_dir, cfg_path)
    lora = stage_train(cfg_path, out_dir, a.gpu)
    print(f"\n✅ LoRA: {lora}")
    if not a.no_wire:
        stage_wire(lora, a.style)
    print(f"\n=== done: '{a.style}' LoRA ready. Load LTX-2.3_T2V_{a.style}_LoRA in ComfyUI (or add "
          f"LoraLoaderModelOnly {a.style}.safetensors) and generate. ===")


if __name__ == "__main__":
    main()
