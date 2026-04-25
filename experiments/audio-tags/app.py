"""Quick-and-dirty Gradio personalization app for audio-tags.

Single-owner research tool. Two tabs:
  1. Record — prompt + style dropdown + mic → save to held-out manifest
  2. Tune & Test — one-button fine-tune on your clips, then record any
     prompt and compare base-model predictions vs personalized.

Launch:
  cd ~/dev/lab && uv run --project . python experiments/audio-tags/app.py
Access:
  http://protolabs:7870/  (on tailnet)  or  http://localhost:7870/
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "training"))
sys.path.insert(0, str(HERE / "labels"))
from model import AudioTagModel, compute_loss  # noqa: E402
from taxonomy import HEADS, HEADS_BY_NAME  # noqa: E402

# ── Paths ───────────────────────────────────────────────────────────────
HELD_OUT = Path("/mnt/data/audio-tags/held_out")
MANIFEST = HELD_OUT / "manifest.jsonl"
BASE_CKPT = Path("/mnt/data/training/audio-tags/v1-full/best.ckpt")
V2_CKPT = Path("/mnt/data/training/audio-tags/v2-whisper/best.ckpt")
PERSONALIZED_CKPT = Path("/mnt/data/training/audio-tags/personalized/best.ckpt")
WHISPER_MODEL = "openai/whisper-tiny"
SR = 16000

# ── Prompts + style → labels ────────────────────────────────────────────
PROMPTS = [
    "The weather has been strange this week.",
    "Set a timer for fifteen minutes please.",
    "What's on the agenda for the meeting?",
    "Remind me to call my mother on Sunday.",
    "Send a message to Alex about the project.",
    "Play some quiet music for the evening.",
    "Tell me what you remember about yesterday.",
    "Could you help me find my keys?",
    "Read me the headlines from this morning.",
    "I'd like to add an event to my calendar.",
]

# Maps style name → label dict that will be written into the manifest.
# Only listed heads are supervised for that clip; everything else is masked
# at fine-tune time.
STYLE_LABELS: dict[str, dict] = {
    "voiced_neutral":  {"voice_quality": "voiced"},
    "whispered":       {"voice_quality": "whispered"},
    "excited":         {"mood_class": "excited", "valence": 0.7, "arousal": 0.6},
    "sad":             {"mood_class": "sad", "valence": -0.5, "arousal": -0.3},
    "tired":           {"mood_class": "neutral", "valence": -0.1, "arousal": -0.5},
    "fast":            {"speaking_speed": "fast"},
    "slow":            {"speaking_speed": "slow"},
    "shouting":        {"volume": "loud", "mood_class": "excited",
                        "valence": 0.4, "arousal": 0.8},
}

STYLES = list(STYLE_LABELS.keys())

# ── Lazy model cache ────────────────────────────────────────────────────
_models: dict[str, tuple] = {}
_feat: WhisperFeatureExtractor | None = None


def _load_feat() -> WhisperFeatureExtractor:
    global _feat
    if _feat is None:
        _feat = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
    return _feat


def _ckpt_for(name: str) -> Path | None:
    for key, path in [("base", BASE_CKPT), ("v2", V2_CKPT),
                      ("personalized", PERSONALIZED_CKPT)]:
        if name == key and path.exists():
            return path
    return None


def get_model(name: str = "base") -> tuple | None:
    """Load & cache a model by name. Returns (model, device) or None."""
    ckpt_path = _ckpt_for(name)
    if ckpt_path is None:
        return None
    # Cache-bust personalized since it retrains
    if name in _models and name != "personalized":
        return _models[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioTagModel(freeze_encoder=True).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[{name}] missing={len(missing)} unexpected={len(unexpected)}")
    _models[name] = (model, device)
    return _models[name]


def _to_16k_mono(audio: tuple[int, np.ndarray]) -> np.ndarray:
    sr, wav = audio
    wav = wav.astype("float32", copy=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.abs(wav).max() or 1.0)
    if peak > 1.5:  # int16 → float
        wav = wav / 32768.0
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


@torch.no_grad()
def predict(audio, model_name: str) -> dict:
    if audio is None:
        return {"error": "no audio"}
    m = get_model(model_name)
    if m is None:
        return {"error": f"no {model_name} checkpoint available"}
    model, device = m
    wav = _to_16k_mono(audio)
    feats = _load_feat()(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        out = model(feats)
    result = {}
    for h in HEADS:
        z = out.logits[h.name][0]
        if h.type == "classification":
            probs = torch.softmax(z.float(), dim=-1)
            idx = int(probs.argmax().item())
            result[h.name] = f"{h.classes[idx]} ({float(probs[idx])*100:.0f}%)"
        else:
            val = float(z.item())
            if h.name == "snr_db":
                val = val * 90.0  # denormalize
            result[h.name] = round(val, 3)
    return result


# ── Recording: save to disk + append to manifest ────────────────────────
def save_recording(audio, prompt_idx, style, user_gender):
    if audio is None:
        return "⚠️ No audio captured.", load_manifest_table()
    if style not in STYLE_LABELS:
        return f"⚠️ Unknown style '{style}'.", load_manifest_table()

    wav = _to_16k_mono(audio)
    duration = len(wav) / SR
    if duration < 0.3:
        return f"⚠️ Too short ({duration:.1f}s). Try again.", load_manifest_table()

    ts = int(time.time())
    style_dir = HELD_OUT / style
    style_dir.mkdir(parents=True, exist_ok=True)
    out_path = style_dir / f"prompt_{prompt_idx:02d}_{ts}.wav"
    sf.write(out_path, wav, SR)

    row = {
        "audio_path": str(out_path),
        "prompt_id": int(prompt_idx),
        "prompt_text": PROMPTS[int(prompt_idx)],
        "style": style,
        "speaker_gender": user_gender,
        "duration": duration,
        "timestamp": ts,
        **STYLE_LABELS[style],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return f"✅ Saved {out_path.name}  ({duration:.1f}s)", load_manifest_table()


def load_manifest_table(limit: int = 60):
    if not MANIFEST.exists():
        return []
    rows = []
    with MANIFEST.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                rows.append([
                    Path(d["audio_path"]).name,
                    d["style"],
                    f"{d['duration']:.1f}s",
                    d["prompt_text"][:50],
                ])
            except Exception:
                continue
    return rows[-limit:]


def manifest_stats():
    if not MANIFEST.exists():
        return "No recordings yet."
    counts: dict[str, int] = {}
    total = 0
    with MANIFEST.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                counts[d["style"]] = counts.get(d["style"], 0) + 1
                total += 1
            except Exception:
                continue
    lines = [f"**Total:** {total} clips"]
    for s in STYLES:
        n = counts.get(s, 0)
        lines.append(f"- {s}: {n}")
    return "\n".join(lines)


# ── Fine-tune: small run on user clips ──────────────────────────────────
def _class_idx(head_name: str, value) -> int | None:
    h = HEADS_BY_NAME[head_name]
    if value in h.classes:
        return h.classes.index(value)
    return None


def _build_user_batch(rows: list[dict], batch_size: int = 8,
                     max_duration: float = 12.0):
    picks = random.sample(rows, min(batch_size, len(rows)))
    feats_list: list[torch.Tensor] = []
    targets: dict[str, list] = {h.name: [] for h in HEADS}
    masks: dict[str, list] = {h.name: [] for h in HEADS}
    feat = _load_feat()
    for r in picks:
        wav, sr = sf.read(r["audio_path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        if len(wav) > SR * max_duration:
            wav = wav[: int(SR * max_duration)]
        f = feat(wav, sampling_rate=SR, return_tensors="pt").input_features[0]
        feats_list.append(f)

        for h in HEADS:
            val = None
            if h.name == "speaker_gender":
                val = r.get("speaker_gender")
            elif h.name in r:
                val = r[h.name]
            present = False
            if h.type == "classification":
                idx = _class_idx(h.name, val) if val is not None else None
                if idx is not None:
                    targets[h.name].append(idx)
                    present = True
                else:
                    targets[h.name].append(0)
            else:
                if val is not None:
                    v = float(val)
                    if h.name == "snr_db":
                        v = v / 90.0
                    targets[h.name].append(v)
                    present = True
                else:
                    targets[h.name].append(0.0)
            masks[h.name].append(present)

    return (
        torch.stack(feats_list),
        {k: torch.tensor(v, dtype=torch.long if HEADS_BY_NAME[k].type == "classification" else torch.float32)
         for k, v in targets.items()},
        {k: torch.tensor(v, dtype=torch.bool) for k, v in masks.items()},
    )


def finetune_personalized(steps: int, lr: float, progress=gr.Progress()):
    if not MANIFEST.exists():
        return "⚠️ No recordings yet."
    rows = []
    with MANIFEST.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if len(rows) < 8:
        return f"⚠️ Only {len(rows)} clips — need at least 8 to fine-tune."

    base = get_model("v2") or get_model("base")
    if base is None:
        return "⚠️ No base checkpoint found."
    # Start from a fresh copy so we don't mutate the cached base
    src_model, device = base
    model = AudioTagModel(freeze_encoder=True).to(device).train()
    model.load_state_dict(src_model.state_dict())

    # Unfreeze trunk + heads (encoder stays frozen — honors "tiny" budget)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trunk.parameters():
        p.requires_grad = True
    for p in model.heads.parameters():
        p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[finetune] trainable: {sum(p.numel() for p in trainable)/1e6:.3f}M params, {len(rows)} clips")

    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
    batch_size = min(8, len(rows))

    losses = []
    for step in progress.tqdm(range(steps), desc="fine-tuning"):
        feats, targets, masks = _build_user_batch(rows, batch_size=batch_size)
        feats = feats.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = model(feats)
            total, per_head = compute_loss(out, targets, masks)
        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        losses.append(float(total.item()))

    model.eval()
    PERSONALIZED_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "source_ckpt": str(BASE_CKPT if not V2_CKPT.exists() else V2_CKPT),
        "user_clips": len(rows),
        "steps": steps,
        "final_loss": losses[-1],
    }, PERSONALIZED_CKPT)
    _models.pop("personalized", None)

    return (f"✅ Personalized on {len(rows)} clips, {steps} steps. "
            f"loss: {losses[0]:.3f} → {losses[-1]:.3f}. "
            f"Saved to {PERSONALIZED_CKPT.name}")


def compare_predict(audio):
    """Run both base and personalized, return side-by-side."""
    base = predict(audio, "v2") if _ckpt_for("v2") else predict(audio, "base")
    personalized = predict(audio, "personalized")
    return base, personalized


def which_base_label() -> str:
    if _ckpt_for("v2"):
        return "v2 (with whisper head)"
    if _ckpt_for("base"):
        return "v1 (no whisper head)"
    return "no checkpoint"


# ── UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="audio-tags personalization") as demo:
    gr.Markdown(f"""
    # audio-tags — personalize on your voice

    Record a handful of clips across speech styles, then click **Personalize**.
    The model's heads + trunk fine-tune on your voice (encoder stays frozen —
    still 8 M params). Test in the second tab.

    Base model: **{which_base_label()}**
    Manifest: `{MANIFEST}`
    """)

    with gr.Tab("1. Record"):
        with gr.Row():
            user_gender = gr.Radio(
                ["female", "male", "unknown"], value="unknown",
                label="Your voice (set once per session)",
            )
        with gr.Row():
            with gr.Column(scale=2):
                prompt_idx = gr.Dropdown(
                    choices=[(f"{i}. {p}", i) for i, p in enumerate(PROMPTS)],
                    value=0, label="Prompt",
                )
                prompt_text = gr.Markdown(value=f"> _{PROMPTS[0]}_")

                def update_prompt(i):
                    return f"> _{PROMPTS[int(i)]}_"
                prompt_idx.change(update_prompt, [prompt_idx], [prompt_text])

                style = gr.Dropdown(choices=STYLES, value="voiced_neutral",
                                    label="Speaking style")
                audio_in = gr.Audio(sources=["microphone"], type="numpy",
                                    label="Record take")
                save_btn = gr.Button("Save recording", variant="primary")
                status = gr.Markdown()

            with gr.Column(scale=1):
                stats = gr.Markdown(value=manifest_stats())
                refresh_btn = gr.Button("Refresh stats")
                refresh_btn.click(lambda: manifest_stats(), None, [stats])

        manifest_tbl = gr.Dataframe(
            headers=["file", "style", "dur", "prompt"],
            value=load_manifest_table(),
            label="Recent recordings",
            interactive=False,
        )

        save_btn.click(
            save_recording,
            [audio_in, prompt_idx, style, user_gender],
            [status, manifest_tbl],
        ).then(
            lambda: manifest_stats(), None, [stats]
        )

    with gr.Tab("2. Tune & Test"):
        with gr.Row():
            steps = gr.Slider(20, 400, value=120, step=20, label="Training steps")
            lr = gr.Slider(1e-5, 5e-4, value=1e-4, step=1e-5, label="Learning rate")
            tune_btn = gr.Button("Personalize on my clips", variant="primary")
        tune_status = gr.Markdown()
        tune_btn.click(finetune_personalized, [steps, lr], [tune_status])

        gr.Markdown("---")
        gr.Markdown("### Test: record anything, compare base vs personalized")
        test_audio = gr.Audio(sources=["microphone"], type="numpy",
                              label="Test take")
        test_btn = gr.Button("Predict with both models")
        with gr.Row():
            out_base = gr.JSON(label=f"Base ({which_base_label()})")
            out_personal = gr.JSON(label="Personalized")
        test_btn.click(compare_predict, [test_audio], [out_base, out_personal])


if __name__ == "__main__":
    HELD_OUT.mkdir(parents=True, exist_ok=True)
    # Eager-load the base model so the first prediction isn't slow
    get_model("v2") or get_model("base")
    demo.queue().launch(server_name="0.0.0.0", server_port=7870, share=False)
