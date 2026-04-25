"""Minimal test UI for the audio-tags models.

Drop in mic or upload, pick a model, see all 11 head predictions
with confidences. No personalization, no recording — just inspection.

Launch:
  cd ~/dev/lab && CUDA_VISIBLE_DEVICES=1 uv run --project . python experiments/audio-tags/test_app.py
Access:
  http://protolabs:7871/    or    https://protolabs.taild25506.ts.net:8445/  (after `tailscale serve`)
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import WhisperFeatureExtractor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "training"))
sys.path.insert(0, str(HERE / "labels"))
from model import AudioTagModel  # noqa: E402
from taxonomy import HEADS  # noqa: E402

WHISPER_MODEL = "openai/whisper-tiny"
SR = 16000

MODELS: dict[str, Path] = {
    "v5-soft (flagship)":      Path("/mnt/data/training/audio-tags/v5-soft-balanced/best.ckpt"),
    "v4-multi":                Path("/mnt/data/training/audio-tags/v4-multidata/best.ckpt"),
    "v3-balanced":             Path("/mnt/data/training/audio-tags/v3-balanced/best.ckpt"),
    "v2":                      Path("/mnt/data/training/audio-tags/v2-whisper/best.ckpt"),
}

_cache: dict[str, tuple] = {}
_feat: WhisperFeatureExtractor | None = None


def _get_feat() -> WhisperFeatureExtractor:
    global _feat
    if _feat is None:
        _feat = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
    return _feat


def _load(model_name: str):
    if model_name in _cache:
        return _cache[model_name]
    ckpt_path = MODELS[model_name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    no_trunk = bool(ckpt.get("args", {}).get("no_trunk", False))
    model = AudioTagModel(freeze_encoder=True, no_trunk=no_trunk).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    _cache[model_name] = (model, device)
    return _cache[model_name]


def _to_16k_mono(audio):
    sr, wav = audio
    wav = wav.astype("float32", copy=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.abs(wav).max() or 1.0)
    if peak > 1.5:
        wav = wav / 32768.0
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


@torch.no_grad()
def tag(audio, model_name: str):
    if audio is None:
        return {"error": "no audio"}, None, None
    model, device = _load(model_name)
    wav = _to_16k_mono(audio)
    feats = _get_feat()(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        out = model(feats)

    structured: dict = {"speaker": {}, "mood": {}, "acoustic": {}, "style": None, "voice_quality": None}
    confidence: dict = {}
    bars: list[tuple[str, float]] = []

    for h in HEADS:
        z = out.logits[h.name][0]
        if h.type == "classification":
            probs = torch.softmax(z.float(), dim=-1)
            idx = int(probs.argmax())
            value = h.classes[idx]
            conf = round(float(probs[idx]), 3)
            confidence[h.name] = conf
            bars.append((f"{h.name}={value}", conf))
        else:
            value = float(z)
            if h.name == "snr_db":
                value *= 90.0
            value = round(value, 3)

        if h.name == "speaker_gender":
            structured["speaker"]["gender"] = value
        elif h.name == "speaker_age":
            structured["speaker"]["age"] = value
        elif h.name == "mood_class":
            structured["mood"]["class"] = value
        elif h.name in ("valence", "arousal"):
            structured["mood"][h.name] = value
        elif h.name in ("volume", "pitch", "speaking_speed", "snr_db", "environment"):
            structured["acoustic"][h.name] = value
        elif h.name == "speech_style":
            structured["style"] = value
        elif h.name == "voice_quality":
            structured["voice_quality"] = value

    structured["confidence"] = confidence
    bars_sorted = sorted(bars, key=lambda kv: -kv[1])
    bar_data = {label: c for label, c in bars_sorted}
    waveform_summary = {
        "duration_s": round(len(wav) / SR, 2),
        "rms": round(float(np.sqrt(np.mean(wav**2))), 4),
    }
    return structured, bar_data, waveform_summary


with gr.Blocks(title="audio-tags test") as demo:
    gr.Markdown(
        """
        # audio-tags — test

        Drop in audio or record from mic. Pick a model. See what the 11 heads predict.
        Use this to compare v2 → v5 on whatever you throw at it.
        """
    )

    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                            label="Audio input", scale=2)
        model_picker = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="v5-soft (flagship)",
            label="Model",
            scale=1,
        )
    btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        with gr.Column():
            structured = gr.JSON(label="Predictions")
        with gr.Column():
            bars = gr.Label(label="Per-head confidence (top → least confident)",
                            num_top_classes=11)
            audio_summary = gr.JSON(label="Audio summary", value={})

    btn.click(tag, [audio_in, model_picker], [structured, bars, audio_summary])

    gr.Markdown("---")
    gr.Markdown(
        """
        **Tips**
        - For voice_quality, try recording a whispered take — see if v5 calls it.
        - For mood_class, try expressive vs neutral takes (excited / sad / tense).
        - Switch the model dropdown to compare predictions on the same clip
          (your audio stays loaded; the result panel re-runs).
        """
    )


if __name__ == "__main__":
    # Pre-load v5 so the first prediction isn't slow
    try:
        _load("v5-soft (flagship)")
    except Exception as e:
        print(f"[warn] could not pre-load v5: {e}")
    demo.queue().launch(server_name="0.0.0.0", server_port=7871, share=False)
