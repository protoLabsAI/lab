"""FastAPI tag service.

POST /tag    multipart audio file → JSON tag dict
GET  /healthz
GET  /schema → taxonomy JSON

Wire format mirrors taxonomy.example_output(). ORBIS calls this once per
utterance (post-VAD) with the same audio bytes it sends to Whisper.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from transformers import WhisperFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from model import AudioTagModel, WHISPER_MODEL  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labels"))
from taxonomy import HEADS, HEADS_BY_NAME, SCHEMA_VERSION  # noqa: E402

CKPT_PATH = Path("/mnt/data/training/audio-tags/v0/best.ckpt")
SR = 16000

app = FastAPI(title="audio-tags", version=SCHEMA_VERSION)
_state: dict = {}


class HealthResp(BaseModel):
    status: str
    schema_version: str
    device: str


def _load():
    ckpt_path = Path(_state.get("ckpt_path", CKPT_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioTagModel(freeze_encoder=True).to(device).eval()
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded ckpt: {ckpt_path}", flush=True)
    else:
        print(f"WARNING: no ckpt at {ckpt_path} — serving random weights", flush=True)
    feat = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
    _state.update(model=model, device=device, feat=feat)


@app.on_event("startup")
def _startup():
    _load()


@app.get("/healthz", response_model=HealthResp)
def healthz():
    return HealthResp(
        status="ok",
        schema_version=SCHEMA_VERSION,
        device=str(_state.get("device", "uninitialized")),
    )


@app.get("/schema")
def schema():
    return {
        "version": SCHEMA_VERSION,
        "heads": [
            {"name": h.name, "type": h.type,
             "classes": list(h.classes), "range": h.range}
            for h in HEADS
        ],
    }


def _format_response(logits: dict[str, torch.Tensor]) -> dict:
    out: dict = {"schema": SCHEMA_VERSION}
    speaker: dict = {}
    mood: dict = {}
    acoustic: dict = {}
    confidence: dict = {}

    for h in HEADS:
        z = logits[h.name][0]
        if h.type == "classification":
            probs = torch.softmax(z, dim=-1)
            idx = int(probs.argmax().item())
            value = h.classes[idx]
            conf = float(probs[idx].item())
        else:
            value = float(z.item())
            conf = None

        # Route to nested groups
        if h.name == "speaker_gender":
            speaker["gender"] = value
        elif h.name == "speaker_age":
            speaker["age"] = value
        elif h.name == "mood_class":
            mood["class"] = value
        elif h.name in ("valence", "arousal"):
            mood[h.name] = value
        elif h.name in ("volume", "pitch", "speaking_speed", "snr_db", "environment"):
            acoustic[h.name] = value
        elif h.name == "speech_style":
            out["style"] = value

        if conf is not None:
            confidence[h.name] = round(conf, 3)

    out["speaker"] = speaker
    out["mood"] = mood
    out["acoustic"] = acoustic
    out["confidence"] = confidence
    return out


@app.post("/tag")
async def tag(audio: UploadFile = File(...)):
    raw = await audio.read()
    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {e}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        try:
            import librosa
        except ImportError:
            raise HTTPException(400, f"sample rate {sr} != 16000 and librosa missing")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)

    feat = _state["feat"]
    model = _state["model"]
    device = _state["device"]

    feats = feat(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        out = model(feats)

    return _format_response(out.logits)
