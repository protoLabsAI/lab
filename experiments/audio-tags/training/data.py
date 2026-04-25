"""Parquet-backed dataset: row → (log-mel features, target dict, mask dict).

Loads audio from disk lazily, runs WhisperFeatureExtractor, encodes
labels using the taxonomy class index. Rows missing a per-head label
are masked out at loss time (see model.compute_loss).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labels"))
from taxonomy import HEADS, HEADS_BY_NAME  # noqa: E402

WHISPER_MODEL = "openai/whisper-tiny"
SR = 16000


class AudioTagDataset(Dataset):
    def __init__(self, parquet_path: str | Path, max_seconds: float = 30.0):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.feat = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
        self.max_samples = int(max_seconds * SR)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio, sr = sf.read(row["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            # LibriSpeech is 16k; this branch is defensive for mixed sources later.
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]

        feats = self.feat(audio, sampling_rate=SR, return_tensors="pt").input_features[0]

        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for h in HEADS:
            v = row.get(h.name)
            present = v is not None and not (isinstance(v, float) and np.isnan(v))
            if h.type == "classification":
                if present and v in h.classes:
                    targets[h.name] = torch.tensor(h.classes.index(v), dtype=torch.long)
                    masks[h.name] = torch.tensor(True)
                else:
                    targets[h.name] = torch.tensor(0, dtype=torch.long)  # placeholder
                    masks[h.name] = torch.tensor(False)
            else:  # regression
                if present:
                    val = float(v)
                    # Normalize regression targets so MSE is on the same
                    # scale as CE (~O(1)).
                    if h.name == "snr_db":
                        val = val / 90.0
                    # valence/arousal already in [-1, 1] (emotion2vec convention)
                    targets[h.name] = torch.tensor(val, dtype=torch.float32)
                    masks[h.name] = torch.tensor(True)
                else:
                    targets[h.name] = torch.tensor(0.0, dtype=torch.float32)
                    masks[h.name] = torch.tensor(False)

        return {"input_features": feats, "targets": targets, "masks": masks}


def collate(batch: list[dict]) -> dict:
    out = {
        "input_features": torch.stack([b["input_features"] for b in batch]),
        "targets": {h.name: torch.stack([b["targets"][h.name] for b in batch]) for h in HEADS},
        "masks": {h.name: torch.stack([b["masks"][h.name] for b in batch]) for h in HEADS},
    }
    return out
