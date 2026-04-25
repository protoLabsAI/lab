"""Whisper-tiny encoder + multi-head classifier.

Frozen Whisper encoder → mean-pool → shared trunk → per-attribute heads
(one Linear per classification class set, one Linear per regression
target). Loss is per-head, summed (uncertainty weighting deferred to v1).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labels"))
from taxonomy import HEADS, HEADS_BY_NAME, SCHEMA_VERSION  # noqa: E402

WHISPER_MODEL = "openai/whisper-tiny"
HIDDEN_DIM = 384  # whisper-tiny encoder hidden size
TRUNK_DIM = 256


@dataclass
class ModelOutput:
    logits: dict[str, torch.Tensor]   # name → (B, n_classes) for classification, (B, 1) for regression
    pooled: torch.Tensor              # (B, HIDDEN_DIM)


class AudioTagModel(nn.Module):
    def __init__(self, freeze_encoder: bool = True, no_trunk: bool = False):
        """
        no_trunk=True turns the trunk + per-head MLP into just Linear(384, n_classes)
        per head — the literal linear probe variant. ~50 K trainable params instead
        of ~110 K, makes the apples-to-apples sklearn-probe comparison clean.
        """
        super().__init__()
        whisper = WhisperModel.from_pretrained(WHISPER_MODEL)
        self.encoder = whisper.encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.no_trunk = no_trunk
        if no_trunk:
            self.trunk = nn.Identity()
            head_in = HIDDEN_DIM
        else:
            self.trunk = nn.Sequential(
                nn.Linear(HIDDEN_DIM, TRUNK_DIM),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            head_in = TRUNK_DIM

        self.heads = nn.ModuleDict()
        for h in HEADS:
            if h.type == "classification":
                self.heads[h.name] = nn.Linear(head_in, len(h.classes))
            else:
                self.heads[h.name] = nn.Linear(head_in, 1)

    def forward(self, input_features: torch.Tensor) -> ModelOutput:
        # input_features: (B, 80, 3000) log-mel
        enc = self.encoder(input_features=input_features).last_hidden_state  # (B, 1500, 384)
        pooled = enc.mean(dim=1)  # (B, 384)
        h = self.trunk(pooled)
        logits = {name: head(h) for name, head in self.heads.items()}
        return ModelOutput(logits=logits, pooled=pooled)


# v0 multi-task loss weights. Classification heads get the highest
# weight; regression heads are normalized to [0,1]-scale targets (see
# data.py) so MSE is comparable to CE. 'unknown' classes get a lower
# weight so the model isn't rewarded for predicting unknown.
LOSS_WEIGHTS: dict[str, float] = {
    "speaker_gender": 2.0,
    "speaker_age": 0.5,       # weak labels
    "mood_class": 1.0,
    "valence": 1.0,
    "arousal": 1.0,
    "volume": 1.0,
    "pitch": 1.0,
    "speaking_speed": 1.0,
    "snr_db": 1.0,
    "environment": 0.5,        # nearly all 'indoor_quiet' on LibriSpeech
    "speech_style": 1.0,
    "voice_quality": 2.0,      # ORBIS-priority head, weight up
}


def compute_loss(
    output: ModelOutput,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None = None,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of per-head losses. `masks[name]` is (B,) bool; rows
    where the label is missing/unknown are masked out of that head's loss.
    If `class_weights[name]` is provided, used as the `weight=` arg to
    cross_entropy — this is the "v3 class-weighted loss" alternative to
    using a WeightedRandomSampler."""
    masks = masks or {}
    class_weights = class_weights or {}
    total = torch.zeros((), device=next(iter(output.logits.values())).device)
    per_head: dict[str, float] = {}

    for h in HEADS:
        if h.name not in targets:
            continue
        logits = output.logits[h.name]
        target = targets[h.name]
        mask = masks.get(h.name)
        if mask is not None and mask.sum() == 0:
            per_head[h.name] = 0.0
            continue

        if h.type == "classification":
            cw = class_weights.get(h.name)
            loss = F.cross_entropy(logits, target, weight=cw, reduction="none")
        else:  # regression
            loss = F.mse_loss(logits.squeeze(-1), target.float(), reduction="none")

        if mask is not None:
            loss = loss[mask]
        loss = loss.mean() if loss.numel() > 0 else torch.zeros((), device=loss.device)

        per_head[h.name] = float(loss.detach().item())
        total = total + LOSS_WEIGHTS.get(h.name, 1.0) * loss

    return total, per_head


def class_index(head_name: str, label: str) -> int:
    h = HEADS_BY_NAME[head_name]
    if label not in h.classes:
        raise KeyError(f"{label!r} not in {head_name} classes {h.classes}")
    return h.classes.index(label)


__all__ = ["AudioTagModel", "ModelOutput", "compute_loss", "class_index",
           "WHISPER_MODEL", "HIDDEN_DIM", "TRUNK_DIM", "SCHEMA_VERSION"]
