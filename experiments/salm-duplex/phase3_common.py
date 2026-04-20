"""
Shared helpers for Phase 3 conversational dataset converters.

All Phase 3 converters (Switchboard, AMI, Emilia) emit the same output shape:
- Per-dialogue MonoCut with concatenated audio
- SupervisionSegments per utterance, speaker="user" or "agent"
- custom.target_audio = same recording (monologue-style duplex target)
- Output: train.jsonl.gz + val.jsonl.gz + shar-train/ + shar-val/
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioSource


TARGET_SAMPLE_RATE = 16000  # source side — training resamples to 22050 for target audio


@dataclass
class Utterance:
    """One turn within a dialogue."""
    id: str
    speaker_role: str      # "user" or "agent"
    text: str
    audio: np.ndarray      # 1-D float32, mono
    sample_rate: int
    start_offset: float = 0.0  # set during concatenation


def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Resample to target_sr if needed. Uses scipy for quality."""
    if sr == target_sr:
        return audio, sr
    from scipy.signal import resample_poly
    g = np.gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    out = resample_poly(audio, up, down).astype(np.float32)
    return out, target_sr


def decode_audio_bytes(raw: bytes, to_mono: bool = True) -> tuple[np.ndarray, int]:
    """Decode a WAV/MP3/FLAC bytes blob to float32 mono @ its native sample rate."""
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if to_mono and audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def build_dialogue_cut(
    cut_id: str,
    utterances: list[Utterance],
    output_audio_dir: Path,
    dataset_name: str,
    gap_seconds: float = 0.1,
) -> MonoCut | None:
    """
    Concatenate utterances in order (each already in the correct sample rate),
    save as a single WAV, return a MonoCut with per-utterance supervisions.

    Inserts a small silence gap between utterances to avoid clicks and
    better reflect real turn transitions.
    """
    if not utterances:
        return None

    sample_rate = utterances[0].sample_rate
    assert all(u.sample_rate == sample_rate for u in utterances), "mixed sample rates not allowed"

    segments: list[np.ndarray] = []
    supervisions: list[SupervisionSegment] = []
    cumulative = 0.0
    gap_samples = int(gap_seconds * sample_rate)
    silence = np.zeros(gap_samples, dtype=np.float32)

    for i, u in enumerate(utterances):
        if i > 0:
            segments.append(silence)
            cumulative += gap_seconds
        duration = len(u.audio) / sample_rate
        supervisions.append(SupervisionSegment(
            id=u.id,
            recording_id=cut_id,
            start=cumulative,
            duration=duration,
            text=u.text,
            speaker=u.speaker_role,
            language="en",
        ))
        segments.append(u.audio)
        cumulative += duration

    full = np.concatenate(segments)
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_audio_dir / f"{cut_id}.wav"
    sf.write(str(wav_path), full, sample_rate)

    # Re-read the file to get the exact on-disk sample count. MP3 decoders and
    # resample_poly can produce lengths that differ slightly from what we
    # compute; Lhotse will error if Recording metadata diverges from the file.
    info = sf.info(str(wav_path))
    actual_samples = info.frames
    actual_duration = actual_samples / sample_rate

    # Clamp any supervision that now extends past the actual file duration.
    for sup in supervisions:
        if sup.start + sup.duration > actual_duration:
            sup.duration = max(0.0, actual_duration - sup.start)

    recording = Recording(
        id=cut_id,
        sources=[AudioSource(type="file", channels=[0], source=str(wav_path))],
        sampling_rate=sample_rate,
        num_samples=actual_samples,
        duration=actual_duration,
    )

    return MonoCut(
        id=cut_id,
        start=0.0,
        duration=actual_duration,
        channel=0,
        recording=recording,
        supervisions=supervisions,
        custom={
            "target_audio": recording,
            "num_turns": len(utterances),
            "dataset": dataset_name,
        },
    )


def split_and_save(
    cuts: list[MonoCut],
    output_manifest_dir: Path,
    train_ratio: float = 0.95,
    seed: int = 42,
    export_shar: bool = True,
    shar_num_jobs: int = 1,
) -> tuple[int, int]:
    """Shuffle, split, and save as JSONL + optional SHAR. Returns (n_train, n_val)."""
    random.seed(seed)
    random.shuffle(cuts)
    split_idx = int(len(cuts) * train_ratio)
    train_cuts = CutSet.from_cuts(cuts[:split_idx])
    val_cuts = CutSet.from_cuts(cuts[split_idx:])

    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = output_manifest_dir / "train.jsonl.gz"
    val_jsonl = output_manifest_dir / "val.jsonl.gz"
    train_cuts.to_jsonl(str(train_jsonl))
    val_cuts.to_jsonl(str(val_jsonl))

    if export_shar:
        for split_name, split in [("shar-train", train_cuts), ("shar-val", val_cuts)]:
            shar_dir = output_manifest_dir / split_name
            shar_dir.mkdir(exist_ok=True)
            split.to_shar(
                output_dir=str(shar_dir),
                fields={"recording": "wav", "target_audio": "wav"},
                shard_size=200,
                num_jobs=shar_num_jobs,
            )

    return len(train_cuts), len(val_cuts)


def report_stats(label: str, cuts: list[MonoCut]) -> None:
    if not cuts:
        print(f"  {label}: 0 cuts")
        return
    total_dur = sum(c.duration for c in cuts)
    avg_turns = sum(len(c.supervisions) for c in cuts) / len(cuts)
    print(f"  {label}: {len(cuts):>6d} cuts  |  {total_dur/3600:>6.1f}h  |  {avg_turns:>4.1f} turns/cut")
