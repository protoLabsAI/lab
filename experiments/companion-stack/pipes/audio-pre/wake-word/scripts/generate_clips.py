#!/usr/bin/env python3
"""Generate synthetic "hey orbis" clips using Fish Audio S2 Pro.

Replaces openWakeWord's default Piper TTS generation with higher-quality
Fish Audio voices. Produces 16kHz mono WAV files in the directory structure
expected by openWakeWord's training pipeline:

    {output_dir}/{model_name}/positive_train/   (80% of positive clips)
    {output_dir}/{model_name}/positive_test/    (20% of positive clips)
    {output_dir}/{model_name}/negative_train/   (80% of adversarial clips)
    {output_dir}/{model_name}/negative_test/    (20% of adversarial clips)

Usage:
    python scripts/generate_clips.py --n-positive 5000 --n-adversarial 2000
    python scripts/generate_clips.py --dry-run
"""

import argparse
import io
import json
import struct
import time
import wave
from pathlib import Path

import requests
from tqdm import tqdm

FISH_URL = "http://localhost:8092/v1/tts"
MODEL_NAME = "hey_orbis"

# Voices with actual audio references on the running Fish Audio instance
VOICES = ["voice_01", "voice_02", "voice_03", "voice_04", "josh_sample_1"]

# Target phrase variations for prosody diversity
POSITIVE_PHRASES = [
    "hey orbis",
    "Hey Orbis",
    "Hey ORBIS",
    "hey orbis!",
    "Hey Orbis.",
    "hey, orbis",
]

# Adversarial negatives — phonetically similar phrases the model should reject
ADVERSARIAL_PHRASES = [
    "hey orbit",
    "hey boris",
    "hey gorgeous",
    "a orbis",
    "hey orbits",
    "hey norse",
    "hey office",
    "hey august",
    "hey ardis",
    "hey or bis",
    "the orbis",
    "say orbis",
    "my orbis",
    "hey oris",
    "hey orbus",
]

# Temperature/top_p combos for prosody variation
GENERATION_PARAMS = [
    {"temperature": 0.6, "top_p": 0.7},
    {"temperature": 0.7, "top_p": 0.8},
    {"temperature": 0.8, "top_p": 0.8},
    {"temperature": 0.9, "top_p": 0.9},
    {"temperature": 1.0, "top_p": 0.9},
]

FISH_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
TRAIN_SPLIT = 0.8  # 80% train, 20% test


def resample_to_16k(pcm_data: bytes, src_rate: int = FISH_SAMPLE_RATE) -> bytes:
    """Resample raw int16 PCM from src_rate to 16kHz using linear interpolation."""
    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_data)

    ratio = TARGET_SAMPLE_RATE / src_rate
    out_len = int(n_samples * ratio)
    resampled = []

    for i in range(out_len):
        src_idx = i / ratio
        idx = int(src_idx)
        frac = src_idx - idx

        if idx + 1 < n_samples:
            val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            val = samples[min(idx, n_samples - 1)]

        resampled.append(int(max(-32768, min(32767, val))))

    return struct.pack(f"<{len(resampled)}h", *resampled)


def generate_clip(
    text: str,
    voice: str,
    params: dict,
    seed: int | None = None,
) -> bytes | None:
    """Generate a single clip via Fish Audio, return 16kHz mono WAV bytes."""
    body = {
        "text": text,
        "format": "wav",
        "streaming": False,
        "reference_id": voice,
        "chunk_length": 200,
        "normalize": True,
        "repetition_penalty": 1.1,
        "max_new_tokens": 512,
        "use_memory_cache": "off",
        **params,
    }
    if seed is not None:
        body["seed"] = seed

    try:
        resp = requests.post(FISH_URL, json=body, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARN: Fish Audio request failed: {e}")
        return None

    audio_bytes = resp.content

    # Parse the WAV to extract raw PCM
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
            assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth() * 8}-bit"
            src_rate = wf.getframerate()
            pcm_data = wf.readframes(wf.getnframes())
    except Exception:
        pcm_data = audio_bytes
        src_rate = FISH_SAMPLE_RATE

    # Resample to 16kHz
    if src_rate != TARGET_SAMPLE_RATE:
        pcm_16k = resample_to_16k(pcm_data, src_rate)
    else:
        pcm_16k = pcm_data

    # Write as 16kHz mono WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm_16k)

    return buf.getvalue()


def generate_and_split(
    phrases: list[str],
    n_total: int,
    train_dir: Path,
    test_dir: Path,
    label: str,
) -> tuple[int, int]:
    """Generate n_total clips, split 80/20 into train/test dirs."""
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(n_total * TRAIN_SPLIT)
    n_test = n_total - n_train

    n_voices = len(VOICES)
    n_params = len(GENERATION_PARAMS)
    n_phrases = len(phrases)

    generated = 0
    failed = 0
    train_count = 0
    test_count = 0

    with tqdm(total=n_total, desc=label) as pbar:
        i = 0
        while generated < n_total:
            phrase = phrases[i % n_phrases]
            voice = VOICES[i % n_voices]
            params = GENERATION_PARAMS[i % n_params]
            seed = i * 7 + 42

            wav_bytes = generate_clip(phrase, voice, params, seed=seed)

            if wav_bytes is not None:
                # First n_train go to train, rest to test
                if train_count < n_train:
                    fname = f"{label}_train_{train_count:05d}.wav"
                    (train_dir / fname).write_bytes(wav_bytes)
                    train_count += 1
                else:
                    fname = f"{label}_test_{test_count:05d}.wav"
                    (test_dir / fname).write_bytes(wav_bytes)
                    test_count += 1
                generated += 1
                pbar.update(1)
            else:
                failed += 1
                if failed > n_total * 0.1:
                    print(f"\nERROR: Too many failures ({failed}), aborting")
                    break

            i += 1

            if i % 20 == 0:
                time.sleep(0.5)

    return train_count, test_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic wake word clips with Fish Audio S2 Pro"
    )
    parser.add_argument("--n-positive", type=int, default=5000,
                        help="Number of positive clips (default: 5000)")
    parser.add_argument("--n-adversarial", type=int, default=2000,
                        help="Number of adversarial negative clips (default: 2000)")
    parser.add_argument("--output-dir", type=str,
                        default="/mnt/data/training/wake-word/output",
                        help="openWakeWord output_dir from config")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help="Model name (directory under output_dir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate 10 clips per category for testing")
    args = parser.parse_args()

    if args.dry_run:
        args.n_positive = 10
        args.n_adversarial = 10
        print("=== DRY RUN: 10 clips per category ===\n")

    base = Path(args.output_dir) / args.model_name
    print(f"Output: {base}")
    print(f"Voices: {VOICES}")
    print(f"Split:  {TRAIN_SPLIT:.0%} train / {1-TRAIN_SPLIT:.0%} test")
    print()

    # Check Fish Audio is reachable
    try:
        health = requests.get("http://localhost:8092/v1/health", timeout=5)
        print(f"Fish Audio health: {health.status_code}")
    except requests.ConnectionError:
        print("ERROR: Fish Audio not reachable at localhost:8092")
        return

    # Generate positive clips → positive_train/ + positive_test/
    print(f"\n=== Generating {args.n_positive} positive clips ===")
    pos_train, pos_test = generate_and_split(
        POSITIVE_PHRASES, args.n_positive,
        base / "positive_train", base / "positive_test",
        "positive",
    )

    # Generate adversarial clips → negative_train/ + negative_test/
    print(f"\n=== Generating {args.n_adversarial} adversarial clips ===")
    neg_train, neg_test = generate_and_split(
        ADVERSARIAL_PHRASES, args.n_adversarial,
        base / "negative_train", base / "negative_test",
        "adversarial",
    )

    # Write manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fish_url": FISH_URL,
        "voices": VOICES,
        "positive_phrases": POSITIVE_PHRASES,
        "adversarial_phrases": ADVERSARIAL_PHRASES,
        "positive_train": pos_train,
        "positive_test": pos_test,
        "negative_train": neg_train,
        "negative_test": neg_test,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "train_split": TRAIN_SPLIT,
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n=== Done ===")
    print(f"Positive:    {pos_train} train + {pos_test} test")
    print(f"Adversarial: {neg_train} train + {neg_test} test")
    print(f"Manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
