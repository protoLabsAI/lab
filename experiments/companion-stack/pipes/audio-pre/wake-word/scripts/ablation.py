#!/usr/bin/env python3
"""
Wake word model ablation study.

Trains multiple variants and evaluates each on the same test set
in streaming mode. Outputs a comparison table.

Usage:
    source /mnt/data/training/wake-word/env/bin/activate
    python scripts/ablation.py
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

BASE_DIR = Path("/mnt/data/training/wake-word")
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "hey_orbis.yml"
TRAIN_SCRIPT = BASE_DIR / "openWakeWord" / "openwakeword" / "train.py"
RESULTS_DIR = BASE_DIR / "ablation_results"


def load_base_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Each ablation: (name, config_overrides)
ABLATIONS = [
    # v0 baseline (reproduce current model)
    ("v0-baseline", {}),

    # More negative weight (push FA penalty harder)
    ("v1-neg3000", {"max_negative_weight": 3000}),

    # Even more negative weight
    ("v2-neg6000", {"max_negative_weight": 6000}),

    # Larger model (64-dim layers instead of 32)
    ("v3-layer64", {"layer_size": 64}),

    # Larger model + more negative weight
    ("v4-layer64-neg3000", {"layer_size": 64, "max_negative_weight": 3000}),

    # More training steps
    ("v5-100k", {"steps": 100000}),

    # More augmentation rounds (needs re-augmentation)
    ("v6-aug4", {"augmentation_rounds": 4}),

    # More adversarial negatives + larger neg weight
    ("v7-moreneg", {
        "max_negative_weight": 3000,
        "custom_negative_phrases": [
            "hey orbit", "hey gorgeous", "a orbis", "hey boris",
            "hey or", "hey orbits", "hey norse", "hey office",
            "hey august", "hey ardis",
            # Additional confusables
            "hey morris", "hey norris", "hey tortoise",
            "hey fortress", "hey porpoise", "hey service",
            "hey courteous", "hey orca", "day orbis",
            "they orbis", "play orbis", "say orbis",
        ],
    }),

    # Kitchen sink: larger model + more neg weight + more steps
    ("v8-kitchen-sink", {
        "layer_size": 64,
        "max_negative_weight": 6000,
        "steps": 100000,
    }),
]


def make_config(name, overrides):
    """Create an ablation config with overrides applied."""
    cfg = load_base_config()
    output_dir = str(BASE_DIR / "ablation_output" / name)
    os.makedirs(output_dir, exist_ok=True)
    cfg["output_dir"] = output_dir

    for k, v in overrides.items():
        cfg[k] = v

    config_dir = RESULTS_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{name}.yml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path, output_dir


def train_model(config_path, name, needs_augment=False):
    """Train a single model variant."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    # Only re-augment if config changes affect augmentation
    if needs_augment:
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--training_config", str(config_path),
            "--augment_clips",
        ]
        print(f"  Augmenting clips...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"  AUGMENT FAILED: {result.stderr[-500:]}")
            return False

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--training_config", str(config_path),
        "--train_model",
    ]
    print(f"  Training model...")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start
    print(f"  Training took {elapsed:.0f}s")

    if result.returncode != 0:
        # TFLite conversion may fail (onnx_tf missing) but ONNX is still saved
        stderr = result.stderr[-500:] if result.stderr else ""
        if "onnx_tf" in stderr or "onnx-tf" in stderr:
            print(f"  TFLite conversion failed (expected), ONNX should be saved")
        else:
            print(f"  TRAIN FAILED: {stderr}")
            return False

    return True


def evaluate_streaming(onnx_path, threshold=0.5):
    """Evaluate model in streaming mode on test clips."""
    from openwakeword.model import Model
    import scipy.io.wavfile as wav

    model = Model(
        wakeword_models=[onnx_path],
        inference_framework="onnx",
    )

    base = BASE_DIR / "output" / "hey_orbis"

    def test_clips(clip_dir, limit=None):
        import glob as g
        clips = sorted(g.glob(os.path.join(clip_dir, "*.wav")))
        if limit:
            clips = clips[:limit]
        scores = []
        for clip_path in clips:
            sr, audio = wav.read(clip_path)
            pre = np.zeros(int(sr * 2.0), dtype=np.int16)
            post = np.zeros(int(sr * 1.0), dtype=np.int16)
            full = np.concatenate([pre, audio, post])
            model.reset()
            max_score = 0.0
            for i in range(0, len(full) - 1280, 1280):
                prediction = model.predict(full[i : i + 1280].astype(np.int16))
                score = list(prediction.values())[0]
                max_score = max(max_score, score)
            scores.append(max_score)
        return np.array(scores)

    pos_scores = test_clips(str(base / "positive_test"))
    neg_scores = test_clips(str(base / "negative_test"))

    results = {}
    for t in [0.3, 0.5, 0.7]:
        recall = (pos_scores > t).sum() / len(pos_scores) * 100
        fa_rate = (neg_scores > t).sum() / len(neg_scores) * 100
        results[f"recall@{t}"] = recall
        results[f"fa@{t}"] = fa_rate

    results["pos_mean"] = float(pos_scores.mean())
    results["pos_median"] = float(np.median(pos_scores))
    results["neg_mean"] = float(neg_scores.mean())
    results["neg_median"] = float(np.median(neg_scores))
    results["n_pos"] = len(pos_scores)
    results["n_neg"] = len(neg_scores)

    return results


def evaluate_features(onnx_path):
    """Evaluate model directly on pre-extracted features."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    base = BASE_DIR / "output" / "hey_orbis"

    results = {}
    for name, path in [
        ("pos_test", base / "positive_features_test.npy"),
        ("neg_test", base / "negative_features_test.npy"),
    ]:
        features = np.load(str(path))
        scores = []
        for i in range(len(features)):
            r = sess.run(None, {"x": features[i : i + 1].astype(np.float32)})
            scores.append(r[0][0][0])
        scores = np.array(scores)
        results[f"{name}_recall@0.5"] = float((scores > 0.5).sum() / len(scores) * 100)
        results[f"{name}_mean"] = float(scores.mean())

    return results


def run_ablations(names=None):
    """Run all (or selected) ablations."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, overrides in ABLATIONS:
        if names and name not in names:
            continue

        config_path, output_dir = make_config(name, overrides)

        # Check if augmentation-affecting params changed
        needs_augment = any(
            k in overrides
            for k in [
                "augmentation_rounds",
                "custom_negative_phrases",
                "n_samples",
                "rir_paths",
            ]
        )

        success = train_model(config_path, name, needs_augment=needs_augment)
        if not success:
            all_results[name] = {"error": "training failed"}
            continue

        onnx_path = os.path.join(output_dir, "hey_orbis.onnx")
        if not os.path.exists(onnx_path):
            all_results[name] = {"error": "no ONNX output"}
            continue

        print(f"  Evaluating (streaming)...")
        streaming = evaluate_streaming(onnx_path)

        print(f"  Evaluating (features)...")
        features = evaluate_features(onnx_path)

        result = {
            "config": overrides,
            "streaming": streaming,
            "features": features,
            "onnx_path": onnx_path,
            "onnx_size_kb": os.path.getsize(onnx_path) / 1024,
        }
        all_results[name] = result

        # Print summary
        s = streaming
        print(f"  RESULTS: recall@0.5={s['recall@0.5']:.1f}% fa@0.5={s['fa@0.5']:.1f}%"
              f" recall@0.7={s['recall@0.7']:.1f}% fa@0.7={s['fa@0.7']:.1f}%")

    # Save results
    results_path = RESULTS_DIR / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    print_comparison(all_results)

    return all_results


def print_comparison(results):
    """Print a markdown comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"| {'Name':<25} | {'Recall@0.5':>10} | {'FA@0.5':>8} | {'Recall@0.7':>10} | {'FA@0.7':>8} | {'Size':>6} |")
    print(f"|{'-'*27}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*8}|")

    for name, data in results.items():
        if "error" in data:
            print(f"| {name:<25} | {'ERROR':>10} | {'':>8} | {'':>10} | {'':>8} | {'':>6} |")
            continue
        s = data["streaming"]
        size = f"{data['onnx_size_kb']:.0f}KB"
        print(f"| {name:<25} | {s['recall@0.5']:>9.1f}% | {s['fa@0.5']:>7.1f}% | {s['recall@0.7']:>9.1f}% | {s['fa@0.7']:>7.1f}% | {size:>6} |")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wake word ablation study")
    parser.add_argument("--names", nargs="*", help="Run only these ablations")
    parser.add_argument("--show", action="store_true", help="Show existing results")
    args = parser.parse_args()

    if args.show:
        results_path = RESULTS_DIR / "ablation_results.json"
        if results_path.exists():
            with open(results_path) as f:
                print_comparison(json.load(f))
        else:
            print("No results yet. Run ablations first.")
    else:
        run_ablations(names=args.names)
