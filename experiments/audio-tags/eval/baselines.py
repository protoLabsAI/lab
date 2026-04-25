"""Tier-0 sanity baselines for v2.

Two baselines, both evaluated on the same test set as v2:

1. Majority class — predict the modal class from train. Floor that any
   real model should clear.

2. Linear probe — Logistic regression / ridge on top of mean-pooled
   Whisper-tiny encoder features. Asks: "does our trunk + multi-head
   architecture add anything over a single linear layer on the same
   frozen features?"

For regression heads, baseline = mean of train target; probe = ridge.

Caches encoder features to .npz so re-runs are fast.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression, Ridge

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "training"))
sys.path.insert(0, str(HERE / "labels"))
from taxonomy import HEADS  # noqa: E402

WHISPER_MODEL = "openai/whisper-tiny"
SR = 16000
HIDDEN = 384

CACHE_DIR = Path("/mnt/data/audio-tags/cache/baseline_feats")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _to_16k_mono(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def extract_features(parquet_path: Path, n_subset: int | None,
                     cache_key: str, device: str = "cuda",
                     batch_size: int = 32) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (features, df) where features is (N, 384). Caches to npz."""
    cache = CACHE_DIR / f"{cache_key}.npz"
    df = pd.read_parquet(parquet_path)
    if n_subset is not None:
        df = df.sample(n=min(n_subset, len(df)), random_state=42).reset_index(drop=True)

    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if len(z["feats"]) == len(df):
            print(f"  cache hit: {cache}", flush=True)
            return z["feats"], df

    print(f"  extracting features for {len(df):,} samples → {cache}", flush=True)
    from transformers import WhisperModel, WhisperFeatureExtractor
    enc = WhisperModel.from_pretrained(WHISPER_MODEL).encoder.to(device).eval()
    feat = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)

    out = np.zeros((len(df), HIDDEN), dtype=np.float32)
    t0 = time.time()
    last_log = t0
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            chunk = df.iloc[i:i + batch_size]
            wavs = [_to_16k_mono(p) for p in chunk["audio_path"].tolist()]
            f = feat(wavs, sampling_rate=SR, return_tensors="pt").input_features.to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                e = enc(input_features=f).last_hidden_state.mean(dim=1).float()
            out[i:i + len(chunk)] = e.cpu().numpy()
            now = time.time()
            if now - last_log >= 30:
                done = i + len(chunk)
                rate = done / (now - t0)
                eta = (len(df) - done) / max(rate, 1e-3)
                print(f"    [{done}/{len(df)}]  {rate:.1f} samples/s  eta {eta:.0f}s", flush=True)
                last_log = now

    np.savez(cache, feats=out)
    return out, df


def f1_macro(preds: np.ndarray, tgts: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((preds == c) & (tgts == c)).sum())
        fp = int(((preds == c) & (tgts != c)).sum())
        fn = int(((preds != c) & (tgts == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))


def encode_classification(values, classes: tuple[str, ...]):
    """Returns (y, mask). y is int idx; mask True where label is valid."""
    y = np.zeros(len(values), dtype=int)
    mask = np.zeros(len(values), dtype=bool)
    for i, v in enumerate(values):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            if v in classes:
                y[i] = classes.index(v)
                mask[i] = True
    return y, mask


def encode_regression(values):
    y = np.zeros(len(values), dtype=np.float32)
    mask = np.zeros(len(values), dtype=bool)
    for i, v in enumerate(values):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            y[i] = float(v)
            mask[i] = True
    return y, mask


def majority_baseline(train_df, test_df) -> dict:
    """For each head, predict the most common train class on the entire
    test set. Regression: predict train mean."""
    out: dict[str, dict] = {}
    for h in HEADS:
        if h.name not in train_df.columns:
            continue
        if h.type == "classification":
            train_y, train_m = encode_classification(train_df[h.name].tolist(), h.classes)
            test_y, test_m = encode_classification(test_df[h.name].tolist(), h.classes)
            if test_m.sum() == 0:
                continue
            modal = int(np.bincount(train_y[train_m], minlength=len(h.classes)).argmax())
            preds = np.full(len(test_df), modal)
            n = int(test_m.sum())
            acc = float((preds[test_m] == test_y[test_m]).mean())
            f1 = f1_macro(preds[test_m], test_y[test_m], len(h.classes))
            out[h.name] = {"accuracy": acc, "f1_macro": f1, "n": n}
        else:
            train_y, train_m = encode_regression(train_df[h.name].tolist())
            test_y, test_m = encode_regression(test_df[h.name].tolist())
            if test_m.sum() == 0:
                continue
            mean_v = float(train_y[train_m].mean())
            preds = np.full(len(test_df), mean_v, dtype=np.float32)
            mae = float(np.mean(np.abs(preds[test_m] - test_y[test_m])))
            out[h.name] = {"mae": mae, "n": int(test_m.sum())}
    return out


def linear_probe(X_train, train_df, X_test, test_df) -> dict:
    out: dict[str, dict] = {}
    for h in HEADS:
        if h.name not in train_df.columns:
            continue
        if h.type == "classification":
            train_y, train_m = encode_classification(train_df[h.name].tolist(), h.classes)
            test_y, test_m = encode_classification(test_df[h.name].tolist(), h.classes)
            if train_m.sum() < 20 or test_m.sum() == 0:
                continue
            # Need at least 2 classes present
            present = set(train_y[train_m].tolist())
            if len(present) < 2:
                continue
            clf = LogisticRegression(max_iter=500, n_jobs=-1, C=1.0)
            clf.fit(X_train[train_m], train_y[train_m])
            preds = clf.predict(X_test[test_m])
            acc = float((preds == test_y[test_m]).mean())
            f1 = f1_macro(preds, test_y[test_m], len(h.classes))
            out[h.name] = {"accuracy": acc, "f1_macro": f1, "n": int(test_m.sum())}
        else:
            train_y, train_m = encode_regression(train_df[h.name].tolist())
            test_y, test_m = encode_regression(test_df[h.name].tolist())
            if train_m.sum() < 20 or test_m.sum() == 0:
                continue
            reg = Ridge(alpha=1.0)
            reg.fit(X_train[train_m], train_y[train_m])
            preds = reg.predict(X_test[test_m])
            mae = float(np.mean(np.abs(preds - test_y[test_m])))
            out[h.name] = {"mae": mae, "n": int(test_m.sum())}
    return out


def merge_v2(eval_json_path: Path) -> dict:
    if not eval_json_path.exists():
        return {}
    j = json.loads(eval_json_path.read_text())
    out = {}
    for k, v in j.get("classification", {}).items():
        out[k] = {"accuracy": v["accuracy"], "f1_macro": v["f1_macro"], "n": v["n"]}
    for k, v in j.get("regression", {}).items():
        out[k] = {"mae": v["mae"], "n": v["n"]}
    return out


def print_table(majority: dict, probe: dict, v2: dict):
    print("\n=== Tier-0 baseline comparison ===\n")
    cls_heads = [h for h in HEADS if h.type == "classification"]
    reg_heads = [h for h in HEADS if h.type == "regression"]

    print(f"{'head':18s}  {'majority':>16s}  {'linear probe':>16s}  {'v2 (ours)':>16s}")
    print("─" * 76)
    for h in cls_heads:
        m = majority.get(h.name, {})
        p = probe.get(h.name, {})
        v = v2.get(h.name, {})
        m_str = f"{m.get('accuracy',0)*100:5.1f}/{m.get('f1_macro',0):.2f}" if m else "—"
        p_str = f"{p.get('accuracy',0)*100:5.1f}/{p.get('f1_macro',0):.2f}" if p else "—"
        v_str = f"{v.get('accuracy',0)*100:5.1f}/{v.get('f1_macro',0):.2f}" if v else "—"
        print(f"{h.name:18s}  {m_str:>16s}  {p_str:>16s}  {v_str:>16s}")
    print("(classification: acc% / F1 macro)\n")
    for h in reg_heads:
        m = majority.get(h.name, {})
        p = probe.get(h.name, {})
        v = v2.get(h.name, {})
        m_str = f"MAE {m['mae']:.3f}" if m else "—"
        p_str = f"MAE {p['mae']:.3f}" if p else "—"
        v_str = f"MAE {v['mae']:.3f}" if v else "—"
        print(f"{h.name:18s}  {m_str:>16s}  {p_str:>16s}  {v_str:>16s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet",
                    default="/mnt/data/audio-tags/labels/labels-v2-train.parquet")
    ap.add_argument("--test-parquet",
                    default="/mnt/data/audio-tags/labels/labels-test-clean-with-dsp.parquet")
    ap.add_argument("--n-train", type=int, default=20000,
                    help="subsample train for the probe (full takes longer)")
    ap.add_argument("--v2-eval",
                    default="/mnt/data/training/audio-tags/v2-whisper/eval-test-clean-with-dsp.json")
    ap.add_argument("--out",
                    default="/mnt/data/training/audio-tags/v2-whisper/baselines.json")
    args = ap.parse_args()

    print("Extracting features (test) …", flush=True)
    X_test, test_df = extract_features(Path(args.test_parquet), None, "test_clean_with_dsp")
    print("Extracting features (train subset) …", flush=True)
    X_train, train_df = extract_features(Path(args.train_parquet), args.n_train,
                                         f"v2_train_subset_n{args.n_train}")

    print("\nMajority baseline …", flush=True)
    maj = majority_baseline(train_df, test_df)
    print("Linear probe …", flush=True)
    probe = linear_probe(X_train, train_df, X_test, test_df)
    v2 = merge_v2(Path(args.v2_eval))

    print_table(maj, probe, v2)

    Path(args.out).write_text(json.dumps(
        {"majority": maj, "linear_probe": probe, "v2": v2,
         "n_train_used": args.n_train},
        indent=2,
    ))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
