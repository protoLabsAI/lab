"""Evaluate a trained audio-tags checkpoint on a labels parquet.

Per-head accuracy / F1 / MAE, plus a confusion matrix per classification
head and a latency micro-bench at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from data import AudioTagDataset, collate  # noqa: E402
from model import AudioTagModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labels"))
from taxonomy import HEADS, HEADS_BY_NAME  # noqa: E402


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


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds: dict[str, list[int]] = defaultdict(list)
    tgts: dict[str, list[int]] = defaultdict(list)
    reg_pred: dict[str, list[float]] = defaultdict(list)
    reg_tgt: dict[str, list[float]] = defaultdict(list)

    for batch in loader:
        feat = batch["input_features"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(feat)
        for h in HEADS:
            mask = batch["masks"][h.name].bool()
            if mask.sum() == 0:
                continue
            if h.type == "classification":
                p = out.logits[h.name].argmax(dim=-1).cpu().numpy()
                t = batch["targets"][h.name].cpu().numpy()
                preds[h.name].extend(p[mask.numpy()].tolist())
                tgts[h.name].extend(t[mask.numpy()].tolist())
            else:
                p = out.logits[h.name].squeeze(-1).float().cpu().numpy()
                t = batch["targets"][h.name].cpu().numpy()
                reg_pred[h.name].extend(p[mask.numpy()].tolist())
                reg_tgt[h.name].extend(t[mask.numpy()].tolist())

    metrics = {"classification": {}, "regression": {}, "confusion": {}}
    for name, ps in preds.items():
        h = HEADS_BY_NAME[name]
        ps_a = np.array(ps)
        ts_a = np.array(tgts[name])
        if len(ps_a) == 0:
            continue
        metrics["classification"][name] = {
            "accuracy": float((ps_a == ts_a).mean()),
            "f1_macro": f1_macro(ps_a, ts_a, len(h.classes)),
            "n": int(len(ps_a)),
        }
        cm = np.zeros((len(h.classes), len(h.classes)), dtype=int)
        for p, t in zip(ps_a, ts_a):
            cm[t, p] += 1
        metrics["confusion"][name] = {
            "classes": list(h.classes),
            "matrix": cm.tolist(),
        }

    for name, ps in reg_pred.items():
        ts = np.array(reg_tgt[name])
        ps_a = np.array(ps)
        if len(ps_a) == 0:
            continue
        metrics["regression"][name] = {
            "mae": float(np.mean(np.abs(ps_a - ts))),
            "rmse": float(np.sqrt(np.mean((ps_a - ts) ** 2))),
            "n": int(len(ps_a)),
        }

    return metrics


def latency_bench(model, device, n: int = 50) -> dict:
    model.eval()
    feat = torch.randn(1, 80, 3000, device=device)
    # warmup
    for _ in range(5):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(feat)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(feat)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"mean_ms": 1000 * elapsed / n, "n": n, "device": str(device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    no_trunk = bool(ckpt.get("args", {}).get("no_trunk", False))
    model = AudioTagModel(freeze_encoder=True, no_trunk=no_trunk).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded ckpt epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_total_loss')} "
          f"no_trunk={no_trunk}", flush=True)

    ds = AudioTagDataset(args.parquet)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    metrics = evaluate(model, loader, device)
    metrics["latency"] = latency_bench(model, device)

    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
