"""Train Whisper-tiny + multi-head model on labels parquet.

Single GPU 1 (CUDA_VISIBLE_DEVICES=1), bf16 mixed precision, AdamW,
cosine LR. Saves best ckpt by val total loss.

Usage:
  CUDA_VISIBLE_DEVICES=1 python training/train.py \
    --train-parquet /mnt/data/audio-tags/labels/labels-31k.parquet \
    --val-parquet   /mnt/data/audio-tags/labels/labels-test-clean.parquet \
    --out-dir /mnt/data/training/audio-tags/v0/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import AudioTagDataset, collate  # noqa: E402
from model import AudioTagModel, compute_loss, SCHEMA_VERSION  # noqa: E402
from torch.utils.data import WeightedRandomSampler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labels"))
from taxonomy import HEADS, HEADS_BY_NAME  # noqa: E402


def compute_class_weights(df, device, power: float = 1.0) -> dict[str, torch.Tensor]:
    """Per-head inverse-frequency weights (raised to `power`), normalized
    to mean 1.0. Used as `weight=` in cross_entropy.

    power=1.0 → vanilla inverse-frequency (full re-weighting)
    power=0.5 → sqrt of inverse-frequency (gentler — keeps majority class
                from being over-corrected away from)
    power=0.0 → uniform (no class weighting)
    """
    import numpy as np
    weights: dict[str, torch.Tensor] = {}
    for h in HEADS:
        if h.type != "classification":
            continue
        if h.name not in df.columns:
            continue
        counts = df[h.name].value_counts()
        n_present = sum(int(counts.get(c, 0)) for c in h.classes)
        if n_present == 0:
            continue
        w = np.ones(len(h.classes), dtype=np.float32)
        for i, c in enumerate(h.classes):
            n = int(counts.get(c, 0))
            if n > 0:
                inv_freq = n_present / (n * len(h.classes))
                w[i] = inv_freq ** power
            else:
                w[i] = 0.0
        if w.sum() > 0:
            w = w * (len(h.classes) / w.sum())
        weights[h.name] = torch.tensor(w, dtype=torch.float32, device=device)
    return weights


def make_balanced_sampler(df) -> WeightedRandomSampler:
    """Per-sample weights balanced across (voice_quality × speaking_speed).
    Rows with NaN on either column fall back to the mean weight so they
    aren't dropped from sampling.
    """
    import numpy as np
    n = len(df)

    def _inv_freq(col: str) -> np.ndarray:
        counts = df[col].value_counts(dropna=False)
        w_per = {k: 1.0 / c for k, c in counts.items()}
        w = df[col].map(w_per).astype(float).to_numpy()
        # Fill NaN weights (from NaN category values) with the mean
        mean_w = np.nanmean(w) if np.isfinite(np.nanmean(w)) else 1.0 / n
        w = np.where(np.isfinite(w), w, mean_w)
        return w

    w_speed = _inv_freq("speaking_speed") if "speaking_speed" in df.columns else np.ones(n) / n
    w_voice = _inv_freq("voice_quality") if "voice_quality" in df.columns else np.ones(n) / n
    sample_weights = w_speed * w_voice
    # Final NaN/Inf guard — multinomial refuses any non-finite weight
    sample_weights = np.where(np.isfinite(sample_weights), sample_weights,
                              float(np.nanmean(sample_weights)))
    # And must be non-negative
    sample_weights = np.clip(sample_weights, 0.0, None)
    return WeightedRandomSampler(sample_weights, num_samples=n, replacement=True)


def move(d: dict, device) -> dict:
    if isinstance(d, torch.Tensor):
        return d.to(device, non_blocking=True)
    return {k: move(v, device) for k, v in d.items()}


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None) -> dict:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    seen: dict[str, int] = defaultdict(int)
    total_loss_sum = 0.0
    n = 0

    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch["input_features"])
            total, per_head = compute_loss(out, batch["targets"], batch["masks"],
                                            class_weights=class_weights)
        total_loss_sum += float(total.item()) * batch["input_features"].size(0)
        n += batch["input_features"].size(0)
        for k, v in per_head.items():
            sums[k] += v
            counts[k] += 1

        # Per-class accuracy
        for h in HEADS:
            if h.type != "classification":
                continue
            mask = batch["masks"][h.name]
            if mask.sum() == 0:
                continue
            preds = out.logits[h.name].argmax(dim=-1)
            tgts = batch["targets"][h.name]
            m = mask.bool()
            correct[h.name] += int((preds[m] == tgts[m]).sum().item())
            seen[h.name] += int(m.sum().item())

    metrics = {
        "val_total_loss": total_loss_sum / max(n, 1),
        "per_head_loss": {k: sums[k] / max(counts[k], 1) for k in sums},
        "per_head_acc": {k: correct[k] / max(seen[k], 1) for k in correct},
    }
    return metrics


def train_one_epoch(model, loader, optim, scheduler, device,
                     log_every: int = 50, class_weights=None) -> dict:
    model.train()
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    total_loss_sum = 0.0
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch["input_features"])
            total, per_head = compute_loss(out, batch["targets"], batch["masks"],
                                            class_weights=class_weights)

        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        total_loss_sum += float(total.item()) * batch["input_features"].size(0)
        n += batch["input_features"].size(0)
        for k, v in per_head.items():
            sums[k] += v
            counts[k] += 1

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = n / max(elapsed, 1e-3)
            print(f"  step {step+1}  loss={total_loss_sum / n:.4f}  "
                  f"{rate:.1f} samples/s  lr={scheduler.get_last_lr()[0]:.2e}",
                  flush=True)

    return {"train_total_loss": total_loss_sum / max(n, 1),
            "per_head_loss": {k: sums[k] / max(counts[k], 1) for k in sums}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--unfreeze-encoder", action="store_true")
    ap.add_argument("--weighted-sampler", action="store_true",
                    help="Oversample by inverse speaking_speed frequency")
    ap.add_argument("--no-trunk", action="store_true",
                    help="Linear probe — drop the trunk, heads operate on encoder features directly")
    ap.add_argument("--class-weighted-loss", action="store_true",
                    help="Apply per-head inverse-frequency class weights inside CE loss "
                         "(alternative to --weighted-sampler).")
    ap.add_argument("--class-weight-power", type=float, default=1.0,
                    help="Exponent on inverse-frequency weights. 1.0=full, "
                         "0.5=sqrt (gentler), 0.0=uniform.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)

    print(f"Loading datasets...", flush=True)
    train_ds = AudioTagDataset(args.train_parquet)
    val_ds = AudioTagDataset(args.val_parquet)
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}", flush=True)

    if args.weighted_sampler:
        sampler = make_balanced_sampler(train_ds.df)
        print(f"Using WeightedRandomSampler (speaking_speed × voice_quality)", flush=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=True, drop_last=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True,
    )

    print(f"Building model (freeze_encoder={not args.unfreeze_encoder}, "
          f"no_trunk={args.no_trunk})", flush=True)
    model = AudioTagModel(
        freeze_encoder=not args.unfreeze_encoder, no_trunk=args.no_trunk,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_trainable/1e6:.2f}M / total: {n_total/1e6:.2f}M", flush=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    total_steps = math.ceil(len(train_loader)) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps, pct_start=0.05, anneal_strategy="cos"
    )

    best_val = float("inf")
    history: list[dict] = []

    class_weights = None
    if args.class_weighted_loss:
        class_weights = compute_class_weights(train_ds.df, device,
                                              power=args.class_weight_power)
        print(f"Using class-weighted CE loss (power={args.class_weight_power}). Weights:", flush=True)
        for name, w in class_weights.items():
            print(f"  {name}: {[round(x, 3) for x in w.tolist()]}", flush=True)

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===", flush=True)
        train_metrics = train_one_epoch(model, train_loader, optim, scheduler, device,
                                         class_weights=class_weights)
        val_metrics = evaluate(model, val_loader, device, class_weights=class_weights)
        print(f"  val_total_loss = {val_metrics['val_total_loss']:.4f}", flush=True)
        for k, v in sorted(val_metrics["per_head_acc"].items()):
            print(f"    acc[{k}] = {v:.3f}", flush=True)

        record = {"epoch": epoch + 1, **train_metrics, **val_metrics}
        history.append(record)
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["val_total_loss"] < best_val:
            best_val = val_metrics["val_total_loss"]
            ckpt_path = out_dir / "best.ckpt"
            torch.save({
                "schema": SCHEMA_VERSION,
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_total_loss": best_val,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ↳ saved {ckpt_path} (val_total_loss={best_val:.4f})", flush=True)

    print(f"\nDone. best val_total_loss={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
