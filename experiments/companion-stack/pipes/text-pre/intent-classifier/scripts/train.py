#!/usr/bin/env python3
"""Train intent classifier: sentence-transformer embeddings + linear head.

Implements the v0 pipeline from PLAN.md:
1. Encode all utterances with all-MiniLM-L6-v2 (384-dim)
2. Train a logistic regression (linear probe) as Tier-0 baseline
3. Train a small MLP as v0 model
4. Report macro F1, per-class metrics, confusion matrix
5. Export model for inference

Usage:
    python scripts/train.py                           # train on data/intent_train.jsonl
    python scripts/train.py --baseline-only            # just Tier-0 baselines
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    """Load JSONL file, return (texts, labels)."""
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            texts.append(sample["text"])
            labels.append(sample["label"])
    return texts, labels


def encode_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Encode texts to embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.array(embeddings)


def run_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict:
    """Run Tier-0 baselines: majority class, random, logistic regression."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score

    results = {}

    # Baseline 1: Majority class
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)
    y_pred = majority.predict(X_test)
    results["majority"] = {
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, target_names=label_names, zero_division=0),
    }
    print(f"\n=== Majority Class Baseline ===")
    print(f"Macro F1: {results['majority']['macro_f1']:.4f}")

    # Baseline 2: Random (stratified)
    rand = DummyClassifier(strategy="stratified", random_state=42)
    rand.fit(X_train, y_train)
    y_pred = rand.predict(X_test)
    results["random"] = {
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, target_names=label_names, zero_division=0),
    }
    print(f"\n=== Random Baseline ===")
    print(f"Macro F1: {results['random']['macro_f1']:.4f}")

    # Baseline 3: Linear probe (logistic regression on frozen embeddings)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results["linear_probe"] = {
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, target_names=label_names, zero_division=0),
    }
    print(f"\n=== Linear Probe (Logistic Regression) ===")
    print(f"Macro F1: {results['linear_probe']['macro_f1']:.4f}")
    print(results["linear_probe"]["report"])

    return results


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    hidden_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
) -> dict:
    """Train a small MLP classifier on frozen embeddings."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(label_names)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Small MLP: 384 → hidden → n_classes
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_size),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size, n_classes),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        probs = torch.softmax(logits, dim=-1)
        y_pred = logits.argmax(dim=-1).cpu().numpy()
        confidences = probs.max(dim=-1).values.cpu().numpy()

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== MLP Classifier (v0) ===")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)

    # Confidence calibration
    print("=== Confidence Calibration ===")
    for threshold in [0.7, 0.8, 0.85, 0.9, 0.95]:
        mask = confidences >= threshold
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_test[mask]).mean()
            coverage = mask.mean()
            print(f"  threshold={threshold:.2f}: accuracy={acc:.4f}, coverage={coverage:.2%}")

    return {
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(__file__).resolve().parent.parent / "data")
    data_dir = Path(args.data_dir)

    train_path = data_dir / "intent_train.jsonl"
    test_path = data_dir / "intent_test.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run scripts/generate_data.py first.")
        return

    # Load data
    print("Loading data...")
    train_texts, train_labels = load_jsonl(train_path)
    test_texts, test_labels = load_jsonl(test_path)
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Build label index
    label_names = sorted(set(train_labels))
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    y_train = np.array([label_to_idx[l] for l in train_labels])
    y_test = np.array([label_to_idx[l] for l in test_labels])
    print(f"  Classes: {label_names}")

    # Encode
    print("\nEncoding with all-MiniLM-L6-v2...")
    t0 = time.time()
    all_texts = train_texts + test_texts
    all_embeddings = encode_texts(all_texts)
    X_train = all_embeddings[: len(train_texts)]
    X_test = all_embeddings[len(train_texts) :]
    print(f"  Encoded {len(all_texts)} texts in {time.time()-t0:.1f}s")
    print(f"  Embedding shape: {X_train.shape}")

    # Tier-0 baselines
    baseline_results = run_baselines(X_train, y_train, X_test, y_test, label_names)

    if args.baseline_only:
        return

    # Train MLP
    print("\nTraining MLP classifier...")
    mlp_results = train_mlp(
        X_train, y_train, X_test, y_test, label_names,
        hidden_size=args.hidden_size, epochs=args.epochs,
    )

    # Save results
    results_path = data_dir.parent / "eval" / "results_v0.json"
    results_path.parent.mkdir(exist_ok=True)
    results = {
        "baselines": {
            k: {"macro_f1": v["macro_f1"]} for k, v in baseline_results.items()
        },
        "mlp": {
            "macro_f1": mlp_results["macro_f1"],
            "confusion_matrix": mlp_results["confusion_matrix"],
        },
        "label_names": label_names,
        "train_size": len(train_texts),
        "test_size": len(test_texts),
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Save model
    import torch

    model_dir = data_dir.parent / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": mlp_results["model"].state_dict(),
            "label_names": label_names,
            "hidden_size": args.hidden_size,
            "embedding_dim": X_train.shape[1],
        },
        model_dir / "intent_classifier_v0.pt",
    )
    print(f"Model saved to {model_dir / 'intent_classifier_v0.pt'}")


if __name__ == "__main__":
    main()
