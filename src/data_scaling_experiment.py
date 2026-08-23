"""Data scaling experiment: train baseline on different dataset sizes.

This script trains the frozen Whisper baseline (Whisper-tiny encoder +
mean pooling + logistic regression) on progressively larger subsets of
training data, keeping validation constant.

Experiment keeps constant:
- Validation set (400 samples from prepared val.csv)
- Architecture (frozen Whisper-tiny + mean pooling + logistic regression)
- Random seed (42)
- Evaluation methodology

Uses pre-cached Whisper features to run scaling experiments efficiently.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configuration
SEED = 42
MANIFEST_DIR = Path("data/manifests")
WHISPER_FEATURES_DIR = Path("data/whisper_features")
OUTPUT_DIR = Path("experiments/data_scaling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cached_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted Whisper features."""
    print("\nLoading cached Whisper features...")
    X_train = np.load(WHISPER_FEATURES_DIR / "X_train.npy")
    y_train = np.load(WHISPER_FEATURES_DIR / "y_train.npy")
    X_val = np.load(WHISPER_FEATURES_DIR / "X_val.npy")
    y_val = np.load(WHISPER_FEATURES_DIR / "y_val.npy")
    
    print(f"Train: {X_train.shape}, labels: {np.bincount(y_train)}")
    print(f"Val:   {X_val.shape}, labels: {np.bincount(y_val)}")
    return X_train, y_train, X_val, y_val


def train_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                    seed: int = SEED) -> Any:
    """Train baseline classifier: StandardScaler + LogisticRegression."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    model.fit(X_train, y_train)
    return model


def evaluate(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """Evaluate classifier on validation set."""
    y_pred = model.predict(X_val)
    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
    }


def main() -> None:
    """Run data scaling experiments."""
    X_train, y_train, X_val, y_val = load_cached_features()
    
    max_train = len(X_train)
    print(f"\nAvailable training data: {max_train:,} samples")
    print(f"Validation data: {len(X_val):,} samples")
    
    # Create subset sizes: 25%, 50%, 75%, 100%
    subset_sizes = []
    for frac in [0.25, 0.50, 0.75, 1.0]:
        size = int(max_train * frac)
        if size not in subset_sizes:
            subset_sizes.append(size)
    
    print(f"Testing subset sizes: {subset_sizes}\n")
    
    results = []
    
    print("=" * 80)
    print("DATA SCALING EXPERIMENT — Frozen Whisper + Mean Pooling + Logistic Regression")
    print("=" * 80 + "\n")
    
    for subset_size in subset_sizes:
        print(f"Training on {subset_size:,} samples...")
        X_sub = X_train[:subset_size]
        y_sub = y_train[:subset_size]
        
        # Train
        start = time.time()
        model = train_classifier(X_sub, y_sub)
        train_time = time.time() - start
        
        # Evaluate
        metrics = evaluate(model, X_val, y_val)
        
        result = {
            "n_train": subset_size,
            "n_val": len(X_val),
            "training_time_seconds": round(train_time, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        results.append(result)
        
        print(f"  Accuracy={result['accuracy']:.4f} F1={result['f1']:.4f} "
              f"Prec={result['precision']:.4f} Rec={result['recall']:.4f} "
              f"Time={result['training_time_seconds']:.3f}s\n")
    
    # Save results
    csv_path = OUTPUT_DIR / "results.csv"
    json_path = OUTPUT_DIR / "results.json"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nn_train | Accuracy | Precision | Recall  | F1     | Time(s)")
    print("-" * 70)
    for r in results:
        print(f"{r['n_train']:7d} | "
              f"{r['accuracy']:8.4f} | "
              f"{r['precision']:9.4f} | "
              f"{r['recall']:7.4f} | "
              f"{r['f1']:7.4f} | "
              f"{r['training_time_seconds']:6.4f}")
    
    print(f"\nResults saved to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
