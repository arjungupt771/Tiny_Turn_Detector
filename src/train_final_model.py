"""
Final model training for the turn-detection classifier.

This script:
  1. Reproduces the two baselines already established
     (hand-crafted acoustic features vs. Whisper-tiny encoder
     embeddings, both evaluated on the *same* 1600/400 manifest
     split so the comparison is apples-to-apples).
  2. Sweeps a small set of classifier heads on top of the Whisper
     embeddings (the representation that actually works) with
     5-fold CV on train + a held-out val check, and picks the best
     by CV F1.
  3. Persists the winning pipeline + a metadata/results file that
     the README and the Gradio app both read from.

The classifier head is intentionally tiny (logistic regression /
linear SVM / a 1-2 layer MLP) because the heavy lifting is already
done by the frozen Whisper-tiny encoder (39M params). We are only
learning a ~400-parameter linear probe on top of it, which is what
keeps the whole thing fast and cheap to train/retrain.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import joblib


MODEL_DIR = Path("models")
EXPERIMENTS_DIR = Path("experiments")
MODEL_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)

WHISPER_DIR = Path("data/whisper_features")
CONTROLLED_ACOUSTIC_DIR = Path("data/controlled_features/acoustic")
LARGE_ACOUSTIC_DIR = Path("data/features")

WHISPER_MODEL_NAME = "openai/whisper-tiny"
EMBEDDING_DIM = 384  # whisper-tiny encoder hidden size


def load_split(feature_dir: Path):
    X_train = np.load(feature_dir / "X_train.npy")
    y_train = np.load(feature_dir / "y_train.npy")
    X_val = np.load(feature_dir / "X_val.npy")
    y_val = np.load(feature_dir / "y_val.npy")
    return X_train, y_train, X_val, y_val


def eval_pipeline(pipe, X_val, y_val):
    pred = pipe.predict(X_val)
    return {
        "accuracy": float(accuracy_score(y_val, pred)),
        "f1": float(f1_score(y_val, pred)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_val, pred).tolist(),
    }


def run_baselines(results):
    """Reproduce the two comparison baselines for the report."""

    print("=" * 70)
    print("BASELINE 1: hand-crafted acoustic features (large set, 8000/2000)")
    print("=" * 70)
    X_train, y_train, X_val, y_val = load_split(LARGE_ACOUSTIC_DIR)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]
    )
    pipe.fit(X_train, y_train)
    metrics = eval_pipeline(pipe, X_val, y_val)
    print(json.dumps(metrics, indent=2))
    results["acoustic_baseline_large"] = {
        "description": "47-dim hand-crafted acoustic features (pitch/energy/pause stats), "
        "logistic regression, larger 8000/2000 split",
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        **metrics,
    }

    print("\n" + "=" * 70)
    print("BASELINE 2: same acoustic features, controlled 1600/400 split")
    print("(same exact samples used for the Whisper comparison)")
    print("=" * 70)
    X_train, y_train, X_val, y_val = load_split(CONTROLLED_ACOUSTIC_DIR)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    metrics = eval_pipeline(pipe, X_val, y_val)
    print(json.dumps(metrics, indent=2))
    results["acoustic_baseline_controlled"] = {
        "description": "Same 54-dim acoustic features, evaluated on the identical "
        "1600/400 split used for Whisper, to remove 'more data' as a confound",
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        **metrics,
    }


def sweep_whisper_heads(results):
    print("\n" + "=" * 70)
    print("WHISPER-TINY EMBEDDINGS: classifier head sweep")
    print("=" * 70)

    X_train, y_train, X_val, y_val = load_split(WHISPER_DIR)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    candidates = {
        "logreg_C0.1": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=0.1, random_state=42))]
        ),
        "logreg_C1": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=42))]
        ),
        "logreg_C10": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=10.0, random_state=42))]
        ),
        "svm_rbf_C1": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]
        ),
        "svm_rbf_C5": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=5.0, probability=True, random_state=42))]
        ),
        "mlp_64": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        alpha=1e-3,
                        max_iter=2000,
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
        "mlp_32_16": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        alpha=1e-3,
                        max_iter=2000,
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sweep_results = {}
    best_name, best_pipe, best_cv_f1 = None, None, -1.0

    for name, pipe in candidates.items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1")
        pipe.fit(X_train, y_train)
        val_metrics = eval_pipeline(pipe, X_val, y_val)

        sweep_results[name] = {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            **val_metrics,
        }

        print(
            f"{name:15s} cv_f1={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}"
        )

        if cv_scores.mean() > best_cv_f1:
            best_cv_f1 = cv_scores.mean()
            best_name = name
            best_pipe = pipe

    print(f"\nSelected head: {best_name} (best 5-fold CV F1 = {best_cv_f1:.4f})")

    results["whisper_head_sweep"] = sweep_results
    results["selected_model"] = {
        "head": best_name,
        "cv_f1_mean": float(best_cv_f1),
        **eval_pipeline(best_pipe, X_val, y_val),
    }

    return best_name, best_pipe


def main():
    results = {}

    run_baselines(results)
    best_name, best_pipe = sweep_whisper_heads(results)

    # --------------------------------------------------
    # Persist final model
    # --------------------------------------------------
    model_path = MODEL_DIR / "turn_detector_whisper.joblib"
    joblib.dump(best_pipe, model_path)

    meta = {
        "embedding_model": WHISPER_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "pooling": "mean over encoder time steps",
        "classifier_head": best_name,
        "sample_rate": 16000,
        "label_map": {"0": "CONTINUE", "1": "END"},
        "trained_on": "pipecat-ai/smart-turn-data-v3.2-train (2000-sample subset, seed=42)",
    }
    with open(MODEL_DIR / "turn_detector_whisper.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(EXPERIMENTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL MODEL SAVED")
    print("=" * 70)
    print(f"Model:    {model_path}")
    print(f"Metadata: {MODEL_DIR / 'turn_detector_whisper.meta.json'}")
    print(f"Results:  {EXPERIMENTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
