"""
Task 12 (completion pass) -- Temperature scaling for the frozen-Whisper
mean-pool + LogisticRegression baseline.

This uses ONLY the already-cached validation features
(data/whisper_features/{X,y}_val.npy) and the already-trained sklearn
pipeline (models/turn_detector_whisper.joblib). No Whisper encoder forward
pass is needed here, so this is NOT blocked by the sandbox's lack of
network access to huggingface.co -- unlike most of the other partial tasks.

Method
------
1. Get the pipeline's raw decision-function logits on the validation set
   (StandardScaler -> LogisticRegression linear score, pre-sigmoid).
2. Fit a single scalar temperature T > 0 that minimizes validation NLL of
   sigmoid(logit / T) against true labels (Platt/temperature scaling).
3. Recompute Brier score and Expected Calibration Error (ECE) before and
   after scaling, on the SAME validation set (never the official test set,
   per the "don't tune on test" rule).
4. Save T and the before/after metrics into reports/calibration.json,
   preserving the original (pre-existing) uncalibrated numbers rather than
   overwriting them.
"""
import json
from pathlib import Path

import numpy as np
import joblib
from scipy.optimize import minimize_scalar

SEED = 42
np.random.seed(SEED)

MODEL_PATH = Path("models/turn_detector_whisper.joblib")
X_VAL_PATH = Path("data/whisper_features/X_val.npy")
Y_VAL_PATH = Path("data/whisper_features/y_val.npy")
OUT_PATH = Path("reports/calibration.json")
N_BINS = 10


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def brier_score(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def ece_and_reliability(probs, labels, n_bins=N_BINS):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted, fraction_positive, bin_counts = [], [], []
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            mean_predicted.append(None)
            fraction_positive.append(None)
            continue
        mp = float(probs[mask].mean())
        fp = float(labels[mask].mean())
        mean_predicted.append(mp)
        fraction_positive.append(fp)
        ece += (count / n) * abs(mp - fp)
    return float(ece), mean_predicted, fraction_positive, bin_counts


def nll(temperature, logits, labels, eps=1e-7):
    probs = sigmoid(logits / temperature)
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def main():
    pipeline = joblib.load(MODEL_PATH)
    X_val = np.load(X_VAL_PATH)
    y_val = np.load(Y_VAL_PATH).astype(np.float64)

    # Raw pre-sigmoid logits: StandardScaler -> LogisticRegression linear score
    logits = pipeline.decision_function(X_val)
    probs_uncalibrated = sigmoid(logits)

    # --- Fit temperature on validation only ---
    result = minimize_scalar(
        nll, bounds=(0.05, 20.0), method="bounded",
        args=(logits, y_val), options={"xatol": 1e-6},
    )
    T = float(result.x)
    probs_calibrated = sigmoid(logits / T)

    brier_before = brier_score(probs_uncalibrated, y_val)
    brier_after = brier_score(probs_calibrated, y_val)

    ece_before, mp_before, fp_before, counts_before = ece_and_reliability(probs_uncalibrated, y_val)
    ece_after, mp_after, fp_after, counts_after = ece_and_reliability(probs_calibrated, y_val)

    out = {
        "scope": (
            "Frozen Whisper-tiny mean-pool + StandardScaler + LogisticRegression "
            "baseline, evaluated on the 400-example cached validation split "
            "(data/whisper_features/X_val.npy, y_val.npy). Uses only the "
            "already-extracted 384-dim feature vectors and the already-trained "
            "sklearn pipeline -- no Whisper encoder forward pass required, so this "
            "task was not blocked by the sandbox's lack of huggingface.co access."
        ),
        "n_val_examples": int(len(y_val)),
        "uncalibrated": {
            "brier_score": brier_before,
            "ece": ece_before,
            "mean_predicted": mp_before,
            "fraction_positive": fp_before,
            "bin_counts": counts_before,
        },
        "temperature_scaling": {
            "method": "Single-parameter temperature scaling: p = sigmoid(logit / T), "
                      "T fit by minimizing validation NLL (scipy bounded scalar "
                      "minimization, bounds=[0.05, 20.0]). Fit and evaluated on "
                      "validation only -- the official test set was not touched.",
            "temperature": T,
            "brier_score": brier_after,
            "ece": ece_after,
            "mean_predicted": mp_after,
            "fraction_positive": fp_after,
            "bin_counts": counts_after,
            "brier_improvement": brier_before - brier_after,
            "ece_improvement": ece_before - ece_after,
        },
        "conclusion": (
            f"T = {T:.3f}. "
            + (
                "T > 1 means the raw logistic-regression probabilities were "
                "overconfident (too close to 0/1); dividing logits by T > 1 "
                "softens them toward 0.5 before comparing to observed frequencies."
                if T > 1.05 else
                "T < 1 means the raw probabilities were underconfident; "
                "temperature scaling sharpened them."
                if T < 0.95 else
                "T is close to 1, meaning the uncalibrated probabilities were "
                "already reasonably well-calibrated on this validation set."
            )
            + f" ECE moved from {ece_before:.4f} to {ece_after:.4f} "
            f"({'improved' if ece_after < ece_before else 'did not improve'}), "
            f"Brier score moved from {brier_before:.4f} to {brier_after:.4f} "
            f"({'improved' if brier_after < brier_before else 'did not improve'})."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
