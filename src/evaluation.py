"""Leakage-safe evaluation, threshold selection, calibration, and error reporting."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score


def metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)), "f1": float(f1_score(labels, predictions, zero_division=0)),
            "false_end_rate": float(fp / max(tn + fp, 1)), "false_continue_rate": float(fn / max(tp + fn, 1)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def select_threshold(labels: np.ndarray, probabilities: np.ndarray, thresholds: Iterable[float] = np.arange(.30, .71, .05), max_false_end_rate: float | None = None) -> tuple[float, list[dict[str, float | int]]]:
    """Select on validation data only; optionally prioritize an interruption-safety ceiling."""
    rows = [metrics_at_threshold(labels, probabilities, float(t)) for t in thresholds]
    eligible = [r for r in rows if max_false_end_rate is None or r["false_end_rate"] <= max_false_end_rate]
    pool = eligible or rows
    best = max(pool, key=lambda row: (row["f1"], -row["false_end_rate"]))
    return float(best["threshold"]), rows


def calibration_summary(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
    observed, predicted = calibration_curve(labels, probabilities, n_bins=bins, strategy="uniform")
    ece = float(np.mean(np.abs(observed - predicted))) if len(observed) else 0.0
    return {"brier_score": float(brier_score_loss(labels, probabilities)), "ece": ece,
            "mean_predicted": predicted.tolist(), "fraction_positive": observed.tolist()}


def targeted_metrics(labels: np.ndarray, probabilities: np.ndarray, metadata: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {"overall": list(range(len(labels)))}
    for key in ("language", "midfiller", "endfiller", "synthetic"):
        for index, row in enumerate(metadata):
            value = row.get(key, "unknown")
            groups.setdefault(f"{key}={value}", []).append(index)
    durations = np.asarray([row.get("duration_seconds", np.nan) for row in metadata], dtype=float)
    for name, selector in {"duration=short": durations < 2, "duration=medium": (durations >= 2) & (durations < 5), "duration=long": durations >= 5}.items():
        if not np.isnan(durations).all(): groups[name] = np.flatnonzero(selector).tolist()
    result = []
    for name, indices in groups.items():
        if indices:
            row = metrics_at_threshold(labels[indices], probabilities[indices], threshold)
            result.append({"group": name, "n": len(indices), **row})
    return result


def write_error_analysis(path: str | Path, labels: np.ndarray, probabilities: np.ndarray, metadata: list[dict[str, Any]], threshold: float) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    fields = ["audio_id", "ground_truth", "prediction", "probability", "error_type", "language", "midfiller", "endfiller", "synthetic", "duration_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, (truth, pred, prob) in enumerate(zip(labels, predictions, probabilities)):
            if truth == pred: continue
            item = metadata[index]
            writer.writerow({"audio_id": item.get("id", item.get("file", str(index))), "ground_truth": "END" if truth else "CONTINUE", "prediction": "END" if pred else "CONTINUE", "probability": float(prob), "error_type": "false_end" if pred else "false_continue", **{key: item.get(key) for key in fields[5:]}})
