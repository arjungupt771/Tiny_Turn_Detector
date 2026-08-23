"""Evaluate cached validation embeddings and create reproducible operational reports.

The default input is validation only, so threshold selection/calibration cannot leak the
official test set. Run this after fitting a classifier artifact.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from evaluation import calibration_summary, select_threshold, targeted_metrics, write_error_analysis


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--features", default=str(ROOT / "data/whisper_features")); parser.add_argument("--model", default=str(ROOT / "models/turn_detector_whisper.joblib")); parser.add_argument("--reports", default=str(ROOT / "reports")); parser.add_argument("--max-false-end-rate", type=float)
    args = parser.parse_args(); features, reports = Path(args.features), Path(args.reports)
    labels = np.load(features / "y_val.npy"); embeddings = np.load(features / "X_val.npy")
    metadata = json.loads((features / "val_metadata.json").read_text(encoding="utf-8"))
    if not (len(labels) == len(embeddings) == len(metadata)): raise ValueError("Validation artifacts have inconsistent lengths")
    probabilities = joblib.load(args.model).predict_proba(embeddings)[:, 1]
    threshold, sweep = select_threshold(labels, probabilities, max_false_end_rate=args.max_false_end_rate)
    calibration = calibration_summary(labels, probabilities)
    write_rows(reports / "threshold_sweep.csv", sweep)
    write_rows(reports / "targeted_metrics.csv", targeted_metrics(labels, probabilities, metadata, threshold))
    write_error_analysis(reports / "error_analysis.csv", labels, probabilities, metadata, threshold)
    (reports / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (reports / "selected_threshold.json").write_text(json.dumps({"threshold": threshold, "selection_split": "validation", "max_false_end_rate": args.max_false_end_rate}, indent=2), encoding="utf-8")
    print(f"Selected validation-only threshold: {threshold:.2f}; reports written to {reports}")


if __name__ == "__main__": main()
