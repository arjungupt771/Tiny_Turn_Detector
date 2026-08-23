"""Validate manifest/feature/model artifact alignment before evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_split(root: Path, split: str, feature_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest = root / "data" / "manifests" / f"{split}.csv"
    x_path = feature_dir / f"X_{split}.npy"
    y_path = feature_dir / f"y_{split}.npy"
    meta_path = feature_dir / f"{split}_metadata.json"

    if not manifest.exists():
        errors.append(f"missing manifest: {manifest}")
        return errors
    if not x_path.exists() or not y_path.exists():
        errors.append(f"missing cached features for {split}: {x_path} / {y_path}")
        return errors

    import csv
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    if len(rows) != len(X):
        errors.append(f"{split}: manifest={len(rows)} but X={len(X)}")
    if len(X) != len(y):
        errors.append(f"{split}: X={len(X)} but y={len(y)}")
    if meta_path.exists():
        metadata = load_json(meta_path)
        if len(metadata) != len(X):
            errors.append(f"{split}: metadata={len(metadata)} but X={len(X)}")
    else:
        errors.append(f"{split}: missing metadata: {meta_path}")

    if X.ndim != 2 or X.shape[1] != 384:
        errors.append(f"{split}: expected pooled Whisper features [N,384], got {X.shape}")
    if set(np.unique(y).tolist()) - {0, 1}:
        errors.append(f"{split}: labels must be binary 0/1")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    feature_dir = Path(args.feature_dir).resolve() if args.feature_dir else root / "data" / "whisper_features"
    model_path = Path(args.model).resolve() if args.model else root / "models" / "turn_detector_whisper.joblib"

    errors: list[str] = []
    for split in ("train", "val"):
        errors.extend(check_split(root, split, feature_dir))

    if model_path.exists():
        model = joblib.load(model_path)
        X_val = np.load(feature_dir / "X_val.npy", mmap_mode="r")
        try:
            model.predict_proba(X_val[:2])
        except Exception as exc:
            errors.append(f"model cannot score 384-dim validation features: {exc}")
    else:
        errors.append(f"missing model: {model_path}")

    threshold_path = root / "reports" / "selected_threshold.json"
    if threshold_path.exists():
        threshold = float(load_json(threshold_path)["threshold"])
        if not 0.0 < threshold < 1.0:
            errors.append(f"invalid selected threshold: {threshold}")
    else:
        errors.append(f"missing selected threshold: {threshold_path}")

    if errors:
        print("ARTIFACT VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("ARTIFACT VALIDATION PASSED")
    print("- train manifest/features/labels/metadata aligned")
    print("- val manifest/features/labels/metadata aligned")
    print("- classifier accepts 384-dim Whisper embeddings")
    print("- selected validation threshold is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
