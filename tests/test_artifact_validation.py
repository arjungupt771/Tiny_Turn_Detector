from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_selected_threshold_is_validation_only_and_valid():
    data = json.loads((ROOT / "reports" / "selected_threshold.json").read_text())
    assert data["selection_split"] == "validation"
    assert 0.0 < float(data["threshold"]) < 1.0


def test_training_feature_artifacts_are_aligned():
    import numpy as np
    for split in ("train", "val"):
        X = np.load(ROOT / "data" / "whisper_features" / f"X_{split}.npy", mmap_mode="r")
        y = np.load(ROOT / "data" / "whisper_features" / f"y_{split}.npy", mmap_mode="r")
        metadata = json.loads((ROOT / "data" / "whisper_features" / f"{split}_metadata.json").read_text())
        assert len(X) == len(y) == len(metadata)
        assert X.shape[1] == 384
