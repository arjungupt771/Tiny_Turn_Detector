"""
Tests for the trained classifier head, using the cached Whisper
embeddings under data/whisper_features/. These do NOT need network
access or the Whisper encoder itself - they check that the saved
model artifact behaves correctly and meets a minimum quality bar.

Run with:  pytest tests/test_classifier.py -v
"""

from pathlib import Path

import numpy as np
import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "turn_detector_whisper.joblib"
FEATURE_DIR = ROOT / "data" / "whisper_features"

# Minimum bar the shipped model must clear on the held-out val split.
# (Measured performance is ~0.81 accuracy / ~0.81 F1 - set thresholds
# a bit below that so small, harmless retraining variance doesn't
# break CI.)
MIN_ACCURACY = 0.72
MIN_F1 = 0.72


@pytest.fixture(scope="module")
def classifier():
    assert MODEL_PATH.exists(), f"Trained model not found at {MODEL_PATH}. Run src/train_final_model.py first."
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def val_data():
    X_val = np.load(FEATURE_DIR / "X_val.npy")
    y_val = np.load(FEATURE_DIR / "y_val.npy")
    return X_val, y_val


def test_model_loads(classifier):
    assert hasattr(classifier, "predict")
    assert hasattr(classifier, "predict_proba")


def test_embedding_dim_matches(classifier, val_data):
    X_val, _ = val_data
    assert X_val.shape[1] == 384, "Whisper-tiny encoder hidden size should be 384"
    # Should not raise - confirms the pipeline was fit on 384-dim input.
    classifier.predict(X_val[:5])


def test_output_is_binary_and_probabilistic(classifier, val_data):
    X_val, _ = val_data
    preds = classifier.predict(X_val)
    proba = classifier.predict_proba(X_val)

    assert set(np.unique(preds)).issubset({0, 1})
    assert proba.shape == (len(X_val), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_val_accuracy_above_threshold(classifier, val_data):
    from sklearn.metrics import accuracy_score, f1_score

    X_val, y_val = val_data
    preds = classifier.predict(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    assert acc >= MIN_ACCURACY, f"Accuracy {acc:.3f} fell below minimum {MIN_ACCURACY}"
    assert f1 >= MIN_F1, f"F1 {f1:.3f} fell below minimum {MIN_F1}"


def test_beats_acoustic_only_baseline():
    """
    Sanity check on the core finding of this project: Whisper
    embeddings should clearly outperform hand-crafted acoustic
    features on the identical train/val split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    acoustic_dir = ROOT / "data" / "controlled_features" / "acoustic"
    if not acoustic_dir.exists():
        pytest.skip("controlled acoustic features not present")

    X_train = np.load(acoustic_dir / "X_train.npy")
    y_train = np.load(acoustic_dir / "y_train.npy")
    X_val = np.load(acoustic_dir / "X_val.npy")
    y_val = np.load(acoustic_dir / "y_val.npy")

    acoustic_model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]
    )
    acoustic_model.fit(X_train, y_train)
    acoustic_acc = accuracy_score(y_val, acoustic_model.predict(X_val))

    whisper_model = joblib.load(MODEL_PATH)
    X_val_w = np.load(FEATURE_DIR / "X_val.npy")
    y_val_w = np.load(FEATURE_DIR / "y_val.npy")
    whisper_acc = accuracy_score(y_val_w, whisper_model.predict(X_val_w))

    assert whisper_acc > acoustic_acc + 0.15, (
        f"Expected Whisper embeddings ({whisper_acc:.3f}) to clearly beat "
        f"acoustic-only features ({acoustic_acc:.3f})"
    )
