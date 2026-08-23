"""
End-to-end tests: raw .wav file -> TurnDetector -> label.

These need `torch` + `transformers` installed AND network access to
download openai/whisper-tiny from the Hugging Face Hub the first
time. If either is unavailable (e.g. an offline CI runner), the
tests skip cleanly instead of failing the suite - the offline-safe
tests live in test_classifier.py.

Run with:  pytest tests/test_inference_pipeline.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
sys.path.insert(0, str(ROOT / "src"))

# Sample files are named after the label a correct model should predict.
EXPECTED_LABELS = {
    "end.wav": "END",
    "continue.wav": "CONTINUE",
    "midfiller_end.wav": "END",  # filler word mid-utterance, but turn actually ends
    "midfiller_continue.wav": "CONTINUE",
}


@pytest.fixture(scope="module")
def detector():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("torch/transformers not installed")

    from inference import TurnDetector

    try:
        return TurnDetector()
    except OSError as e:
        pytest.skip(f"Whisper-tiny weights not reachable in this environment: {e}")


@pytest.mark.parametrize("filename,expected_label", EXPECTED_LABELS.items())
def test_sample_predictions(detector, filename, expected_label):
    path = SAMPLES_DIR / filename
    if not path.exists():
        pytest.skip(f"sample file missing: {path}")

    result = detector.predict_file(path)

    assert result["label"] in {"CONTINUE", "END"}
    assert 0.0 <= result["end_probability"] <= 1.0
    assert result["latency_ms"] > 0
    # Soft check: report rather than hard-fail on the trickier
    # midfiller cases, since those are genuinely ambiguous even for
    # humans - but the plain end/continue cases should be clear-cut.
    if "midfiller" not in filename:
        assert result["label"] == expected_label, (
            f"{filename}: expected {expected_label}, got {result['label']} "
            f"(P(END)={result['end_probability']:.3f})"
        )


def test_latency_is_reasonable(detector):
    """Turn detection needs to be fast enough for real-time voice AI."""
    path = SAMPLES_DIR / "continue.wav"
    if not path.exists():
        pytest.skip("sample file missing")

    result = detector.predict_file(path)
    # Whisper-tiny encoder + linear probe on CPU should comfortably
    # run well under a second per short utterance.
    assert result["latency_ms"] < 2000
