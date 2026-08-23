"""
Inference wrapper for the turn-detection model.

Usage
-----
    from inference import TurnDetector

    detector = TurnDetector()
    result = detector.predict_file("data/samples/end.wav")
    # {"label": "END", "end_probability": 0.94, "latency_ms": 42.1}

Design notes
------------
- The encoder (Whisper-tiny) is frozen and loaded once; only the tiny
  classifier head (a scikit-learn Pipeline, <1KB of real parameters)
  was trained. This keeps the model "tiny" as the brief asks for:
  ~39M frozen encoder params (reused, not ours) + a ~400-parameter
  logistic-regression probe that we own and can retrain in seconds.
- We mean-pool the encoder's last_hidden_state across time, exactly
  matching the feature extraction used at training time
  (see extract_whisper_features.py) - this consistency matters a lot
  for a linear probe, since it was fit on this exact representation.
- Audio is resampled to 16 kHz mono, which is what Whisper expects.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Union

import numpy as np
import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "turn_detector_whisper.joblib"
WHISPER_MODEL_NAME = "openai/whisper-tiny"
TARGET_SAMPLE_RATE = 16000

LABEL_MAP = {0: "CONTINUE", 1: "END"}


class TurnDetector:
    """Loads the frozen Whisper-tiny encoder + trained classifier head."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH, device: str | None = None):
        import torch
        from transformers import WhisperProcessor, WhisperModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
        self.encoder = WhisperModel.from_pretrained(WHISPER_MODEL_NAME).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        model_path = Path(model_path)
        if "random_weights" in model_path.name.lower():
            raise ValueError("Random-weight encoder artifacts are for architecture/latency checks only, not accuracy inference.")
        self.classifier = joblib.load(model_path)
        meta_path = model_path.with_suffix(".meta.json")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        threshold_path = model_path.parent.parent / "reports" / "selected_threshold.json"
        if threshold_path.exists():
            threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
            self.threshold = float(threshold_data["threshold"])
            self.threshold_source = str(threshold_path)
        else:
            self.threshold = float(self.metadata.get("threshold", 0.5))
            self.threshold_source = "model metadata/default"
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"Invalid decision threshold: {self.threshold}")
        self._torch = torch

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _load_waveform(self, audio_path: Union[str, Path]) -> tuple[np.ndarray, int]:
        import soundfile as sf

        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform, sample_rate

    def _resample(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == TARGET_SAMPLE_RATE:
            return waveform
        import librosa

        return librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Waveform (float32, any sample rate) -> 384-dim mean-pooled embedding."""
        waveform = self._resample(waveform.astype(np.float32), sample_rate)

        inputs = self.processor(waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with self._torch.inference_mode():
            hidden = self.encoder.encoder(input_features).last_hidden_state
            embedding = hidden.mean(dim=1)

        return embedding.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_embedding(self, embedding: np.ndarray) -> dict:
        proba = self.classifier.predict_proba(embedding.reshape(1, -1))[0]
        end_prob = float(proba[1])
        label = LABEL_MAP[int(end_prob >= self.threshold)]
        return {"label": label, "end_probability": end_prob, "threshold": self.threshold,
                "model_version": self.metadata.get("model_version", "whisper-mean-logreg"),
                "threshold_source": self.threshold_source}

    def predict_waveform(self, waveform: np.ndarray, sample_rate: int) -> dict:
        t0 = time.perf_counter()
        embedding = self.embed(waveform, sample_rate)
        result = self.predict_embedding(embedding)
        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def predict_file(self, audio_path: Union[str, Path]) -> dict:
        waveform, sample_rate = self._load_waveform(audio_path)
        return self.predict_waveform(waveform, sample_rate)


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Run turn-detection inference on a wav file.")
    parser.add_argument("audio_path", type=str, help="Path to a .wav file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    detector = TurnDetector(model_path=args.model)
    result = detector.predict_file(args.audio_path)
    print(f"{args.audio_path}: {result['label']}  (P(END)={result['end_probability']:.3f}, "
          f"{result['latency_ms']:.1f} ms)")


if __name__ == "__main__":
    _cli()
