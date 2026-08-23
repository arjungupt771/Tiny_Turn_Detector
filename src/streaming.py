"""A deterministic sliding-window turn-detection simulation with an energy VAD gate."""
from __future__ import annotations

from collections import deque

import numpy as np


class StreamingTurnDetector:
    """Only runs the classifier after a configurable energy-based pause trigger."""

    def __init__(self, detector, sample_rate: int = 16000, window_seconds: float = 6.0,
                 stride_seconds: float = 0.2, silence_rms: float = 0.008,
                 min_silence_seconds: float = 0.3):
        self.detector, self.sample_rate = detector, sample_rate
        self.window_samples = int(window_seconds * sample_rate)
        self.stride_samples = int(stride_seconds * sample_rate)
        self.silence_rms = silence_rms
        self.min_silence_samples = int(min_silence_seconds * sample_rate)
        self.reset()

    def reset(self) -> None:
        self.buffer: deque[float] = deque(maxlen=self.window_samples)
        self.pending = np.empty(0, dtype=np.float32)
        self.silence_samples = 0

    def add_audio(self, chunk: np.ndarray) -> list[dict]:
        """Ingest mono audio and return any pause-triggered turn decisions."""
        self.pending = np.concatenate([self.pending, np.asarray(chunk, dtype=np.float32).reshape(-1)])
        events = []
        while len(self.pending) >= self.stride_samples:
            step, self.pending = self.pending[:self.stride_samples], self.pending[self.stride_samples:]
            self.buffer.extend(step.tolist())
            rms = float(np.sqrt(np.mean(step ** 2)))
            self.silence_samples = self.silence_samples + len(step) if rms < self.silence_rms else 0
            if self.buffer and self.silence_samples >= self.min_silence_samples:
                event = self.detector.predict_waveform(np.asarray(self.buffer, dtype=np.float32), self.sample_rate)
                event.update({"vad_triggered": True, "buffer_seconds": len(self.buffer) / self.sample_rate})
                events.append(event)
        return events
