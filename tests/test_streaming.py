import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from streaming import StreamingTurnDetector


class StubDetector:
    def predict_waveform(self, waveform, sample_rate): return {"label": "CONTINUE", "end_probability": .1, "latency_ms": 1.0}


def test_streaming_only_runs_after_pause_and_resets():
    stream = StreamingTurnDetector(StubDetector(), sample_rate=10, window_seconds=2, stride_seconds=.2, min_silence_seconds=.4)
    assert not stream.add_audio(np.ones(10, dtype=np.float32))
    events = stream.add_audio(np.zeros(4, dtype=np.float32))
    assert events and events[-1]["vad_triggered"]
    stream.reset(); assert not stream.buffer
