"""
Task 17 (completion pass) -- export the Whisper-tiny ENCODER architecture
to ONNX and verify PyTorch vs ONNX numerical parity.

IMPORTANT SCOPE NOTE (read before trusting any number here):
This uses the same architecturally-identical, RANDOMLY-INITIALIZED
whisper-tiny encoder as scripts/measure_encoder_size.py (built from
transformers.WhisperConfig with zero network calls -- huggingface.co is
still unreachable in this sandbox, so the real pretrained checkpoint could
not be downloaded).

What this DOES verify (validly, regardless of weight values):
  - The encoder's computation graph exports to ONNX at all.
  - PyTorch and ONNXRuntime produce numerically identical outputs for the
    SAME (random) weights, i.e. the export mechanics are correct.
  - The export's file size and forward-pass latency, which -- like the
    param count -- depend on architecture/shape, not weight values.

What this does NOT verify: any accuracy/F1 claim for the encoder+classifier
pipeline. Random weights make the encoder's actual output meaningless for
turn-detection; only the export mechanics, size, and latency are real here.
"""
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import WhisperConfig, WhisperModel

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
REPORTS = ROOT / "reports"
CKPT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

WHISPER_TINY_CONFIG = dict(
    vocab_size=51865, num_mel_bins=80, encoder_layers=4,
    encoder_attention_heads=6, decoder_layers=4, decoder_attention_heads=6,
    d_model=384, encoder_ffn_dim=1536, decoder_ffn_dim=1536,
    max_source_positions=1500,
)

ONNX_PATH = CKPT / "whisper_tiny_encoder_fp32_random_weights.onnx"


class EncoderWrapper(torch.nn.Module):
    """Whisper's HF encoder returns a ModelOutput; wrap it so the ONNX
    graph has a single plain tensor output."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state


def main():
    torch.manual_seed(42)
    config = WhisperConfig(**WHISPER_TINY_CONFIG)
    model = WhisperModel(config)
    model.eval()
    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    # Whisper's feature extractor always produces (batch, n_mels, 3000)
    # log-mel frames for up to a 30s clip (fixed padding/truncation to 30s
    # is the default behavior this project's own
    # src/extract_whisper_features.py relies on via WhisperProcessor).
    dummy_input = torch.randn(1, 80, 3000)

    with torch.no_grad():
        torch_out = wrapper(dummy_input).numpy()

    torch.onnx.export(
        wrapper, dummy_input, str(ONNX_PATH),
        input_names=["input_features"], output_names=["last_hidden_state"],
        dynamic_axes={"input_features": {0: "batch"}, "last_hidden_state": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(str(ONNX_PATH))

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["last_hidden_state"], {"input_features": dummy_input.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))

    # Latency: PyTorch vs ONNXRuntime, same random weights, same input.
    n_runs = 100
    for _ in range(5):  # warmup
        with torch.no_grad():
            wrapper(dummy_input)
        sess.run(["last_hidden_state"], {"input_features": dummy_input.numpy()})

    torch_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            wrapper(dummy_input)
        torch_times.append((time.perf_counter() - t0) * 1000)

    onnx_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(["last_hidden_state"], {"input_features": dummy_input.numpy()})
        onnx_times.append((time.perf_counter() - t0) * 1000)

    def stats(times):
        arr = np.array(times)
        return {
            "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)),
            "stdev_ms": float(arr.std()), "n_runs": n_runs,
        }

    onnx_graph_bytes = ONNX_PATH.stat().st_size
    external_data_path = ONNX_PATH.with_name(ONNX_PATH.name + ".data")
    external_data_bytes = external_data_path.stat().st_size if external_data_path.exists() else 0
    onnx_size_bytes = onnx_graph_bytes + external_data_bytes

    out = {
        "scope": (
            "Architecturally-identical whisper-tiny ENCODER with RANDOM weights "
            "(huggingface.co unreachable in this sandbox -- see module docstring). "
            "Verifies export mechanics, numerical parity, size, and latency, which "
            "are weight-value-independent. Does NOT verify or claim any "
            "accuracy/F1 number for the encoder or the full pipeline."
        ),
        "onnx_opset": 17,
        "input_shape": [1, 80, 3000],
        "max_abs_diff_pytorch_vs_onnx": max_abs_diff,
        "numerically_consistent": max_abs_diff < 1e-3,
        "onnx_graph_file_bytes": onnx_graph_bytes,
        "onnx_external_weights_data_bytes": external_data_bytes,
        "onnx_total_file_size_bytes": onnx_size_bytes,
        "onnx_total_file_size_mb": round(onnx_size_bytes / (1024 * 1024), 2),
        "note_on_external_data": (
            "PyTorch's dynamo-based ONNX exporter stores large weight tensors in a "
            "separate '<model>.onnx.data' file alongside the small graph/metadata "
            "'.onnx' file; both must ship together and their sizes are summed above."
        ),
        "pytorch_fp32_latency_ms": stats(torch_times),
        "onnxruntime_fp32_latency_ms": stats(onnx_times),
    }

    out_path = REPORTS / "onnx_encoder_export_verification.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
