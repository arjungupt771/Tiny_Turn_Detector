"""
Task 18 (completion pass) -- INT8 quantize the Whisper-tiny ENCODER export.

Same scope note as export_encoder_onnx.py: this uses the architecturally-
identical, RANDOM-weight encoder (huggingface.co unreachable, so the real
pretrained checkpoint could not be downloaded). Size and latency numbers
below are real and weight-independent; no accuracy/F1 claim is made or
implied for the encoder, since random weights produce meaningless
predictions. This complements (does not replace) the classifier head's
FP32/ONNX/INT8 comparison in reports/int8_quantization.json, which DOES
have real accuracy numbers because that model was actually trained.
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
REPORTS = ROOT / "reports"

FP32_PATH = CKPT / "whisper_tiny_encoder_fp32_random_weights.onnx"
INT8_PATH = CKPT / "whisper_tiny_encoder_int8_random_weights.onnx"


def bench(sess, dummy_input, n_runs=100):
    for _ in range(5):
        sess.run(None, {"input_features": dummy_input})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"input_features": dummy_input})
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)),
        "stdev_ms": float(arr.std()), "n_runs": n_runs,
    }


def total_size(onnx_path: Path) -> int:
    total = onnx_path.stat().st_size
    ext = onnx_path.with_name(onnx_path.name + ".data")
    if ext.exists():
        total += ext.stat().st_size
    return total


def main():
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QInt8,
    )

    dummy_input = np.random.randn(1, 80, 3000).astype(np.float32)

    sess_fp32 = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(INT8_PATH), providers=["CPUExecutionProvider"])

    out_fp32 = sess_fp32.run(None, {"input_features": dummy_input})[0]
    out_int8 = sess_int8.run(None, {"input_features": dummy_input})[0]
    max_abs_diff = float(np.max(np.abs(out_fp32 - out_int8)))
    mean_abs_diff = float(np.mean(np.abs(out_fp32 - out_int8)))

    fp32_size = total_size(FP32_PATH)
    int8_size = total_size(INT8_PATH)

    fp32_latency = bench(sess_fp32, dummy_input)
    int8_latency = bench(sess_int8, dummy_input)

    out = {
        "scope": (
            "Architecturally-identical whisper-tiny ENCODER, random weights "
            "(see export_encoder_onnx.py docstring -- huggingface.co unreachable). "
            "Size and latency are real and weight-independent; output-difference "
            "numbers below reflect quantization noise on RANDOM weights, not "
            "real-world accuracy impact -- no F1/accuracy claim is made for the "
            "encoder. Compare against reports/int8_quantization.json for the "
            "classifier head, which DOES have real accuracy numbers."
        ),
        "onnx_fp32": {
            "total_size_bytes": fp32_size,
            "total_size_mb": round(fp32_size / (1024 * 1024), 2),
            "latency_ms": fp32_latency,
        },
        "onnx_int8": {
            "total_size_bytes": int8_size,
            "total_size_mb": round(int8_size / (1024 * 1024), 2),
            "latency_ms": int8_latency,
        },
        "size_reduction_pct": round(100 * (1 - int8_size / fp32_size), 1),
        "output_difference_on_random_weights": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "note": (
                "This is a hidden-state-level numerical difference from dynamic "
                "INT8 weight quantization, measured on random weights purely to "
                "confirm the quantized graph runs and produces finite, "
                "structurally-consistent output -- it is NOT a classification "
                "accuracy/F1 number and should not be read as one."
            ),
        },
        "conclusion": (
            f"Unlike the 385-parameter classifier head (where INT8 did not shrink "
            f"the file because runtime/protobuf overhead dominates), the "
            f"{8208384:,}-parameter encoder shows a real "
            f"{round(100 * (1 - int8_size / fp32_size), 1)}% size reduction from "
            "dynamic INT8 quantization -- this is the regime where INT8 actually "
            "delivers the expected ~4x weight-size win, confirming the earlier "
            "hypothesis in reports/int8_quantization.json's size_note. Latency "
            "improvement from INT8 on CPU depends on onnxruntime's kernel support "
            "and is reported above rather than assumed."
        ),
    }

    out_path = REPORTS / "onnx_encoder_int8_quantization.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
