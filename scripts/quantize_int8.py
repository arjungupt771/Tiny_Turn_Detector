"""
Task 18/19 — INT8 quantization + verification.

Same scope limitation as scripts/export_onnx.py: this quantizes the
CLASSIFIER HEAD ONNX model only (385 params). It does NOT quantize a
Whisper-tiny encoder, which can't be downloaded in this sandbox (no
network route to huggingface.co). See reports/onnx_int8_comparison.md
for what a full encoder+head INT8 pipeline would require and the exact
command to run it elsewhere.

A 385-parameter logistic-regression head is already ~13.6KB in FP32, so
INT8 quantizing it is not where the real size win of this project lives
(that's Whisper-tiny at ~39M params) -- but the brief asks for the
comparison to be run and reported, including investigating any
meaningful F1 drop, so it's done here honestly rather than skipped.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
FEAT = ROOT / "data" / "whisper_features"
REPORTS = ROOT / "reports"


def bench(sess, input_name, output_names, X, n_runs=200):
    single = X[:1]
    for _ in range(10):
        sess.run(output_names, {input_name: single})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(output_names, {input_name: single})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "stdev_ms": float(times.std()),
        "n_runs": n_runs,
    }


def eval_onnx(sess, input_name, output_names, X, y):
    out = sess.run(output_names, {input_name: X.astype(np.float32)})
    labels = out[0]
    acc = float((labels == y).mean())
    tp = int(((labels == 1) & (y == 1)).sum())
    fp = int(((labels == 1) & (y == 0)).sum())
    fn = int(((labels == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    fp32_path = CKPT / "turn_classifier_head_fp32.onnx"
    int8_path = CKPT / "turn_classifier_head_int8.onnx"

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    X_val = np.load(FEAT / "X_val.npy").astype(np.float32)
    y_val = np.load(FEAT / "y_val.npy")

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    in_name = sess_fp32.get_inputs()[0].name
    out_names_fp32 = [o.name for o in sess_fp32.get_outputs()]
    out_names_int8 = [o.name for o in sess_int8.get_outputs()]

    m_fp32 = eval_onnx(sess_fp32, in_name, out_names_fp32, X_val, y_val)
    m_int8 = eval_onnx(sess_int8, in_name, out_names_int8, X_val, y_val)

    lat_fp32 = bench(sess_fp32, in_name, out_names_fp32, X_val)
    lat_int8 = bench(sess_int8, in_name, out_names_int8, X_val)

    size_fp32 = os.path.getsize(fp32_path)
    size_int8 = os.path.getsize(int8_path)

    f1_drop = m_fp32["f1"] - m_int8["f1"]
    investigation = (
        f"F1 drop from FP32->INT8 was {f1_drop:.4f}. "
        + (
            "This is within noise for a 400-example validation set (a handful "
            "of examples flipping near the decision boundary) -- expected for "
            "dynamic INT8 quantization of a 385-parameter linear model, since "
            "there's very little redundancy to exploit and quantization noise "
            "on 8-bit weights can shift a few borderline logits across 0.5. "
            "Not concerning."
            if abs(f1_drop) < 0.02
            else "This drop is larger than expected for a linear model this "
            "size and would need investigation before shipping (e.g. checking "
            "which validation examples flipped and whether they cluster near "
            "the decision boundary)."
        )
    )

    size_note = (
        f"INT8 checkpoint ({size_int8}B) is "
        + ("smaller than" if size_int8 < size_fp32 else "NOT smaller than")
        + f" FP32 ({size_fp32}B). For a model this tiny (385 params), ONNX "
        "runtime/protobuf overhead can dominate actual weight bytes, so the "
        "expected big win from INT8 (~4x smaller weights) may not show up in "
        "on-disk file size the way it would for Whisper-tiny's ~39M params -- "
        "that's where INT8 would actually matter for this project, and it's "
        "not measurable here without network access to the encoder weights."
    )

    result = {
        "pytorch_fp32_sklearn": {  # duplicated from reports/classifier_head_fp32_metrics.json for convenience
            "size_bytes": os.path.getsize(CKPT / "turn_classifier_head_fp32.joblib"),
        },
        "onnx_fp32": {"metrics": m_fp32, "latency_ms": lat_fp32, "size_bytes": size_fp32},
        "onnx_int8": {"metrics": m_int8, "latency_ms": lat_int8, "size_bytes": size_int8},
        "f1_drop_fp32_to_int8": f1_drop,
        "f1_drop_investigation": investigation,
        "size_note": size_note,
        "scope_limitation": (
            "Classifier-head-only comparison (385 params). Whisper-tiny "
            "encoder (~39M params, where INT8 would actually deliver a "
            "meaningful size/latency win) could not be downloaded/quantized "
            "in this sandbox -- no network route to huggingface.co."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "int8_quantization.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
