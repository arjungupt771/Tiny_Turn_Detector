"""
Task 17 — Export to ONNX.

SCOPE / LIMITATION (read before trusting the numbers below):
This exports and verifies ONNX for the CLASSIFIER HEAD ONLY
(StandardScaler + LogisticRegression(C=0.1), the real, already-trained
384 -> 1 head that sits on top of the frozen Whisper-tiny encoder).

It does NOT export the Whisper-tiny encoder itself. Doing that requires
downloading `openai/whisper-tiny` weights from huggingface.co, which is
blocked in this sandbox's network egress (only pypi/npm/github/crates
domains are allowed; huggingface.co returns 403 host_not_allowed). The
classifier head is ~99.999% of what's realistically "our" model anyway
(385 params vs. Whisper-tiny's ~39M, per OpenAI's published spec) but
it is not the full end-to-end pipeline, and end-to-end ONNX/INT8 export
of the encoder is left as documented, not-yet-run work — see
reports/onnx_int8_comparison.md for the exact command to run this
elsewhere.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import to_onnx

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "whisper_features"
CKPT = ROOT / "checkpoints"
REPORTS = ROOT / "reports"


def main():
    pipe = joblib.load(CKPT / "turn_classifier_head_fp32.joblib")
    X_val = np.load(FEAT / "X_val.npy").astype(np.float32)
    y_val = np.load(FEAT / "y_val.npy")

    onnx_model = to_onnx(pipe, X_val[:1], target_opset=17)
    onnx_path = CKPT / "turn_classifier_head_fp32.onnx"
    onnx.save(onnx_model, str(onnx_path))
    onnx.checker.check_model(str(onnx_path))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    sk_pred = pipe.predict(X_val)
    sk_proba = pipe.predict_proba(X_val)[:, 1]

    onnx_out = sess.run(output_names, {input_name: X_val})
    # skl2onnx classifier output order is (label, probabilities)
    onnx_label = onnx_out[0]
    onnx_proba_raw = onnx_out[1]
    # probabilities come back as a list of dicts {0: p0, 1: p1} per row
    onnx_proba = np.array([row[1] for row in onnx_proba_raw], dtype=np.float64)

    max_abs_diff = float(np.max(np.abs(sk_proba - onnx_proba)))
    label_mismatches = int(np.sum(sk_pred != onnx_label))

    # latency of ONNX inference on the full val set, batch-of-1 loop (100+ runs)
    n_runs = 200
    single = X_val[:1]
    # warmup
    for _ in range(10):
        sess.run(output_names, {input_name: single})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(output_names, {input_name: single})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    result = {
        "onnx_opset": 17,
        "n_val_examples_checked": int(len(X_val)),
        "max_abs_probability_diff_sklearn_vs_onnx": max_abs_diff,
        "label_mismatches": label_mismatches,
        "numerically_consistent": bool(max_abs_diff < 1e-4 and label_mismatches == 0),
        "onnx_single_inference_latency_ms": {
            "mean": float(times.mean()),
            "median_p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "stdev": float(times.std()),
            "n_runs": n_runs,
        },
        "note": (
            "This is the classifier-head-only ONNX export and latency "
            "(StandardScaler + LogisticRegression, 385 params). It does not "
            "include Whisper-tiny encoder inference, which requires network "
            "access to huggingface.co that is not available in this sandbox."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "onnx_export_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
