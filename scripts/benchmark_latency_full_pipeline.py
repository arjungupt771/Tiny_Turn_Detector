"""
Task 15 (completion pass) -- full-pipeline latency benchmark with a real
1/2/4/6/8-second audio-duration sweep, cold-start vs. warm latency, and
peak memory.

SCOPE (read this before trusting any single number):
------------------------------------------------------
Three of the four pipeline stages below are measured with the REAL,
already-trained artifacts and are fully trustworthy for accuracy-relevant
conclusions:
  1. Audio preprocessing (resample to 16 kHz mono)      -- real
  2. Whisper log-mel feature extraction (WhisperFeatureExtractor,
     built fully offline from its public config -- no hub download)
                                                          -- real
  4. Classifier head (StandardScaler + LogisticRegression, real trained
     weights, real ONNX/INT8 exports from prior tasks)   -- real

Stage 3, the Whisper-tiny ENCODER forward pass, uses the same
architecturally-identical, RANDOM-weight encoder as
scripts/measure_encoder_size.py / export_encoder_onnx.py, because the
pretrained checkpoint cannot be downloaded in this sandbox (no route to
huggingface.co). Forward-pass COMPUTE TIME depends only on architecture
and input shape, not on weight values, so this latency number is real and
valid -- but it is not paired with a real accuracy number, since random
weights produce meaningless encoder representations.

A key real finding from this benchmark (not an artifact of the random
weights): this project's own feature-extraction code
(src/extract_whisper_features.py, via WhisperProcessor's default
padding behavior) always pads/truncates audio to Whisper's fixed 30-second
context (3000 mel frames) before the encoder sees it. That means the
ENCODER's forward-pass latency is expected to be roughly CONSTANT across
the audio durations swept below (1-8s, all well under 30s) -- only the
feature-extraction stage's compute scales with actual input length. This
is a real property of the current implementation, worth noting for anyone
building true low-latency streaming on top of it (a non-padded / chunked
encoder input would trade this constant cost for one that scales with
window size).
"""
import json
import os
import resource
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import joblib
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DURATIONS_SEC = [1, 2, 4, 6, 8]
SOURCE_SR = 22050  # simulate a realistic non-16kHz source, matching the
                    # project's own resample-on-load code path
TARGET_SR = 16000
N_RUNS_PER_DURATION = 30

WHISPER_TINY_CONFIG = dict(
    vocab_size=51865, num_mel_bins=80, encoder_layers=4,
    encoder_attention_heads=6, decoder_layers=4, decoder_attention_heads=6,
    d_model=384, encoder_ffn_dim=1536, decoder_ffn_dim=1536,
    max_source_positions=1500,
)


def make_test_audio(duration_sec, sr=SOURCE_SR, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # a few harmonics + noise floor -- stands in for real speech-band content
    signal = (
        0.3 * np.sin(2 * np.pi * 180 * t)
        + 0.15 * np.sin(2 * np.pi * 540 * t)
        + 0.05 * rng.standard_normal(len(t))
    ).astype(np.float32)
    return signal, sr


def resample_stage(waveform, sr):
    if sr == TARGET_SR:
        return waveform
    gcd = np.gcd(sr, TARGET_SR)
    return resample_poly(waveform, TARGET_SR // gcd, sr // gcd).astype(np.float32)


def main():
    torch.manual_seed(42)
    config = WhisperConfig(**WHISPER_TINY_CONFIG)
    encoder = WhisperModel(config).encoder
    encoder.eval()
    feature_extractor = WhisperFeatureExtractor(
        feature_size=80, sampling_rate=TARGET_SR, hop_length=160, chunk_length=30, n_fft=400
    )
    head_pipeline = joblib.load(ROOT / "models" / "turn_detector_whisper.joblib")

    results = []
    cold_start_recorded = False
    cold_start_ms = None

    for dur in DURATIONS_SEC:
        raw_wave, src_sr = make_test_audio(dur)

        stage_times = {"resample_ms": [], "feature_extract_ms": [], "encoder_ms": [], "head_ms": []}

        for run_idx in range(N_RUNS_PER_DURATION):
            t0 = time.perf_counter()
            wave_16k = resample_stage(raw_wave, src_sr)
            t1 = time.perf_counter()

            fe_out = feature_extractor(wave_16k, sampling_rate=TARGET_SR, return_tensors="pt")
            input_features = fe_out.input_features
            t2 = time.perf_counter()

            with torch.no_grad():
                hidden = encoder(input_features).last_hidden_state
            pooled = hidden.mean(dim=1).numpy().astype(np.float32)
            t3 = time.perf_counter()

            _ = head_pipeline.predict_proba(pooled)[:, 1]
            t4 = time.perf_counter()

            if not cold_start_recorded:
                cold_start_ms = (t4 - t0) * 1000
                cold_start_recorded = True
                continue  # exclude the cold run from the warm stats below

            stage_times["resample_ms"].append((t1 - t0) * 1000)
            stage_times["feature_extract_ms"].append((t2 - t1) * 1000)
            stage_times["encoder_ms"].append((t3 - t2) * 1000)
            stage_times["head_ms"].append((t4 - t3) * 1000)

        def summarize(vals):
            arr = np.array(vals)
            return {
                "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)), "n_runs": len(vals),
            }

        total_mean = sum(summarize(v)["mean_ms"] for v in stage_times.values())
        results.append({
            "duration_sec": dur,
            "resample": summarize(stage_times["resample_ms"]),
            "feature_extract": summarize(stage_times["feature_extract_ms"]),
            "encoder_forward_random_weights": summarize(stage_times["encoder_ms"]),
            "classifier_head_real_weights": summarize(stage_times["head_ms"]),
            "total_warm_mean_ms": total_mean,
        })

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    out = {
        "scope": (
            "Stages 1 (resample), 2 (feature extraction), 4 (classifier head) use "
            "real, already-trained/real-code artifacts. Stage 3 (encoder forward "
            "pass) uses an architecturally-identical, random-weight whisper-tiny "
            "encoder (huggingface.co unreachable in this sandbox) -- its LATENCY "
            "is real and weight-independent, but no accuracy claim is made for it. "
            "See module docstring for the full explanation, including the "
            "constant-encoder-latency finding caused by fixed 30s padding."
        ),
        "cold_start_full_pipeline_ms": cold_start_ms,
        "warm_by_duration": results,
        "gpu": "NOT RUN -- no GPU present in this sandbox (CPU-only container).",
        "peak_rss_kb": peak_rss_kb,
        "n_runs_per_duration_after_cold_start_exclusion": N_RUNS_PER_DURATION - 1,
    }

    out_path = REPORTS / "latency_benchmark_full_pipeline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    # CSV for the required plot
    import csv
    csv_path = REPORTS / "latency_benchmark_full_pipeline.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["duration_sec", "resample_ms", "feature_extract_ms", "encoder_ms", "head_ms", "total_ms"])
        for r in results:
            w.writerow([
                r["duration_sec"], round(r["resample"]["mean_ms"], 4),
                round(r["feature_extract"]["mean_ms"], 4),
                round(r["encoder_forward_random_weights"]["mean_ms"], 4),
                round(r["classifier_head_real_weights"]["mean_ms"], 4),
                round(r["total_warm_mean_ms"], 4),
            ])
    print(f"\nSaved: {out_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
