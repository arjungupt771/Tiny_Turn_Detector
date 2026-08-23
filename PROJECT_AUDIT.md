# Project audit — 2026-08-20

## Current architecture

The shipped baseline is `openai/whisper-tiny` encoder → mean temporal pooling → `StandardScaler` → regularized logistic regression. The encoder is frozen. The cached experiment uses a seeded 2,000-item training-dataset subset (1,600 train / 400 validation), with the classifier artifact in `models/turn_detector_whisper.joblib`.

## Existing scripts and results

Data/feature extraction is in `src/create_experiment_manifest.py`, `src/extract_whisper_features.py`, `src/extract_acoustic_features.py`, and `src/extract_controlled_features.py`. Training/evaluation is in `src/train_final_model.py`, `src/train_whisper_baseline.py`, `src/train_acoustic_baseline.py`, and `src/analyze_whisper_errors.py`. Serving is `src/inference.py` and `app/app.py`; tests are in `tests/`.

`experiments/results.json` records 0.815 accuracy / 0.813 F1 for Whisper mean pooling + logistic regression on validation. The controlled acoustic baseline records 0.503 accuracy / 0.488 F1. These are validation results only.

## Limitations and additions

The prior manifest does not separate the official test set or detect content duplicates, inference uses a fixed 0.5 threshold, and cached features are pooled so they cannot train temporal attention. No real Hinglish evaluation set, official-test result, attention/fine-tuning result, calibration result, ONNX/INT8 result, or latency benchmark is present.

This production pass adds reusable attention pooling/MLP modules, threshold/calibration and target-slice/error-reporting utilities, and a VAD-gated sliding-window simulation. These utilities require official dataset access and new frame-level feature extraction before results can be reported; no unrun metric is claimed.
