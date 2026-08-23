# Bugfix: stale `experiments/results.json` / shipped model

## What was found

`experiments/results.json` (from a prior pass) recorded the Whisper-embedding
classifier-head sweep at near-chance performance (best: `svm_rbf_C5` at
58.5% accuracy / 0.574 F1), and that sweep result selected `svm_rbf_C5` as
the model actually saved to `models/turn_detector_whisper.joblib`.

This contradicted the 81.5% / 0.813 F1 baseline figure quoted everywhere
else in the repo (README, PROJECT_AUDIT.md, pending_tasks.txt) for
"frozen Whisper-tiny + mean pooling + logistic regression."

## Root cause

Re-running the exact same code (`src/train_final_model.py`, unchanged) against
the **current** `data/whisper_features/{X,y}_{train,val}.npy` reproduces
81.5% / 0.813 F1 for `logreg_C0.1` exactly — matching the widely-quoted
number. This means `experiments/results.json` was generated against an
earlier version of the cached feature arrays (before whatever fix made the
Whisper embeddings actually separable) and was never regenerated after that
fix, even though the feature `.npy` files on disk were updated. The stale
JSON in turn caused an inferior model (`svm_rbf_C5`, chosen by 5-fold CV F1
on the *old*, weak features) to be the one actually shipped in
`models/turn_detector_whisper.joblib`.

## What was fixed

- Backed up the stale files: `experiments/results.json.stale_backup_20260821`,
  `models/turn_detector_whisper.joblib.stale_backup_20260821`.
- Re-ran `python src/train_final_model.py` end-to-end (fully reproducible
  locally, no network needed — it only touches cached `.npy` feature files).
- `experiments/results.json`, `models/turn_detector_whisper.joblib`, and
  `models/turn_detector_whisper.meta.json` now correctly reflect
  `logreg_C0.1` (5-fold CV F1 = 0.780, val accuracy = 0.815, val F1 = 0.813)
  as the selected head — consistent with every other doc in the repo.

## What this means for downstream reports

`reports/threshold_sweep.csv`, `reports/calibration.json`,
`reports/targeted_metrics.csv`, and `reports/error_analysis.csv` were
generated in an earlier pass and are not necessarily against this
corrected model — this was already flagged as a known gap in
`pending_tasks.txt` (they were computed against the OLD 400-example
validation split before the manifest sizes changed, and now separately
we know they may also predate this classifier-head fix). They should be
regenerated against `models/turn_detector_whisper.joblib` (now fixed) and
`data/manifests/val.csv` before being trusted as final. Not redone in this
pass — scope was the 12 tasks in the request, and this bugfix was
uncovered while sourcing real numbers for the master results table.
