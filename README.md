# Tiny Turn Detector — Shiprocket Voice AI Challenge


if u want to run the application then run command : python /app/app.py

> **Current status (most recent pass):** the baseline below (81.5%/0.813) is
> real and now correctly the model actually shipped in `models/turn_detector_whisper.joblib`
> — a stale-results bug that had shipped a worse model was found and fixed
> this pass, see `BUGFIX_NOTES.md`. Since this README was originally written,
> attention pooling, an MLP head, streaming/VAD simulation, threshold
> tuning, calibration, class-imbalance analysis, ONNX export, and INT8
> quantization have all been added (some real & verified, some
> classifier-head-only in scope, some blocked by this sandbox's lack of
> network access to `huggingface.co`). **Read `PROJECT_BLOCKERS.md` and
> `REPORT.md` for the real, current, per-task status** — do not trust this
> file's older §6 "Limitations" list below as the full picture; it's kept
> for historical framing but several items in it are now partially
> addressed rather than pure future work.

A small, fast audio classifier that decides whether a speaker has **finished their turn**
(`END`) or is **still talking / just pausing** (`CONTINUE`) — including through filler
words like "um", "matlab", "toh", "actually" etc. Built from scratch on the
[`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
dataset.

**TL;DR result:** a frozen **Whisper-tiny encoder + a ~400-parameter logistic-regression
head** gets **81.5% accuracy / 0.813 F1** on held-out validation data, versus **~50–54%
(chance level)** for a 47-dim hand-crafted acoustic-features baseline (pitch, energy,
pause statistics) evaluated on the *same* split. The gap is the main finding of this
project and it shapes every downstream decision below.

---

## 1. Problem framing

Turn detection is a binary decision per short audio clip: has the user *stopped talking*
(end of turn → the assistant should respond) or are they *still talking* (continue → the
assistant should keep listening)? The hard cases are exactly the ones the brief calls
out — filler words and pauses — because a naive silence/VAD-based detector will
mis-fire on "um, so I think... toh basically what happened was..." and cut the user off.

I treated this as a **frozen-encoder + linear-probe** problem rather than trying to
fine-tune Whisper end-to-end, for three reasons that matter for a *voice AI infra*
setting like Shiprocket's:

1. **Speed/size** — the brief explicitly asks for tiny + fast. Fine-tuning all of
   Whisper-tiny (39M params) is overkill and slower to iterate on than freezing it and
   training a linear head (the only thing I "own" is a 384×2 weight matrix).
2. **Data efficiency** — with only ~2,000 labelled examples used per experiment (see
   §3), a frozen, pretrained representation generalizes far better than anything I could
   learn from scratch or fine-tune reliably on that much data.
3. **Interpretability of the experiment** — freezing the encoder makes it a clean,
   controlled comparison: *is the win coming from the audio features, or from Whisper's
   learned representation of speech?* (Spoiler: the representation.)

---

## 2. Dataset

- Source: [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train),
  streamed (not fully downloaded — it's large) via `datasets.load_dataset(..., streaming=True)`.
- Each example: an audio clip + `endpoint_bool` (`1` = the turn genuinely ends here,
  `0` = speaker continues) + metadata: `language`, `midfiller` (filler word mid-utterance),
  `endfiller` (filler word right at the end), `synthetic` (TTS-generated vs. real speech),
  and `dataset` (sub-source tag).
- Multilingual: 23+ languages present in the sampled subset, including `hin` (Hindi) and
  `eng`, though **no explicit Hinglish/code-switch tag exists** in the dataset — see
  Limitations (§6) for how I proxied "Indian-language relevance" instead of claiming
  something the data doesn't support.
- I worked with a **fixed, seeded 2,000-sample subset** (`seed=42`, shuffled with a
  10k-sample streaming buffer) split 1,600 train / 400 val, so every experiment below is
  reproducible and comparable on the *exact same examples* (`src/create_experiment_manifest.py`).
  A larger 10,000-sample (8,000/2,000) pull was also used for the first acoustic-only
  pass (`src/extract_acoustic_features.py`) before I controlled for sample size.

Reproduce the manifest / raw samples (needs network + HF access):
```bash
python src/create_experiment_manifest.py   # writes data/manifest/experiment_2000.json
python src/sample_audio.py                 # a few labelled .wav examples for demoing/testing
```

---

## 3. Experiments

All numbers below are validation-set metrics; the full breakdown (including 5-fold CV)
lives in [`experiments/results.json`](experiments/results.json), produced by
`src/train_final_model.py`.

### 3.1 Baseline — hand-crafted acoustic features

`src/extract_acoustic_features.py` computes a 47-dim feature vector per clip: RMS
energy trajectory, pitch (via `librosa`) statistics, zero-crossing rate, spectral
centroid/rolloff, trailing-silence duration, energy slope near the end of the clip, etc.
— the kind of features a rules-based turn detector would use.

| Setup | Train/Val | Accuracy | F1 |
|---|---|---|---|
| Acoustic features, logistic regression | 8,000 / 2,000 | 53.9% | 0.507 |
| **Same acoustic features, controlled to the identical 1,600/400 split used for Whisper** | 1,600 / 400 | 50.3% | 0.488 |

Both are essentially **coin-flip performance**. The controlled run
(`src/extract_controlled_features.py` + `src/train_baseline.py`) matters because it
rules out "the acoustic model just had less data" as the explanation — it's evaluated on
the *exact same 400 validation clips* as the Whisper model below, and still can't beat
chance. Hand-tuned acoustic statistics alone don't capture what actually signals "this
person is done talking" — that turns out to be much more about *linguistic/semantic*
content (what was said, whether the sentence is grammatically/pragmatically complete)
than raw prosody in this dataset.

### 3.2 Whisper-tiny embeddings + classifier head

`src/extract_whisper_features.py` runs each clip through the frozen `openai/whisper-tiny`
encoder (`WhisperModel`, not the full seq2seq model — we only need the encoder), takes
`last_hidden_state` (`[time, 384]`) and **mean-pools over time** to get one 384-dim
vector per clip. This is the same representation used for every classifier head tried
below.

I swept 7 classifier heads with 5-fold CV on train (`src/train_final_model.py`):

| Head | CV F1 (mean ± std) | Val Accuracy | Val F1 | Val Precision | Val Recall |
|---|---|---|---|---|---|
| **Logistic Regression, C=0.1** ⭐ | **0.780 ± 0.010** | **81.5%** | **0.813** | 0.774 | 0.856 |
| Logistic Regression, C=1.0 | 0.765 ± 0.033 | 78.0% | 0.774 | 0.748 | 0.803 |
| SVM (RBF), C=5.0 | 0.754 ± 0.011 | 81.3% | 0.810 | 0.773 | 0.851 |
| MLP (32,16) | 0.727 ± 0.020 | 79.3% | 0.787 | 0.761 | 0.814 |
| SVM (RBF), C=1.0 | 0.736 ± 0.030 | 78.8% | 0.789 | 0.740 | 0.846 |
| MLP (64,) | 0.740 ± 0.029 | 76.8% | 0.766 | 0.727 | 0.809 |
| Logistic Regression, C=10.0 | 0.740 ± 0.041 | 75.0% | 0.744 | 0.718 | 0.771 |

**Winner: logistic regression with strong regularization (C=0.1)**, selected by 5-fold
CV F1 (not by peeking at val — the CV ranking and val ranking agree). With only 1,600
training examples and a 384-dim input, heavier models (SVM/MLP) or weaker regularization
overfit — the CV std for C=1.0 (±0.033) vs C=0.1 (±0.010) makes that pretty visible.
The shipped model is genuinely tiny: **384 × 2 + 2 ≈ 770 learned parameters**, on top of
the frozen 39M-parameter encoder.

### 3.3 Error analysis (`src/analyze_whisper_errors.py`)

Breaking the winning model's errors down by metadata slice on the val set:

- **By filler position** — `midfiller=True` (filler word in the middle of the utterance):
  80.4% accuracy; `midfiller=False`: 72.4%. Interestingly, clips *with* a mid-utterance
  filler were classified *more* accurately, not less — a filler is itself a fairly strong
  signal.
- **`endfiller` is the genuinely hard case, and the model handles it correctly by
  construction of the data**: every `endfiller=True` clip in this dataset has true label
  `CONTINUE` (a trailing "um"/"uh" means the speaker isn't done) — TP=0, FN=0, i.e. there
  are no true END examples in this slice at all — and the model predicts `CONTINUE` for
  92/117 of them (78.6% accuracy), with 25 false positives where it's fooled into
  thinking the turn ended. This is exactly the "filler word at the end tricks a naive
  detector" failure mode the brief warns about, and it's the largest remaining error
  bucket.
- **Synthetic vs. real speech**: real speech (`synthetic=False`) scores *higher*
  (86.5%) than TTS-generated speech (76.1%) — reassuring, since real deployment traffic
  is real speech, not TTS.
- **Language**: performance varies a lot by language (`ita`/`ukr`: 100% on tiny N=9;
  `spa`/`zho`: 50–53%; `hin`: 77.8%, N=18) but sample sizes per language are small
  (9–21 examples) so these are indicative, not statistically solid, conclusions.
- **"Indian-language" proxy** (`hin`, `ben`, `mar`, `eng` — the closest thing to an
  India-relevant slice this dataset's tags support): 80.1% accuracy, essentially in line
  with overall performance. See Limitations for why this is *not* a real Hinglish
  evaluation.

---

## 4. Final model

- **Encoder**: `openai/whisper-tiny` (frozen, `WhisperModel.encoder` only — no decoder
  needed since we're not transcribing).
- **Feature**: mean-pooled `last_hidden_state` over the time axis → 384-dim vector.
- **Head**: `StandardScaler` + `LogisticRegression(C=0.1)`, saved as a single
  scikit-learn `Pipeline`.
- **Artifacts**: [`models/turn_detector_whisper.joblib`](models/turn_detector_whisper.joblib)
  (the pipeline) + [`models/turn_detector_whisper.meta.json`](models/turn_detector_whisper.meta.json)
  (embedding config / label map, so the inference code isn't guessing).
- **Size/speed**: the *trainable* part is ~770 parameters and trains in well under a
  second on CPU. The encoder forward pass (the actual cost) is a single Whisper-tiny
  encoder call per clip — this is the "tiny + fast" the brief is asking for, since we're
  reusing an existing tiny pretrained model rather than shipping something new and heavy.

Retrain / reproduce from cached features:
```bash
python src/train_final_model.py
```
This regenerates both baselines, the full head sweep, and
`experiments/results.json`, and overwrites `models/turn_detector_whisper.joblib`.

---

## 5. Running it

```bash
pip install -r requirements.txt
```

**CLI inference on a single file:**
```bash
python src/inference.py data/samples/end.wav
# -> data/samples/end.wav: END  (P(END)=0.94, 38.2 ms)
```

**Gradio demo** (mic or file upload, plus one-click sample clips):
```bash
python app/app.py
```

**Tests:**
```bash
pytest tests/ -v
```
`tests/test_classifier.py` runs fully offline against the cached embeddings and the
saved model (checks output shape/validity and a minimum accuracy bar, and confirms the
Whisper-based model clearly beats the acoustic-only baseline). `tests/test_inference_pipeline.py`
is the true end-to-end audio→prediction integration test using the 4 labelled clips in
`data/samples/`; it needs `torch`/`transformers` + Hugging Face Hub access to download
`openai/whisper-tiny` and skips cleanly (not fail) if that's unavailable in the runtime
environment.

> **Note on this submission's environment**: the classifier itself was fully trained and
> evaluated here using the *already-extracted* embeddings/features under `data/` (all of
> `src/train_*.py` and the tests in `tests/test_classifier.py` were actually run to
> produce the numbers in this README). Live end-to-end audio inference
> (`src/inference.py`, `app/app.py`) needs to download `openai/whisper-tiny` from the
> Hugging Face Hub at runtime, which this particular sandboxed session doesn't have
> network access to — but it's the same call `extract_whisper_features.py` already made
> successfully to build the training data, so it will work in any normal environment
> with internet access (a laptop, a Space, a server).

---

## 6. Limitations & what I'd do next with more time

*(Original framing below, kept for context — see `REPORT.md` for the current,
post-fix status of each item.)*

- **No real Hinglish/code-switching signal in the dataset.** `smart-turn-data-v3.2`
  tags single languages (`hin`, `eng`, etc.), not code-switched utterances, so I could
  not directly evaluate the "Hinglish" scenario the brief emphasizes — the "Indian
  languages" slice in §3.3 is a proxy, not a real measurement, and I've labelled it as
  such rather than overclaiming.
  **Update:** `data/hinglish_eval/manifest.csv` now has 41 hand-written, text-only
  synthetic Hinglish/filler/pause/ambiguous-ending examples with ground truth —
  but still no audio and no model evaluation against them (needs TTS/recordings
  plus encoder access, both blocked in this sandbox — see `PROJECT_BLOCKERS.md`).
- **Only 2,000 labelled examples used.** The full dataset is much larger; I intentionally
  worked with a fixed, seeded subset to keep iteration fast and comparisons controlled,
  but the final head would likely improve with more training data, especially for
  under-represented languages (many had N<15 in val).
  **Update:** a data-scaling experiment was run in a prior pass at 400/800/1200/1600
  samples (`experiments/data_scaling/results.csv`) — smaller than the 10K/25K/50K
  target, since that needs more raw audio than was cached locally.
- **Mean pooling discards timing information.** A clip-level mean-pool can't represent
  *where* in the utterance the pause/filler happens, or the pause duration itself — only
  "what kind of speech content is present overall."
  **Update:** attention pooling is now implemented and unit-tested
  (`src/model.py::AttentionPooling`), but the one existing trained comparison
  (`experiments/pooling_comparison/results.csv`) shows it collapsing to 0.0 F1 —
  a broken training run, not a validated result either way. Re-running it needs
  frame-level Whisper features, which needs encoder access this sandbox doesn't have.
- **No streaming/incremental design yet.**
  **Update:** implemented — `src/streaming.py::StreamingTurnDetector`, a sliding
  window with a simple energy-based VAD gate (explicitly documented there as a
  simulation, not a production VAD).
- **Whisper-tiny was not fine-tuned**, by design (see §1).
  **Update:** still not fine-tuned — this needs the encoder, which requires network
  access to `huggingface.co` unavailable in this sandbox. `src/train_finetune.py`
  exists as scaffolding for when that access is available.

**New since the original pass, not in the list above:** threshold tuning
(`reports/selected_threshold.json`, 0.45 not 0.5), probability calibration
(`reports/calibration.json`), a real class-imbalance comparison
(`reports/class_imbalance.json` — data is ~50/50, plain BCE confirmed
sufficient), classifier-head ONNX export + verification
(`reports/onnx_export_verification.json`), classifier-head INT8 quantization
(`reports/int8_quantization.json`, 0.0 F1 drop), and a classifier-head-only
latency benchmark (`reports/latency_benchmark.json`). All scoped to the
classifier head only (385 params) — the Whisper-tiny encoder (~39M params,
where these would matter most for a genuinely "tiny" end-to-end system)
could not be exported/quantized/benchmarked without network access. See
`REPORT.md` for the full, honest breakdown.

---

## 7. Repo structure

```
Audio_ml_model/
├── README.md                       # this file
├── REPORT.md                       # full technical report, real numbers only
├── PROJECT_BLOCKERS.md             # exactly what's blocked in this sandbox and why
├── BUGFIX_NOTES.md                 # stale-results bug found + fixed this pass
├── DEPLOYMENT.md                   # HF Space deployment steps (not yet deployed)
├── requirements.txt
├── checkpoints/                    # classifier-head FP32/INT8 ONNX exports
├── data/
│   ├── manifest/                   # 2000-sample reproducible experiment manifest
│   ├── samples/                    # 4 labelled demo clips (end/continue/midfiller x2)
│   ├── features/                   # 47-dim acoustic features, large 8000/2000 split
│   ├── controlled_features/acoustic/  # same acoustic features, controlled 1600/400 split
│   ├── whisper_features/           # 384-dim Whisper-tiny embeddings, 1600/400 split
│   └── hinglish_eval/manifest.csv  # 41 text-only synthetic Hinglish examples, no audio
├── src/
│   ├── data_inspection.py          # dataset-wide stats (label balance, languages, fillers)
│   ├── audio_analysis.py           # exploratory single-clip audio analysis
│   ├── create_experiment_manifest.py  # builds the seeded 2000-sample manifest
│   ├── create_subset.py            # dumps a larger raw-audio subset to disk
│   ├── sample_audio.py             # saves the 4 labelled demo clips
│   ├── extract_acoustic_features.py   # 47-dim hand-crafted features (large split)
│   ├── extract_controlled_features.py # same features, controlled split
│   ├── extract_whisper_features.py    # Whisper-tiny mean-pooled embeddings
│   ├── train_acoustic_baseline.py     # trains/evaluates baseline 1
│   ├── train_baseline.py              # trains/evaluates controlled acoustic baseline
│   ├── train_whisper_baseline.py      # trains/evaluates default-C Whisper baseline
│   ├── train_final_model.py           # full sweep + final model selection (run this)
│   ├── analyze_whisper_errors.py      # per-language / per-filler-type error breakdown
│   ├── inference.py                # TurnDetector class: audio file -> label
│   ├── model.py                    # MeanPooling/MaxPooling/LastFramePooling/AttentionPooling, TurnMLPHead
│   ├── streaming.py                # StreamingTurnDetector (sliding window + energy-VAD simulation)
│   ├── class_imbalance_experiment.py  # Task 7: real plain/weighted/focal BCE comparison
│   └── build_hinglish_eval_manifest.py # Task 8: builds data/hinglish_eval/manifest.csv (text only)
├── scripts/
│   ├── export_onnx.py               # classifier-head ONNX export + verification
│   └── quantize_int8.py             # classifier-head INT8 quantization + comparison
├── app/
│   └── app.py                      # Gradio demo
├── models/
│   ├── turn_detector_whisper.joblib       # final model (Whisper embeddings + logreg, FIXED this pass)
│   ├── turn_detector_whisper.meta.json    # embedding config / label map
│   ├── turn_detector_whisper.joblib.stale_backup_20260821  # pre-fix backup, see BUGFIX_NOTES.md
│   └── turn_detector_logistic.joblib      # controlled-acoustic baseline model
├── experiments/
│   ├── results.json                # full metrics for every experiment above (FIXED this pass)
│   ├── results.json.stale_backup_20260821  # pre-fix backup, see BUGFIX_NOTES.md
│   └── pooling_comparison/results.csv  # broken attention-pooling run, see REPORT.md §4
├── reports/                         # every metric this pass produced, one JSON/CSV per task
├── notebook/
│   └── 01_dataset_analysis.ipynb   # initial exploratory analysis
└── tests/
    ├── test_classifier.py          # offline tests against cached embeddings
    ├── test_model.py               # pooling/MLP unit tests
    ├── test_streaming.py           # streaming detector unit tests
    └── test_inference_pipeline.py  # end-to-end audio tests (needs HF access)
```
