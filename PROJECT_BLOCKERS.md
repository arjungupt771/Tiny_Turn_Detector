# Environment blockers and what they mean for each task

## The root blocker

This sandbox's network egress allowlist is: `api.anthropic.com`,
`api.github.com`, `archive.ubuntu.com`, `codeload.github.com`, `crates.io`,
`files.pythonhosted.org`, `github.com`, `index.crates.io`, `npmjs.com`,
`npmjs.org`, `pypi.org`, `pythonhosted.org`, `raw.githubusercontent.com`,
`registry.npmjs.org`, `registry.yarnpkg.com`, `release-assets.githubusercontent.com`,
`security.ubuntu.com`, `static.crates.io`, `www.npmjs.com`, `www.npmjs.org`,
`yarnpkg.com`.

**`huggingface.co` is not on this list** (confirmed: `HTTP 403,
x-deny-reason: host_not_allowed`). No `torch` was installed either at the
start of this pass. Two consequences cascade through almost every task:

1. **The `openai/whisper-tiny` encoder cannot be downloaded**, so nothing
   requiring a forward pass through the actual Whisper model (fine-tuning,
   frame-level feature extraction, real end-to-end latency, encoder ONNX
   export/quantization) can be run for real here.
2. **The `pipecat-ai/smart-turn-data-v3.2-*` dataset audio cannot be
   downloaded**, so nothing requiring new audio (official test-set features,
   Hinglish audio) can be run for real here either.

What *is* available locally and genuinely usable: the already-cached, real
Whisper-tiny mean-pooled feature vectors for the existing 1600-train /
400-val split (`data/whisper_features/{X,y}_{train,val}.npy`), the trained
classifier joblib artifacts, and 4 sample audio clips. Everything below that
touches only these was run for real, with real numbers. Everything that
needed the encoder or new audio was not run, and is marked as such rather
than faked.

## Per-task status (this pass)

| Task | Status | What was actually done |
|---|---|---|
| 6 — Whisper fine-tuning | **Blocked, not run** | Needs the encoder. `src/train_finetune.py` already existed as scaffolding from a prior pass; not executable here. Command to run elsewhere: `python src/train_finetune.py --experiment B --unfreeze-last-n 4` once network access exists. |
| 7 — Class imbalance | **Done, real** | `src/class_imbalance_experiment.py` run against real cached features. Balance is ~50/50; plain BCE, weighted BCE, and a focal-style reweighting were all trained and compared. See `reports/class_imbalance.json`. |
| 8 — Hinglish eval set | **Partially done, honestly scoped** | `data/hinglish_eval/manifest.csv` — 41 text-only, hand-written synthetic Hinglish/code-switch/filler/pause/ambiguous-ending examples with ground-truth labels. No audio exists (no TTS access, no encoder to evaluate against), so `model_evaluated=False` on every row and no results were reported. |
| 15 — Latency benchmark | **Partially done, real but narrow scope** | `reports/latency_benchmark.json` — real classifier-head-only latency (200 runs, mean/p50/p95/p99/stdev). The 1/2/4/6/8s-duration sweep and the encoder's contribution to latency could not be run (needs the encoder). |
| 16 — Model size analysis | **Partially done, real + cited** | Classifier head measured directly (385 params, byte sizes for joblib/ONNX/INT8). Whisper-tiny's 39M-param total is cited from public documentation (verified via web search this session, not from memory), not measured locally — see `reports/model_size_analysis.json`. |
| 17 — ONNX export | **Partially done, real for the head** | `scripts/export_onnx.py` exports and verifies the classifier head (max prob diff ~7e-7, 0 label mismatches against 400 real val examples). Encoder ONNX export not run — needs the encoder. |
| 18 — INT8 quantization | **Partially done, real for the head** | `scripts/quantize_int8.py` — real dynamic INT8 quantization of the head. F1 drop = 0.0000; file size did NOT shrink for a model this tiny (protobuf/runtime overhead dominates at 385 params) — see `reports/int8_quantization.json` for the investigation. Encoder INT8 (where the real win would be) not run. |
| 19 — Verify quantized model | **Done for the head** | Same script; FP32 vs INT8 classification/latency/output validity all checked and logged for the head. Not checked for the (unavailable) encoder. |
| 21 — HF deployment | **Blocked, not run — deployment-ready instead** | No network route to `huggingface.co`, no HF credentials in this sandbox. `DEPLOYMENT.md` documents exact deploy commands and confirms the repo already has everything (`app/app.py`, `requirements.txt`, model artifacts) needed once run elsewhere. |
| 24 — Master results table | **Done, populated honestly** | See `REPORT.md` §"Master results table" — only rows with real numbers are filled in; blocked rows are explicitly marked, not fabricated. |
| 25 — Official test-set evaluation | **Blocked, not run** | `data/manifests/test.csv` has metadata for 2000 official test examples but no cached feature vectors (`X_test.npy` does not exist) — extracting them needs the encoder. Command to run elsewhere: `python src/extract_whisper_features.py --split test` then `python scripts/evaluate.py --split test`. |
| 26 — README | **Done** | Rewritten to reflect actual current state (this pass's real results + explicit blockers), replacing stale "future work" framing. |
| 27 — Technical report | **Done** | `REPORT.md` — answers all 15 questions from the original brief, grounded only in numbers that were actually produced (real ones from cached-feature experiments, cited ones for Whisper-tiny's public spec, explicit "not run" for the rest). |
| 28 — Final quality check | **Done** | See checklist at the bottom of `REPORT.md`. |

## If you have HF network access and want to finish this for real

Run, in order:
1. `pip install torch transformers datasets` (needs `huggingface.co` + `pypi.org`)
2. `python src/create_experiment_manifest.py` (if re-pulling the dataset) or reuse
   existing `data/manifests/*.csv`
3. `python src/extract_frame_features.py` — frame-level (unpooled) Whisper hidden
   states, needed to actually fix Task 5's broken attention-pooling result and to
   run Task 6 fine-tuning
4. `python src/train_pooling_comparison.py` — re-run with real frame-level features
5. `python src/train_finetune.py` — Experiments A/B/C
6. `python src/extract_whisper_features.py --split test` — official test features
7. `python scripts/evaluate.py --split test` — the one-time official evaluation
8. `python scripts/export_onnx.py --include-encoder` (needs writing — currently
   head-only) and `python scripts/quantize_int8.py --include-encoder`
9. Follow `DEPLOYMENT.md` to push a real HF Space
10. Re-run `python src/build_hinglish_eval_manifest.py`'s output through TTS,
    then `python src/inference.py --manifest data/hinglish_eval/manifest.csv`

---

## Completion pass (this session) -- Tasks 8, 12, 15, 16, 17, 18, 20

A follow-up session picked up the "Partially done" tasks left over from the
prior pass (Task 2 -- data scaling -- was explicitly left untouched by
request). The root network blocker described above is unchanged:
`huggingface.co` is still unreachable (`403 host_not_allowed`), so the real
pretrained `openai/whisper-tiny` checkpoint still cannot be downloaded, and
nothing that requires its actual trained weights (real end-to-end accuracy,
real Hinglish model evaluation, the official test set) could be produced.

What changed this session, and how each task now stands:

| Task | New status | What was actually done |
|---|---|---|
| 8 -- Hinglish eval set | **Audio now real, model-eval still blocked** | Installed `espeak-ng` via `apt` (`archive.ubuntu.com` is on the allowlist -- no hub access needed). Synthesized a real `.wav` clip for all 41 manifest rows (`data/hinglish_eval/audio/`), replacing `...` pause markers with SSML `<break>` tags sized to each row's `pause_length_conceptual`, resampled to 16kHz mono. Manifest now has `has_audio=True`, `audio_path`, `audio_duration_sec`, `tts_engine` for every row. **Still not run through the model**: `model_evaluated=False` remains on every row, since scoring them needs the real pretrained encoder. Explicitly labeled as `espeak-ng` formant-synthesis, English voice, romanized text -- not authentic Hinglish prosody. |
| 12 -- Calibration | **Done, real** | Temperature scaling fit on the cached validation logits (`reports/calibration.json`, via `scripts/calibrate_temperature.py`). No encoder needed -- uses the already-trained sklearn pipeline's `decision_function` on `data/whisper_features/X_val.npy`. Result: T ≈ 1.001 -- the logistic-regression baseline was already close to well-calibrated on this validation set; temperature scaling made a negligible difference (Brier basically unchanged, ECE moved by <0.001 in either direction depending on bin convention). |
| 15 -- Latency benchmark | **Materially more complete, real** | `scripts/benchmark_latency_full_pipeline.py` now sweeps real 1/2/4/6/8s audio through resample -> feature-extraction -> encoder -> classifier-head stages (`reports/latency_benchmark_full_pipeline.{json,csv}`, plot in `reports/figures/latency_vs_duration.png`). Resample and feature-extraction stages are fully real. The encoder stage necessarily uses the architecture-equivalent random-weight encoder (see Task 16 below) since the pretrained checkpoint is still undownloadable -- its *latency* is real (compute time depends on shape, not weight values) but carries no accuracy claim. **Real finding**: because this project's own `extract_whisper_features.py` uses `WhisperProcessor`'s default 30-second padding, encoder latency is roughly *constant* (~335-342ms) across all five durations -- only resample (1.4ms->3.4ms) and feature extraction (~9-10ms) actually scale with input length. Cold-start vs. warm and peak RSS are both reported. GPU: still not run (no GPU in this sandbox). |
| 16 -- Model size | **Encoder now measured exactly, not cited** | Realized that `transformers.WhisperModel(WhisperConfig(...))` builds the architecture locally with **zero network calls** (confirmed via `HF_HUB_OFFLINE=1`) -- no download needed, just the public config. Since param count and serialized size depend only on tensor shape/dtype (not weight values), instantiating this randomly-initialized-but-architecturally-identical model gives an exact, real measurement: encoder-only = 8,208,384 params (7,632,384 trainable; the ~576K gap is the non-trainable sinusoidal-position buffer), 31.34MB fp32 on disk. Full encoder+decoder = 37,760,640 params, consistent with the ~39M commonly cited figure. `reports/model_size_analysis.json` updated via `scripts/measure_encoder_size.py`. |
| 17 -- ONNX export | **Encoder export + parity now done, mechanics-only** | `scripts/export_encoder_onnx.py` exports the same architecture-equivalent encoder to ONNX and confirms PyTorch vs. ONNXRuntime numerical parity (max abs diff ~2e-6) on a real (30s-padded) input shape. `reports/onnx_encoder_export_verification.json`. Export mechanics, latency, and total file size (28.6KB graph + 32.9MB external weights = 31.4MB) are real; no accuracy claim, since weights are random. The existing classifier-head-only export (`scripts/export_onnx.py`, `reports/onnx_export_verification.json`) is untouched and still has real accuracy numbers. |
| 18 -- INT8 quantization | **Encoder quantized, mechanics + real size/latency win** | `scripts/quantize_encoder_int8.py` dynamically INT8-quantizes the encoder ONNX export. Unlike the 385-parameter classifier head (where INT8 didn't shrink the file, per the prior session's finding), the 8.2M-parameter encoder shows a real **74.9% size reduction** (31.4MB -> 7.89MB) and a real latency improvement (~332ms -> ~279ms mean) -- confirming the earlier hypothesis in `reports/int8_quantization.json`'s `size_note` that INT8 would actually matter at encoder scale. `reports/onnx_encoder_int8_quantization.json`. Output-difference numbers are reported only to confirm the quantized graph runs correctly on random weights -- explicitly not read as an accuracy/F1 result. |
| 19 -- Verify quantized model | **Encoder-side verification added** | Covered by the same two scripts above: numerical parity (FP32 PyTorch vs ONNX) and FP32-vs-INT8 size/latency for the encoder are both checked and logged, extending the existing head-only verification from the prior pass. |
| 20 -- Gradio demo | **UI additions written, still can't launch live here** | `app/app.py` now renders a waveform plot (`matplotlib`), a plain-language "why this verdict" explanation (distance from the decision threshold), and a Hinglish/filler/pause examples section pulling from the newly-generated `data/hinglish_eval/audio/` clips (clearly labeled synthetic/TTS in the UI copy). Syntax-checked (`python -m py_compile`) but **not run as a live Gradio session**, since `TurnDetector.__init__` still calls `WhisperModel.from_pretrained(...)`, which needs the still-unreachable `huggingface.co`. |

### New packages installed this session (all from allowed hosts)
`espeak-ng` (via `apt-get`, `archive.ubuntu.com`), `torch`, `transformers`,
`onnx`, `onnxscript`, `skl2onnx`, `soundfile`, `matplotlib` (all via `pip`,
`pypi.org`/`files.pythonhosted.org`). No package required `huggingface.co`.

### Tasks intentionally left untouched this session
Task 2 (data scaling beyond 1,600 samples) -- left alone per explicit
instruction. Tasks 5 (pooling comparison), 6 (Whisper fine-tuning), 21 (HF
deployment), 25 (official test-set evaluation) -- left alone per explicit
instruction; all four remain blocked by the same `huggingface.co`/dataset-
audio unavailability as before, unchanged by this session's work.
