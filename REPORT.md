# Technical Report — Tiny Turn Detector (Shiprocket Data Scientist Challenge)

**Read `PROJECT_BLOCKERS.md` first.** This sandbox cannot reach `huggingface.co`
(confirmed 403, `host_not_allowed`), so it cannot download `openai/whisper-tiny`
or new dataset audio. Everything below that needed either is marked "not run"
rather than faked. Everything computed from the already-cached, real
Whisper-tiny features (1600 train / 400 val examples) is real, and is marked
as such.

---

## 1. Why this problem?

Turn detection is the decision that gates every response in a voice assistant:
has the user actually stopped talking, or are they just pausing / using a
filler word mid-thought? Get it wrong in the "cut them off" direction and the
assistant feels rude and unusable; get it wrong in the "wait too long"
direction and it feels laggy and unresponsive. For a market like India, this
is sharper than for English-only systems: Hinglish code-switching ("haan
toh...", "matlab...", "basically...") produces exactly the acoustic and
lexical patterns — filler words, mid-clause pauses — that a naive
silence/VAD-based detector mis-reads as endpoints.

## 2. Why acoustic features were insufficient?

A 47–54-dim hand-crafted acoustic baseline (pitch, energy, pause statistics)
scores **50.3–54.0% accuracy / 0.488–0.508 F1** on this task (`experiments/results.json`,
`acoustic_baseline_large` and `acoustic_baseline_controlled`) — indistinguishable
from chance on a roughly-balanced binary problem. Pitch/energy/pause statistics
don't carry the lexical/semantic content that actually signals a completed
thought ("...and that's basically it" vs. "...and that's basically"); you need
something that has learned language structure, not just prosody.

## 3. Why Whisper representations helped?

Swapping in frozen Whisper-tiny encoder embeddings (mean-pooled, 384-dim) on
the *exact same* 1600/400 split and moving to a simple logistic-regression
head jumps to **81.5% accuracy / 0.813 F1** (`experiments/results.json`,
`selected_model`, freshly re-verified this pass — see §12 "Data integrity fix").
That's a ~30-point accuracy jump from swapping only the representation, with
everything else (classifier, split, sample count) held constant — strong
evidence the win is in what Whisper's encoder has learned about speech, not
in the classifier or the data.

## 4. Why mean pooling was insufficient?

Mean pooling treats every audio frame as equally informative for the
"has this person finished" decision, which isn't true — the frames right
around a potential endpoint (or a filler word) plausibly matter more than
frames from the middle of a long, clearly-still-talking utterance. This
motivated implementing attention pooling (`src/model.py::AttentionPooling`)
as an alternative. **Caveat on what we can say here:** the controlled
pooling-strategy comparison that exists in this repo
(`experiments/pooling_comparison/results.csv`, from a prior pass) shows
attention pooling collapsing to 0.0 F1/precision/recall — that's a broken
training run (likely a masking or gradient-flow bug), not a genuine negative
result, and re-running it needs frame-level (unpooled) Whisper features that
require the encoder to extract, which this sandbox can't do. So: the
*architectural* argument for attention pooling stands (it's a strictly more
expressive pooling function), but there is currently no valid trained result
either supporting or refuting it — this is flagged, not resolved, in this
pass.

## 5. Why attention pooling was selected [or wasn't]?

Not resolved with real data this pass — see §4. It was implemented and
unit-tested (`tests/test_model.py`) as the more principled choice, but "we
implemented it and it's more expressive in principle" is different from "we
proved it helps," and this report doesn't claim the latter.

## 6. Why fine-tuning was or wasn't useful?

**Not run.** Fine-tuning any part of Whisper-tiny requires loading the actual
encoder weights, which requires `huggingface.co` access unavailable in this
sandbox. `src/train_finetune.py` exists as scaffolding from a prior pass but
was not executed here. No fine-tuning result — real or fabricated — exists
for this pass.

## 7. What are the major error modes?

From the existing `reports/error_analysis.csv` and `reports/targeted_metrics.csv`
(computed on the earlier 400-example validation split, before this pass's
classifier-head fix — see §12, so treat these as indicative rather than final
until regenerated against the corrected model): errors cluster around
mid-utterance filler words and short pauses, exactly the cases the brief
flags as hard. A more complete picture needs the targeted subsets regenerated
against the fixed model (`models/turn_detector_whisper.joblib`) — not done
in this pass; flagged in `BUGFIX_NOTES.md`.

## 8. How does Hinglish affect the problem?

The dataset has language tags (`hin`, `eng`, 21 others) but **no explicit
Hinglish/code-switch tag** — a real limitation, not something to paper over.
This pass built `data/hinglish_eval/manifest.csv`, 41 hand-written,
text-only synthetic Hinglish examples spanning code-switching, mid/end
fillers (um, uh, hmm, actually, basically, like, matlab, toh, haan, acha,
wait), conceptual pause lengths, and ambiguous endings — but **no audio
exists for these and no model was evaluated against them** (needs TTS or
real recordings, plus the Whisper encoder, neither available here). This is
explicitly a labeled manifest for someone with network access to turn into
audio and evaluate, not a completed evaluation.

## 9. How do fillers affect predictions?

Not independently re-verified this pass beyond what's in the existing
(pre-fix, see §12) `reports/targeted_metrics.csv`. The Hinglish manifest in
§8 specifically constructs filler-adjacent-to-true-END examples (e.g. "That's
the update, matlab.") precisely because these are the cases most likely to
trip up a model that's learned "filler word nearby → keep listening" as a
shortcut rather than genuinely modeling completion.

## 10. How did threshold selection affect conversational behavior?

`reports/selected_threshold.json` records **threshold = 0.45** (not the
naive 0.5), chosen on validation only, reasoning that false-END (interrupting
the user) is more costly than false-CONTINUE (waiting a bit longer) in a
conversational-AI setting — so far so good in principle. **But** this was
tuned against the pre-fix model and the earlier 400-example val split (see
§12) — it should be re-tuned against the corrected model before being
trusted as final. Not redone in this pass.

## 11. How fast is the final model?

**Classifier head only** (real, measured this pass): mean 0.155ms, p50
0.145ms, p95 0.215ms, p99 0.241ms over 200 runs, batch size 1
(`reports/latency_benchmark.json`). ONNX export of the same head: mean
0.0083-0.0093ms per inference (`reports/onnx_export_verification.json`).
**This is not end-to-end latency** — it excludes audio loading and the
Whisper-tiny encoder forward pass, which dominates real-world latency and
which this sandbox cannot benchmark (no encoder access). The README's
earlier anecdotal 38.2ms end-to-end number was measured in a session with
model access and could not be reproduced or re-verified here.

## 12. How small is it?

Classifier head: **385 parameters** (384 weights + 1 bias), 13.6KB as a
joblib pipeline, 8.3KB as ONNX FP32, 8.4KB as ONNX INT8
(`reports/model_size_analysis.json`). Whisper-tiny itself is publicly
documented at **~39M parameters total** (verified via web search this
session, multiple independent sources agree — not loaded or measured
locally). Stated plainly: **the "tiny" claim for the full deployed system
rests almost entirely on Whisper-tiny's own published footprint**, not on
anything this project added — the classifier head this project trained is
0.001% of the total parameter count. This is worth being honest about rather
than implying the whole pipeline was engineered to be small.

### Data integrity fix found this pass

While sourcing real numbers for this report, `experiments/results.json` was
found to be **stale**: it recorded the Whisper-embedding classifier sweep at
near-chance performance (58.5% acc / 0.574 F1 for the selected head), which
caused an inferior classifier (`svm_rbf_C5`) to be the one actually shipped
in `models/turn_detector_whisper.joblib`. Re-running the exact same,
unmodified sweep code against the current cached feature files reproduces
the widely-quoted 81.5% / 0.813 F1 baseline exactly. This was a real bug —
the cached results file didn't match what its own code produces against the
data currently on disk — not a model-selection judgment call. Fixed this
pass; see `BUGFIX_NOTES.md` for the full root-cause writeup and backup file
locations. **Practical implication:** any report/metric file computed before
this fix (threshold sweep, calibration, targeted metrics, error analysis)
was generated against the old, inferior model and should be regenerated
against the corrected one before being trusted as final — flagged, not done,
in this pass (out of the original 12-task scope).

## 13. What happens after quantization?

For the classifier head (the only component quantizable here): **F1 drop =
0.0000** — INT8 and FP32 ONNX produce identical val-set predictions
(`reports/int8_quantization.json`). Interestingly, the INT8 file (8372
bytes) is *not* smaller than FP32 (8259 bytes) — for a model this tiny,
ONNX/protobuf structural overhead dominates actual weight bytes, so the
expected ~4x compression from INT8 doesn't show up in on-disk size the way
it would for a large model. **This tells you nothing about whether
quantizing Whisper-tiny's encoder (the ~39M-param component where
compression would actually matter) is worthwhile** — that's not measurable
in this sandbox.

## 14. What are the limitations?

- No network access to `huggingface.co` in the environment this pass ran in
  → no fine-tuning result, no encoder ONNX/INT8, no official test-set
  evaluation, no real Hinglish audio evaluation, no full end-to-end latency.
- The attention-pooling result on file is broken (0.0 F1), not a validated
  negative finding.
- Threshold/calibration/error-analysis reports predate this pass's
  classifier-head bugfix and need to be regenerated.
- The Hinglish eval set is text-only and hand-written by one person — 41
  examples is small, and "synthetic Hinglish" (even at the text level) is
  not the same claim as "validated against real Indian speakers."
- No GPU was available in this sandbox even for the parts that were run.

## 15. What would you do with more data/time?

In priority order: (1) get network access to `huggingface.co` and re-run the
now-clearly-scoped blocked list in `PROJECT_BLOCKERS.md` — extract
frame-level features to fix the attention-pooling result, run the three
fine-tuning experiments, do the one-time official test-set evaluation; (2)
regenerate threshold/calibration/error-analysis against the now-corrected
classifier head; (3) turn the 41-row Hinglish text manifest into actual
audio (TTS at minimum, ideally some real human Hinglish recordings) and
report real numbers against it; (4) benchmark true end-to-end latency
(encoder + head) across the 1/2/4/6/8s duration sweep the brief asks for;
(5) only then attempt encoder-level ONNX/INT8, where quantization would
actually matter for the "tiny" claim.

---

## Master results table

Only rows with real, run numbers are populated. Blocked rows are marked, not
fabricated.

| Experiment | Data | Pooling | Fine-tuned | Accuracy | Precision | Recall | F1 | Size | Latency |
|---|---|---|---|---|---|---|---|---|---|
| Acoustic LR (controlled) | 1600/400 | — | No | 0.503 | 0.473 | 0.505 | 0.488 | — | — |
| Whisper + LR (baseline, **fixed this pass**) | 1600/400 | Mean | No | 0.815 | 0.774 | 0.856 | 0.813 | 385 params / 13.6KB | 0.155ms (head only) |
| Whisper + MLP | — | Mean | No | *not run* | | | | | |
| Whisper + Attention | — | Attention | No | **broken run** (0.0 F1 — masking/training bug, not a valid result) | | | | | |
| Fine-tuned Whisper | — | — | — | *blocked — no encoder access* | | | | | |
| Final FP32 (classifier head) | 1600/400 | Mean | No | 0.815 | 0.774 | 0.856 | 0.813 | 13.6KB (joblib) | 0.155ms |
| Final ONNX (classifier head) | 1600/400 | Mean | No | 0.815 | 0.774 | 0.856 | 0.813 | 8.3KB | 0.008-0.009ms |
| Final INT8 (classifier head) | 1600/400 | Mean | No | 0.815 | 0.774 | 0.856 | 0.813 | 8.4KB | 0.008ms |

Notes: "Size"/"Latency" for the FP32/ONNX/INT8 rows are classifier-head-only
(§11-13 explain why the encoder can't be included here). The Whisper+MLP and
fine-tuned rows are genuinely not run, not zero-filled as a placeholder for
"bad." The attention-pooling row's 0.0 F1 is a known-broken training run
carried over from a prior pass, reported as-is per the "do not fabricate"
rule rather than hidden or silently re-labeled "not run."

---

## Final quality checklist (this pass)

**Data**
- [x] No train/test leakage (unchanged from prior pass — `TASK1_SPLIT_REPORT.md`)
- [ ] Official test set used — blocked, no cached test features exist locally
- [x] Reproducible manifests (unchanged, still present)
- [x] Class distributions documented (`reports/class_imbalance.json`, this pass)

**Model**
- [x] Baseline preserved and **corrected** (`BUGFIX_NOTES.md`)
- [x] Attention pooling implemented (prior pass); **not validly evaluated** (this pass, honestly flagged)
- [ ] Fine-tuning tested — blocked
- [x] Final architecture documented (README, this pass)

**Evaluation**
- [ ] Hinglish metrics — manifest built (this pass), no audio/no model run
- [ ] Official test evaluation — blocked
- [x] Threshold optimized (prior pass; flagged as needing re-tuning against fixed model)

**Deployment**
- [x] Latency benchmark — real, classifier-head scope only (this pass)
- [x] ONNX export — real, classifier-head scope only (this pass)
- [x] INT8 model — real, classifier-head scope only (this pass)
- [x] ONNX inference verified — real (this pass)
- [ ] Hugging Face deployment — blocked; deployment-ready repo + exact commands provided (`DEPLOYMENT.md`)

**Engineering**
- [x] No fabricated metrics — every number in this report traces to a saved
  `reports/*.json`/`*.csv` file or is explicitly marked "not run" / "cited
  from public docs, not measured"
