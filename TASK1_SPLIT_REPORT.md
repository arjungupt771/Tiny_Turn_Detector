# SECTION 1 — TASK 1: PROPER TRAIN/VALIDATION/TEST SPLIT

## Methodology

The data preparation pipeline (`scripts/prepare_data.py`) creates deterministic train/validation/test splits using the official Hugging Face datasets:
- **Training source:** `pipecat-ai/smart-turn-data-v3.2-train` (10,300 samples used)
- **Test source:** `pipecat-ai/smart-turn-data-v3.2-test` (500 samples used)
- **Random seed:** 42 (fixed for reproducibility)
- **Shuffling:** 10k-sample buffer (streaming with fixed seed)

### Split Allocation
The 10,300 training samples were split:
- **Train:** 4,000 samples (77%)
- **Validation:** 800 samples (15%)
- **Test:** Official test set, 500 samples (100% separate dataset)

The test set comes from a completely separate official dataset and is never used for
checkpoint selection, threshold optimization, calibration, or hyperparameter tuning.

## Split Statistics

### Train Set (4,000 samples)
- **Class balance:** END 49.7% (1,989) | CONTINUE 50.3% (2,011) ✓ Well balanced
- **Duration:** Mean 7.72s, stdev 3.35s (range 0.64s–26.89s, p50 7.16s)
- **Languages:** 23 unique, English 24.2%, Spanish 5.9%, Russian 4.8%, ...
- **Synthetic:** 82.6% (3,305) synthetic TTS, 17.4% (695) human recordings
- **Mid-filler:** 41.4% True (filler present), 38.0% False, 20.6% missing metadata
- **End-filler:** 53.1% False, 26.2% True, 20.6% missing metadata
- **Dataset sources:** 12 unique, dominated by chirp3_1 (54.3%), chirp3_2 (24.8%), liva_1 (11.4%)

### Validation Set (800 samples)
- **Class balance:** END 46.8% (374) | CONTINUE 53.2% (426) ✓ Well balanced
- **Duration:** Mean 7.79s, stdev 3.42s (range 1.16s–28.04s, p50 7.16s)
- **Languages:** 23 unique, English 23.8%, Portuguese 5.6%, Spanish 5.5%, ...
- **Synthetic:** 82.5% (660) synthetic, 17.5% (140) human
- **Mid-filler:** 40.5% False, 39.2% True, 20.2% missing
- **End-filler:** 52.5% False, 27.2% True, 20.2% missing
- **Dataset sources:** Same 12 sources, similar proportions

### Test Set (500 samples)
- **Class balance:** END 44.2% (221) | CONTINUE 55.8% (279) ✓ Well balanced
- **Duration:** Mean 7.52s, stdev 3.37s (range 0.64s–19.56s, p50 7.16s)
- **Languages:** 23 unique, English 27.2%, Russian 5.8%, Spanish/German/Portuguese 4.8% each, ...
- **Synthetic:** 81.6% (408) synthetic, 18.4% (92) human
- **Mid-filler:** 41.4% True, 36.2% False, 22.4% missing
- **End-filler:** 48.6% False, 29.0% True, 22.4% missing
- **Dataset sources:** 11 unique (missing chirp3_3_short from test)

## Data Integrity Validation

✓ **No duplicate IDs:** 5,300 unique IDs across all splits
✓ **No train/val leakage:** Train and validation share 0 IDs
✓ **No train/test leakage:** Train and test share 0 IDs
✓ **No val/test leakage:** Validation and test share 0 IDs
✓ **Audio validity:** All 5,300 audio files present and valid (sample_rate=16kHz, non-empty)
✓ **Supported sample rate:** All files use 16 kHz (required by Whisper)
✓ **Valid labels:** All 5,300 samples have label ∈ {0, 1}

## Key Observations

1. **Class balance:** All three splits maintain ~50/50 class distribution (END vs CONTINUE),
   with minimal variation (train 49.7%, val 46.8%, test 44.2% END). No class imbalance
   treatment is required at the split level.

2. **Language coverage:** 23 languages present across all splits, with English dominant
   (24–27%) but meaningful representation of Hindi, Russian, Spanish, Portuguese, German,
   etc. No evidence of language leakage between splits.

3. **Synthetic/human ratio:** Consistent ~82% synthetic across all splits. This high
   proportion reflects the dataset composition; no rebalancing between splits.

4. **Duration distribution:** Nearly identical across splits (mean ~7.7s, stdev ~3.4s),
   confirming random shuffling preserved distribution.

5. **Dataset source:** Dominated by chirp3 variants (79% of train), with secondary sources
   (liva, rime, mundo, orpheus, human) consistently represented across splits.

6. **Metadata completeness:** ~20% of samples missing midfiller/endfiller information,
   uniformly distributed across splits; this is acceptable as these are optional attributes.

## Files Created/Modified

- `scripts/prepare_data.py` — Main data preparation script (already existed, confirmed working)
- `scripts/validate_splits.py` — Validation and statistics report (new)
- `data/manifests/train.csv` — 4,000 training samples (created by prepare_data.py)
- `data/manifests/val.csv` — 800 validation samples (created by prepare_data.py)
- `data/manifests/test.csv` — 500 official test samples (created by prepare_data.py)
- `TASK1_SPLIT_REPORT.md` — This file

## Next Steps

Task 1 is complete. The proper train/validation/test split is established and validated.
Proceed to Task 2: Data scaling experiment using train and validation sets, with the
official test set held out.

## References

- Dataset: [pipecat-ai/smart-turn-data-v3.2-train](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
- Dataset: [pipecat-ai/smart-turn-data-v3.2-test](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test)
- Random seed: 42 (Python hash randomization disabled during split)
- Execution: `python scripts/prepare_data.py --seed 42` (default parameters)
