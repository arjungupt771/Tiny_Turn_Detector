"""Extract unpooled Whisper encoder states for temporal pooling experiments.

Outputs [N, max_frames, 384] plus an attention mask. These are real frame-level
encoder states and must not be confused with the existing pooled X_*.npy files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "whisper_frame_features"
MODEL_NAME = "openai/whisper-tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model


def extract_frame_states(waveform, sample_rate, processor, model, max_frames: int = 1500):
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0 if waveform.shape[0] <= waveform.shape[1] else 1)
    waveform = waveform.astype(np.float32)
    if sample_rate != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)
    with torch.inference_mode():
        hidden = model.encoder(input_features).last_hidden_state.squeeze(0).cpu().numpy()

    valid = min(hidden.shape[0], max_frames)
    out = np.zeros((max_frames, hidden.shape[1]), dtype=np.float32)
    mask = np.zeros(max_frames, dtype=np.int64)
    out[:valid] = hidden[:valid].astype(np.float32)
    mask[:valid] = 1
    return out, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pipecat-ai/smart-turn-data-v3.2-train")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    processor, model = load_model()

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10_000)

    total = args.train_size + args.val_size if args.split == "train" else None
    train_states, train_masks, train_y, train_meta = [], [], [], []
    val_states, val_masks, val_y, val_meta = [], [], [], []

    iterator = tqdm(dataset, total=total, desc=f"Extracting frame states ({args.split})")
    for i, sample in enumerate(iterator):
        if total is not None and i >= total:
            break
        audio = sample["audio"]
        decoded = audio.get_all_samples()
        states, mask = extract_frame_states(decoded.data.numpy(), decoded.sample_rate, processor, model, args.max_frames)
        label = int(sample["endpoint_bool"])
        meta = {k: sample.get(k) for k in ("language", "midfiller", "endfiller", "synthetic", "dataset")}
        if args.split == "train" and i < args.train_size:
            train_states.append(states); train_masks.append(mask); train_y.append(label); train_meta.append(meta)
        elif args.split == "train":
            val_states.append(states); val_masks.append(mask); val_y.append(label); val_meta.append(meta)
        else:
            train_states.append(states); train_masks.append(mask); train_y.append(label); train_meta.append(meta)

    if args.split == "train":
        np.save(output / "X_train_frames.npy", np.stack(train_states))
        np.save(output / "train_attention_mask.npy", np.stack(train_masks))
        np.save(output / "y_train.npy", np.asarray(train_y, dtype=np.int64))
        np.save(output / "X_val_frames.npy", np.stack(val_states))
        np.save(output / "val_attention_mask.npy", np.stack(val_masks))
        np.save(output / "y_val.npy", np.asarray(val_y, dtype=np.int64))
        (output / "train_metadata.json").write_text(json.dumps(train_meta, indent=2), encoding="utf-8")
        (output / "val_metadata.json").write_text(json.dumps(val_meta, indent=2), encoding="utf-8")
        print(f"Saved train={len(train_y)}, val={len(val_y)} frame-level samples")
    else:
        np.save(output / "X_test_frames.npy", np.stack(train_states))
        np.save(output / "test_attention_mask.npy", np.stack(train_masks))
        np.save(output / "y_test.npy", np.asarray(train_y, dtype=np.int64))
        (output / "test_metadata.json").write_text(json.dumps(train_meta, indent=2), encoding="utf-8")
        print(f"Saved test={len(train_y)} frame-level samples")


if __name__ == "__main__":
    main()
