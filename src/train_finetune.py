# src/train_finetune.py
"""
Task 6 — fine-tune Whisper-tiny for turn detection.
  A: frozen encoder + attention pooling + MLP
  B: unfreeze last N encoder layers
  C: fully fine-tuned encoder (only if compute allows)

Usage:
    python src/train_finetune.py --experiment A
    python src/train_finetune.py --experiment B --unfreeze-last-n 4
    python src/train_finetune.py --experiment C
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import WhisperModel, WhisperProcessor, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import configure_whisper_finetuning, TemporalTurnClassifier

MODEL_NAME = "openai/whisper-tiny"
TARGET_SR = 16000
DATASET_TRAIN = "pipecat-ai/smart-turn-data-v3.2-train"

MANIFEST_DIR = Path("data/manifests")
CKPT_DIR = Path("checkpoints/finetune"); CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("experiments/finetuning"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 8
HEAD_LR = 1e-3
ENCODER_LR = 1e-5
GRAD_CLIP_NORM = 1.0


class RawAudioManifestDataset(Dataset):
    """Loads raw waveforms for a manifest split into memory once, matched by id
    against the streaming HF dataset. ~4000/800 examples at ~7.7s avg is a
    few GB float32 -- switch to on-disk caching if that doesn't fit in RAM."""
    def __init__(self, split):
        import csv
        with open(MANIFEST_DIR / f"{split}.csv") as f:
            wanted = {row["id"]: row for row in csv.DictReader(f)}
        self.examples = []
        ds = load_dataset(DATASET_TRAIN, split="train", streaming=True)
        found = 0
        for sample in ds:
            sid = str(sample["id"])
            if sid not in wanted:
                continue
            audio = sample["audio"].get_all_samples()
            waveform = audio.data.numpy().astype(np.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=0)
            sr = int(audio.sample_rate)
            if sr != TARGET_SR:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
            self.examples.append((waveform, int(wanted[sid]["label"])))
            found += 1
            if found >= len(wanted):
                break
        print(f"{split}: loaded {found}/{len(wanted)} raw audio examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, processor):
    waveforms = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    inputs = processor(waveforms, sampling_rate=TARGET_SR, return_tensors="pt")
    return inputs.input_features, labels


def run_experiment(mode, unfreeze_last_n=4, pooling="attention"):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    encoder = WhisperModel.from_pretrained(MODEL_NAME).encoder.to(device)
    trainable_encoder_params = configure_whisper_finetuning(
        encoder, mode={"A": "frozen", "B": "partial", "C": "full"}[mode],
        final_layers=unfreeze_last_n,
    )
    head = TemporalTurnClassifier(hidden_size=384, pooling=pooling).to(device)
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"Experiment {mode}: trainable_encoder_params={trainable_encoder_params}, total_params={total_params}")

    train_ds, val_ds = RawAudioManifestDataset("train"), RawAudioManifestDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=lambda b: collate(b, processor))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda b: collate(b, processor))

    param_groups = [{"params": head.parameters(), "lr": HEAD_LR}]
    if trainable_encoder_params > 0:
        param_groups.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": ENCODER_LR})
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = MAX_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_f1, best_state, no_improve = -1.0, None, 0
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        encoder.train(mode != "A" and trainable_encoder_params > 0)
        head.train()
        for input_features, labels in train_loader:
            input_features, labels = input_features.to(device), labels.to(device)
            optimizer.zero_grad()
            hidden = encoder(input_features).last_hidden_state
            loss = loss_fn(head(hidden), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), GRAD_CLIP_NORM)
            optimizer.step(); scheduler.step()

        encoder.eval(); head.eval()
        all_probs, all_labels = [], []
        with torch.inference_mode():
            for input_features, labels in val_loader:
                hidden = encoder(input_features.to(device)).last_hidden_state
                all_probs.append(torch.sigmoid(head(hidden)).cpu().numpy())
                all_labels.append(labels.numpy())
        probs, labels_np = np.concatenate(all_probs), np.concatenate(all_labels)
        preds = (probs >= 0.5).astype(int)
        val_f1 = f1_score(labels_np, preds, zero_division=0)
        print(f"[{mode}] epoch {epoch+1}/{MAX_EPOCHS} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {"encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                           "head": {k: v.cpu().clone() for k, v in head.state_dict().items()}}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    torch.save(best_state, CKPT_DIR / f"experiment_{mode}_best.pt")

    encoder.load_state_dict(best_state["encoder"]); head.load_state_dict(best_state["head"])
    encoder.eval(); head.eval()
    all_probs, all_labels = [], []
    with torch.inference_mode():
        for input_features, labels in val_loader:
            hidden = encoder(input_features.to(device)).last_hidden_state
            all_probs.append(torch.sigmoid(head(hidden)).cpu().numpy())
            all_labels.append(labels.numpy())
    probs, labels_np = np.concatenate(all_probs), np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)

    result = {
        "experiment": mode, "pooling": pooling,
        "unfreeze_last_n": unfreeze_last_n if mode == "B" else None,
        "trainable_encoder_params": trainable_encoder_params, "total_params": total_params,
        "training_time_seconds": training_time, "best_val_f1": float(best_val_f1),
        "val_accuracy": float(accuracy_score(labels_np, preds)),
        "val_precision": float(precision_score(labels_np, preds, zero_division=0)),
        "val_recall": float(recall_score(labels_np, preds, zero_division=0)),
    }
    with open(RESULTS_DIR / f"experiment_{mode}_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "C"], required=True)
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument("--pooling", default="attention")
    args = parser.parse_args()
    run_experiment(args.experiment, args.unfreeze_last_n, args.pooling)