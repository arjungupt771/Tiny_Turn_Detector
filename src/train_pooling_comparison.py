"""Controlled comparison of mean/max/last/attention pooling on real frame states."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TemporalTurnClassifier

ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = ROOT / "data" / "whisper_frame_features"
RESULTS_DIR = ROOT / "experiments" / "pooling_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_data():
    required = ["X_train_frames.npy", "train_attention_mask.npy", "y_train.npy", "X_val_frames.npy", "val_attention_mask.npy", "y_val.npy"]
    missing = [str(FRAME_DIR / f) for f in required if not (FRAME_DIR / f).exists()]
    if missing:
        raise FileNotFoundError("Frame-level Whisper features are missing. Run `python src/extract_frame_features.py` first.\n" + "\n".join(missing))
    Xtr = np.load(FRAME_DIR / "X_train_frames.npy", mmap_mode="r")
    Mtr = np.load(FRAME_DIR / "train_attention_mask.npy", mmap_mode="r")
    ytr = np.load(FRAME_DIR / "y_train.npy", mmap_mode="r")
    Xv = np.load(FRAME_DIR / "X_val_frames.npy", mmap_mode="r")
    Mv = np.load(FRAME_DIR / "val_attention_mask.npy", mmap_mode="r")
    yv = np.load(FRAME_DIR / "y_val.npy", mmap_mode="r")
    if not (len(Xtr) == len(Mtr) == len(ytr) and len(Xv) == len(Mv) == len(yv)):
        raise ValueError("Frame features, masks, and labels are misaligned")
    if Xtr.ndim != 3 or Xtr.shape[-1] != 384:
        raise ValueError(f"Expected [N,T,384] frame states, got {Xtr.shape}")
    return [torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in (Xtr, Mtr, ytr, Xv, Mv, yv)]


def train_one(pooling, data):
    seed_everything()
    Xtr, Mtr, ytr, Xv, Mv, yv = data
    train_loader = DataLoader(TensorDataset(Xtr, Mtr, ytr.float()), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model = TemporalTurnClassifier(hidden_size=384, widths=(128,), dropout=0.1, pooling=pooling).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state, best_f1 = None, -1.0
    start = time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, mb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.inference_mode():
            probs = torch.sigmoid(model(Xv.to(DEVICE), Mv.to(DEVICE))).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(yv.numpy(), preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - start
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        probs = torch.sigmoid(model(Xv.to(DEVICE), Mv.to(DEVICE))).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y = yv.numpy()
    result = {"pooling": pooling, "accuracy": accuracy_score(y, preds), "precision": precision_score(y, preds, zero_division=0), "recall": recall_score(y, preds, zero_division=0), "f1": f1_score(y, preds, zero_division=0), "train_time_seconds": train_time, "device": DEVICE}
    torch.save({"state_dict": model.state_dict(), "pooling": pooling}, RESULTS_DIR / f"{pooling}.pt")
    return result


def main():
    seed_everything()
    data = load_data()
    results = [train_one(pooling, data) for pooling in ("mean", "max", "last", "attention")]
    with (RESULTS_DIR / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
