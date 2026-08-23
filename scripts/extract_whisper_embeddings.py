from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset
from transformers import WhisperModel, WhisperProcessor
from tqdm import tqdm


DATASET = "pipecat-ai/smart-turn-data-v3.2-train"

SEED = 42

# Increase to 16 if your GPU has enough VRAM.
# If CUDA OOM happens, use 4 or 2.
BATCH_SIZE = 8

MODEL_NAME = "openai/whisper-tiny"


def load_manifests(
    train_path: str,
    val_path: str,
):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    train["split"] = "train"
    val["split"] = "val"

    combined = pd.concat(
        [train, val],
        ignore_index=True,
    )

    required_ids = {
        str(x)
        for x in combined["id"]
    }

    return train, val, combined, required_ids


def extract_all(
    combined_manifest,
    required_ids,
    processor,
    model,
    device,
):
    """
    Single streaming pass over the Hugging Face dataset.

    For every required audio sample:
        audio
          ↓
        Whisper encoder
          ↓
        temporal hidden states
          ↓
        mean / max / last pooling

    We save only pooled 384-dim embeddings,
    keeping disk/RAM usage very small.
    """

    print("\n" + "=" * 60)
    print("WHISPER FEATURE EXTRACTION")
    print("=" * 60)

    print(f"Samples required : {len(required_ids)}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Device            : {device}")
    print(f"Model             : {MODEL_NAME}")

    dataset = load_dataset(
        DATASET,
        split="train",
        streaming=True,
    )

    # Explicitly decode audio.
    dataset = dataset.cast_column(
        "audio",
        Audio(decode=True),
    )

    # --------------------------------------------------------
    # Output dictionaries
    # --------------------------------------------------------

    mean_embeddings = {}
    max_embeddings = {}
    last_embeddings = {}

    found = set()

    # --------------------------------------------------------
    # Batch buffers
    # --------------------------------------------------------

    audio_batch = []
    id_batch = []

    start = time.time()

    def process_batch():
        if not audio_batch:
            return

        # WhisperProcessor must produce the standard
        # Whisper mel length of 3000.
        inputs = processor(
            audio_batch,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        input_features = inputs.input_features.to(
            device,
            non_blocking=True,
        )

        with torch.inference_mode():

            hidden = model.encoder(
                input_features
            ).last_hidden_state

            # hidden:
            # [batch, time, 384]

            mean_pool = hidden.mean(dim=1)

            max_pool = hidden.max(dim=1).values

            last_pool = hidden[:, -1, :]

        mean_pool = mean_pool.cpu().numpy()
        max_pool = max_pool.cpu().numpy()
        last_pool = last_pool.cpu().numpy()

        for i, sample_id in enumerate(id_batch):

            mean_embeddings[sample_id] = (
                mean_pool[i].astype(np.float32)
            )

            max_embeddings[sample_id] = (
                max_pool[i].astype(np.float32)
            )

            last_embeddings[sample_id] = (
                last_pool[i].astype(np.float32)
            )

            found.add(sample_id)

        audio_batch.clear()
        id_batch.clear()

    # --------------------------------------------------------
    # SINGLE HF DATASET PASS
    # --------------------------------------------------------

    progress = tqdm(
        total=len(required_ids),
        desc="Whisper extraction",
        unit="audio",
    )

    for sample in dataset:

        sample_id = str(sample["id"])

        if sample_id not in required_ids:
            continue

        if sample_id in found:
            continue

        try:

            audio = sample["audio"]

            waveform = np.asarray(
                audio["array"],
                dtype=np.float32,
            )

            sample_rate = int(
                audio["sampling_rate"]
            )

            # Stereo → mono
            if waveform.ndim == 2:

                # Handle either [channels, time]
                # or [time, channels].
                if waveform.shape[0] <= 4:
                    waveform = waveform.mean(axis=0)
                else:
                    waveform = waveform.mean(axis=1)

            # Whisper requires 16 kHz.
            if sample_rate != 16000:

                import librosa

                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=16000,
                ).astype(np.float32)

            audio_batch.append(waveform)
            id_batch.append(sample_id)

            if len(audio_batch) >= BATCH_SIZE:

                process_batch()

                progress.update(
                    len(id_batch)
                    if False
                    else 0
                )

                # Since process_batch clears id_batch,
                # update based on found count.
                progress.n = len(found)
                progress.refresh()

            if len(found) >= len(required_ids):
                break

        except Exception as exc:

            print(
                f"\nSkipping {sample_id}: {exc}"
            )

    # Process final partial batch.
    if audio_batch:
        before = len(found)
        process_batch()
        progress.n = len(found)
        progress.refresh()

    progress.close()

    # --------------------------------------------------------
    # Check missing
    # --------------------------------------------------------

    missing = (
        required_ids
        - found
    )

    if missing:

        print(
            f"\nWARNING: {len(missing)} samples "
            f"were not found."
        )

        print(
            "Examples:",
            list(missing)[:10],
        )

    else:

        print(
            "\nAll requested samples extracted successfully."
        )

    elapsed = time.time() - start

    print(
        f"Extraction time: {elapsed:.2f} seconds"
    )

    return (
        mean_embeddings,
        max_embeddings,
        last_embeddings,
    )


def build_arrays(
    manifest,
    embedding_dict,
):
    """
    Convert dictionary of embeddings into
    X/y arrays following manifest order.
    """

    X = []
    y = []
    metadata = []

    for _, row in manifest.iterrows():

        sample_id = str(row["id"])

        if sample_id not in embedding_dict:
            continue

        X.append(
            embedding_dict[sample_id]
        )

        y.append(
            int(row["label"])
        )

        metadata.append(
            {
                "id": sample_id,
                "label": int(row["label"]),
                "language": row.get("language"),
                "midfiller": row.get("midfiller"),
                "endfiller": row.get("endfiller"),
                "synthetic": row.get("synthetic"),
            }
        )

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        metadata,
    )


def save_pooling(
    output,
    pooling_name,
    train_manifest,
    val_manifest,
    embedding_dict,
):

    X_train, y_train, train_metadata = build_arrays(
        train_manifest,
        embedding_dict,
    )

    X_val, y_val, val_metadata = build_arrays(
        val_manifest,
        embedding_dict,
    )

    pool_dir = output / pooling_name

    pool_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        pool_dir / "X_train.npy",
        X_train,
    )

    np.save(
        pool_dir / "y_train.npy",
        y_train,
    )

    np.save(
        pool_dir / "X_val.npy",
        X_val,
    )

    np.save(
        pool_dir / "y_val.npy",
        y_val,
    )

    (pool_dir / "train_metadata.json").write_text(
        json.dumps(
            train_metadata,
            indent=2,
        )
    )

    (pool_dir / "val_metadata.json").write_text(
        json.dumps(
            val_metadata,
            indent=2,
        )
    )

    print(
        f"\n{pooling_name.upper()} POOLING"
    )

    print(
        "  Train:",
        X_train.shape,
    )

    print(
        "  Val:",
        X_val.shape,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-manifest",
        default="data/manifests/train.csv",
    )

    parser.add_argument(
        "--val-manifest",
        default="data/manifests/val.csv",
    )

    parser.add_argument(
        "--output",
        default="data/whisper_scaling",
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    output = Path(args.output)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 60
    )

    print(
        "FAST WHISPER FEATURE EXTRACTION"
    )

    print(
        "=" * 60
    )

    print(
        "Device:",
        device,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

    train_manifest = pd.read_csv(
        args.train_manifest
    )

    val_manifest = pd.read_csv(
        args.val_manifest
    )

    combined_manifest = pd.concat(
        [
            train_manifest,
            val_manifest,
        ],
        ignore_index=True,
    )

    required_ids = {
        str(x)
        for x in combined_manifest["id"]
    }

    print(
        f"Train samples: {len(train_manifest)}"
    )

    print(
        f"Validation samples: {len(val_manifest)}"
    )

    print(
        f"Total samples: {len(required_ids)}"
    )

    # --------------------------------------------------------
    # Load Whisper ONCE
    # --------------------------------------------------------

    print(
        "\nLoading Whisper..."
    )

    processor = (
        WhisperProcessor
        .from_pretrained(args.model)
    )

    model = (
        WhisperModel
        .from_pretrained(args.model)
        .to(device)
        .eval()
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # ONE PASS THROUGH HF DATASET
    # --------------------------------------------------------

    (
        mean_embeddings,
        max_embeddings,
        last_embeddings,
    ) = extract_all(
        combined_manifest,
        required_ids,
        processor,
        model,
        device,
    )

    # --------------------------------------------------------
    # Save all three pooling strategies
    # --------------------------------------------------------

    save_pooling(
        output,
        "mean",
        train_manifest,
        val_manifest,
        mean_embeddings,
    )

    save_pooling(
        output,
        "max",
        train_manifest,
        val_manifest,
        max_embeddings,
    )

    save_pooling(
        output,
        "last",
        train_manifest,
        val_manifest,
        last_embeddings,
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "EXTRACTION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Output: {output}"
    )

    print(
        "\nAttention pooling is NOT extracted here."
    )

    print(
        "It should be learned directly from the "
        "cached Whisper temporal states."
    )


if __name__ == "__main__":
    main()