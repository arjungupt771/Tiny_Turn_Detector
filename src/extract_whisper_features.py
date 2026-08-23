from pathlib import Path
import json

import numpy as np
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperModel
from tqdm import tqdm


DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"

OUTPUT_DIR = Path("data/whisper_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SIZE = 1600
VAL_SIZE = 400

SEED = 42

MODEL_NAME = "openai/whisper-tiny"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():

    print(f"Loading {MODEL_NAME}...")
    print(f"Device: {DEVICE}")

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME
    )

    model = WhisperModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(DEVICE)

    # Freeze Whisper
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return processor, model




def extract_embedding(
    waveform,
    sample_rate,
    processor,
    model,
):
    """
    Convert audio into a Whisper encoder representation.

    We mean-pool across the encoder time dimension,
    producing one fixed-size vector per audio sample.
    """

    # TorchCodec returns [channels, samples]
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)

    waveform = waveform.astype(np.float32)

    # Whisper expects 16 kHz
    if sample_rate != 16000:

        import librosa

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=16000,
        )

        sample_rate = 16000

    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(
        DEVICE
    )

    with torch.inference_mode():

        outputs = model.encoder(
            input_features
        )

        hidden_states = outputs.last_hidden_state

        # [batch, time, hidden]
        embedding = hidden_states.mean(
            dim=1
        )

    return embedding.squeeze(0).cpu().numpy()


def main():

    processor, model = load_model()

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=10_000,
    )

    print("Dataset loaded.")

    total_samples = TRAIN_SIZE + VAL_SIZE

    X_train = []
    y_train = []
    metadata_train = []

    X_val = []
    y_val = []
    metadata_val = []

    for i, sample in enumerate(
        tqdm(
            dataset,
            total=total_samples,
        )
    ):

        if i >= total_samples:
            break

        audio = sample["audio"]

        decoded = audio.get_all_samples()

        waveform = decoded.data.numpy()
        sample_rate = decoded.sample_rate

        embedding = extract_embedding(
            waveform,
            sample_rate,
            processor,
            model,
        )

        label = int(
            sample["endpoint_bool"]
        )

        metadata = {
            "language": sample["language"],
            "midfiller": sample["midfiller"],
            "endfiller": sample["endfiller"],
            "synthetic": sample["synthetic"],
            "dataset": sample["dataset"],
        }

        if i < TRAIN_SIZE:

            X_train.append(embedding)
            y_train.append(label)
            metadata_train.append(metadata)

        else:

            X_val.append(embedding)
            y_val.append(label)
            metadata_val.append(metadata)

    X_train = np.stack(X_train)
    y_train = np.asarray(y_train)

    X_val = np.stack(X_val)
    y_val = np.asarray(y_val)

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        OUTPUT_DIR / "X_val.npy",
        X_val,
    )

    np.save(
        OUTPUT_DIR / "y_val.npy",
        y_val,
    )

    with open(
        OUTPUT_DIR / "train_metadata.json",
        "w",
    ) as f:

        json.dump(
            metadata_train,
            f,
            indent=2,
        )

    with open(
        OUTPUT_DIR / "val_metadata.json",
        "w",
    ) as f:

        json.dump(
            metadata_val,
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("WHISPER FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Train shape: {X_train.shape}"
    )

    print(
        f"Validation shape: {X_val.shape}"
    )

    print(
        f"Train labels: {np.bincount(y_train)}"
    )

    print(
        f"Validation labels: {np.bincount(y_val)}"
    )


if __name__ == "__main__":
    main()