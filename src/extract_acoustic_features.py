from pathlib import Path
import json

import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"

OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SIZE = 8_000
VAL_SIZE = 2_000

SEED = 42


def extract_features(waveform, sample_rate):
    """
    Extract compact acoustic features from one audio sample.

    Returns a fixed-length feature vector.
    """

    # Ensure mono
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)

    waveform = waveform.astype(np.float32)

    features = []

    # --------------------------------------------------
    # 1. RMS Energy
    # --------------------------------------------------

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=512,
        hop_length=160,
    )

    features.extend([
        rms.mean(),
        rms.std(),
        rms.min(),
        rms.max(),
    ])

    # --------------------------------------------------
    # 2. Zero Crossing Rate
    # --------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        waveform,
        frame_length=512,
        hop_length=160,
    )

    features.extend([
        zcr.mean(),
        zcr.std(),
        zcr.min(),
        zcr.max(),
    ])

    # --------------------------------------------------
    # 3. Spectral Centroid
    # --------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=waveform,
        sr=sample_rate,
        n_fft=512,
        hop_length=160,
    )

    features.extend([
        centroid.mean(),
        centroid.std(),
        centroid.min(),
        centroid.max(),
    ])

    # --------------------------------------------------
    # 4. Spectral Bandwidth
    # --------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=waveform,
        sr=sample_rate,
        n_fft=512,
        hop_length=160,
    )

    features.extend([
        bandwidth.mean(),
        bandwidth.std(),
        bandwidth.min(),
        bandwidth.max(),
    ])

    # --------------------------------------------------
    # 5. Spectral Rolloff
    # --------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=waveform,
        sr=sample_rate,
        n_fft=512,
        hop_length=160,
    )

    features.extend([
        rolloff.mean(),
        rolloff.std(),
        rolloff.min(),
        rolloff.max(),
    ])

    # --------------------------------------------------
    # 6. MFCC
    # --------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=512,
        hop_length=160,
    )

    # Mean + std for each MFCC
    for coefficient in mfcc:

        features.append(
            coefficient.mean()
        )

        features.append(
            coefficient.std()
        )

    # --------------------------------------------------
    # 7. Duration
    # --------------------------------------------------

    duration = len(waveform) / sample_rate

    features.append(duration)

    return np.asarray(
        features,
        dtype=np.float32,
    )


def process_dataset():

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Shuffle the stream.
    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=10_000,
    )

    print("Dataset loaded.")

    total_samples = TRAIN_SIZE + VAL_SIZE

    X_train = []
    y_train = []

    X_val = []
    y_val = []

    metadata_train = []
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

        # Decode audio
        decoded = audio.get_all_samples()

        waveform = decoded.data.numpy()
        sample_rate = decoded.sample_rate

        # Extract features
        features = extract_features(
            waveform,
            sample_rate,
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

        # ------------------------------------------
        # Train
        # ------------------------------------------

        if i < TRAIN_SIZE:

            X_train.append(features)
            y_train.append(label)

            metadata_train.append(metadata)

        # ------------------------------------------
        # Validation
        # ------------------------------------------

        else:

            X_val.append(features)
            y_val.append(label)

            metadata_val.append(metadata)

    # Convert to NumPy
    X_train = np.stack(X_train)
    y_train = np.asarray(y_train)

    X_val = np.stack(X_val)
    y_val = np.asarray(y_val)

    # Save
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
    print("FEATURE EXTRACTION COMPLETE")
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
    process_dataset()