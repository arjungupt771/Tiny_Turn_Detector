import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"

MANIFEST_FILE = Path(
    "data/manifest/experiment_2000.json"
)

OUTPUT_DIR = Path(
    "data/controlled_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def extract_features(waveform, sample_rate):
    """
    Same 47-dimensional acoustic representation
    used in our first acoustic baseline.
    """

    import librosa

    waveform = waveform.astype(
        np.float32
    )

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)

    if sample_rate != 16000:

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=16000,
        )

        sample_rate = 16000

    features = []

    # --------------------------------------------------
    # Basic waveform statistics
    # --------------------------------------------------

    features.append(
        np.mean(waveform)
    )

    features.append(
        np.std(waveform)
    )

    features.append(
        np.min(waveform)
    )

    features.append(
        np.max(waveform)
    )

    features.append(
        np.sqrt(
            np.mean(waveform ** 2)
        )
    )

    # --------------------------------------------------
    # Zero crossing rate
    # --------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        waveform
    )[0]

    features.append(
        np.mean(zcr)
    )

    features.append(
        np.std(zcr)
    )

    # --------------------------------------------------
    # Spectral features
    # --------------------------------------------------

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=waveform,
            sr=sample_rate,
        )[0]
    )

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=waveform,
            sr=sample_rate,
        )[0]
    )

    spectral_rolloff = (
        librosa.feature.spectral_rolloff(
            y=waveform,
            sr=sample_rate,
        )[0]
    )

    spectral_contrast = (
        librosa.feature.spectral_contrast(
            y=waveform,
            sr=sample_rate,
        )
    )

    features.extend(
        [
            np.mean(spectral_centroid),
            np.std(spectral_centroid),

            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),

            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
        ]
    )

    features.extend(
        np.mean(
            spectral_contrast,
            axis=1,
        )
    )

    features.extend(
        np.std(
            spectral_contrast,
            axis=1,
        )
    )

    # --------------------------------------------------
    # MFCC
    # --------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=13,
    )

    features.extend(
        np.mean(
            mfcc,
            axis=1,
        )
    )

    features.extend(
        np.std(
            mfcc,
            axis=1,
        )
    )

    # --------------------------------------------------
    # Duration
    # --------------------------------------------------

    duration = (
        len(waveform) /
        sample_rate
    )

    features.append(
        duration
    )

    return np.asarray(
        features,
        dtype=np.float32,
    )


def main():

    print("Loading manifest...")

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    manifest_by_id = {
        item["id"]: item
        for item in manifest
    }

    print(
        f"Manifest samples: {len(manifest)}"
    )

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    print("Dataset loaded.")
    print("Searching for manifest samples...")

    features_by_id = {}

    for sample in tqdm(dataset):

        sample_id = sample["id"]

        if sample_id not in manifest_by_id:
            continue

        audio = sample["audio"]

        decoded = audio.get_all_samples()

        waveform = decoded.data.numpy()

        sample_rate = decoded.sample_rate

        features = extract_features(
            waveform,
            sample_rate,
        )

        features_by_id[sample_id] = (
            features
        )

        if (
            len(features_by_id)
            == len(manifest)
        ):
            break

    print(
        f"\nFound: {len(features_by_id)} / "
        f"{len(manifest)}"
    )

    missing = [
        item["id"]
        for item in manifest
        if item["id"] not in features_by_id
    ]

    if missing:

        print(
            f"WARNING: {len(missing)} "
            "samples missing."
        )

    # --------------------------------------------------
    # Reconstruct exact manifest order
    # --------------------------------------------------

    X = np.stack(
        [
            features_by_id[item["id"]]
            for item in manifest
            if item["id"] in features_by_id
        ]
    )

    y = np.asarray(
        [
            item["label"]
            for item in manifest
            if item["id"] in features_by_id
        ],
        dtype=np.int64,
    )

    splits = np.asarray(
        [
            item["split"]
            for item in manifest
            if item["id"] in features_by_id
        ]
    )

    train_mask = (
        splits == "train"
    )

    val_mask = (
        splits == "validation"
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    output = (
        OUTPUT_DIR / "acoustic"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output / "X_train.npy",
        X_train,
    )

    np.save(
        output / "y_train.npy",
        y_train,
    )

    np.save(
        output / "X_val.npy",
        X_val,
    )

    np.save(
        output / "y_val.npy",
        y_val,
    )

    print("\n" + "=" * 60)
    print("CONTROLLED ACOUSTIC FEATURES")
    print("=" * 60)

    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"Validation: {X_val.shape}"
    )

    print(
        f"Train labels: "
        f"{np.bincount(y_train)}"
    )

    print(
        f"Validation labels: "
        f"{np.bincount(y_val)}"
    )


if __name__ == "__main__":
    main()