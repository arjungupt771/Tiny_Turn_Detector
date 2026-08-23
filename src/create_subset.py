from pathlib import Path

import json
import soundfile as sf
from datasets import load_dataset


DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"

OUTPUT_DIR = Path("data/subset")

TRAIN_SIZE = 8_000
VAL_SIZE = 2_000

SEED = 42


def save_sample(sample, output_path):

    audio = sample["audio"]

    decoded = audio.get_all_samples()

    waveform = decoded.data.numpy()
    sample_rate = decoded.sample_rate

    if waveform.ndim == 2:
        waveform = waveform.T

    sf.write(
        output_path,
        waveform,
        sample_rate,
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Shuffle streaming dataset.
    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=10_000,
    )

    print("Dataset ready.")

    train_metadata = []
    val_metadata = []

    total_needed = TRAIN_SIZE + VAL_SIZE

    for i, sample in enumerate(dataset):

        if i >= total_needed:
            break

        if i % 500 == 0:
            print(
                f"Processed {i}/{total_needed}"
            )

        split = (
            "train"
            if i < TRAIN_SIZE
            else "val"
        )

        if split == "train":
            index = i
            output_dir = train_dir
            metadata = train_metadata

        else:
            index = i - TRAIN_SIZE
            output_dir = val_dir
            metadata = val_metadata

        filename = f"{index:06d}.wav"

        output_path = output_dir / filename

        save_sample(
            sample,
            output_path,
        )

        metadata.append(
            {
                "file": filename,
                "label": int(
                    sample["endpoint_bool"]
                ),
                "language": sample["language"],
                "midfiller": sample["midfiller"],
                "endfiller": sample["endfiller"],
                "synthetic": sample["synthetic"],
                "dataset": sample["dataset"],
            }
        )

    with open(
        train_dir / "metadata.json",
        "w",
    ) as f:

        json.dump(
            train_metadata,
            f,
            indent=2,
        )

    with open(
        val_dir / "metadata.json",
        "w",
    ) as f:

        json.dump(
            val_metadata,
            f,
            indent=2,
        )

    print("\nSubset creation complete.")

    print(
        f"Train samples: {len(train_metadata)}"
    )

    print(
        f"Validation samples: {len(val_metadata)}"
    )


if __name__ == "__main__":
    main()