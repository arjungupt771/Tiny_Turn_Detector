import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from collections import Counter


DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"

OUTPUT_DIR = Path("data/manifest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "experiment_2000.json"

TOTAL_SAMPLES = 2000
SEED = 42


def main():

    print("Loading dataset in streaming mode...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Shuffle with a fixed seed so that the experiment
    # can be reproduced.
    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=10_000,
    )

    print("Dataset loaded.")
    print(f"Collecting {TOTAL_SAMPLES:,} samples...")

    manifest = []

    for i, sample in enumerate(
        tqdm(dataset, total=TOTAL_SAMPLES)
    ):

        if i >= TOTAL_SAMPLES:
            break

        item = {
            "id": sample["id"],
            "label": int(sample["endpoint_bool"]),

            "language": sample["language"],
            "midfiller": sample["midfiller"],
            "endfiller": sample["endfiller"],
            "synthetic": sample["synthetic"],
            "dataset": sample["dataset"],
        }

        manifest.append(item)
        train_size=1600
        for i, item in enumerate(manifest):
            if i < train_size:
                item["split"]="train"
            else:
                item["split"]="validation"

    with open(
        OUTPUT_FILE,
        "w",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT MANIFEST CREATED")
    print("=" * 60)

    print(f"Samples: {len(manifest):,}")
    print(f"Output : {OUTPUT_FILE}")

    end_count = sum(
        item["label"] == 1
        for item in manifest
    )

    continue_count = sum(
        item["label"] == 0
        for item in manifest
    )

    print("\n--- Split distribution---")
    split_counts=Counter(
        item["split"]
        for item in manifest
    )

    for split, count in split_counts.items():
        print(f"{split:12s}: {count:,}")

    print("\n--- Language distribution---")

    language_counts = Counter(
        item["language"]
        for item in manifest
    )

    for language, count in language_counts.most_common():
        print(f"{language:10s}: {count:,}")


    print("\n---Midfiller---")
    midfiller_counts = Counter(
        item["midfiller"]
        for item in manifest
    )

    for value, count in midfiller_counts.items():
        print(f"{str(value):10s}: {count:,}")

    print("\n--- Endfiller ---")

    endfiller_counts = Counter(
        item["endfiller"]
        for item in manifest
    )

    for value, count in endfiller_counts.items():
        print(f"{str(value):10s}: {count:,}")

    print("\n--- Synthetic ---")

    synthetic_counts = Counter(
        item['synthetic']
        for item in manifest
    )

    for value, count in synthetic_counts.items():
        print(f"{str(value):10s}: {count:,}")



    

    print("\nLabels:")

    print(
        f"CONTINUE: {continue_count:,}"
    )

    print(
        f"END     : {end_count:,}"
    )

    print("\nFirst 5 samples:")

    for item in manifest[:5]:
        print(item)


if __name__ == "__main__":
    main()