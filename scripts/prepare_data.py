from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset


DATASET_TRAIN = "pipecat-ai/smart-turn-data-v3.2-train"
DATASET_TEST = "pipecat-ai/smart-turn-data-v3.2-test"

FIELDS = [
    "id",
    "split",
    "label",
    "language",
    "midfiller",
    "endfiller",
    "synthetic",
    "dataset",
]


def load_metadata(
    dataset_name: str,
    limit: int,
    seed: int,
):
    """
    Fast metadata-only collection.

    IMPORTANT:
    We do NOT decode audio here.
    Audio validation happens later during
    feature extraction.
    """

    print()
    print("=" * 60)
    print(f"Loading: {dataset_name}")
    print("=" * 60)

    print("Opening streaming dataset...")

    ds = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )

    print("Dataset opened.")

    # Small shuffle buffer.
    # Do not use 10,000 here.
    ds = ds.shuffle(
        seed=seed,
        buffer_size=max(
            1000,
            min(limit * 2, 5000),
        ),
    )

    rows = []
    seen_ids = set()

    print(
        f"Collecting {limit} samples..."
    )

    for sample in ds:

        sample_id = str(
            sample.get("id")
        )

        if not sample_id:
            continue

        if sample_id in seen_ids:
            continue

        # Some datasets expose the label
        # under endpoint_bool.
        if "endpoint_bool" not in sample:
            raise KeyError(
                "endpoint_bool not found "
                "in dataset sample"
            )

        label = int(
            sample["endpoint_bool"]
        )

        if label not in (0, 1):
            continue

        rows.append(
            {
                "id": sample_id,
                "split": "",
                "label": label,
                "language": sample.get(
                    "language"
                ),
                "midfiller": sample.get(
                    "midfiller"
                ),
                "endfiller": sample.get(
                    "endfiller"
                ),
                "synthetic": sample.get(
                    "synthetic"
                ),
                "dataset": sample.get(
                    "dataset"
                ),
            }
        )

        seen_ids.add(sample_id)

        if len(rows) >= limit:
            break

        if len(rows) % 250 == 0:
            print(
                f"Collected "
                f"{len(rows)}/{limit}"
            )

    print(
        f"Finished: {len(rows)} samples"
    )

    if len(rows) < limit:
        raise RuntimeError(
            f"Requested {limit} samples "
            f"but only collected "
            f"{len(rows)}."
        )

    return rows


def stratified_split(
    rows,
    train_size,
    val_size,
    seed,
):
    """
    Create deterministic stratified
    train/validation split.
    """

    required = (
        train_size
        + val_size
    )

    if len(rows) < required:
        raise ValueError(
            f"Need {required} samples, "
            f"got {len(rows)}"
        )

    rng = np.random.default_rng(
        seed
    )

    class_0 = [
        row
        for row in rows
        if row["label"] == 0
    ]

    class_1 = [
        row
        for row in rows
        if row["label"] == 1
    ]

    print()
    print("Class distribution:")
    print(
        f"Class 0: {len(class_0)}"
    )
    print(
        f"Class 1: {len(class_1)}"
    )

    rng.shuffle(class_0)
    rng.shuffle(class_1)

    # Exact validation ratio
    val_ratio = (
        val_size / required
    )

    val_0 = round(
        len(class_0)
        * val_ratio
    )

    val_1 = round(
        len(class_1)
        * val_ratio
    )

    val = (
        class_0[:val_0]
        + class_1[:val_1]
    )

    train = (
        class_0[val_0:]
        + class_1[val_1:]
    )

    rng.shuffle(train)
    rng.shuffle(val)

    # Correct rounding.
    while len(val) > val_size:
        train.append(
            val.pop()
        )

    while len(val) < val_size:
        val.append(
            train.pop()
        )

    # Make exact sizes.
    if len(train) > train_size:
        train = train[:train_size]

    if len(train) < train_size:
        raise RuntimeError(
            "Could not construct "
            "requested train split."
        )

    if len(val) != val_size:
        raise RuntimeError(
            "Could not construct "
            "requested validation split."
        )

    for row in train:
        row["split"] = "train"

    for row in val:
        row["split"] = "validation"

    return train, val


def verify_no_leakage(
    train,
    val,
    test,
):

    train_ids = {
        row["id"]
        for row in train
    }

    val_ids = {
        row["id"]
        for row in val
    }

    test_ids = {
        row["id"]
        for row in test
    }

    train_val = (
        train_ids & val_ids
    )

    train_test = (
        train_ids & test_ids
    )

    val_test = (
        val_ids & test_ids
    )

    if train_val:
        raise RuntimeError(
            f"Train/validation leakage: "
            f"{len(train_val)} IDs"
        )

    if train_test:
        raise RuntimeError(
            f"Train/test leakage: "
            f"{len(train_test)} IDs"
        )

    if val_test:
        raise RuntimeError(
            f"Validation/test leakage: "
            f"{len(val_test)} IDs"
        )

    print()
    print(
        "✓ No ID leakage detected"
    )


def distribution(rows):

    def count(field):

        return {
            str(key): int(value)
            for key, value in Counter(
                row.get(field)
                for row in rows
            ).items()
        }

    return {
        "samples": len(rows),

        "labels": count("label"),

        "languages": count(
            "language"
        ),

        "midfiller": count(
            "midfiller"
        ),

        "endfiller": count(
            "endfiller"
        ),

        "synthetic": count(
            "synthetic"
        ),
    }


def save_csv(
    path: Path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Saved: {path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-size",
        type=int,
        default=1600,
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default="data/manifests",
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    # ==================================================
    # 1. TRAIN + VALIDATION
    # ==================================================

    train_val_size = (
        args.train_size
        + args.val_size
    )

    print()
    print(
        "STEP 1/4: TRAIN + VALIDATION"
    )

    train_val = load_metadata(
        DATASET_TRAIN,
        train_val_size,
        args.seed,
    )

    train, val = stratified_split(
        train_val,
        args.train_size,
        args.val_size,
        args.seed,
    )

    # ==================================================
    # 2. OFFICIAL TEST
    # ==================================================

    print()
    print(
        "STEP 2/4: OFFICIAL TEST"
    )

    test = load_metadata(
        DATASET_TEST,
        args.test_size,
        args.seed,
    )

    for row in test:
        row["split"] = "test"

    # ==================================================
    # 3. LEAKAGE
    # ==================================================

    print()
    print(
        "STEP 3/4: LEAKAGE CHECK"
    )

    verify_no_leakage(
        train,
        val,
        test,
    )

    # ==================================================
    # 4. SAVE
    # ==================================================

    print()
    print(
        "STEP 4/4: SAVING"
    )

    save_csv(
        output / "train.csv",
        train,
    )

    save_csv(
        output / "val.csv",
        val,
    )

    save_csv(
        output / "test.csv",
        test,
    )

    report = {
        "seed": args.seed,

        "train": distribution(
            train
        ),

        "validation": distribution(
            val
        ),

        "test": distribution(
            test
        ),

        "datasets": {
            "train_validation":
                DATASET_TRAIN,

            "test":
                DATASET_TEST,
        },

        "audio_validation": (
            "Performed during "
            "feature extraction, "
            "not manifest creation."
        ),
    }

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output
        / "statistics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Train      : {len(train)}"
    )

    print(
        f"Validation : {len(val)}"
    )

    print(
        f"Test       : {len(test)}"
    )

    print()
    print(
        "Official test data has "
        "NOT been used for training "
        "or validation."
    )


if __name__ == "__main__":
    main()