from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset


DATASET_TRAIN = "pipecat-ai/smart-turn-data-v3.2-train"
DATASET_TEST = "pipecat-ai/smart-turn-data-v3.2-test"

SEED = 42

FIELDS = [
    "id",
    "split",
    "label",
    "language",
    "midfiller",
    "endfiller",
    "synthetic",
    "dataset",
    "duration_seconds",
    "sample_rate",
    "audio_ok",
]


def collect_samples(
    dataset_name: str,
    limit: int | None,
    seed: int,
    metadata_only: bool,
):
    """
    Collect unique samples from a Hugging Face streaming dataset.

    The official test dataset is NEVER mixed with training data.
    """

    ds = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )

    ds = ds.shuffle(
        seed=seed,
        buffer_size=10_000,
    )

    if metadata_only:
        ds = ds.cast_column(
            "audio",
            Audio(decode=False),
        )

    rows = []
    seen_ids = set()

    for sample in ds:

        if limit is not None and len(rows) >= limit:
            break

        sample_id = str(sample["id"])

        # Duplicate protection
        if sample_id in seen_ids:
            continue

        # Label validation
        label = int(sample["endpoint_bool"])

        if label not in (0, 1):
            print(
                f"Skipping invalid label for {sample_id}: "
                f"{label}"
            )
            continue

        duration = None
        sample_rate = None
        audio_ok = False

        if not metadata_only:

            try:
                audio = sample["audio"].get_all_samples()

                data = audio.data
                sample_rate = int(audio.sample_rate)

                if data.numel() == 0:
                    raise ValueError("empty audio")

                if sample_rate <= 0:
                    raise ValueError(
                        f"invalid sample rate: {sample_rate}"
                    )

                duration = (
                    data.shape[-1] / sample_rate
                )

                audio_ok = True

            except Exception as exc:

                print(
                    f"Skipping invalid audio "
                    f"{sample_id}: {exc}"
                )

                continue

        rows.append(
            {
                "id": sample_id,
                "split": "",
                "label": label,
                "language": sample.get("language"),
                "midfiller": sample.get("midfiller"),
                "endfiller": sample.get("endfiller"),
                "synthetic": sample.get("synthetic"),
                "dataset": sample.get("dataset"),
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "audio_ok": audio_ok,
            }
        )

        seen_ids.add(sample_id)

    return rows


def stratified_split(
    rows,
    train_size,
    val_size,
    seed,
):
    """
    Deterministic stratified train/validation split.
    """

    required = train_size + val_size

    if len(rows) < required:
        raise ValueError(
            f"Need {required} samples, "
            f"but only collected {len(rows)}"
        )

    rng = np.random.default_rng(seed)

    class_0 = [
        row for row in rows
        if row["label"] == 0
    ]

    class_1 = [
        row for row in rows
        if row["label"] == 1
    ]

    rng.shuffle(class_0)
    rng.shuffle(class_1)

    total = len(class_0) + len(class_1)

    validation_ratio = val_size / required

    val_0 = round(
        len(class_0) * validation_ratio
    )

    val_1 = round(
        len(class_1) * validation_ratio
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

    # Correct rounding differences
    while len(val) > val_size:
        train.append(val.pop())

    while len(val) < val_size:
        val.append(train.pop())

    train = train[:train_size]

    if len(train) != train_size:
        raise RuntimeError(
            "Failed to construct train split"
        )

    if len(val) != val_size:
        raise RuntimeError(
            "Failed to construct validation split"
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
        row["id"] for row in train
    }

    val_ids = {
        row["id"] for row in val
    }

    test_ids = {
        row["id"] for row in test
    }

    assert train_ids.isdisjoint(
        val_ids
    ), "Train/validation leakage"

    assert train_ids.isdisjoint(
        test_ids
    ), "Train/test leakage"

    assert val_ids.isdisjoint(
        test_ids
    ), "Validation/test leakage"

    print("✓ No ID leakage detected")


def distribution(rows):

    def count(field):
        return {
            str(key): int(value)
            for key, value in Counter(
                row.get(field)
                for row in rows
            ).items()
        }

    durations = [
        float(row["duration_seconds"])
        for row in rows
        if row["duration_seconds"]
        is not None
    ]

    return {
        "samples": len(rows),

        "labels": count("label"),

        "languages": count("language"),

        "midfiller": count("midfiller"),

        "endfiller": count("endfiller"),

        "synthetic": count("synthetic"),

        "duration": {
            "min": min(durations)
            if durations else None,

            "max": max(durations)
            if durations else None,

            "mean": float(
                np.mean(durations)
            )
            if durations else None,
        },
    }


def save_csv(path, rows):

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
        help="-1 = full official test set",
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

    parser.add_argument(
        "--metadata-only",
        action="store_true",
    )

    args = parser.parse_args()

    output = Path(args.output)

    # ---------------------------
    # TRAIN + VALIDATION
    # ---------------------------

    train_val = collect_samples(
        DATASET_TRAIN,
        args.train_size + args.val_size,
        args.seed,
        args.metadata_only,
    )

    train, val = stratified_split(
        train_val,
        args.train_size,
        args.val_size,
        args.seed,
    )

    # ---------------------------
    # OFFICIAL TEST
    # ---------------------------

    test_limit = (
        None
        if args.test_size < 0
        else args.test_size
    )

    test = collect_samples(
        DATASET_TEST,
        test_limit,
        args.seed,
        args.metadata_only,
    )

    for row in test:
        row["split"] = "test"

    # ---------------------------
    # LEAKAGE CHECK
    # ---------------------------

    verify_no_leakage(
        train,
        val,
        test,
    )

    # ---------------------------
    # SAVE MANIFESTS
    # ---------------------------

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

        "train": distribution(train),

        "validation": distribution(val),

        "test": distribution(test),

        "official_test_dataset": DATASET_TEST,
    }

    with (
        output / "statistics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Official test data must remain "
        "untouched until final model selection."
    )


if __name__ == "__main__":
    main()