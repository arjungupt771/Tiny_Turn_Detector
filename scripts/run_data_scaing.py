import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(
    model,
    X,
    y,
):

    prediction = model.predict(X)

    return {
        "accuracy": accuracy_score(
            y,
            prediction,
        ),

        "precision": precision_score(
            y,
            prediction,
            zero_division=0,
        ),

        "recall": recall_score(
            y,
            prediction,
            zero_division=0,
        ),

        "f1": f1_score(
            y,
            prediction,
            zero_division=0,
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default="data/whisper_scaling",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[
            2000,
            10000,
            25000,
            50000,
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/"
            "data_scaling/"
            "results.csv"
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.features
    )

    X_train = np.load(
        root / "X_train.npy"
    )

    y_train = np.load(
        root / "y_train.npy"
    )

    X_val = np.load(
        root / "X_val.npy"
    )

    y_val = np.load(
        root / "y_val.npy"
    )

    results = []

    for size in args.sizes:

        if size > len(X_train):

            raise ValueError(
                f"Requested {size}, "
                f"but only {len(X_train)} "
                f"training samples exist."
            )

        rng = np.random.default_rng(
            args.seed
        )

        indices = rng.choice(
            len(X_train),
            size=size,
            replace=False,
        )

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    LogisticRegression(
                        C=0.1,
                        max_iter=2000,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

        start = time.perf_counter()

        model.fit(
            X_train[indices],
            y_train[indices],
        )

        training_time = (
            time.perf_counter()
            - start
        )

        metrics = evaluate(
            model,
            X_val,
            y_val,
        )

        result = {
            "train_samples": size,
            "training_seconds": training_time,
            **metrics,
        }

        results.append(result)

        print(result)

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        results
    ).to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()