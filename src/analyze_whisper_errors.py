import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


FEATURE_DIR = Path("data/whisper_features")


def evaluate_group(
    name,
    indices,
    y_true,
    y_pred,
):

    if len(indices) == 0:
        return

    true = y_true[indices]
    pred = y_pred[indices]

    f1 = f1_score(
        true,
        pred,
        zero_division=0,
    )

    accuracy = np.mean(
        true == pred
    )

    print(
        f"{name:25s} "
        f"N={len(indices):4d} "
        f"Accuracy={accuracy:.3f} "
        f"F1={f1:.3f}"
    )


def main():

    X_train = np.load(
        FEATURE_DIR / "X_train.npy"
    )

    y_train = np.load(
        FEATURE_DIR / "y_train.npy"
    )

    X_val = np.load(
        FEATURE_DIR / "X_val.npy"
    )

    y_val = np.load(
        FEATURE_DIR / "y_val.npy"
    )

    with open(
        FEATURE_DIR / "val_metadata.json"
    ) as f:

        metadata = json.load(f)

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_val
    )

    print("\n" + "=" * 70)
    print("WHISPER ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------
    # Overall
    # --------------------------------------------------

    evaluate_group(
        "Overall",
        np.arange(len(y_val)),
        y_val,
        y_pred,
    )

    # --------------------------------------------------
    # Languages
    # --------------------------------------------------

    languages = sorted(
        set(
            item["language"]
            for item in metadata
        )
    )

    print("\n--- Language ---")

    for language in languages:

        indices = np.array(
            [
                i
                for i, item in enumerate(metadata)
                if item["language"] == language
            ]
        )

        evaluate_group(
            language,
            indices,
            y_val,
            y_pred,
        )

    # --------------------------------------------------
    # Midfiller
    # --------------------------------------------------

    print("\n--- Midfiller ---")

    for value in [
        True,
        False,
        None,
    ]:

        indices = np.array(
            [
                i
                for i, item in enumerate(metadata)
                if item["midfiller"] is value
            ]
        )

        evaluate_group(
            f"midfiller={value}",
            indices,
            y_val,
            y_pred,
        )

    # --------------------------------------------------
    # Endfiller
    # --------------------------------------------------

    print("\n--- Endfiller ---")

    for value in [
        True,
        False,
        None,
    ]:

        indices = np.array(
            [
                i
                for i, item in enumerate(metadata)
                if item["endfiller"] is value
            ]
        )

        evaluate_group(
            f"endfiller={value}",
            indices,
            y_val,
            y_pred,
        )

    # --------------------------------------------------
    # Synthetic
    # --------------------------------------------------

    print("\n--- Synthetic ---")

    for value in [
        True,
        False,
    ]:

        indices = np.array(
            [
                i
                for i, item in enumerate(metadata)
                if item["synthetic"] == value
            ]
        )

        evaluate_group(
            f"synthetic={value}",
            indices,
            y_val,
            y_pred,
        )


    # Endfiller Confusion Matrix

    print("\n --- Endfiller confusion ---")

    for value in [True, False, None]:

        indices = np.array(
            [
                i
                for i, item in enumerate(metadata)
                if item["endfiller"] is value
            ]
        )

        if len(indices) ==0:
            continue

        true = y_val[indices]
        pred = y_pred[indices]

        tp = np.sum(
            (true ==1) & (pred==1)
        )

        tn = np.sum(
            (true==0) & (pred==0)
        )

        fp = np.sum(
            (true==0) & (pred==1)
        )

        fn = np.sum(
            (true==1) & (pred==0)
        )

        print(f"\n endfiller={value}")
        print(f"N={len(indices)}")
        print(f"TP={tp}")
        print(f"TN={tn}")
        print(f"FP={fp}")
        print(f"FN={fn}")



    
    # --------------------------------------------------
    # Hinglish proxy
    # --------------------------------------------------
    
    print("\n--- Indian language subset ---")

    indian_languages = {
        "hin",
        "eng",
        "ben",
        "mar",
    }

    indices = np.array(
        [
            i
            for i, item in enumerate(metadata)
            if item["language"]
            in indian_languages
        ]
    )

    evaluate_group(
        "Indian languages",
        indices,
        y_val,
        y_pred,
    )




if __name__ == "__main__":
    main()