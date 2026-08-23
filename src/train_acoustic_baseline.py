from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURE_DIR = Path("data/features")


def main():

    print("Loading features...")

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

    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"Validation: {X_val.shape}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

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

    print("\nTraining acoustic baseline...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = model.predict(X_val)

    # Probability of END
    y_prob = model.predict_proba(X_val)[:, 1]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred,
    )

    precision = precision_score(
        y_val,
        y_pred,
    )

    recall = recall_score(
        y_val,
        y_pred,
    )

    f1 = f1_score(
        y_val,
        y_pred,
    )

    print("\n" + "=" * 60)
    print("ACOUSTIC BASELINE RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print("\n--- Classification Report ---")

    print(
        classification_report(
            y_val,
            y_pred,
            target_names=[
                "CONTINUE",
                "END",
            ],
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_val,
        y_pred,
    )

    print("--- Confusion Matrix ---")

    print(
        "              Predicted"
    )

    print(
        "              CONT  END"
    )

    print(
        f"Actual CONT   {cm[0, 0]:4d} {cm[0, 1]:4d}"
    )

    print(
        f"Actual END    {cm[1, 0]:4d} {cm[1, 1]:4d}"
    )

    # --------------------------------------------------
    # Probability examples
    # --------------------------------------------------

    print("\n--- Example END probabilities ---")

    for i in range(
        min(10, len(y_prob))
    ):

        print(
            f"True={y_val[i]} "
            f"END_probability={y_prob[i]:.4f}"
        )


if __name__ == "__main__":
    main()