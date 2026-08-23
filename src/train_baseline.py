import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
import joblib


# ============================================================
# Configuration
# ============================================================

FEATURE_DIR = "data/controlled_features/acoustic"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Load features
# ============================================================

print("=" * 60)
print("LOADING CONTROLLED ACOUSTIC FEATURES")
print("=" * 60)

X_train = np.load(f"{FEATURE_DIR}/X_train.npy")
X_val = np.load(f"{FEATURE_DIR}/X_val.npy")

y_train = np.load(f"{FEATURE_DIR}/y_train.npy")
y_val = np.load(f"{FEATURE_DIR}/y_val.npy")

print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val:   {y_val.shape}")


# ============================================================
# Build model
# ============================================================

print("\n" + "=" * 60)
print("BUILDING BASELINE MODEL")
print("=" * 60)

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])


# ============================================================
# Train
# ============================================================

print("\nTraining...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_val)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=["NOT_END", "END"],
        zero_division=0,
    )
)


# ============================================================
# Save model
# ============================================================

model_path = f"{MODEL_DIR}/turn_detector_logistic.joblib"

joblib.dump(model, model_path)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(model_path)