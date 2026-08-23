"""Create plots for data scaling experiment results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Load results
results_csv = Path("experiments/data_scaling/results.csv")

with open(results_csv) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Extract data
n_train = [int(r["n_train"]) for r in rows]
accuracy = [float(r["accuracy"]) for r in rows]
precision = [float(r["precision"]) for r in rows]
recall = [float(r["recall"]) for r in rows]
f1 = [float(r["f1"]) for r in rows]

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Data Scaling Experiment – Frozen Whisper + Mean Pooling + Logistic Regression")

# F1 vs training samples
axes[0, 0].plot(n_train, f1, "o-", linewidth=2, markersize=8, label="F1", color="C0")
axes[0, 0].set_xlabel("Number of Training Samples")
axes[0, 0].set_ylabel("F1 Score")
axes[0, 0].set_title("F1 Score vs Training Set Size")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xscale("log")

# Accuracy vs training samples
axes[0, 1].plot(n_train, accuracy, "s-", linewidth=2, markersize=8, label="Accuracy", color="C1")
axes[0, 1].set_xlabel("Number of Training Samples")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Accuracy vs Training Set Size")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xscale("log")

# Precision vs training samples
axes[1, 0].plot(n_train, precision, "^-", linewidth=2, markersize=8, label="Precision", color="C2")
axes[1, 0].set_xlabel("Number of Training Samples")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].set_title("Precision vs Training Set Size")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xscale("log")

# Recall vs training samples
axes[1, 1].plot(n_train, recall, "v-", linewidth=2, markersize=8, label="Recall", color="C3")
axes[1, 1].set_xlabel("Number of Training Samples")
axes[1, 1].set_ylabel("Recall")
axes[1, 1].set_title("Recall vs Training Set Size")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xscale("log")

plt.tight_layout()
plt.savefig("experiments/data_scaling/scaling_plots.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/data_scaling/scaling_plots.png")

# Also create a combined comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(n_train, accuracy, "o-", linewidth=2.5, markersize=10, label="Accuracy")
ax.plot(n_train, f1, "s-", linewidth=2.5, markersize=10, label="F1")
ax.plot(n_train, precision, "^-", linewidth=2.5, markersize=10, label="Precision")
ax.plot(n_train, recall, "v-", linewidth=2.5, markersize=10, label="Recall")

ax.set_xlabel("Training Set Size", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Data Scaling Experiment Results\n(Frozen Whisper + Mean Pooling + Logistic Regression)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale("log")

plt.tight_layout()
plt.savefig("experiments/data_scaling/combined_metrics.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/data_scaling/combined_metrics.png")

plt.show()
