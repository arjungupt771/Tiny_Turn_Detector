"""Plot: audio duration vs. per-stage latency (Task 15 completion)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "reports" / "latency_benchmark_full_pipeline.csv"
fig_path = ROOT / "reports" / "figures" / "latency_vs_duration.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

durations = [float(r["duration_sec"]) for r in rows]
resample = [float(r["resample_ms"]) for r in rows]
feat = [float(r["feature_extract_ms"]) for r in rows]
enc = [float(r["encoder_ms"]) for r in rows]
head = [float(r["head_ms"]) for r in rows]
total = [float(r["total_ms"]) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(durations, total, marker="o", label="Total (warm)", color="black", linewidth=2)
ax.plot(durations, enc, marker="s", label="Encoder forward (random weights)", color="tab:orange")
ax.set_xlabel("Audio duration (s)")
ax.set_ylabel("Latency (ms)")
ax.set_title("Total & encoder latency vs. duration\n(encoder padded to fixed 30s -> ~flat)")
ax.legend()
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(durations, resample, marker="o", label="Resample")
ax2.plot(durations, feat, marker="^", label="Feature extraction")
ax2.plot(durations, head, marker="d", label="Classifier head")
ax2.set_xlabel("Audio duration (s)")
ax2.set_ylabel("Latency (ms)")
ax2.set_title("Non-encoder stages vs. duration\n(these actually scale with input length)")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(fig_path, dpi=150)
print(f"Saved {fig_path}")
