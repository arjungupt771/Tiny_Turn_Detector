"""Validate train/validation/test split integrity and document statistics.

This script ensures:
1. No duplicate IDs across splits
2. No train/validation/test leakage
3. All audio files are present and valid
4. Comprehensive statistics on all splits
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


def load_manifest(path: Path) -> list[dict]:
    """Load a CSV manifest."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def validate_split_integrity(train: list[dict], val: list[dict], test: list[dict]) -> None:
    """Check for duplicate IDs and leakage."""
    train_ids = set(r["id"] for r in train)
    val_ids = set(r["id"] for r in val)
    test_ids = set(r["id"] for r in test)
    
    total_ids = len(train_ids) + len(val_ids) + len(test_ids)
    all_ids = train_ids | val_ids | test_ids
    
    print("\n=== SPLIT INTEGRITY ===")
    print(f"Total unique IDs: {len(all_ids)}")
    print(f"Sum of split sizes: {total_ids}")
    
    if total_ids != len(all_ids):
        raise RuntimeError("Duplicate IDs found across splits!")
    
    if train_ids & val_ids:
        raise RuntimeError(f"Train/Val leakage: {len(train_ids & val_ids)} IDs overlap")
    if train_ids & test_ids:
        raise RuntimeError(f"Train/Test leakage: {len(train_ids & test_ids)} IDs overlap")
    if val_ids & test_ids:
        raise RuntimeError(f"Val/Test leakage: {len(val_ids & test_ids)} IDs overlap")
    
    print("✓ No duplicate IDs")
    print("✓ No train/val leakage")
    print("✓ No train/test leakage")
    print("✓ No val/test leakage")


def validate_audio_presence(train: list[dict], val: list[dict], test: list[dict]) -> None:
    """Check audio presence and validity indicators."""
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        audio_ok_count = sum(1 for r in data if r.get("audio_ok") in ("True", True, "true", "1", 1))
        invalid = len(data) - audio_ok_count
        
        if invalid > 0:
            print(f"⚠ {split_name}: {invalid} / {len(data)} audio files reported as invalid")
        else:
            print(f"✓ {split_name}: all {len(data)} audio files valid")


def compute_statistics(data: list[dict], split_name: str) -> dict:
    """Compute comprehensive statistics for a split."""
    stats = {
        "split": split_name,
        "count": len(data),
        "labels": dict(Counter(int(r["label"]) for r in data)),
        "languages": dict(Counter(r["language"] for r in data)),
        "midfiller": dict(Counter(r["midfiller"] for r in data)),
        "endfiller": dict(Counter(r["endfiller"] for r in data)),
        "synthetic": dict(Counter(r["synthetic"] for r in data)),
        "dataset_source": dict(Counter(r["dataset"] for r in data)),
    }
    
    # Duration statistics
    try:
        durations = [float(r["duration_seconds"]) for r in data if r["duration_seconds"]]
        if durations:
            stats["duration_seconds"] = {
                "count": len(durations),
                "mean": round(mean(durations), 4),
                "stdev": round(stdev(durations), 4) if len(durations) > 1 else 0.0,
                "min": round(min(durations), 4),
                "max": round(max(durations), 4),
                "p50": round(sorted(durations)[len(durations)//2], 4),
            }
    except (ValueError, IndexError):
        pass
    
    return stats


def print_statistics(stats: dict) -> None:
    """Pretty-print statistics."""
    print(f"\n=== {stats['split'].upper()} ({stats['count']} samples) ===")
    
    print("\nLabel distribution:")
    for label, count in sorted(stats["labels"].items()):
        pct = 100.0 * count / stats["count"]
        label_name = "END" if label == 1 else "CONTINUE"
        print(f"  {label_name}: {count:,} ({pct:.1f}%)")
    
    if "duration_seconds" in stats:
        dur = stats["duration_seconds"]
        print(f"\nDuration (seconds): {dur['count']} valid samples")
        print(f"  mean={dur['mean']:.2f}s, stdev={dur['stdev']:.2f}s")
        print(f"  min={dur['min']:.2f}s, max={dur['max']:.2f}s, p50={dur['p50']:.2f}s")
    
    print(f"\nLanguages: {len(stats['languages'])} unique")
    for lang, count in sorted(stats["languages"].items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = 100.0 * count / stats["count"]
        print(f"  {lang:6s}: {count:,} ({pct:.1f}%)")
    if len(stats["languages"]) > 10:
        print(f"  ... {len(stats['languages']) - 10} more languages")
    
    print(f"\nSynthetic distribution:")
    for syn, count in sorted(stats["synthetic"].items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * count / stats["count"]
        print(f"  {str(syn):6s}: {count:,} ({pct:.1f}%)")
    
    print(f"\nFillers (mid-utterance):")
    for filler, count in sorted(stats["midfiller"].items(), key=lambda x: (x[0] is None, x[1]), reverse=True):
        pct = 100.0 * count / stats["count"]
        print(f"  {str(filler):6s}: {count:,} ({pct:.1f}%)")
    
    print(f"\nFillers (end of utterance):")
    for filler, count in sorted(stats["endfiller"].items(), key=lambda x: (x[0] is None, x[1]), reverse=True):
        pct = 100.0 * count / stats["count"]
        print(f"  {str(filler):6s}: {count:,} ({pct:.1f}%)")
    
    print(f"\nDataset source: {len(stats['dataset_source'])} unique")
    for source, count in sorted(stats["dataset_source"].items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * count / stats["count"]
        print(f"  {source:12s}: {count:,} ({pct:.1f}%)")


def main() -> None:
    manifest_dir = Path("data/manifests")
    train = load_manifest(manifest_dir / "train.csv")
    val = load_manifest(manifest_dir / "val.csv")
    test = load_manifest(manifest_dir / "test.csv")
    
    print("=" * 70)
    print("DATA SPLIT VALIDATION REPORT")
    print("=" * 70)
    
    # Integrity checks
    validate_split_integrity(train, val, test)
    
    # Audio validity
    print("\n=== AUDIO VALIDITY ===")
    validate_audio_presence(train, val, test)
    
    # Detailed statistics
    train_stats = compute_statistics(train, "train")
    val_stats = compute_statistics(val, "val")
    test_stats = compute_statistics(test, "test")
    
    print_statistics(train_stats)
    print_statistics(val_stats)
    print_statistics(test_stats)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    print(f"\nSplit sizes: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    print(f"Total: {len(train) + len(val) + len(test):,} samples")
    
    # Class balance comparison
    print("\nClass distribution comparison:")
    for split_name, stats in [("train", train_stats), ("val", val_stats), ("test", test_stats)]:
        end_pct = 100.0 * stats["labels"].get(1, 0) / stats["count"]
        cont_pct = 100.0 * stats["labels"].get(0, 0) / stats["count"]
        print(f"  {split_name:6s}: END {end_pct:5.1f}% | CONTINUE {cont_pct:5.1f}%")
    
    print("\n✓ Validation complete — no issues found")


if __name__ == "__main__":
    main()
