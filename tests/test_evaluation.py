import csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation import metrics_at_threshold, select_threshold, write_error_analysis


def test_threshold_metrics_classify_false_end_and_continue():
    result = metrics_at_threshold(np.array([0, 0, 1, 1]), np.array([.9, .1, .2, .8]), .5)
    assert (result["fp"], result["fn"], result["false_end_rate"], result["false_continue_rate"]) == (1, 1, .5, .5)


def test_threshold_selection_and_error_csv(tmp_path):
    labels, probabilities = np.array([0, 0, 1, 1]), np.array([.1, .4, .6, .9])
    threshold, _ = select_threshold(labels, probabilities, thresholds=[.3, .5, .7])
    assert threshold == .5
    output = tmp_path / "errors.csv"; write_error_analysis(output, labels, np.array([.9, .1, .2, .8]), [{"id": str(i)} for i in range(4)], .5)
    with output.open() as handle: rows = list(csv.DictReader(handle))
    assert {row["error_type"] for row in rows} == {"false_end", "false_continue"}
