from __future__ import annotations

import numpy as np

from src.eval.binary import evaluate_binary_classification


def test_threshold_fixed_vs_youden_changes_value():
    # Construct a simple case where Youden's J prefers a threshold away from 0.5
    # y_true coded with 1 as the positive class
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    # Well-separated probabilities but not symmetric around 0.5
    y_prob = np.array([0.05, 0.15, 0.2, 0.55, 0.7, 0.9], dtype=float)

    res_fixed = evaluate_binary_classification(
        y_true, y_prob, threshold_cfg={"strategy": "fixed", "value": 0.5}
    )
    res_youden = evaluate_binary_classification(
        y_true, y_prob, threshold_cfg={"strategy": "youden_j"}
    )

    assert abs(res_fixed.threshold - 0.5) < 1e-9
    # Expect Youden to choose one of the ROC thresholds present in data
    assert 0.0 <= res_youden.threshold <= 1.0
    assert not np.isclose(res_youden.threshold, res_fixed.threshold)


def test_threshold_f1_uses_validation_when_provided():
    # Make validation prefer a different operating point than test
    y_true_test = np.array([0, 0, 1, 1], dtype=int)
    y_prob_test = np.array([0.2, 0.6, 0.55, 0.9], dtype=float)
    y_true_val = np.array([0, 0, 1, 1], dtype=int)
    y_prob_val = np.array([0.45, 0.49, 0.51, 0.52], dtype=float)

    res = evaluate_binary_classification(
        y_true_test,
        y_prob_test,
        threshold_cfg={"strategy": "f1"},
        y_true_val=y_true_val,
        y_prob_val=y_prob_val,
    )

    # Validation has probabilities clustered around 0.5; F1 should choose near 0.5
    assert res.threshold_source == "validation"
    assert 0.45 <= res.threshold <= 0.55


def test_pos_label_passthrough_identity():
    # The evaluator expects y_true as 0/1 with 1 being positive. Base pipeline
    # handles alignment when pos_label != 1. Here we verify no surprises when pos_label=1.
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.9, 0.6, 0.4], dtype=float)
    res = evaluate_binary_classification(y_true, y_prob, threshold_cfg={"strategy": "fixed", "value": 0.5}, pos_label=1)
    # Metrics should be sane and threshold applied as-is
    assert 0.0 <= res.metrics["roc_auc"] <= 1.0
    assert res.threshold == 0.5

