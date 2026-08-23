"""Small numerical helpers used by evaluation metrics."""

from __future__ import annotations

from typing import Any


def safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    if true_positive == 0 and false_positive == 0 and false_negative == 0:
        f1 = None
    elif true_positive == 0:
        f1 = 0.0
    elif precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
