"""Metrics for deciding whether a query requires one or more tool calls."""

from __future__ import annotations

from typing import Any

from src.evaluation.metric_utils import prf, safe_divide
from src.evaluation.types import GoldRecord, PredictionRecord


def compute_detection_metrics(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    for gold_record, prediction in zip(gold, predictions, strict=True):
        if gold_record.is_positive and prediction.predicts_call:
            true_positive += 1
        elif not gold_record.is_positive and prediction.predicts_call:
            false_positive += 1
        elif gold_record.is_positive and not prediction.predicts_call:
            false_negative += 1
        else:
            true_negative += 1
    metrics = prf(true_positive, false_positive, false_negative)
    metrics.update(
        {
            "true_negative": true_negative,
            "accuracy": safe_divide(true_positive + true_negative, len(gold)),
            "hallucinated_call_rate": safe_divide(false_positive, false_positive + true_negative),
            "missed_call_rate": safe_divide(false_negative, true_positive + false_negative),
            "positive_count": true_positive + false_negative,
            "negative_count": true_negative + false_positive,
        }
    )
    return metrics
