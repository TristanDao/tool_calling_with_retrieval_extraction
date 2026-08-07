"""Tests for detection, retrieval, selection, and extraction metrics."""

from __future__ import annotations

import pytest
from src.evaluation.config import NormalizationConfig
from src.evaluation.detection_metrics import compute_detection_metrics
from src.evaluation.extraction_metrics import compute_extraction_metrics
from src.evaluation.normalization import ArgumentNormalizer
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.selection_metrics import compute_selection_metrics


def test_detection_and_selection_metrics(gold_records, prediction_records) -> None:
    detection = compute_detection_metrics(gold_records, prediction_records)
    selection = compute_selection_metrics(gold_records, prediction_records)
    assert detection["precision"] == pytest.approx(2 / 3)
    assert detection["recall"] == 1.0
    assert detection["f1"] == pytest.approx(0.8)
    assert detection["hallucinated_call_rate"] == 0.5
    assert detection["missed_call_rate"] == 0.0
    assert selection["tool_set_accuracy_positive"] == 1.0
    assert selection["micro"]["f1"] == 1.0


def test_multi_call_retrieval_metrics(gold_records, prediction_records) -> None:
    metrics = compute_retrieval_metrics(gold_records, prediction_records, (1, 3))
    assert metrics["coverage"] == 1.0
    assert metrics["recall_at_1"] == pytest.approx(2 / 3)
    assert metrics["full_recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["full_recall_at_3"] == 1.0
    assert metrics["selection_conversion_at_3"] == 1.0
    assert metrics["mrr"] == 1.0


def test_normalized_extraction_metrics(gold_records, prediction_records) -> None:
    metrics, rows = compute_extraction_metrics(
        gold_records,
        prediction_records,
        ArgumentNormalizer(NormalizationConfig()),
    )
    assert metrics["normalized_arg_em_given_correct_tool"] == 1.0
    assert metrics["key"]["f1"] == 1.0
    assert metrics["argument_pair"]["f1"] == 1.0
    assert metrics["value_accuracy"] == 1.0
    assert metrics["per_type"]["string"]["key_f1"] == 1.0
    assert metrics["per_type"]["string"]["value_accuracy_on_gold"] == 1.0
    assert len(rows) == len(gold_records)
