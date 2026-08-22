"""Strict normalized end-to-end correctness for tool calling."""

from __future__ import annotations

from typing import Any

from src.evaluation.extraction_metrics import normalize_calls
from src.evaluation.matching import calls_exact, tool_names_exact
from src.evaluation.normalization import ArgumentNormalizer
from src.evaluation.types import GoldRecord, PredictionRecord


def compute_end_to_end_metrics(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
    normalizer: ArgumentNormalizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positive_count = 0
    positive_correct = 0
    negative_count = 0
    negative_correct = 0
    rows: list[dict[str, Any]] = []
    for record, prediction in zip(gold, predictions, strict=True):
        normalized_gold = normalize_calls(record, record.function_calls, normalizer)
        normalized_pred = normalize_calls(record, prediction.function_calls, normalizer)
        tool_exact = tool_names_exact(normalized_gold, normalized_pred)
        full_exact = prediction.parse_valid and calls_exact(normalized_gold, normalized_pred)
        if record.is_positive:
            positive_count += 1
            positive_correct += int(full_exact)
        else:
            negative_count += 1
            negative_correct += int(full_exact)
        rows.append(
            {
                "id": record.id,
                "is_positive": record.is_positive,
                "predicted_call": prediction.predicts_call,
                "tool_set_exact": tool_exact,
                "n_fcem": full_exact if record.is_positive else None,
                "correct_no_call": full_exact if not record.is_positive else None,
                "overall_success": full_exact,
            }
        )
    return (
        {
            "n_fcem_positive": positive_correct / positive_count if positive_count else None,
            "overall_success": (
                (positive_correct + negative_correct) / len(gold) if gold else None
            ),
            "positive_count": positive_count,
            "positive_correct": positive_correct,
            "negative_count": negative_count,
            "negative_correct": negative_correct,
        },
        rows,
    )
