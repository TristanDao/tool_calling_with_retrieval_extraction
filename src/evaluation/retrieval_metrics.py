"""Multi-call-aware metrics for ranked tool retrieval."""

from __future__ import annotations

from typing import Any

from src.evaluation.matching import tool_names_exact
from src.evaluation.types import GoldRecord, PredictionRecord


def compute_retrieval_metrics(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    positive_pairs = [
        (gold_record, prediction)
        for gold_record, prediction in zip(gold, predictions, strict=True)
        if gold_record.is_positive
    ]
    covered = [(item, pred) for item, pred in positive_pairs if pred.ranked_tools is not None]
    result: dict[str, Any] = {
        "positive_count": len(positive_pairs),
        "covered_positive_count": len(covered),
        "coverage": len(covered) / len(positive_pairs) if positive_pairs else None,
    }
    if not covered:
        for k in ks:
            result[f"recall_at_{k}"] = None
            result[f"full_recall_at_{k}"] = None
            result[f"selection_conversion_at_{k}"] = None
        result["mrr"] = None
        return result

    reciprocal_ranks: list[float] = []
    total_gold_tools = sum(len({call.name for call in item.function_calls}) for item, _ in covered)
    for item, prediction in covered:
        gold_names = {call.name for call in item.function_calls}
        ranks = [
            rank
            for rank, name in enumerate(prediction.ranked_tools or (), start=1)
            if name in gold_names
        ]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
    result["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)

    for k in ks:
        retrieved_gold_tools = 0
        full_coverage_count = 0
        conversion_eligible = 0
        conversion_correct = 0
        for item, prediction in covered:
            top_k = set((prediction.ranked_tools or ())[:k])
            gold_tool_names = {call.name for call in item.function_calls}
            retrieved_gold_tools += sum(name in top_k for name in gold_tool_names)
            full = all(name in top_k for name in gold_tool_names)
            full_coverage_count += int(full)
            if full:
                conversion_eligible += 1
                conversion_correct += int(tool_names_exact(item.function_calls, prediction.function_calls))
        result[f"recall_at_{k}"] = (
            retrieved_gold_tools / total_gold_tools if total_gold_tools else None
        )
        result[f"full_recall_at_{k}"] = full_coverage_count / len(covered)
        result[f"selection_conversion_at_{k}"] = (
            conversion_correct / conversion_eligible if conversion_eligible else None
        )
    return result
