"""Metrics for final tool-name selection after call detection or retrieval."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.evaluation.matching import tool_name_counter, tool_names_exact
from src.evaluation.metric_utils import prf
from src.evaluation.types import GoldRecord, PredictionRecord


def compute_selection_metrics(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
) -> dict[str, Any]:
    positive_pairs = [
        (gold_record, prediction)
        for gold_record, prediction in zip(gold, predictions, strict=True)
        if gold_record.is_positive
    ]
    exact_count = sum(
        tool_names_exact(item.function_calls, prediction.function_calls)
        for item, prediction in positive_pairs
    )
    per_tool_counts: dict[str, Counter[str]] = {}
    micro_tp = micro_fp = micro_fn = 0
    for item, prediction in positive_pairs:
        gold_counter = tool_name_counter(item.function_calls)
        pred_counter = tool_name_counter(prediction.function_calls)
        for name in set(gold_counter) | set(pred_counter):
            counts = per_tool_counts.setdefault(name, Counter())
            true_positive = min(gold_counter[name], pred_counter[name])
            false_positive = max(pred_counter[name] - gold_counter[name], 0)
            false_negative = max(gold_counter[name] - pred_counter[name], 0)
            counts["tp"] += true_positive
            counts["fp"] += false_positive
            counts["fn"] += false_negative
            counts["support"] += gold_counter[name]
            micro_tp += true_positive
            micro_fp += false_positive
            micro_fn += false_negative
    tool_metrics: dict[str, Any] = {}
    f1_values: list[float] = []
    weighted_total = 0.0
    support_total = 0
    for name, counts in sorted(per_tool_counts.items()):
        metrics = prf(counts["tp"], counts["fp"], counts["fn"])
        metrics["support"] = counts["support"]
        tool_metrics[name] = metrics
        if metrics["f1"] is not None:
            f1_values.append(metrics["f1"])
            weighted_total += metrics["f1"] * counts["support"]
            support_total += counts["support"]
    return {
        "tool_set_accuracy_positive": exact_count / len(positive_pairs) if positive_pairs else None,
        "positive_count": len(positive_pairs),
        "micro": prf(micro_tp, micro_fp, micro_fn),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "weighted_f1": weighted_total / support_total if support_total else None,
        "per_tool": tool_metrics,
    }
