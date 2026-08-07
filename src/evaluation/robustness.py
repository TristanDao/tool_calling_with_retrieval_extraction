"""Stratified robustness metrics and max-minus-min performance gaps."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.evaluation.types import GoldRecord


def _bucket(value: int, boundaries: tuple[tuple[int, str], ...], final_label: str) -> str:
    for upper, label in boundaries:
        if value <= upper:
            return label
    return final_label


def _derived_value(record: GoldRecord, field: str) -> str | None:
    if field == "source":
        return record.source
    if field == "call_count":
        count = len(record.function_calls)
        return "no_call" if count == 0 else "single" if count == 1 else "multi"
    if field == "candidate_tool_count":
        return _bucket(len(record.tools), ((4, "1-4"), (10, "5-10"), (50, "11-50")), ">50")
    if field == "schema_parameter_count":
        schemas = record.tool_schemas
        count = sum(
            len(schemas.get(call.name, {}).get("parameters", {}).get("properties", {}))
            for call in record.function_calls
        )
        return _bucket(count, ((0, "0"), (1, "1"), (3, "2-3"), (5, "4-5")), ">5")
    if field == "feature_group":
        schemas = record.tool_schemas
        groups = {
            str(schemas.get(call.name, {}).get("feature_group", "unknown"))
            for call in record.function_calls
        }
        if not groups:
            return "no_call"
        return next(iter(groups)) if len(groups) == 1 else "mixed"
    return None


def _nested_value(record: GoldRecord, field: str) -> str | None:
    derived = _derived_value(record, field)
    if derived is not None:
        return derived
    parts = field.split(".")
    if parts[0] == "metadata":
        value: Any = record.metadata
        parts = parts[1:]
    else:
        value = record.metadata.get(parts[0])
        parts = parts[1:]
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None or isinstance(value, (dict, list)):
        return None
    return str(value)


def _mean(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_robustness_metrics(
    gold: list[GoldRecord],
    per_sample: list[dict[str, Any]],
    slice_fields: tuple[str, ...],
) -> dict[str, Any]:
    sample_by_id = {item["id"]: item for item in per_sample}
    result: dict[str, Any] = {}
    for field in slice_fields:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in gold:
            value = _nested_value(record, field)
            if value is not None:
                groups[value].append(sample_by_id[record.id])
        group_metrics: dict[str, Any] = {}
        for value, rows in sorted(groups.items()):
            positives = [row for row in rows if row["is_positive"]]
            tool_correct = [bool(row["tool_set_exact"]) for row in positives]
            arg_values = [
                bool(row["arg_exact_given_tool"])
                for row in positives
                if row.get("arg_exact_given_tool") is not None
            ]
            group_metrics[value] = {
                "sample_count": len(rows),
                "positive_count": len(positives),
                "overall_success": _mean([bool(row["overall_success"]) for row in rows]),
                "n_fcem_positive": _mean([bool(row["n_fcem"]) for row in positives]),
                "tool_set_accuracy_positive": _mean(tool_correct),
                "normalized_arg_em_given_correct_tool": _mean(arg_values),
            }
        gaps: dict[str, float | None] = {}
        metric_names = (
            "overall_success",
            "n_fcem_positive",
            "tool_set_accuracy_positive",
            "normalized_arg_em_given_correct_tool",
        )
        for metric_name in metric_names:
            values = [
                metrics[metric_name]
                for metrics in group_metrics.values()
                if metrics[metric_name] is not None
            ]
            gaps[metric_name] = max(values) - min(values) if len(values) >= 2 else None
        result[field] = {"groups": group_metrics, "gaps": gaps}
    return result
