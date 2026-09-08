"""Normalized argument extraction metrics with optimal multi-call alignment."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.evaluation.matching import align_calls, tool_names_exact
from src.evaluation.metric_utils import prf, safe_divide
from src.evaluation.normalization import ArgumentNormalizer, canonical_json, flatten_arguments
from src.evaluation.types import FunctionCall, GoldRecord, PredictionRecord


def normalize_calls(
    record: GoldRecord,
    calls: tuple[FunctionCall, ...],
    normalizer: ArgumentNormalizer,
) -> tuple[FunctionCall, ...]:
    schemas = record.tool_schemas
    return tuple(normalizer.normalize_call(call, schemas.get(call.name)) for call in calls)


def _schema_for_path(tool: dict[str, Any] | None, path: str) -> dict[str, Any]:
    schema = (tool or {}).get("parameters", {})
    for part in path.replace("[]", "").split("."):
        if not isinstance(schema, dict):
            return {}
        if schema.get("type") == "array":
            schema = schema.get("items", {})
        if schema.get("type") == "object" or "properties" in schema:
            schema = schema.get("properties", {}).get(part, {})
    return schema if isinstance(schema, dict) else {}


def _add_counts(
    gold_flat: dict[str, Any],
    pred_flat: dict[str, Any],
    counts: Counter[str],
) -> None:
    gold_keys = set(gold_flat)
    pred_keys = set(pred_flat)
    counts["key_tp"] += len(gold_keys & pred_keys)
    counts["key_fp"] += len(pred_keys - gold_keys)
    counts["key_fn"] += len(gold_keys - pred_keys)
    gold_pairs = Counter((key, canonical_json(value)) for key, value in gold_flat.items())
    pred_pairs = Counter((key, canonical_json(value)) for key, value in pred_flat.items())
    pair_tp = sum((gold_pairs & pred_pairs).values())
    counts["pair_tp"] += pair_tp
    counts["pair_fp"] += sum(pred_pairs.values()) - pair_tp
    counts["pair_fn"] += sum(gold_pairs.values()) - pair_tp
    common_keys = gold_keys & pred_keys
    counts["value_comparable"] += len(common_keys)
    counts["value_correct"] += sum(
        canonical_json(gold_flat[key]) == canonical_json(pred_flat[key]) for key in common_keys
    )


def _add_type_counts(
    record: GoldRecord,
    gold_call: FunctionCall | None,
    pred_call: FunctionCall | None,
    gold_flat: dict[str, Any],
    pred_flat: dict[str, Any],
    per_type_counts: dict[str, Counter[str]],
) -> None:
    gold_schema = record.tool_schemas.get(gold_call.name) if gold_call is not None else None
    pred_schema = record.tool_schemas.get(pred_call.name) if pred_call is not None else None
    for path, gold_value in gold_flat.items():
        type_name = str(_schema_for_path(gold_schema, path).get("type", "unknown"))
        type_counts = per_type_counts.setdefault(type_name, Counter())
        type_counts["support"] += 1
        if path in pred_flat:
            type_counts["key_tp"] += 1
            type_counts["value_comparable"] += 1
            if canonical_json(gold_value) == canonical_json(pred_flat[path]):
                type_counts["correct_value"] += 1
        else:
            type_counts["key_fn"] += 1
    for path in set(pred_flat) - set(gold_flat):
        type_name = str(_schema_for_path(pred_schema, path).get("type", "unknown"))
        per_type_counts.setdefault(type_name, Counter())["key_fp"] += 1


def _metric_from_pairs(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
    normalizer: ArgumentNormalizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    per_type_counts: dict[str, Counter[str]] = {}
    per_sample: list[dict[str, Any]] = []
    tool_correct_count = 0
    arg_exact_count = 0
    positive_count = 0
    for record, prediction in zip(gold, predictions, strict=True):
        if not record.is_positive:
            per_sample.append({"id": record.id, "arg_exact_given_tool": None})
            continue
        positive_count += 1
        normalized_gold = normalize_calls(record, record.function_calls, normalizer)
        normalized_pred = normalize_calls(record, prediction.function_calls, normalizer)
        tool_correct = tool_names_exact(normalized_gold, normalized_pred)
        if tool_correct:
            tool_correct_count += 1
        alignments = align_calls(normalized_gold, normalized_pred)
        all_aligned_exact = tool_correct and prediction.parse_valid
        for alignment in alignments:
            gold_flat = flatten_arguments(alignment.gold.arguments) if alignment.gold else {}
            pred_flat = flatten_arguments(alignment.prediction.arguments) if alignment.prediction else {}
            _add_counts(gold_flat, pred_flat, counts)
            _add_type_counts(
                record,
                alignment.gold,
                alignment.prediction,
                gold_flat,
                pred_flat,
                per_type_counts,
            )
            if alignment.gold is None or alignment.prediction is None:
                all_aligned_exact = False
                continue
            if canonical_json(alignment.gold.arguments) != canonical_json(alignment.prediction.arguments):
                all_aligned_exact = False
        arg_exact = all_aligned_exact if tool_correct else False
        arg_exact_count += int(arg_exact)
        per_sample.append(
            {
                "id": record.id,
                "arg_exact_given_tool": arg_exact if tool_correct else None,
                "tool_correct_for_extraction": tool_correct,
            }
        )
    key_metrics = prf(counts["key_tp"], counts["key_fp"], counts["key_fn"])
    pair_metrics = prf(counts["pair_tp"], counts["pair_fp"], counts["pair_fn"])
    per_type: dict[str, Any] = {}
    for type_name, type_counts in sorted(per_type_counts.items()):
        type_key_metrics = prf(
            type_counts["key_tp"],
            type_counts["key_fp"],
            type_counts["key_fn"],
        )
        per_type[type_name] = {
            "support": type_counts["support"],
            "key_precision": type_key_metrics["precision"],
            "key_recall": type_key_metrics["recall"],
            "key_f1": type_key_metrics["f1"],
            "value_accuracy_given_key": safe_divide(
                type_counts["correct_value"], type_counts["value_comparable"]
            ),
            "value_accuracy_on_gold": safe_divide(
                type_counts["correct_value"], type_counts["support"]
            ),
        }
    metrics = {
        "positive_count": positive_count,
        "tool_correct_positive_count": tool_correct_count,
        "normalized_arg_em_given_correct_tool": (
            arg_exact_count / tool_correct_count if tool_correct_count else None
        ),
        "key": key_metrics,
        "argument_pair": pair_metrics,
        "value_accuracy": safe_divide(counts["value_correct"], counts["value_comparable"]),
        "per_type": per_type,
    }
    return metrics, per_sample


def compute_extraction_metrics(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
    normalizer: ArgumentNormalizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _metric_from_pairs(gold, predictions, normalizer)
