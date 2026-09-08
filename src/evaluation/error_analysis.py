"""Call-aligned error events with explicit heuristic labels for human review."""

from __future__ import annotations

from typing import Any

from src.evaluation.config import NormalizationConfig
from src.evaluation.extraction_metrics import normalize_calls
from src.evaluation.io import parse_gold_record, parse_prediction_record
from src.evaluation.matching import align_calls
from src.evaluation.normalization import ArgumentNormalizer, canonical_json


def classify_errors(
    sample: dict[str, Any], prediction: dict[str, Any], raw: dict[str, Any]
) -> list[dict[str, Any]]:
    gold = parse_gold_record({"source": "unknown", "tools": [], **sample})
    pred = parse_prediction_record({"id": gold.id, **prediction})
    normalizer = ArgumentNormalizer(NormalizationConfig())
    gold_calls = normalize_calls(gold, gold.function_calls, normalizer)
    pred_calls = normalize_calls(gold, pred.function_calls, normalizer)
    events: list[dict[str, Any]] = []
    query = normalizer.normalize_text(gold.query)

    def record(label: str, **detail: Any) -> None:
        events.append({"id": gold.id, "error_class": label, "unit": "event",
                       "label_source": "heuristic", "reviewed": False, **detail})

    if not pred.parse_valid:
        record("invalid_output", detail=pred.parse_error)
    for index, pair in enumerate(align_calls(gold_calls, pred_calls)):
        left, right = pair.gold, pair.prediction
        detail = {"alignment_index": index, "tool": (left or right).name}
        if left is None:
            record("hallucinated_call" if not gold_calls else "wrong_tool", **detail,
                   reason="extra_call", predicted=right.arguments)
            continue
        if right is None:
            record("missed_call" if not pred_calls else "wrong_tool", **detail,
                   reason="missing_call", gold=left.arguments)
            continue
        for key in sorted(set(left.arguments) | set(right.arguments)):
            if key not in right.arguments:
                record("I", **detail, param=key, gold=left.arguments[key], reason="missing_argument")
            elif key not in left.arguments:
                record("extra_argument", **detail, param=key, predicted=right.arguments[key])
            else:
                expected, actual = left.arguments[key], right.arguments[key]
                if canonical_json(expected) == canonical_json(actual):
                    continue
                label = "W"
                reason = "wrong_value"
                if isinstance(expected, str) and isinstance(actual, str):
                    if expected not in query:
                        label, reason = "T", "gold_not_verbatim_requires_review"
                    elif expected and actual and (expected in actual or actual in expected):
                        label, reason = "P", "substring_overlap_requires_review"
                record(label, **detail, param=key, gold=expected, predicted=actual, reason=reason)
    return events
