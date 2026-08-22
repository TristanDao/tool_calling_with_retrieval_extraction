"""JSON Schema validation for predicted function calls."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from src.evaluation.types import GoldRecord, PredictionRecord


def validate_prediction_schema(
    gold: GoldRecord,
    prediction: PredictionRecord,
) -> dict[str, Any]:
    schemas = gold.tool_schemas
    call_results: list[dict[str, Any]] = []
    for call in prediction.function_calls:
        tool = schemas.get(call.name)
        errors: list[str] = []
        if tool is None:
            errors.append("tool is not present in the candidate schema set")
        else:
            parameters = tool.get("parameters", {})
            if not isinstance(parameters, dict):
                errors.append("tool parameters is not a JSON Schema object")
            else:
                properties = parameters.get("properties", {})
                if isinstance(properties, dict):
                    unknown = sorted(set(call.arguments) - set(properties))
                    errors.extend(f"unknown parameter: {name}" for name in unknown)
                try:
                    validator = Draft202012Validator(parameters)
                    errors.extend(
                        error.message
                        for error in sorted(
                            validator.iter_errors(call.arguments), key=lambda item: item.json_path
                        )
                    )
                except SchemaError as exc:
                    errors.append(f"invalid tool JSON Schema: {exc.message}")
        call_results.append({"name": call.name, "valid": not errors, "errors": errors})
    all_calls_valid = all(item["valid"] for item in call_results)
    return {
        "parse_valid": prediction.parse_valid,
        "parse_error": prediction.parse_error,
        "has_predicted_call": prediction.predicts_call,
        "all_calls_valid": prediction.parse_valid and all_calls_valid,
        "calls": call_results,
    }


def aggregate_schema_validity(results: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_count = len(results)
    parse_valid_count = sum(bool(item["parse_valid"]) for item in results)
    calls = [call for item in results for call in item["calls"]]
    predicted_call_samples = [item for item in results if item["has_predicted_call"]]
    valid_calls = sum(bool(call["valid"]) for call in calls)
    valid_samples = sum(bool(item["all_calls_valid"]) for item in predicted_call_samples)
    return {
        "prediction_parse_validity": parse_valid_count / prediction_count if prediction_count else None,
        "call_schema_validity": valid_calls / len(calls) if calls else None,
        "sample_schema_validity": (
            valid_samples / len(predicted_call_samples) if predicted_call_samples else None
        ),
        "prediction_count": prediction_count,
        "predicted_call_count": len(calls),
        "predicted_call_sample_count": len(predicted_call_samples),
    }
