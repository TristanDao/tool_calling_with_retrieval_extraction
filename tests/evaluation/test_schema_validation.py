"""Tests for strict candidate-tool and parameter schema validation."""

from __future__ import annotations

from src.evaluation.io import parse_gold_record, parse_prediction_record
from src.evaluation.schema_validation import validate_prediction_schema


def test_unknown_parameter_is_invalid_when_schema_omits_additional_properties() -> None:
    gold = parse_gold_record(
        {
            "id": "schema_1",
            "source": "xlam",
            "query": "Thời tiết Huế",
            "function_calls": [{"name": "get_weather", "arguments": {"city": "Huế"}}],
            "tools": [
                {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
    )
    prediction = parse_prediction_record(
        {
            "id": "schema_1",
            "function_calls": [
                {
                    "name": "get_weather",
                    "arguments": {"city": "Huế", "invented": "value"},
                }
            ],
        }
    )
    result = validate_prediction_schema(gold, prediction)
    assert not result["all_calls_valid"]
    assert "unknown parameter: invented" in result["calls"][0]["errors"]
