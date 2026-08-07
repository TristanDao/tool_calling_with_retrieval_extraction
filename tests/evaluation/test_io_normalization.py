"""Tests for schema adapters and deterministic normalization."""

from __future__ import annotations

from src.evaluation.config import NormalizationConfig
from src.evaluation.io import parse_gold_record, parse_prediction_record
from src.evaluation.normalization import ArgumentNormalizer
from src.evaluation.types import FunctionCall


def test_legacy_conversation_gold_adapter() -> None:
    record = parse_gold_record(
        {
            "id": "legacy_1",
            "source": "glaive",
            "conversation": [
                {"role": "user", "content": "Thời tiết Hà Nội"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_calls": [
                        {"name": "get_weather", "arguments": {"city": "Hà Nội"}}
                    ],
                },
            ],
            "tools": [],
        }
    )
    assert record.query == "Thời tiết Hà Nội"
    assert record.function_calls[0].name == "get_weather"


def test_openai_and_raw_output_prediction_adapters() -> None:
    openai_record = parse_prediction_record(
        {
            "id": "p1",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": "{\"city\": \"Huế\"}"}}
            ],
        }
    )
    raw_record = parse_prediction_record(
        {
            "id": "p2",
            "raw_output": (
                "<tool_call>{\"name\":\"get_weather\","
                "\"arguments\":{\"city\":\"Huế\"}}</tool_call>"
            ),
        }
    )
    malformed = parse_prediction_record({"id": "p3", "raw_output": "không parse được"})
    assert openai_record.parse_valid
    assert openai_record.function_calls == raw_record.function_calls
    assert not malformed.parse_valid
    assert malformed.function_calls == ()


def test_schema_aware_normalization_preserves_string_identifier() -> None:
    normalizer = ArgumentNormalizer(
        NormalizationConfig(
            global_aliases={"đô la mỹ": "USD"},
            unordered_array_paths=("search.tags",),
        )
    )
    call = FunctionCall(
        name="search",
        arguments={
            "account": "001234",
            "user_id": "AbC-001",
            "amount": "5 nghìn",
            "active": "Có",
            "currency": "Đô la Mỹ",
            "tags": ["B", "a"],
            "date": "05/08/2026",
        },
    )
    tool = {
        "parameters": {
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "user_id": {"type": "string"},
                "amount": {"type": "integer"},
                "active": {"type": "boolean"},
                "currency": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "date": {"type": "string", "format": "date"},
            },
        }
    }
    normalized = normalizer.normalize_arguments(call, tool)
    assert normalized["account"] == "001234"
    assert normalized["user_id"] == "AbC-001"
    assert normalized["amount"] == 5000
    assert normalized["active"] is True
    assert normalized["currency"] == "usd"
    assert normalized["tags"] == ["a", "b"]
    assert normalized["date"] == "2026-08-05"


def test_fractional_number_is_not_silently_coerced_to_integer() -> None:
    normalizer = ArgumentNormalizer(NormalizationConfig())
    assert normalizer.normalize_value("5.5", {"type": "integer"}, "tool", "amount") == "5.5"
