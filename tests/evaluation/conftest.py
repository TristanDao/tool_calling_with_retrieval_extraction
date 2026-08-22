"""Shared fixtures for evaluation tests."""

from __future__ import annotations

from typing import Any

import pytest
from src.evaluation.io import parse_gold_record, parse_prediction_record


@pytest.fixture
def gold_records() -> list[Any]:
    raw = [
        {
            "id": "sample_1",
            "source": "glaive",
            "query": "Đổi 5 nghìn đồng sang đô la Mỹ",
            "function_calls": [
                {
                    "name": "convert_currency",
                    "arguments": {"amount": 5000, "target_currency": "USD"},
                }
            ],
            "tools": [
                {
                    "name": "convert_currency",
                    "feature_group": "Tài chính",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "integer"},
                            "target_currency": {"type": "string", "enum": ["USD", "VND"]},
                        },
                        "required": ["amount", "target_currency"],
                        "additionalProperties": False,
                    },
                }
            ],
            "metadata": {"tool_split": "seen"},
        },
        {
            "id": "sample_2",
            "source": "xlam",
            "query": "Tìm chuyến bay và khách sạn ở Huế",
            "function_calls": [
                {"name": "search_flights", "arguments": {"destination": "Huế"}},
                {"name": "search_hotels", "arguments": {"city": "Huế"}},
            ],
            "tools": [
                {
                    "name": "search_flights",
                    "feature_group": "Du lịch",
                    "parameters": {
                        "type": "object",
                        "properties": {"destination": {"type": "string"}},
                        "required": ["destination"],
                    },
                },
                {
                    "name": "search_hotels",
                    "feature_group": "Du lịch",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            ],
            "metadata": {"tool_split": "unseen"},
        },
        {
            "id": "sample_3",
            "source": "glaive",
            "query": "Xin chào",
            "function_calls": [],
            "tools": [],
            "metadata": {"tool_split": "seen"},
        },
        {
            "id": "sample_4",
            "source": "xlam",
            "query": "Cảm ơn bạn",
            "function_calls": [],
            "tools": [],
            "metadata": {"tool_split": "unseen"},
        },
    ]
    return [parse_gold_record(item) for item in raw]


@pytest.fixture
def prediction_records() -> list[Any]:
    raw = [
        {
            "id": "sample_1",
            "function_calls": [
                {
                    "name": "convert_currency",
                    "arguments": {"amount": "5 nghìn", "target_currency": "usd"},
                }
            ],
            "ranked_tools": ["convert_currency", "get_exchange_rate"],
            "telemetry": {
                "latency_ms": 10,
                "input_tokens": 20,
                "output_tokens": 5,
                "cost_usd": 0.01,
            },
        },
        {
            "id": "sample_2",
            "function_calls": [
                {"name": "search_hotels", "arguments": {"city": "huế"}},
                {"name": "search_flights", "arguments": {"destination": "HUẾ"}},
            ],
            "ranked_tools": ["search_hotels", "search_flights", "get_weather"],
            "telemetry": {
                "latency_ms": 20,
                "input_tokens": 30,
                "output_tokens": 10,
                "cost_usd": 0.02,
            },
        },
        {
            "id": "sample_3",
            "function_calls": [],
            "telemetry": {
                "latency_ms": 30,
                "input_tokens": 10,
                "output_tokens": 2,
                "cost_usd": 0.01,
            },
        },
        {
            "id": "sample_4",
            "function_calls": [{"name": "hallucinated_tool", "arguments": {}}],
            "telemetry": {
                "latency_ms": 40,
                "input_tokens": 10,
                "output_tokens": 3,
                "cost_usd": 0.01,
            },
        },
    ]
    return [parse_prediction_record(item) for item in raw]
