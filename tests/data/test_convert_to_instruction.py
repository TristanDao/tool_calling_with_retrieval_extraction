"""Tests for the native Qwen3.5 data adapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.data.convert_to_instruction import convert_file, convert_sample

_SAMPLE_SINGLE_CALL = {
    "id": "glaive_00000",
    "source": "glaive",
    "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
    "function_calls": [
        {"name": "search_tutors", "arguments": {"subject": "Toán", "location": "Hà Nội"}},
    ],
    "tools": [
        {
            "name": "search_tutors",
            "description": "Tìm gia sư theo môn học và khu vực.",
            "feature_group": "Tìm kiếm & Kết nối",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Môn học cần tìm"},
                    "location": {"type": "string", "description": "Thành phố hoặc khu vực"},
                },
                "required": ["subject", "location"],
            },
        },
    ],
}

_SAMPLE_MULTI_CALL = {
    "id": "xlam_00001",
    "source": "xlam",
    "query": "Tìm chuyến bay và khách sạn ở Tokyo",
    "function_calls": [
        {"name": "search_flights", "arguments": {"destination": "Tokyo"}},
        {"name": "search_hotels", "arguments": {"city": "Tokyo"}},
    ],
    "tools": [
        {"name": "search_flights", "description": "Tìm chuyến bay", "parameters": {"type": "object", "properties": {}}},
        {"name": "search_hotels", "description": "Tìm khách sạn", "parameters": {"type": "object", "properties": {}}},
    ],
}


def test_convert_sample_single_call_uses_native_shape():
    result = convert_sample(_SAMPLE_SINGLE_CALL)
    assert result is not None
    assert result["id"] == "glaive_00000"
    assert result["language"] == "vi"
    assert [message["role"] for message in result["messages"]] == ["system", "user", "assistant"]
    assert result["messages"][1]["content"] == _SAMPLE_SINGLE_CALL["query"]
    assistant = result["messages"][2]
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["type"] == "function"
    assert assistant["tool_calls"][0]["function"]["name"] == "search_tutors"
    assert assistant["tool_calls"][0]["function"]["arguments"] == {
        "subject": "Toán",
        "location": "Hà Nội",
    }


def test_convert_sample_multi_call_preserves_calls():
    result = convert_sample(_SAMPLE_MULTI_CALL)
    assert result is not None
    calls = result["messages"][2]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["search_flights", "search_hotels"]


def test_convert_sample_negative_uses_natural_response():
    sample = {
        "id": "glaive_00002",
        "source": "glaive",
        "query": "Hôm nay thời tiết thật đẹp.",
        "function_calls": [],
        "tools": [{"name": "get_weather", "description": "Xem thời tiết.", "parameters": {"type": "object", "properties": {}}}],
    }
    result = convert_sample(sample)
    assert result is not None
    assistant = result["messages"][2]
    assert "tool_calls" not in assistant
    assert assistant["content"]
    assert "<no_tool_call>" not in assistant["content"]


def test_convert_sample_english_prompt():
    result = convert_sample(_SAMPLE_SINGLE_CALL, language="en")
    assert result is not None
    assert result["language"] == "en"
    assert result["messages"][0]["content"].startswith("You are an AI assistant")


def test_convert_sample_requires_tools():
    assert convert_sample({"query": "hi", "function_calls": [], "tools": []}) is None


def test_convert_file():
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "in.jsonl"
        output_path = Path(tmp) / "instruction" / "out.jsonl"
        input_path.write_text(
            json.dumps(_SAMPLE_SINGLE_CALL, ensure_ascii=False) + "\n"
            + json.dumps(_SAMPLE_MULTI_CALL, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        converted, skipped = convert_file(input_path, output_path)
        assert converted == 2
        assert skipped == 0
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        assert [row["id"] for row in rows] == ["glaive_00000", "xlam_00001"]


def test_convert_file_skips_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "in.jsonl"
        output_path = Path(tmp) / "instruction" / "out.jsonl"
        input_path.write_text(
            json.dumps(_SAMPLE_SINGLE_CALL, ensure_ascii=False) + "\n"
            + "not valid json\n"
            + json.dumps({"query": "", "tools": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        converted, skipped = convert_file(input_path, output_path)
        assert converted == 1
        assert skipped == 2
