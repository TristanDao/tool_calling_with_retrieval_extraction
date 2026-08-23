"""Tests for convert_to_instruction.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.data.convert_to_instruction import convert_sample, convert_file


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
        {"name": "search_flights", "description": "Tìm chuyến bay", "feature_group": "Du lịch",
         "parameters": {"type": "object", "properties": {}, "required": []}},
        {"name": "search_hotels", "description": "Tìm khách sạn", "feature_group": "Du lịch",
         "parameters": {"type": "object", "properties": {}, "required": []}},
    ],
}


def test_convert_sample_single_call():
    result = convert_sample(_SAMPLE_SINGLE_CALL)
    assert result is not None
    assert len(result["conversations"]) == 3
    assert result["conversations"][0]["from"] == "system"
    assert result["conversations"][1]["from"] == "human"
    assert result["conversations"][2]["from"] == "gpt"

    system_val = result["conversations"][0]["value"]
    assert "search_tutors" in system_val
    assert "Tìm gia sư" in system_val

    assert result["conversations"][1]["value"] == _SAMPLE_SINGLE_CALL["query"]

    assistant_val = result["conversations"][2]["value"]
    assert assistant_val.startswith("<tool_call>")
    assert assistant_val.endswith("</tool_call>")
    assert "search_tutors" in assistant_val
    assert "Toán" in assistant_val


def test_convert_sample_multi_call():
    result = convert_sample(_SAMPLE_MULTI_CALL)
    assert result is not None
    assistant_val = result["conversations"][2]["value"]
    lines = assistant_val.strip().split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("<tool_call>")
    assert lines[1].startswith("<tool_call>")
    assert "search_flights" in lines[0]
    assert "search_hotels" in lines[1]


def test_convert_sample_empty_query():
    result = convert_sample({"query": "", "function_calls": [], "tools": []})
    assert result is None


def test_convert_sample_no_function_calls():
    result = convert_sample({"query": "hi", "function_calls": [], "tools": []})
    assert result is None


def test_convert_sample_tool_call_json_valid():
    result = convert_sample(_SAMPLE_SINGLE_CALL)
    assistant_val = result["conversations"][2]["value"]
    json_str = assistant_val[len("<tool_call>"):-len("</tool_call>")]
    fc = json.loads(json_str)
    assert fc["name"] == "search_tutors"
    assert fc["arguments"] == {"subject": "Toán", "location": "Hà Nội"}


def test_convert_sample_feature_group_not_in_prompt():
    result = convert_sample(_SAMPLE_SINGLE_CALL)
    system_val = result["conversations"][0]["value"]
    assert "feature_group" not in system_val


def test_convert_file():
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.jsonl"
        out = Path(tmp) / "instruction" / "out.jsonl"
        inp.write_text(
            json.dumps(_SAMPLE_SINGLE_CALL, ensure_ascii=False) + "\n" +
            json.dumps(_SAMPLE_MULTI_CALL, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        converted, skipped, contaminated = convert_file(inp, out)
        assert converted == 2
        assert skipped == 0
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            item = json.loads(line)
            assert len(item["conversations"]) == 3


def test_convert_file_skips_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.jsonl"
        out = Path(tmp) / "instruction" / "out.jsonl"
        inp.write_text(
            json.dumps(_SAMPLE_SINGLE_CALL, ensure_ascii=False) + "\n" +
            "not valid json\n" +
            json.dumps({"query": "", "tools": []}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        converted, skipped, contaminated = convert_file(inp, out)
        assert converted == 1
        assert skipped == 2


def test_convert_file_drops_queries_belonging_to_a_higher_split():
    """Method 1 phải dùng cùng decontamination index với Method 2.

    Nếu Qwen/SFT train từ benchmark nguyên bản thì Method 1 có train→test
    leakage trong khi Method 2 không, và bảng so sánh hai method mất hiệu lực.
    """
    from src.models.sources import DecontaminationIndex, normalize_query_key

    index = DecontaminationIndex(
        effective_split={normalize_query_key(_SAMPLE_SINGLE_CALL["query"]): "test"}
    )
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.jsonl"
        out = Path(tmp) / "out.jsonl"
        inp.write_text(
            json.dumps(_SAMPLE_SINGLE_CALL, ensure_ascii=False) + "\n" +
            json.dumps(_SAMPLE_MULTI_CALL, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        converted, _, contaminated = convert_file(inp, out, index, "train")

        assert contaminated == 1
        assert converted == 1


def test_convert_file_keeps_sample_in_its_own_split():
    from src.models.sources import DecontaminationIndex, normalize_query_key

    index = DecontaminationIndex(
        effective_split={normalize_query_key(_SAMPLE_SINGLE_CALL["query"]): "test"}
    )
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.jsonl"
        out = Path(tmp) / "out.jsonl"
        inp.write_text(json.dumps(_SAMPLE_SINGLE_CALL, ensure_ascii=False), encoding="utf-8")

        converted, _, contaminated = convert_file(inp, out, index, "test")

        assert contaminated == 0
        assert converted == 1
