"""Tests for normalize_en.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.data.normalize_en import _parse_glaive, _parse_xlam, normalize_dataset


def test_parse_glaive_basic():
    raw = {
        "system": 'SYSTEM: {"name": "search_tutors", "description": "Find tutors", '
                  '"parameters": {"type": "object", "properties": {}}}',
        "chat": (
            "USER: Find a Math tutor\n\n"
            "A: <functioncall> {\"name\": \"search_tutors\", \"arguments\": "
            "'{\"subject\": \"Math\"}'} <|endoftext|>\n\n"
            "FUNCTION RESPONSE: [...]\n\n"
            "A: Done <|endoftext|>"
        ),
    }
    sample = _parse_glaive(raw, 0)
    assert sample is not None
    assert sample["id"] == "glaive_00000"
    assert sample["source"] == "glaive"
    assert sample["query"] == "Find a Math tutor"
    assert len(sample["function_calls"]) == 1
    assert sample["function_calls"][0]["name"] == "search_tutors"
    assert sample["function_calls"][0]["arguments"] == {"subject": "Math"}
    assert len(sample["tools"]) == 1
    assert sample["tools"][0]["name"] == "search_tutors"
    assert sample["tools"][0]["description"] == "Find tutors"


def test_parse_glaive_multi_call():
    raw = {
        "system": 'SYSTEM: {"name": "multi", "description": "x", "parameters": {"type": "object", "properties": {}}}',
        "chat": (
            "USER: Find flights and hotels\n\n"
            "A: <functioncall> {\"name\": \"search_flights\", \"arguments\": "
            "'{\"origin\": \"HN\"}'} <|endoftext|>\n\n"
            "A: <functioncall> {\"name\": \"search_hotels\", \"arguments\": "
            "'{\"city\": \"HN\"}'} <|endoftext|>\n\n"
            "FUNCTION RESPONSE: []\n\n"
            "A: Done <|endoftext|>"
        ),
    }
    sample = _parse_glaive(raw, 0)
    assert sample is not None
    assert len(sample["function_calls"]) == 2
    assert sample["function_calls"][0]["name"] == "search_flights"
    assert sample["function_calls"][1]["name"] == "search_hotels"


def test_parse_glaive_no_fc_returns_none():
    raw = {
        "system": "",
        "chat": "USER: Hi\n\nA: Hello! <|endoftext|>",
    }
    assert _parse_glaive(raw, 0) is None


def test_parse_glaive_second_turn_fc_only_returns_none():
    raw = {
        "system": 'SYSTEM: {"name": "f", "description": "x", "parameters": {"type": "object", "properties": {}}}',
        "chat": (
            "USER: Hi\n\n"
            "A: Hello! <|endoftext|>\n\n"
            "USER: Do something\n\n"
            "A: <functioncall> {\"name\": \"f\", \"arguments\": '{}'} <|endoftext|>"
        ),
    }
    assert _parse_glaive(raw, 0) is None


def test_parse_glaive_no_system_tool_adds_stub():
    raw = {
        "system": "",
        "chat": (
            "USER: Call func\n\n"
            "A: <functioncall> {\"name\": \"some_function\", \"arguments\": '{}'} <|endoftext|>"
        ),
    }
    sample = _parse_glaive(raw, 0)
    assert sample is not None
    assert len(sample["tools"]) == 1
    assert sample["tools"][0]["name"] == "some_function"
    assert sample["tools"][0]["description"] == ""


def test_parse_xlam_basic():
    raw = {
        "id": 0,
        "query": "Where can I find live giveaways?",
        "answers": json.dumps([
            {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
        ]),
        "tools": json.dumps([{
            "name": "live_giveaways_by_type",
            "description": "Find giveaways",
            "parameters": {
                "type": {"type": "str", "description": "Type of giveaway"},
            },
        }]),
    }
    sample = _parse_xlam(raw, 0)
    assert sample is not None
    assert sample["id"] == "xlam_00000"
    assert sample["source"] == "xlam"
    assert sample["query"] == "Where can I find live giveaways?"
    assert len(sample["function_calls"]) == 1
    assert sample["tools"][0]["parameters"]["properties"]["type"]["type"] == "string"


def test_parse_xlam_multi_call():
    raw = {
        "id": 0,
        "query": "Find beta and game giveaways",
        "answers": json.dumps([
            {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
            {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
        ]),
        "tools": json.dumps([{
            "name": "live_giveaways_by_type",
            "description": "Find giveaways",
            "parameters": {"type": {"type": "str", "description": "Type"}},
        }]),
    }
    sample = _parse_xlam(raw, 0)
    assert sample is not None
    assert len(sample["function_calls"]) == 2


def test_parse_xlam_type_optional():
    raw = {
        "id": 0,
        "query": "test",
        "answers": json.dumps([{"name": "func", "arguments": {}}]),
        "tools": json.dumps([{
            "name": "func",
            "description": "",
            "parameters": {
                "x": {"type": "str, optional", "description": "desc"},
            },
        }]),
    }
    sample = _parse_xlam(raw, 0)
    assert sample is not None
    params = sample["tools"][0]["parameters"]
    assert "x" in params["properties"]
    assert params["properties"]["x"]["type"] == "string"
    assert "x" not in params.get("required", [])


def test_parse_xlam_empty_query():
    raw = {"id": 0, "query": "", "answers": "[]", "tools": "[]"}
    assert _parse_xlam(raw, 0) is None


def test_parse_xlam_no_answers():
    raw = {"id": 0, "query": "test", "answers": "[]", "tools": "[]"}
    assert _parse_xlam(raw, 0) is None


def test_normalize_dataset_glaive():
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.jsonl"
        out = Path(tmp) / "out.jsonl"
        inp.write_text(json.dumps({
            "system": 'SYSTEM: {"name": "f", "description": "desc", "parameters": {"type": "object", "properties": {}}}',
            "chat": (
                "USER: Hi\n\n"
                "A: <functioncall> {\"name\": \"f\", \"arguments\": '{}'} <|endoftext|>"
            ),
        }) + "\n", encoding="utf-8")

        stats = normalize_dataset(inp, out, _parse_glaive, "glaive")
        assert stats["source"] == "glaive"
        assert stats["parsed"] == 1
        assert stats["total_raw"] == 1

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        sample = json.loads(lines[0])
        assert sample["id"] == "glaive_00000"
        assert sample["query"] == "Hi"


def test_normalize_dataset_skips():
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.jsonl"
        out = Path(tmp) / "out.jsonl"
        inp.write_text(
            json.dumps({"system": "", "chat": "USER: Hi\n\nA: Hello!"}) + "\n",  # no FC
            encoding="utf-8",
        )
        stats = normalize_dataset(inp, out, _parse_glaive, "glaive")
        assert stats["parsed"] == 0
        assert stats["skipped_no_fc"] == 1
