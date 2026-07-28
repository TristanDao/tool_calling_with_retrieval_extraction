"""Tests for normalize_schema, build_benchmark parsers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def test_normalize_type_basic():
    from src.data.normalize_schema import normalize_type
    assert normalize_type("str") == ("string", False)
    assert normalize_type("int") == ("integer", False)
    assert normalize_type("float") == ("number", False)
    assert normalize_type("bool") == ("boolean", False)
    assert normalize_type("list") == ("array", False)
    assert normalize_type("object") == ("object", False)
    assert normalize_type("") == ("string", False)
    assert normalize_type("xxx_unknown") == ("string", False)


def test_normalize_type_optional():
    from src.data.normalize_schema import normalize_type
    assert normalize_type("str, optional") == ("string", True)
    assert normalize_type("int, optional") == ("integer", True)
    assert normalize_type("bool, optional") == ("boolean", True)
    assert normalize_type("List[int], optional") == ("array", True)
    assert normalize_type("str,Optional") == ("string", True)


def test_normalize_type_list_typed():
    from src.data.normalize_schema import normalize_type
    assert normalize_type("List[int]") == ("array", False)
    assert normalize_type("List[str]") == ("array", False)
    assert normalize_type("List[float]") == ("array", False)
    assert normalize_type("List[Union[int, float]]") == ("array", False)


def test_normalize_parameter_schema_string():
    from src.data.normalize_schema import normalize_parameter_schema
    out, optional = normalize_parameter_schema("x", {"type": "str", "description": "abc"})
    assert out["type"] == "string"
    assert out["description"] == "abc"
    assert optional is False


def test_normalize_parameter_schema_int_optional():
    from src.data.normalize_schema import normalize_parameter_schema
    out, optional = normalize_parameter_schema("x", {"type": "int, optional", "description": "y"})
    assert out["type"] == "integer"
    assert optional is True


def test_normalize_parameter_schema_list():
    from src.data.normalize_schema import normalize_parameter_schema
    out, optional = normalize_parameter_schema("x", {"type": "List[int]", "description": "y"})
    assert out["type"] == "array"
    assert "items" in out
    assert out["items"]["type"] == "integer"
    assert optional is False


def test_normalize_parameter_schema_list_fallback():
    from src.data.normalize_schema import normalize_parameter_schema
    out, _ = normalize_parameter_schema("x", {"type": "list", "description": "y"})
    assert out["type"] == "array"
    assert out["items"]["type"] == "string"


def test_normalize_parameter_schema_list_union():
    from src.data.normalize_schema import normalize_parameter_schema
    out, _ = normalize_parameter_schema("x", {"type": "List[Union[int, float]]", "description": "y"})
    assert out["type"] == "array"
    assert out["items"]["type"] == "number"


def test_normalize_parameter_schema_list_of_list():
    from src.data.normalize_schema import normalize_parameter_schema
    out, _ = normalize_parameter_schema("x", {"type": "List[List[int]]", "description": "y"})
    assert out["type"] == "array"
    assert out["items"]["type"] == "array"
    assert out["items"]["items"]["type"] == "integer"


def test_normalize_parameter_schema_list_of_tuple():
    from src.data.normalize_schema import normalize_parameter_schema
    out, _ = normalize_parameter_schema("x", {"type": "List[Tuple[int, int]]", "description": "y"})
    assert out["type"] == "array"
    assert out["items"]["type"] == "array"


def test_normalize_parameter_schema_enum():
    from src.data.normalize_schema import normalize_parameter_schema
    out, _ = normalize_parameter_schema("x", {
        "type": "str",
        "enum": ["red", "green", "blue"],
        "description": "color",
    })
    assert out["type"] == "enum"
    assert out["enum"] == ["red", "green", "blue"]


def test_normalize_parameter_schema_default_empty_dropped():
    from src.data.normalize_schema import normalize_parameter_schema
    out, _ = normalize_parameter_schema("x", {"type": "bool", "default": "", "description": "y"})
    assert "default" not in out


def test_normalize_parameter_schema_default_kept():
    from src.data.normalize_schema import normalize_parameter_schema
    out, _ = normalize_parameter_schema("x", {"type": "str", "default": "hello", "description": "y"})
    assert out["default"] == "hello"


def test_normalize_xlam_tool_basic():
    from src.data.normalize_schema import normalize_xlam_tool
    tool = {
        "name": "search_tutors",
        "description": "Find tutors",
        "parameters": {
            "subject": {"type": "str", "description": "Subject"},
            "location": {"type": "str, optional", "description": "City"},
        },
    }
    out = normalize_xlam_tool(tool)
    assert out["name"] == "search_tutors"
    assert out["parameters"]["type"] == "object"
    assert "subject" in out["parameters"]["properties"]
    assert "location" in out["parameters"]["properties"]
    assert out["parameters"]["properties"]["subject"]["type"] == "string"
    assert out["parameters"]["properties"]["location"]["type"] == "string"
    assert out["parameters"]["required"] == ["subject"]


def test_normalize_xlam_tool_empty():
    from src.data.normalize_schema import normalize_xlam_tool
    out = normalize_xlam_tool({"name": "x", "description": ""})
    assert out["parameters"]["properties"] == {}
    assert "required" not in out["parameters"]


def test_normalize_glaive_tool_passthrough():
    from src.data.normalize_schema import normalize_glaive_tool
    tool = {
        "name": "search_tutors",
        "description": "Find tutors",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject"},
            },
            "required": ["subject"],
        },
    }
    out = normalize_glaive_tool(tool)
    assert out["name"] == "search_tutors"
    assert out["parameters"]["required"] == ["subject"]


def test_normalize_tool_auto_detect_xlam():
    from src.data.normalize_schema import normalize_tool
    xlam = {
        "name": "x",
        "description": "",
        "parameters": {"a": {"type": "str", "description": "A"}},
    }
    out = normalize_tool(xlam, dataset="auto")
    assert out["parameters"]["type"] == "object"
    assert "a" in out["parameters"]["properties"]


def test_normalize_tool_auto_detect_glaive():
    from src.data.normalize_schema import normalize_tool
    glaive = {
        "name": "x",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "string", "description": "A"}},
            "required": ["a"],
        },
    }
    out = normalize_tool(glaive, dataset="auto")
    assert out["parameters"]["required"] == ["a"]


# ============ build_benchmark parsers ============


def test_extract_function_calls_from_chat_simple():
    from src.data.build_benchmark import _extract_function_calls_from_chat
    chat = 'A: <functioncall> {"name": "search_tutors", "arguments": \'{"subject": "Math"}\'} <|endoftext|>'
    calls = _extract_function_calls_from_chat(chat)
    assert len(calls) == 1
    assert calls[0]["name"] == "search_tutors"
    assert calls[0]["arguments"] == {"subject": "Math"}


def test_extract_function_calls_from_chat_multicall():
    from src.data.build_benchmark import _extract_function_calls_from_chat
    chat = (
        'A: <functioncall> {"name": "search_tutors", "arguments": \'{"subject": "Math"}\'} <|endoftext|>\n\n'
        'A: <functioncall> {"name": "search_hotels", "arguments": \'{"city": "Hanoi"}\'} <|endoftext|>'
    )
    calls = _extract_function_calls_from_chat(chat)
    assert len(calls) == 2
    assert calls[0]["name"] == "search_tutors"
    assert calls[1]["name"] == "search_hotels"


def test_split_glaive_chat_turns_simple():
    from src.data.build_benchmark import _split_glaive_chat_turns
    chat = (
        "USER: Find tutors\n\n"
        "A: <functioncall> {\"name\": \"search_tutors\", \"arguments\": '{\"subject\": \"Math\"}'} <|endoftext|>\n\n"
        "FUNCTION RESPONSE: [{\"id\": 1}]\n\n"
        "A: I found 1 tutor. <|endoftext|>"
    )
    turns = _split_glaive_chat_turns(chat)
    roles = [t["role"] for t in turns]
    assert roles == ["user", "assistant", "function", "assistant"]
    assert turns[0]["content"] == "Find tutors"
    assert turns[1]["content"] is None
    assert len(turns[1]["function_calls"]) == 1
    assert turns[2]["name"] == "search_tutors"
    assert turns[3]["content"] == "I found 1 tutor."


def test_split_glaive_chat_turns_vi_markers():
    from src.data.build_benchmark import _split_glaive_chat_turns
    chat = (
        "NGƯỜI DÙNG: Tìm gia sư\n\n"
        "TRỢ LÝ: <functioncall> {\"name\": \"search_tutors\", \"arguments\": '{}'} <|endoftext|>\n\n"
        "PHẢN HỒI HÀM: []\n\n"
        "TRỢ LÝ: OK <|endoftext|>"
    )
    turns = _split_glaive_chat_turns(chat)
    roles = [t["role"] for t in turns]
    assert roles == ["user", "assistant", "function", "assistant"]


def test_split_glaive_chat_turns_no_function_call():
    from src.data.build_benchmark import _split_glaive_chat_turns
    chat = "USER: Hi\n\nA: Hello there!"
    turns = _split_glaive_chat_turns(chat)
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"
    assert turns[1]["content"] == "Hello there!"


def test_split_glaive_chat_turns_multi_call():
    from src.data.build_benchmark import _split_glaive_chat_turns
    chat = (
        "USER: Tìm chuyến bay và khách sạn\n\n"
        "A: <functioncall> {\"name\": \"search_flights\", \"arguments\": '{\"a\": 1}'} <|endoftext|>\n\n"
        "A: <functioncall> {\"name\": \"search_hotels\", \"arguments\": '{\"b\": 2}'} <|endoftext|>"
    )
    turns = _split_glaive_chat_turns(chat)
    assert len(turns) == 3
    assert turns[1]["role"] == "assistant"
    assert len(turns[1]["function_calls"]) == 1
    assert turns[1]["function_calls"][0]["name"] == "search_flights"
    assert turns[2]["function_calls"][0]["name"] == "search_hotels"


def test_extract_glaive_tool_from_system_en():
    from src.data.build_benchmark import _extract_glaive_tool_from_system
    system = 'SYSTEM: {"name": "search_tutors", "description": "Find tutors", "parameters": {}}'
    tool = _extract_glaive_tool_from_system(system)
    assert tool is not None
    assert tool["name"] == "search_tutors"


def test_extract_glaive_tool_from_system_vi():
    from src.data.build_benchmark import _extract_glaive_tool_from_system
    system = 'HỆ THỐNG: {"name": "search_tutors", "description": "Tìm gia sư", "parameters": {}}'
    tool = _extract_glaive_tool_from_system(system)
    assert tool is not None
    assert tool["name"] == "search_tutors"


def test_extract_glaive_tool_from_system_no_json():
    from src.data.build_benchmark import _extract_glaive_tool_from_system
    assert _extract_glaive_tool_from_system("HỆ THỐNG: Bạn không có quyền truy cập hàm") is None


def test_parse_glaive_sample_full():
    from src.data.build_benchmark import parse_glaive_sample
    raw = {
        "system": 'SYSTEM: {"name": "search_tutors", "description": "Find tutors", "parameters": {"type": "object", "properties": {"subject": {"type": "string", "description": "Subject"}}, "required": ["subject"]}}',
        "chat": (
            "USER: Find a Math tutor\n\n"
            "A: <functioncall> {\"name\": \"search_tutors\", \"arguments\": '{\"subject\": \"Math\"}'} <|endoftext|>\n\n"
            "FUNCTION RESPONSE: [{\"id\": 1, \"name\": \"John\"}]\n\n"
            "A: I found John. <|endoftext|>"
        ),
    }
    sample = parse_glaive_sample(raw, 0)
    assert sample is not None
    assert sample["id"] == "glaive_00000"
    assert sample["source"] == "glaive"
    assert len(sample["conversation"]) == 4
    assert sample["conversation"][0]["role"] == "user"
    assert sample["conversation"][0]["content"] == "Find a Math tutor"
    assert sample["conversation"][1]["role"] == "assistant"
    assert sample["conversation"][1]["function_calls"][0]["name"] == "search_tutors"
    assert len(sample["tools"]) == 1
    assert sample["tools"][0]["name"] == "search_tutors"


def test_parse_xlam_sample_basic():
    from src.data.build_benchmark import parse_xlam_sample
    raw = {
        "id": 0,
        "query": "Find live giveaways",
        "answers": json.dumps([
            {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
            {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
        ]),
        "tools": json.dumps([{
            "name": "live_giveaways_by_type",
            "description": "Find giveaways",
            "parameters": {
                "type": {"type": "str", "description": "Type", "default": "game"},
            },
        }]),
    }
    sample = parse_xlam_sample(raw, 0)
    assert sample is not None
    assert sample["source"] == "xlam"
    assert len(sample["conversation"]) == 2
    assert sample["conversation"][0]["content"] == "Find live giveaways"
    assert len(sample["conversation"][1]["function_calls"]) == 2
    assert sample["tools"][0]["parameters"]["properties"]["type"]["type"] == "string"
    assert sample["tools"][0]["parameters"]["required"] == ["type"]


def test_parse_xlam_sample_invalid_query():
    from src.data.build_benchmark import parse_xlam_sample
    raw = {"id": 0, "query": "", "answers": "[]", "tools": "[]"}
    assert parse_xlam_sample(raw, 0) is None


def test_parse_xlam_sample_invalid_answers():
    from src.data.build_benchmark import parse_xlam_sample
    raw = {"id": 0, "query": "x", "answers": "not json", "tools": "[]"}
    assert parse_xlam_sample(raw, 0) is None


def test_attach_feature_group():
    from src.data.build_benchmark import _attach_feature_group
    tools = [{"name": "a", "description": ""}, {"name": "b", "description": ""}]
    cache = {"a": "Tìm kiếm & Kết nối"}
    out = _attach_feature_group(tools, cache)
    assert out[0]["feature_group"] == "Tìm kiếm & Kết nối"
    assert out[1]["feature_group"] == "Khác"


def test_build_tool_pool_dedup():
    from src.data.build_benchmark import _build_tool_pool
    samples = [
        {"tools": [{"name": "a", "description": "A1", "parameters": {"type": "object", "properties": {"x": {"type": "string", "description": "X"}}}}]},
        {"tools": [{"name": "a", "description": "A2", "parameters": {"type": "object", "properties": {}}}]},
        {"tools": [{"name": "b", "description": "B", "parameters": {"type": "object", "properties": {}}}]},
    ]
    pool = _build_tool_pool(samples, {"a": "Tìm kiếm & Kết nối", "b": "Khác"})
    assert len(pool) == 2
    a_tool = next(t for t in pool if t["name"] == "a")
    assert a_tool["description"] == "A1"
    assert a_tool["parameters"]["properties"]["x"]["type"] == "string"


def test_validate_sample_ok():
    from src.data.build_benchmark import _validate_sample
    sample = {
        "id": "x_0",
        "source": "glaive",
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "function_calls": [{"name": "a", "arguments": {}}]},
            {"role": "function", "name": "a", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [
            {"name": "a", "description": "A", "feature_group": "Khác",
             "parameters": {"type": "object", "properties": {}}},
        ],
    }
    ok, errs = _validate_sample(sample)
    assert ok is True, errs


def test_validate_sample_missing_id():
    from src.data.build_benchmark import _validate_sample
    sample = {"conversation": [], "tools": []}
    ok, errs = _validate_sample(sample)
    assert ok is False
    assert "missing id" in errs


def test_validate_sample_function_call_not_in_tools():
    from src.data.build_benchmark import _validate_sample
    sample = {
        "id": "x_0",
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "function_calls": [{"name": "unknown", "arguments": {}}]},
        ],
        "tools": [],
    }
    ok, errs = _validate_sample(sample)
    assert ok is False
    assert any("not in tools" in e for e in errs)


def test_validate_sample_tool_not_snake_case():
    from src.data.build_benchmark import _validate_sample
    sample = {
        "id": "x_0",
        "conversation": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "BadName", "description": "x", "feature_group": "Khác",
             "parameters": {"type": "object", "properties": {}}},
        ],
    }
    ok, errs = _validate_sample(sample)
    assert ok is False
    assert any("snake_case" in e for e in errs)


def test_validate_sample_tool_missing_feature_group():
    from src.data.build_benchmark import _validate_sample
    sample = {
        "id": "x_0",
        "conversation": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "good_name", "description": "x", "parameters": {"type": "object", "properties": {}}},
        ],
    }
    ok, errs = _validate_sample(sample)
    assert ok is False
    assert any("feature_group" in e for e in errs)


def test_split_dataset():
    from src.data.build_benchmark import BuildConfig, _split_dataset
    cfg = BuildConfig(
        input_glaive=Path("/x"), input_xlam=Path("/x"),
        output_dir=Path("/x"), tool_schema_dir=Path("/x"),
        tool_pool_path=Path("/x"), train_path=Path("/x"),
        val_path=Path("/x"), test_path=Path("/x"),
        metadata_path=Path("/x"),
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        seed=42, shuffle=True, feature_group_config=Path("/x"),
    )
    samples = [{"id": f"s_{i}"} for i in range(100)]
    train, val, test = _split_dataset(samples, cfg)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    assert train[0]["id"] != samples[0]["id"] or len(set(s["id"] for s in train + val + test)) == 100


def test_load_failed_indices():
    from src.data.build_benchmark import _load_failed_indices
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "failed.jsonl"
        path.write_text('{"source_index": 5}\n{"source_index": 10}\n', encoding="utf-8")
        idx = _load_failed_indices(path)
        assert idx == {5, 10}


def test_load_failed_indices_nonexistent():
    from src.data.build_benchmark import _load_failed_indices
    assert _load_failed_indices(Path("/nonexistent.jsonl")) == set()


def test_safe_json_loads():
    from src.data.build_benchmark import _safe_json_loads
    assert _safe_json_loads('{"a": 1}') == {"a": 1}
    assert _safe_json_loads([1, 2, 3]) == [1, 2, 3]
    assert _safe_json_loads({"k": "v"}) == {"k": "v"}
    assert _safe_json_loads("not json") is None
    assert _safe_json_loads(None) is None
