"""Test canonicalize tool pool (§1.1 method2_plan)."""

import json

import pytest

pytest.importorskip("torch")

from src.models.biencoder.tool_pool import (
    ToolPoolConfig,
    build_document_text,
    build_tool_pool,
    load_tool_pool,
)
from src.models.sources import SourceSpec


def _tool(description: str, properties: dict, required: list[str], group: str = "Khác") -> dict:
    return {
        "name": "search_books",
        "description": description,
        "feature_group": group,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


@pytest.fixture
def source_file(tmp_path):
    def _make(samples: list[dict], key: str = "s", split: str = "train", **kwargs) -> SourceSpec:
        path = tmp_path / f"{key}.jsonl"
        path.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples), encoding="utf-8"
        )
        return SourceSpec(key, path, split, **kwargs)

    return _make


def test_variants_of_one_tool_merge_into_one_entry(source_file, tmp_path):
    # Glaive: cùng tool_name nhưng description/parameters khác nhau do dịch khác.
    samples = [
        {
            "id": "a",
            "query": "tìm sách",
            "source": "glaive",
            "function_calls": [{"name": "search_books", "arguments": {"title": "Dế Mèn"}}],
            "tools": [_tool("Tìm sách", {"title": {"type": "string"}}, ["title"])],
        },
        {
            "id": "b",
            "query": "tìm sách khác",
            "source": "glaive",
            "function_calls": [{"name": "search_books", "arguments": {"title": "Số đỏ"}}],
            "tools": [_tool("Tìm sách", {"title": {"type": "string"}}, ["title"])],
        },
        {
            "id": "c",
            "query": "tìm sách nữa",
            "source": "glaive",
            "function_calls": [],
            "tools": [
                _tool(
                    "Tra cứu sách theo tiêu đề",
                    {"title": {"type": "string"}, "author": {"type": "string"}},
                    [],
                )
            ],
        },
    ]
    spec = source_file(samples)
    config = ToolPoolConfig(
        output_path=tmp_path / "pool.json", stats_path=tmp_path / "stats.json"
    )

    pool, stats = build_tool_pool((spec,), config)

    assert stats["n_tools"] == 1
    assert stats["n_signatures_before_merge"] == 2
    tool = pool[0]
    # Description phổ biến nhất thắng (2 lần vs 1 lần).
    assert tool["description"] == "Tìm sách"
    # Parameters là union theo key.
    assert set(tool["parameters"]["properties"]) == {"title", "author"}
    # `title` required ở 2/3 biến thể → giữ required; `author` thì không.
    assert tool["parameters"]["required"] == ["title"]


def test_gold_in_splits_tracks_where_a_tool_was_supervised(source_file, tmp_path):
    train_spec = source_file(
        [
            {
                "id": "t1",
                "query": "q",
                "function_calls": [{"name": "search_books", "arguments": {}}],
                "tools": [_tool("Tìm sách", {"title": {"type": "string"}}, [])],
            }
        ],
        key="train",
        split="train",
    )
    test_spec = source_file(
        [
            {
                "id": "t2",
                "query": "q2",
                "function_calls": [{"name": "unseen_tool", "arguments": {}}],
                "tools": [
                    {
                        "name": "unseen_tool",
                        "description": "Tool chưa từng train",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    }
                ],
            }
        ],
        key="test",
        split="test",
    )

    pool, _ = build_tool_pool(
        (train_spec, test_spec),
        ToolPoolConfig(output_path=tmp_path / "p.json", stats_path=tmp_path / "s.json"),
    )
    by_name = {t["name"]: t for t in pool}

    # Tool unseen PHẢI có trong pool (để retrieve được), nhưng không được gold ở train.
    assert by_name["unseen_tool"]["gold_in_splits"] == ["test"]
    assert by_name["search_books"]["gold_in_splits"] == ["train"]


def test_document_text_toggles_param_names():
    tool = _tool("Tìm sách", {"title": {"type": "string"}, "author": {"type": "string"}}, [])

    with_params = build_document_text(tool, include_param_names=True)
    without_params = build_document_text(tool, include_param_names=False)

    assert "title" in with_params and "author" in with_params
    assert "title" not in without_params
    assert without_params == "search_books. Tìm sách"


def test_load_tool_pool_returns_name_index(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps([{"name": "a"}, {"name": "b"}]), encoding="utf-8")

    pool = load_tool_pool(path)

    assert set(pool) == {"a", "b"}
