"""Tests for frozen paired benchmark rebuilding and experiment quotas."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.normalize_schema import normalize_tool
from src.data.prepare_experiments import _take_quota
from src.data.rebuild_benchmark import RevisionBuildConfig, build_revision


def _tool(name: str = "lookup") -> dict:
    return {
        "name": name,
        "description": "Look up a value.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }


def _record(sample_id: str, source: str, query: str, value: str | None) -> dict:
    return {
        "id": sample_id,
        "source": source,
        "query": query,
        "function_calls": (
            [{"name": "lookup", "arguments": {"value": value}}] if value is not None else []
        ),
        "tools": [_tool()],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_xlam_defaults_are_optional_and_flat_properties_are_not_misread():
    standard = normalize_tool(
        {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "optional": {"type": "string", "default": "x"},
                    "required": {"type": "string"},
                },
                "required": ["optional", "required"],
            },
        },
        dataset="xlam",
    )
    assert standard["parameters"]["required"] == ["required"]

    flat = normalize_tool(
        {
            "name": "zipcodesbyids",
            "parameters": {
                "properties": {"type": "str", "default": "zip"},
            },
        },
        dataset="xlam",
    )
    assert list(flat["parameters"]["properties"]) == ["properties"]


def test_rebuild_pairs_deduplicates_scenarios_and_freezes_splits(tmp_path: Path):
    en_glaive = [
        _record("glaive_00001", "glaive", "same query", "one"),
        _record("glaive_00002", "glaive", "same query", "one"),
        _record("glaive_00003", "glaive", "another query", "three"),
        _record("glaive_00004", "glaive", "no tool", None),
    ]
    en_xlam = [
        _record("xlam_00001", "xlam", "x one", "one"),
        _record("xlam_00002", "xlam", "x two", "two"),
        _record("xlam_00003", "xlam", "x three", "three"),
        _record("xlam_00004", "xlam", "x four", "four"),
    ]
    vi_glaive = [
        {**row, "query": row["query"].replace("same query", "cùng truy vấn").replace("another query", "truy vấn khác").replace("no tool", "không có công cụ")}
        for row in en_glaive
    ]
    vi_xlam = [{**row, "query": f"VI {row['query']}"} for row in en_xlam]

    paths = {}
    for name, rows in (
        ("en_glaive", en_glaive),
        ("en_glaive_negative", []),
        ("en_xlam", en_xlam),
        ("vi_glaive", vi_glaive),
        ("vi_glaive_negative", []),
        ("vi_xlam", vi_xlam),
    ):
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, rows)
        paths[name] = path

    config = RevisionBuildConfig(
        input_en_glaive=paths["en_glaive"],
        input_en_glaive_negative=paths["en_glaive_negative"],
        input_en_xlam=paths["en_xlam"],
        input_vi_glaive=paths["vi_glaive"],
        input_vi_glaive_negative=paths["vi_glaive_negative"],
        input_vi_xlam=paths["vi_xlam"],
        benchmark_root=tmp_path / "benchmark_core",
        active_dir=tmp_path / "benchmark_vi",
        revision="test",
        feature_group_cache=None,
        promote=False,
    )
    build_revision(config)

    revision = config.benchmark_root / "test"
    metadata = json.loads((revision / "metadata.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((revision / "split_manifest.json").read_text(encoding="utf-8"))
    assert metadata["deduplication"]["removed"] == 1
    assert len(split_manifest) == 7
    assert set(metadata["counts"]["en"]["label"]) == {"negative", "positive"}
    assert (revision / "en" / "train.jsonl").exists()
    assert (revision / "vi" / "test.jsonl").exists()


def test_quota_falls_back_only_when_requested_class_is_unavailable():
    rows = [
        _record("glaive_00001", "glaive", "one", "one"),
        _record("xlam_00001", "xlam", "two", "two"),
        _record("glaive_00002", "glaive", "three", None),
    ]
    selected, stats = _take_quota(
        rows,
        total=3,
        positive_by_source={"glaive": 2, "xlam": 2},
        negative_target=2,
        seed=42,
    )
    assert len(selected) == 3
    assert stats["selected_negative"] == 1
    assert stats["fallback_used"] is True
