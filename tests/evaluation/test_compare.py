"""Integration test for multi-method reports and paired comparisons."""

from __future__ import annotations

import json

from src.evaluation.compare import compare_methods
from src.evaluation.config import EvaluationConfig


def _write_records(path, records) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


def test_compare_methods_writes_tables_and_paired_tests(
    tmp_path,
    gold_records,
    prediction_records,
) -> None:
    gold_path = tmp_path / "gold.jsonl"
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"
    gold_json = [
        {
            "id": item.id,
            "source": item.source,
            "query": item.query,
            "function_calls": [
                {"name": call.name, "arguments": call.arguments} for call in item.function_calls
            ],
            "tools": list(item.tools),
            "metadata": item.metadata,
        }
        for item in gold_records
    ]
    first_json = [
        {
            "id": item.id,
            "function_calls": [
                {"name": call.name, "arguments": call.arguments} for call in item.function_calls
            ],
            "ranked_tools": list(item.ranked_tools) if item.ranked_tools is not None else None,
            "telemetry": item.telemetry,
        }
        for item in prediction_records
    ]
    second_json = json.loads(json.dumps(first_json))
    second_json[0]["function_calls"][0]["arguments"]["amount"] = 99
    _write_records(gold_path, gold_json)
    _write_records(first_path, first_json)
    _write_records(second_path, second_json)
    _write_records(oracle_path, first_json)
    output_dir = tmp_path / "comparison"
    comparison = compare_methods(
        gold_path,
        {"first": first_path, "second": second_path},
        output_dir,
        EvaluationConfig(bootstrap_samples=20),
        oracle_prediction_paths={"first": oracle_path},
    )
    assert comparison["methods"]["first"]["oracle_tool_arg_em"] == 1.0
    assert comparison["methods"]["second"]["oracle_tool_arg_em"] is None
    assert "first__vs__second" in comparison["pairwise"]
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "comparison.csv").exists()
    assert (output_dir / "comparison.md").exists()
