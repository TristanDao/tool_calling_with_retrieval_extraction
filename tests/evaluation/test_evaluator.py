"""Integration tests for end-to-end reports and schema validity."""

from __future__ import annotations

import json

import pytest
from src.evaluation.config import EvaluationConfig
from src.evaluation.evaluator import evaluate, evaluate_records
from src.evaluation.io import align_predictions


def test_evaluate_records_full_protocol(gold_records, prediction_records) -> None:
    config = EvaluationConfig(
        retrieval_ks=(1, 3, 5),
        slice_fields=("source", "metadata.tool_split", "call_count"),
        bootstrap_samples=100,
        seed=7,
    )
    report, rows = evaluate_records(
        gold_records,
        prediction_records,
        config,
        oracle_predictions=prediction_records,
    )
    metrics = report["metrics"]
    assert metrics["end_to_end"]["n_fcem_positive"] == 1.0
    assert metrics["end_to_end"]["overall_success"] == 0.75
    assert metrics["schema_validity"]["prediction_parse_validity"] == 1.0
    assert metrics["schema_validity"]["call_schema_validity"] == 0.5
    assert metrics["oracle_extraction"]["normalized_arg_em_given_correct_tool"] == 1.0
    assert metrics["efficiency"]["latency_ms"]["p95"] == pytest.approx(38.5)
    assert metrics["efficiency"]["cost"]["usd_per_correct_call"] == pytest.approx(0.05 / 3)
    assert "metadata.tool_split" in metrics["robustness"]
    assert metrics["confidence_intervals"]["overall_success"]["estimate"] == 0.75
    assert len(rows) == 4


def test_strict_prediction_coverage(gold_records, prediction_records) -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        align_predictions(gold_records, prediction_records[:-1], strict_coverage=True)
    aligned = align_predictions(gold_records, prediction_records[:-1], strict_coverage=False)
    assert not aligned[-1].parse_valid


def test_evaluate_writes_all_outputs(tmp_path, gold_records, prediction_records) -> None:
    gold_path = tmp_path / "gold.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    raw_gold = []
    for item in gold_records:
        raw_gold.append(
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
        )
    raw_predictions = [
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
    gold_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in raw_gold),
        encoding="utf-8",
    )
    prediction_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in raw_predictions),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    evaluate(
        gold_path,
        prediction_path,
        output_dir,
        EvaluationConfig(bootstrap_samples=10),
    )
    assert (output_dir / "report.json").exists()
    assert (output_dir / "per_sample.jsonl").exists()
    assert (output_dir / "summary.md").exists()
    written = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert written["dataset"]["sample_count"] == 4
