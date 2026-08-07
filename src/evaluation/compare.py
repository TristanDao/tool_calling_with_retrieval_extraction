"""Evaluation and paired comparison for multiple tool-calling methods."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from src.evaluation.config import EvaluationConfig
from src.evaluation.evaluator import evaluate_records
from src.evaluation.io import file_sha256, load_gold, load_predictions
from src.evaluation.report import write_evaluation_outputs
from src.evaluation.statistics import exact_mcnemar, paired_bootstrap_difference


def _headline(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    oracle = metrics["oracle_extraction"]
    return {
        "call_f1": metrics["detection"]["f1"],
        "recall_at_5": metrics["retrieval"].get("recall_at_5"),
        "tool_set_accuracy_positive": metrics["selection"]["tool_set_accuracy_positive"],
        "normalized_arg_em_given_correct_tool": metrics["extraction"][
            "normalized_arg_em_given_correct_tool"
        ],
        "oracle_tool_arg_em": (
            oracle["normalized_arg_em_given_correct_tool"] if oracle is not None else None
        ),
        "n_fcem_positive": metrics["end_to_end"]["n_fcem_positive"],
        "overall_success": metrics["end_to_end"]["overall_success"],
        "schema_validity": metrics["schema_validity"]["call_schema_validity"],
        "p95_latency_ms": metrics["efficiency"]["latency_ms"]["p95"],
        "cost_per_correct_usd": metrics["efficiency"]["cost"]["usd_per_correct_call"],
    }


def _write_comparison_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["method", *[key for key in rows[0] if key != "method"]] if rows else ["method"]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Method Comparison", "", "| " + " | ".join(fields) + " |"]
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(fields) - 1)) + "|")
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            values.append("—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_methods(
    gold_path: Path,
    prediction_paths: dict[str, Path],
    output_dir: Path,
    config: EvaluationConfig | None = None,
    oracle_prediction_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    resolved_config = config or EvaluationConfig()
    gold = load_gold(gold_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_reports: dict[str, Any] = {}
    method_rows: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for method, prediction_path in prediction_paths.items():
        predictions = load_predictions(prediction_path)
        oracle_path = (oracle_prediction_paths or {}).get(method)
        oracle_predictions = load_predictions(oracle_path) if oracle_path is not None else None
        report, per_sample = evaluate_records(
            gold,
            predictions,
            resolved_config,
            oracle_predictions,
        )
        report["inputs"] = {
            "gold_path": str(gold_path),
            "gold_sha256": file_sha256(gold_path),
            "predictions_path": str(prediction_path),
            "predictions_sha256": file_sha256(prediction_path),
            "oracle_predictions_path": str(oracle_path) if oracle_path is not None else None,
            "oracle_predictions_sha256": file_sha256(oracle_path) if oracle_path is not None else None,
        }
        write_evaluation_outputs(output_dir / method, report, per_sample)
        method_reports[method] = report
        method_rows[method] = per_sample
        summary_rows.append({"method": method, **_headline(report)})
    pairwise: dict[str, Any] = {}
    for first_name, second_name in combinations(prediction_paths, 2):
        first_by_id = {row["id"]: row for row in method_rows[first_name]}
        second_by_id = {row["id"]: row for row in method_rows[second_name]}
        common_ids = [record.id for record in gold if record.id in first_by_id and record.id in second_by_id]
        first_success = [bool(first_by_id[item]["overall_success"]) for item in common_ids]
        second_success = [bool(second_by_id[item]["overall_success"]) for item in common_ids]
        positive_ids = [item for item in common_ids if first_by_id[item]["is_positive"]]
        first_fcem = [bool(first_by_id[item]["n_fcem"]) for item in positive_ids]
        second_fcem = [bool(second_by_id[item]["n_fcem"]) for item in positive_ids]
        pairwise[f"{first_name}__vs__{second_name}"] = {
            "overall_success": {
                "mcnemar": exact_mcnemar(first_success, second_success),
                "paired_bootstrap_difference_first_minus_second": paired_bootstrap_difference(
                    first_success,
                    second_success,
                    resolved_config.bootstrap_samples,
                    resolved_config.confidence_level,
                    resolved_config.seed,
                ),
            },
            "n_fcem_positive": {
                "mcnemar": exact_mcnemar(first_fcem, second_fcem),
                "paired_bootstrap_difference_first_minus_second": paired_bootstrap_difference(
                    first_fcem,
                    second_fcem,
                    resolved_config.bootstrap_samples,
                    resolved_config.confidence_level,
                    resolved_config.seed + 1,
                ),
            },
        }
    comparison = {
        "config": resolved_config.to_dict(),
        "methods": {method: _headline(report) for method, report in method_reports.items()},
        "pairwise": pairwise,
    }
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)
    _write_comparison_table(output_dir, summary_rows)
    return comparison
