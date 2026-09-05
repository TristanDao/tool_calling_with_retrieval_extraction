"""Orchestrator for component, end-to-end, robustness, and efficiency metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.config import EvaluationConfig, NormalizationConfig
from src.evaluation.detection_metrics import compute_detection_metrics
from src.evaluation.efficiency_metrics import compute_efficiency_metrics
from src.evaluation.end_to_end_metrics import compute_end_to_end_metrics
from src.evaluation.extraction_metrics import compute_extraction_metrics
from src.evaluation.io import align_predictions, file_sha256, load_gold, load_predictions
from src.evaluation.normalization import ArgumentNormalizer
from src.evaluation.report import write_evaluation_outputs
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.robustness import compute_robustness_metrics
from src.evaluation.schema_validation import aggregate_schema_validity, validate_prediction_schema
from src.evaluation.selection_metrics import compute_selection_metrics
from src.evaluation.statistics import bootstrap_mean_interval
from src.evaluation.types import GoldRecord, PredictionRecord


def evaluate_records(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
    config: EvaluationConfig,
    oracle_predictions: list[PredictionRecord] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aligned = align_predictions(gold, predictions, config.strict_coverage)
    aligned_oracle = (
        align_predictions(gold, oracle_predictions, config.strict_coverage)
        if oracle_predictions is not None
        else None
    )
    normalizer = ArgumentNormalizer(config.normalization)
    detection = compute_detection_metrics(gold, aligned)
    retrieval = compute_retrieval_metrics(gold, aligned, config.retrieval_ks)
    selection = compute_selection_metrics(gold, aligned)
    extraction, extraction_rows = compute_extraction_metrics(gold, aligned, normalizer)
    oracle_extraction = None
    if aligned_oracle is not None:
        oracle_extraction, _ = compute_extraction_metrics(gold, aligned_oracle, normalizer)
    end_to_end, end_rows = compute_end_to_end_metrics(gold, aligned, normalizer)

    # Bản strict: cùng dữ liệu, tắt mọi nới lỏng mức giá trị. Chênh lệch giữa
    # `extraction` và `strict_extraction` = phần công normalizer đang gánh
    # (§11 method2_plan yêu cầu báo cáo cả hai).
    strict_normalizer = ArgumentNormalizer(NormalizationConfig.strict())
    strict_extraction, _ = compute_extraction_metrics(gold, aligned, strict_normalizer)
    strict_end_to_end, _ = compute_end_to_end_metrics(gold, aligned, strict_normalizer)
    strict_oracle_extraction = (
        compute_extraction_metrics(gold, aligned_oracle, strict_normalizer)[0]
        if aligned_oracle is not None
        else None
    )
    schema_rows = [
        validate_prediction_schema(record, prediction)
        for record, prediction in zip(gold, aligned, strict=True)
    ]
    schema_validity = aggregate_schema_validity(schema_rows)
    extraction_by_id = {item["id"]: item for item in extraction_rows}
    per_sample: list[dict[str, Any]] = []
    for record, prediction, end_row, schema_row in zip(
        gold, aligned, end_rows, schema_rows, strict=True
    ):
        extraction_row = extraction_by_id[record.id]
        row = {
            **end_row,
            "source": record.source,
            "parse_valid": prediction.parse_valid,
            "parse_error": prediction.parse_error,
            "schema_valid": schema_row["all_calls_valid"],
            "arg_exact_given_tool": extraction_row.get("arg_exact_given_tool"),
            "ranked_tools_available": prediction.ranked_tools is not None,
            "latency_ms": prediction.telemetry.get("latency_ms"),
            "cost_usd": prediction.telemetry.get("cost_usd"),
        }
        per_sample.append(row)
    robustness = compute_robustness_metrics(gold, per_sample, config.slice_fields)
    efficiency = compute_efficiency_metrics(
        aligned, [bool(item["overall_success"]) for item in per_sample]
    )
    positive_rows = [item for item in per_sample if item["is_positive"]]
    confidence_intervals = {
        "overall_success": bootstrap_mean_interval(
            [bool(item["overall_success"]) for item in per_sample],
            config.bootstrap_samples,
            config.confidence_level,
            config.seed,
        ),
        "n_fcem_positive": bootstrap_mean_interval(
            [bool(item["n_fcem"]) for item in positive_rows],
            config.bootstrap_samples,
            config.confidence_level,
            config.seed + 1,
        ),
        "tool_set_accuracy_positive": bootstrap_mean_interval(
            [bool(item["tool_set_exact"]) for item in positive_rows],
            config.bootstrap_samples,
            config.confidence_level,
            config.seed + 2,
        ),
    }
    report = {
        "evaluation_schema_version": "1.0",
        "dataset": {
            "sample_count": len(gold),
            "positive_count": sum(item.is_positive for item in gold),
            "negative_count": sum(not item.is_positive for item in gold),
            "source_counts": {
                source: sum(item.source == source for item in gold)
                for source in sorted({item.source for item in gold})
            },
        },
        "config": config.to_dict(),
        "metrics": {
            "detection": detection,
            "retrieval": retrieval,
            "selection": selection,
            "extraction": extraction,
            "oracle_extraction": oracle_extraction,
            "strict_extraction": strict_extraction,
            "strict_oracle_extraction": strict_oracle_extraction,
            "strict_end_to_end": strict_end_to_end,
            "schema_validity": schema_validity,
            "end_to_end": end_to_end,
            "robustness": robustness,
            "efficiency": efficiency,
            "confidence_intervals": confidence_intervals,
        },
    }
    return report, per_sample


def evaluate(
    gold_path: Path,
    predictions_path: Path,
    output_dir: Path,
    config: EvaluationConfig | None = None,
    oracle_predictions_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved_config = config or EvaluationConfig()
    gold = load_gold(gold_path)
    predictions = load_predictions(predictions_path)
    oracle_predictions = (
        load_predictions(oracle_predictions_path) if oracle_predictions_path is not None else None
    )
    report, per_sample = evaluate_records(
        gold,
        predictions,
        resolved_config,
        oracle_predictions,
    )
    report["inputs"] = {
        "gold_path": str(gold_path),
        "gold_sha256": file_sha256(gold_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        "oracle_predictions_path": (
            str(oracle_predictions_path) if oracle_predictions_path is not None else None
        ),
        "oracle_predictions_sha256": (
            file_sha256(oracle_predictions_path) if oracle_predictions_path is not None else None
        ),
    }
    write_evaluation_outputs(output_dir, report, per_sample)
    return report, per_sample
