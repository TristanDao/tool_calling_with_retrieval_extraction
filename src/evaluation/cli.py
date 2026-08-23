"""Command-line interface for evaluating and comparing tool-calling predictions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.evaluation.compare import compare_methods
from src.evaluation.config import EvaluationConfig, NormalizationConfig
from src.evaluation.evaluator import evaluate

_METHOD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization-config", type=Path)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--slice", action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true")


def _config(args: argparse.Namespace) -> EvaluationConfig:
    normalization = NormalizationConfig()
    if args.normalization_config is not None:
        normalization = NormalizationConfig.from_alias_file(args.normalization_config, normalization)
    defaults = EvaluationConfig()
    slice_fields = tuple(dict.fromkeys((*defaults.slice_fields, *args.slice)))
    ks = tuple(sorted(set(args.k)))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("All retrieval K values must be positive")
    if not 0 < args.confidence_level < 1:
        raise ValueError("confidence-level must be between 0 and 1")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be non-negative")
    return EvaluationConfig(
        retrieval_ks=ks,
        slice_fields=slice_fields,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        strict_coverage=not args.allow_missing,
        normalization=normalization,
    )


def _prediction_mapping(values: list[str], minimum_count: int = 2) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected METHOD=PATH, received: {value}")
        method, path = value.split("=", 1)
        if not _METHOD_NAME_RE.fullmatch(method) or not path or method in result:
            raise ValueError(f"Invalid or duplicate prediction mapping: {value}")
        result[method] = Path(path)
    if len(result) < minimum_count:
        raise ValueError(f"Expected at least {minimum_count} METHOD=PATH values")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese tool-calling predictions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    _common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--oracle-predictions", type=Path)
    compare_parser = subparsers.add_parser("compare")
    _common_arguments(compare_parser)
    compare_parser.add_argument("--prediction", action="append", required=True)
    compare_parser.add_argument("--oracle-prediction", action="append", default=[])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _config(args)
    if args.command == "evaluate":
        evaluate(
            gold_path=args.gold,
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            config=config,
            oracle_predictions_path=args.oracle_predictions,
        )
    else:
        prediction_paths = _prediction_mapping(args.prediction)
        oracle_paths = (
            _prediction_mapping(args.oracle_prediction, minimum_count=0)
            if args.oracle_prediction
            else {}
        )
        unknown_oracle_methods = sorted(set(oracle_paths) - set(prediction_paths))
        if unknown_oracle_methods:
            raise ValueError(f"Oracle predictions reference unknown methods: {unknown_oracle_methods}")
        compare_methods(
            gold_path=args.gold,
            prediction_paths=prediction_paths,
            output_dir=args.output_dir,
            config=config,
            oracle_prediction_paths=oracle_paths,
        )


if __name__ == "__main__":
    main()
