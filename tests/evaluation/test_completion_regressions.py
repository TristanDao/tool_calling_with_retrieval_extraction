"""Regression cases for strict values, oracle denominators and repeated calls."""

from src.evaluation.config import EvaluationConfig
from src.evaluation.error_analysis import classify_errors
from src.evaluation.evaluator import evaluate_records
from src.evaluation.io import parse_gold_record, parse_prediction_record


def test_strict_preserves_whitespace_and_boolean_type() -> None:
    sample = {"id": "a", "query": "x", "source": "custom_vi", "tools": [],
              "function_calls": [{"name": "t", "arguments": {"x": " a ", "b": True}}]}
    prediction = {"id": "a", "function_calls": [{"name": "t", "arguments": {"x": "a", "b": 1}}]}
    report, _ = evaluate_records([parse_gold_record(sample)], [parse_prediction_record(prediction)],
                                  EvaluationConfig(bootstrap_samples=0))
    assert report["metrics"]["strict_end_to_end"]["n_fcem_positive"] == 0
    assert report["metrics"]["strict_extraction"]["normalized_arg_em_given_correct_tool"] == 0


def test_repeated_call_alignment_preserves_missing_call() -> None:
    calls = [{"name": "t", "arguments": {"x": value}} for value in (1, 2)]
    sample = {"id": "a", "query": "1 and 2", "source": "test", "function_calls": calls}
    errors = classify_errors(sample, {"function_calls": [calls[1]]}, {})
    assert len(errors) == 1
    assert errors[0]["reason"] == "missing_call"
    assert errors[0]["gold"] == {"x": 1}
    assert classify_errors(sample, {"function_calls": calls[::-1]}, {}) == []


def test_oracle_gap_uses_all_positive_samples() -> None:
    gold = [parse_gold_record({"id": str(i), "query": "x", "source": "test", "tools": [],
                               "function_calls": [{"name": "t", "arguments": {}}]}) for i in range(2)]
    oracle = [parse_prediction_record({"id": str(i), "function_calls": [{"name": "t", "arguments": {}}]}) for i in range(2)]
    predictions = [oracle[0], parse_prediction_record({"id": "1", "function_calls": []})]
    report, rows = evaluate_records(gold, predictions, EvaluationConfig(bootstrap_samples=0), oracle)
    assert report["metrics"]["oracle_pipeline_gap_positive"] == 0.5
    assert report["metrics"]["extraction"]["normalized_arg_em_given_correct_tool"] == 1
    assert all(row["oracle_fcem"] for row in rows)
