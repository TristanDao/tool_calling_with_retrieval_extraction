"""Test phân loại lỗi W/T/P/I (§9 experimental_plan) và tổng hợp latency."""

import pytest

from src.models.pipeline.method2 import (
    ERROR_INCOMPLETE,
    ERROR_PARAPHRASE,
    ERROR_TRANSLATION,
    ERROR_WRONG_VALUE,
    StageTimings,
    classify_errors,
    summarize_latency,
)

QUERY = "Tìm quán phở bò ở Hà Nội, giá dưới 50.000 đồng"


def _sample(gold_args: dict) -> dict:
    return {
        "id": "s1",
        "query": QUERY,
        "function_calls": [{"name": "vi_search_restaurants", "arguments": gold_args}],
    }


def _prediction(pred_args: dict) -> dict:
    return {"function_calls": [{"name": "vi_search_restaurants", "arguments": pred_args}]}


def _raw(status: str = "ok") -> dict:
    return {"calls": [{"name": "vi_search_restaurants", "status": status}]}


def _classes(errors: list[dict]) -> list[str]:
    return [e["error_class"] for e in errors]


def test_correct_prediction_yields_no_errors():
    errors = classify_errors(
        _sample({"location": "Hà Nội"}), _prediction({"location": "Hà Nội"}), _raw()
    )

    assert errors == []


def test_wrong_value_when_gold_is_verbatim_in_query():
    errors = classify_errors(
        _sample({"location": "Hà Nội"}), _prediction({"location": "phở bò"}), _raw()
    )

    assert _classes(errors) == [ERROR_WRONG_VALUE]


def test_translation_class_when_gold_not_verbatim_in_query():
    # Giá trị canonical tiếng Anh — giới hạn kiến trúc của span head, không phải
    # lỗi chất lượng model.
    errors = classify_errors(
        _sample({"country": "United States"}), _prediction({"country": "Hoa Kỳ"}), _raw()
    )

    assert _classes(errors) == [ERROR_TRANSLATION]


def test_paraphrase_class_for_substring_overlap():
    errors = classify_errors(
        _sample({"dish": "phở bò"}), _prediction({"dish": "phở"}), _raw()
    )

    assert _classes(errors) == [ERROR_PARAPHRASE]


def test_missing_param_is_incomplete():
    errors = classify_errors(_sample({"location": "Hà Nội"}), _prediction({}), _raw())

    assert ERROR_INCOMPLETE in _classes(errors)


def test_incomplete_status_recorded_once_plus_missing_param():
    errors = classify_errors(
        _sample({"location": "Hà Nội"}), _prediction({}), _raw(status="incomplete")
    )

    assert _classes(errors).count(ERROR_INCOMPLETE) == 2


def test_missed_and_hallucinated_calls():
    missed = classify_errors(_sample({"location": "Hà Nội"}), {"function_calls": []}, {"calls": []})
    hallucinated = classify_errors(
        {"id": "n1", "query": QUERY, "function_calls": []}, _prediction({}), _raw()
    )

    assert _classes(missed) == ["missed_call"]
    assert _classes(hallucinated) == ["hallucinated_call"]


def test_wrong_tool_when_both_sides_have_calls():
    sample = {"id": "s", "query": QUERY, "function_calls": [{"name": "tool_a", "arguments": {}}]}
    prediction = {"function_calls": [{"name": "tool_b", "arguments": {}}]}

    errors = classify_errors(sample, prediction, {"calls": []})

    assert _classes(errors) == ["wrong_tool", "wrong_tool"]


def test_latency_summary_splits_four_stages():
    timings = [
        StageTimings(t_query_embed=0.001, t_retrieve=0.002, t_cross_encode=0.010, t_validate=0.001),
        StageTimings(t_query_embed=0.002, t_retrieve=0.003, t_cross_encode=0.020, t_validate=0.002),
    ]

    report = summarize_latency(timings)

    assert report["n"] == 2
    assert set(report) >= {"t_query_embed", "t_retrieve", "t_cross_encode", "t_validate", "total"}
    assert report["t_cross_encode"]["mean_ms"] == pytest.approx(15.0)
    assert report["total"]["mean_ms"] == pytest.approx(20.5)
