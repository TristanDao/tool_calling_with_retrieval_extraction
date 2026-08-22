"""Test abstention, chọn số call và hiệu chỉnh ngưỡng (§5 method2_plan)."""

import json

import pytest

pytest.importorskip("numpy")

from src.models.biencoder.evaluate import (
    SampleRanking,
    calibrate_call_selection,
    calibrate_tau,
    group_rows_by_sample,
    retrieval_metrics,
)
from src.models.biencoder.retrieve import RetrievalThresholds, select_tools

RANKED = [("a", 0.90), ("b", 0.72), ("c", 0.40)]


def test_absolute_strategy_respects_threshold_and_k_max():
    thresholds = RetrievalThresholds(tau_call=0.70, k_max=3, strategy="absolute")

    assert select_tools(RANKED, thresholds) == ["a", "b"]
    assert select_tools(RANKED, RetrievalThresholds(tau_call=0.70, k_max=1)) == ["a"]


def test_absolute_strategy_always_returns_at_least_top1():
    # Đã qua abstention rồi thì phải gọi ít nhất một tool.
    thresholds = RetrievalThresholds(tau_call=0.99, k_max=3, strategy="absolute")

    assert select_tools(RANKED, thresholds) == ["a"]


def test_gap_strategy_adds_runner_up_only_when_close():
    close = RetrievalThresholds(gap_delta=0.25, k_max=3, strategy="gap")
    strict = RetrievalThresholds(gap_delta=0.10, k_max=3, strategy="gap")

    assert select_tools(RANKED, close) == ["a", "b"]
    assert select_tools(RANKED, strict) == ["a"]


def _ranking(sample_id: str, gold: list[str], ranked: list[tuple[str, float]]) -> SampleRanking:
    return SampleRanking(
        sample_id=sample_id,
        query="q",
        gold=gold,
        ranked=ranked,
        source="custom_vi",
        source_key="custom_train",
        tool_split="seen",
    )


def test_calibrate_tau_separates_positive_from_negative():
    rankings = [
        _ranking("p1", ["a"], [("a", 0.9), ("b", 0.2)]),
        _ranking("p2", ["a"], [("a", 0.8), ("b", 0.1)]),
        _ranking("n1", [], [("a", 0.3), ("b", 0.2)]),
        _ranking("n2", [], [("a", 0.2), ("b", 0.1)]),
    ]

    tau, metrics = calibrate_tau(rankings, [round(0.01 * i, 2) for i in range(101)])

    assert 0.30 < tau <= 0.80
    assert metrics["macro_f1"] == 1.0
    assert metrics["negative_recall"] == 1.0


def test_calibrate_call_selection_compares_both_strategies():
    rankings = [
        _ranking("m1", ["a", "b"], [("a", 0.9), ("b", 0.88), ("c", 0.2)]),
        _ranking("s1", ["a"], [("a", 0.9), ("b", 0.3), ("c", 0.1)]),
    ]

    result = calibrate_call_selection(
        rankings,
        tau=0.0,
        k_max=3,
        absolute_grid=[round(0.05 * i, 2) for i in range(21)],
        gap_grid=[round(0.01 * i, 2) for i in range(31)],
    )

    assert result["winner"] in ("absolute", "gap")
    # Gap-based tách được multi-call khỏi single-call trong ví dụ này.
    assert result["gap"]["tool_set_accuracy"] == 1.0


def test_retrieval_metrics_micro_and_full_recall():
    rankings = [
        _ranking("s1", ["a", "b"], [("a", 0.9), ("x", 0.5), ("b", 0.4)]),
        _ranking("s2", ["c"], [("x", 0.9), ("c", 0.5)]),
        _ranking("neg", [], [("x", 0.1)]),
    ]

    metrics = retrieval_metrics(rankings, k_values=(1, 3))

    assert metrics["n_positive_samples"] == 2
    assert metrics["n_gold_tools"] == 3
    assert metrics["micro_recall@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["micro_recall@3"] == 1.0
    assert metrics["full_recall@1"] == 0.0
    assert metrics["full_recall@3"] == 1.0
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_group_rows_by_sample_merges_multi_call_rows(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"sample_id": "s1", "query": "q", "positive": "a", "all_gold": ["a", "b"], "candidates": ["a", "b", "c"]},
        {"sample_id": "s1", "query": "q", "positive": "b", "all_gold": ["a", "b"], "candidates": ["a", "b", "c"]},
        {"sample_id": "s2", "query": "q2", "positive": None, "candidates": ["a"]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    samples = group_rows_by_sample(path)

    assert len(samples) == 2
    assert samples[0]["gold"] == ["a", "b"]
    assert samples[1]["gold"] == []


def test_thresholds_roundtrip(tmp_path):
    path = tmp_path / "thresholds.json"
    RetrievalThresholds(tau=0.42, tau_call=0.6, k_max=2, strategy="gap").save(path)

    loaded = RetrievalThresholds.load(path)

    assert loaded.tau == 0.42
    assert loaded.strategy == "gap"
    assert loaded.k_max == 2
