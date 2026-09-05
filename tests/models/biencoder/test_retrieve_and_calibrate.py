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


def test_calibrate_thresholds_auto_uses_the_actual_winner():
    """`strategy: auto` phải đọc `selection["winner"]`, không mặc định absolute."""
    from unittest.mock import patch

    from src.models.biencoder.evaluate import calibrate_thresholds

    rankings = [
        _ranking("m1", ["a", "b"], [("a", 0.9), ("b", 0.88), ("c", 0.2)]),
        _ranking("s1", ["a"], [("a", 0.9), ("b", 0.3), ("c", 0.1)]),
    ]

    with (
        patch("src.models.biencoder.evaluate.group_rows_by_sample", return_value=[]),
        patch("src.models.biencoder.evaluate.rank_samples", return_value=rankings),
    ):
        thresholds = calibrate_thresholds(
            retriever=None,
            val_path="unused.jsonl",
            config={"strategy": "auto"},
        )

    assert thresholds.strategy == thresholds.metrics["call_selection"]["winner"]


def test_calibrate_thresholds_non_auto_string_is_used_verbatim():
    """`strategy` khai rõ (không phải "auto") thì giữ nguyên, kể cả khi thua."""
    from unittest.mock import patch

    from src.models.biencoder.evaluate import calibrate_thresholds

    rankings = [_ranking("s1", ["a"], [("a", 0.9), ("b", 0.3)])]

    with (
        patch("src.models.biencoder.evaluate.group_rows_by_sample", return_value=[]),
        patch("src.models.biencoder.evaluate.rank_samples", return_value=rankings),
    ):
        thresholds = calibrate_thresholds(
            retriever=None,
            val_path="unused.jsonl",
            config={"strategy": "absolute"},
        )

    assert thresholds.strategy == "absolute"


def test_reconcile_strategy_fixes_mismatch_using_data_already_in_the_file():
    """Bug thật: config cũ hardcode "absolute" dù calibration tự chọn "gap".

    `thresholds.json` khi đó đã có sẵn `gap_delta` đúng — sửa chỉ cần đổi lại
    field `strategy`, không cần GPU, không cần chạy lại calibration.
    """
    from src.models.biencoder.evaluate import reconcile_strategy

    raw = {
        "strategy": "absolute",
        "gap_delta": 0.2,
        "metrics": {"call_selection": {"winner": "gap"}},
    }

    fixed, changed = reconcile_strategy(raw)

    assert changed
    assert fixed["strategy"] == "gap"
    assert fixed["gap_delta"] == 0.2
    assert raw["strategy"] == "absolute", "không được sửa in-place bản gốc"


def test_reconcile_strategy_leaves_already_correct_file_alone():
    from src.models.biencoder.evaluate import reconcile_strategy

    raw = {"strategy": "gap", "metrics": {"call_selection": {"winner": "gap"}}}

    fixed, changed = reconcile_strategy(raw)

    assert not changed
    assert fixed is raw


def test_reconcile_strategy_is_a_noop_without_a_recorded_winner():
    """File cũ (trước khi calibrate ghi `winner`) không được sửa mù quáng."""
    from src.models.biencoder.evaluate import reconcile_strategy

    raw = {"strategy": "absolute", "metrics": {}}

    fixed, changed = reconcile_strategy(raw)

    assert not changed
    assert fixed is raw


def _raw(sample_id: str, ranked: list[tuple[str, float]]) -> dict:
    return {
        "id": sample_id,
        "ranked_tools": [{"name": n, "score": s} for n, s in ranked],
    }


def _gold(sample_id: str, tools: list[str]) -> dict:
    return {"id": sample_id, "function_calls": [{"name": t} for t in tools]}


def test_replay_selection_reproduces_selection_without_a_model():
    """`ranked_tools` đã lưu kèm score nên đổi ngưỡng là tính lại được offline."""
    from src.models.biencoder.evaluate import replay_selection

    gold = [_gold("s1", ["a"]), _gold("s2", ["a", "b"])]
    raw = [
        _raw("s1", [("a", 0.90), ("b", 0.40)]),
        _raw("s2", [("a", 0.90), ("b", 0.85), ("c", 0.20)]),
    ]

    # gap 0.25: s1 giữ 1 tool (khoảng cách 0.50 > 0.25), s2 lấy cả 2 (0.05).
    result = replay_selection(
        gold, raw, RetrievalThresholds(tau=0.0, gap_delta=0.25, k_max=3, strategy="gap")
    )

    assert result["tool_set_accuracy_positive"] == 1.0
    assert result["n_positive"] == 2


def test_replay_selection_exposes_absolute_over_selection():
    """Đúng cơ chế nghi ngờ: τ_call thấp thì mọi candidate đều lọt.

    Hard negative của CustomTools-VI cùng `feature_group` nên điểm sát nhau —
    `absolute` gom cả cụm, `gap` thì không.
    """
    from src.models.biencoder.evaluate import replay_selection

    gold = [_gold("s1", ["a"])]
    raw = [_raw("s1", [("a", 0.80), ("b", 0.55), ("c", 0.50)])]

    absolute = replay_selection(
        gold, raw, RetrievalThresholds(tau=0.0, tau_call=0.34, k_max=3, strategy="absolute")
    )
    gap = replay_selection(
        gold, raw, RetrievalThresholds(tau=0.0, gap_delta=0.20, k_max=3, strategy="gap")
    )

    assert absolute["mean_selected_positive"] == 3.0
    assert absolute["tool_set_accuracy_positive"] == 0.0
    assert gap["mean_selected_positive"] == 1.0
    assert gap["tool_set_accuracy_positive"] == 1.0


def test_replay_selection_counts_abstention_on_negatives():
    from src.models.biencoder.evaluate import replay_selection

    gold = [_gold("n1", []), _gold("n2", [])]
    raw = [_raw("n1", [("a", 0.10)]), _raw("n2", [("a", 0.90)])]

    result = replay_selection(
        gold, raw, RetrievalThresholds(tau=0.35, gap_delta=0.2, strategy="gap")
    )

    assert result["negative_recall"] == 0.5
    assert result["n_negative"] == 2


def test_replay_selection_reports_samples_missing_from_raw():
    from src.models.biencoder.evaluate import replay_selection

    result = replay_selection(
        [_gold("s1", ["a"]), _gold("s2", ["b"])],
        [_raw("s1", [("a", 0.9)])],
        RetrievalThresholds(tau=0.0, gap_delta=0.2, strategy="gap"),
    )

    assert result["n_missing_raw"] == 1
    assert result["n_positive"] == 1
