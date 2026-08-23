"""Test decontamination theo normalized query giữa các split.

Rủi ro phương pháp luận chính: benchmark gốc chia split theo **sample** chứ
không theo query, nên cùng một query nằm ở cả val lẫn test. Dù không train trên
query đó, việc chọn checkpoint bằng val vẫn làm metric test lạc quan lên.
"""

import json

import pytest

from src.models.sources import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    DecontaminationIndex,
    SourceSpec,
    build_decontamination_index,
    iter_samples,
    normalize_query_key,
    pairwise_overlap,
)


def _source(tmp_path, key: str, split: str, queries: list[str]) -> SourceSpec:
    path = tmp_path / f"{key}.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"{key}_{i}", "query": q}, ensure_ascii=False)
            for i, q in enumerate(queries)
        ),
        encoding="utf-8",
    )
    return SourceSpec(key, path, split)


def test_normalize_query_key_folds_case_and_whitespace():
    assert normalize_query_key("Tìm quán   phở ") == normalize_query_key("tìm quán phở")
    assert normalize_query_key("Hà Nội") != normalize_query_key("Đà Nẵng")


def test_test_split_wins_over_val_and_train(tmp_path):
    shared = "Tìm quán phở bò"
    specs = (
        _source(tmp_path, "train", SPLIT_TRAIN, [shared, "chỉ có ở train"]),
        _source(tmp_path, "val", SPLIT_VAL, [shared, "chỉ có ở val"]),
        _source(tmp_path, "test", SPLIT_TEST, [shared]),
    )

    index = build_decontamination_index(specs)

    assert index.effective_split[normalize_query_key(shared)] == SPLIT_TEST
    assert index.effective_split[normalize_query_key("chỉ có ở val")] == SPLIT_VAL
    assert index.effective_split[normalize_query_key("chỉ có ở train")] == SPLIT_TRAIN
    assert index.stats["n_overlapping_queries"] == 1


def test_val_wins_over_train_when_test_absent(tmp_path):
    shared = "Đặt vé máy bay"
    specs = (
        _source(tmp_path, "train", SPLIT_TRAIN, [shared]),
        _source(tmp_path, "val", SPLIT_VAL, [shared]),
    )

    index = build_decontamination_index(specs)

    assert index.effective_split[normalize_query_key(shared)] == SPLIT_VAL


def test_iter_samples_drops_contaminated_rows_and_counts_them(tmp_path):
    shared = "Tìm quán phở bò"
    specs = (
        _source(tmp_path, "train", SPLIT_TRAIN, [shared, "riêng train"]),
        _source(tmp_path, "val", SPLIT_VAL, [shared]),
        _source(tmp_path, "test", SPLIT_TEST, [shared]),
    )
    index = build_decontamination_index(specs)

    kept = list(iter_samples(specs, decontamination=index))

    queries_per_split = {}
    for sample in kept:
        queries_per_split.setdefault(sample["_split"], set()).add(
            normalize_query_key(sample["query"])
        )
    # Test giữ nguyên; bản sao ở train và val bị loại.
    assert queries_per_split[SPLIT_TEST] == {normalize_query_key(shared)}
    assert queries_per_split[SPLIT_TRAIN] == {normalize_query_key("riêng train")}
    assert SPLIT_VAL not in queries_per_split

    report = index.report()
    assert report["rows_dropped_total"] == 2
    assert report["rows_dropped_by_transition"] == {"train->test": 1, "val->test": 1}


def test_no_pairwise_overlap_remains_after_decontamination(tmp_path):
    specs = (
        _source(tmp_path, "train", SPLIT_TRAIN, ["a", "b", "c"]),
        _source(tmp_path, "val", SPLIT_VAL, ["b", "d"]),
        _source(tmp_path, "test", SPLIT_TEST, ["c", "d", "e"]),
    )
    index = build_decontamination_index(specs)

    queries_per_split: dict[str, set[str]] = {}
    for sample in iter_samples(specs, decontamination=index):
        queries_per_split.setdefault(sample["_split"], set()).add(
            normalize_query_key(sample["query"])
        )

    overlap = pairwise_overlap(queries_per_split)
    assert set(overlap.values()) == {0}
    # test không mất gì; val chỉ còn phần không đụng test.
    assert queries_per_split[SPLIT_TEST] == {"c", "d", "e"}
    assert queries_per_split[SPLIT_VAL] == {"b"}
    assert queries_per_split[SPLIT_TRAIN] == {"a"}


def test_pairwise_overlap_detects_contamination():
    overlap = pairwise_overlap({"val": {"a", "b"}, "test": {"b", "c"}, "train": {"z"}})

    assert overlap["test∩val"] == 1
    assert overlap["train∩val"] == 0


def test_index_roundtrip(tmp_path):
    index = DecontaminationIndex(effective_split={"a": SPLIT_TEST}, stats={"n": 1})
    path = tmp_path / "decon.json"

    index.save(path)
    loaded = DecontaminationIndex.load(path)

    assert loaded.effective_split == {"a": SPLIT_TEST}
    assert loaded.stats == {"n": 1}
    assert loaded.report()["rows_dropped_total"] == 0
