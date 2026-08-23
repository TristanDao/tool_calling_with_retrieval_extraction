"""Test loader nguồn dữ liệu và chia split xác định."""

import json

from src.models.sources import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SourceSpec,
    build_manifest,
    hash_bucket,
    iter_samples,
    load_jsonl,
    sha256_file,
    write_jsonl,
)

RATIOS = {SPLIT_TRAIN: 0.8, SPLIT_VAL: 0.1, SPLIT_TEST: 0.1}


def _write(path, samples):
    path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples), encoding="utf-8"
    )
    return path


def test_hash_bucket_is_deterministic_and_respects_ratios():
    assert hash_bucket("abc", RATIOS) == hash_bucket("abc", RATIOS)

    buckets = [hash_bucket(f"id-{i}", RATIOS) for i in range(2000)]
    train_share = buckets.count(SPLIT_TRAIN) / len(buckets)
    assert 0.75 < train_share < 0.85
    assert set(buckets) == set(RATIOS)


def test_hash_split_on_query_keeps_duplicate_queries_together(tmp_path):
    # glaive_negative lặp cùng một query dưới nhiều id khác nhau. Chia theo id thì
    # query đó vừa ở train vừa ở test → bước dedupe sẽ xoá sạch khỏi train.
    duplicated = "Bạn có thể đặt giúp tôi một chuyến bay không?"
    samples = [
        {"id": f"neg_{i}", "query": duplicated, "function_calls": [], "tools": []}
        for i in range(40)
    ]
    path = _write(tmp_path / "neg.jsonl", samples)

    by_query = SourceSpec(
        "neg", path, SPLIT_TRAIN, negative_only=True, hash_split=RATIOS, hash_split_on="query"
    )
    by_id = SourceSpec(
        "neg", path, SPLIT_TRAIN, negative_only=True, hash_split=RATIOS, hash_split_on="id"
    )

    assert len({s["_split"] for s in iter_samples((by_query,))}) == 1
    assert len({s["_split"] for s in iter_samples((by_id,))}) > 1


def test_iter_samples_tags_source_split_and_tool_split(tmp_path):
    path = _write(tmp_path / "s.jsonl", [{"id": "a", "query": "q", "function_calls": []}])
    spec = SourceSpec("custom_test_unseen", path, SPLIT_TEST, tool_split="unseen")

    sample = next(iter(iter_samples((spec,))))

    assert sample["_source_key"] == "custom_test_unseen"
    assert sample["_split"] == SPLIT_TEST
    assert sample["_tool_split"] == "unseen"


def test_iter_samples_filters_by_split_and_limit(tmp_path):
    path = _write(tmp_path / "s.jsonl", [{"id": f"i{i}", "query": "q"} for i in range(10)])
    spec = SourceSpec("s", path, SPLIT_TRAIN)

    assert len(list(iter_samples((spec,), limit_per_source=3))) == 3
    assert list(iter_samples((spec,), splits={SPLIT_TEST})) == []


def test_iter_samples_skips_missing_files(tmp_path):
    spec = SourceSpec("missing", tmp_path / "nope.jsonl", SPLIT_TRAIN)

    assert list(iter_samples((spec,))) == []


def test_jsonl_roundtrip_and_manifest(tmp_path):
    path = tmp_path / "out" / "rows.jsonl"
    rows = [{"id": "a", "query": "Hà Nội"}, {"id": "b", "query": "Đà Nẵng"}]

    write_jsonl(path, rows)
    assert list(load_jsonl(path)) == rows

    manifest = build_manifest((SourceSpec("k", path, SPLIT_TRAIN),))
    assert manifest["k"]["present"] is True
    assert manifest["k"]["sha256"] == sha256_file(path)
