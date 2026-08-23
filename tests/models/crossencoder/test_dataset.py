"""Test pair builder + quy char span → token index (§1.4, §1.5 method2_plan)."""

import json

import pytest

torch = pytest.importorskip("torch")

from src.models.crossencoder.data_collator import CollatorConfig, CrossEncoderCollator
from src.models.crossencoder.dataset import (
    CrossEncoderDataset,
    CrossEncoderPairsConfig,
    build_pairs,
)
from src.models.sources import SourceSpec

TOOL = {
    "name": "vi_search_restaurants",
    "description": "Tìm quán ăn",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "Địa điểm"},
            "dish": {"type": "string", "description": "Món ăn"},
            "max_price_vnd": {"type": "integer", "description": "Giá tối đa"},
            "min_rating": {"type": "number", "description": "Điểm tối thiểu"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["location"],
    },
}

QUERY = "Tìm quán phở bò ở Hà Nội giá dưới 50.000 đồng"


@pytest.fixture
def source(tmp_path):
    sample = {
        "id": "s1",
        "query": QUERY,
        "source": "custom_vi",
        "function_calls": [
            {
                "name": "vi_search_restaurants",
                "arguments": {"location": "Hà Nội", "dish": "phở bò", "max_price_vnd": 50000},
            }
        ],
        "tools": [TOOL],
    }
    path = tmp_path / "src.jsonl"
    path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
    return SourceSpec("custom_train", path, "train", tool_split="seen")


def test_pairs_cover_every_param_including_absent_ones(source, tmp_path):
    config = CrossEncoderPairsConfig(output_dir=tmp_path, stats_path=tmp_path / "stats.json")

    rows_by_split, stats = build_pairs(config, (source,))

    rows = rows_by_split["train"]
    by_param = {r["param"]["name"]: r for r in rows}
    # location + dish căn được span; min_rating vắng mặt → has_value=0.
    assert by_param["location"]["labels"]["has_value"] == 1
    assert by_param["min_rating"]["labels"]["has_value"] == 0
    # max_price_vnd = 50000 nhưng query viết "50.000" → non-verbatim → SKIP.
    assert "max_price_vnd" not in by_param
    # array không được hỗ trợ → SKIP với lý do riêng để báo cáo coverage.
    assert "tags" not in by_param
    assert stats["skip_by_reason"]["non_verbatim"] == 1
    assert stats["unsupported_type_coverage"]["n_pairs"] == 1


def test_labels_on_disk_are_char_spans_not_token_indices(source, tmp_path):
    rows_by_split, _ = build_pairs(
        CrossEncoderPairsConfig(output_dir=tmp_path, stats_path=tmp_path / "s.json"), (source,)
    )

    location = next(r for r in rows_by_split["train"] if r["param"]["name"] == "location")

    labels = location["labels"]
    assert QUERY[labels["char_start"] : labels["char_end"]] == "Hà Nội"
    # Chưa gắn token index — chỉ tính ở dataset theo backbone thật.
    assert labels["span_start"] == 0 and labels["span_end"] == 0


def test_dataset_converts_char_span_to_token_span(source, tmp_path, fake_tokenizer):
    rows_by_split, _ = build_pairs(
        CrossEncoderPairsConfig(output_dir=tmp_path, stats_path=tmp_path / "s.json"), (source,)
    )
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_by_split["train"]),
        encoding="utf-8",
    )
    collator = CrossEncoderCollator(CollatorConfig(max_length=64), tokenizer=fake_tokenizer)

    dataset = CrossEncoderDataset(path, collator, max_length=64)

    location = next(r for r in dataset.rows if r["param"]["name"] == "location")
    labels = location["labels"]
    assert labels["span_start"] > 0
    assert labels["span_start"] <= labels["span_end"]
    # "Hà Nội" là 2 từ → 2 token với tokenizer word-level của test.
    assert labels["span_end"] - labels["span_start"] == 1


def test_dataset_batch_is_collatable(source, tmp_path, fake_tokenizer):
    rows_by_split, _ = build_pairs(
        CrossEncoderPairsConfig(output_dir=tmp_path, stats_path=tmp_path / "s.json"), (source,)
    )
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_by_split["train"]),
        encoding="utf-8",
    )
    collator = CrossEncoderCollator(CollatorConfig(max_length=64), tokenizer=fake_tokenizer)
    dataset = CrossEncoderDataset(path, collator, max_length=64)

    batch = collator([dataset[i] for i in range(len(dataset))])

    assert batch["input_ids"].shape[0] == len(dataset)
    assert len(batch["labels"]["schema_type"]) == len(dataset)
    assert batch["labels"]["span_end"].max().item() < batch["input_ids"].shape[1]
