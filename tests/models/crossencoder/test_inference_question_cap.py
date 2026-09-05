"""Inference phải cắt schema question GIỐNG HỆT lúc train.

Bug thật, giết job Phase 5 ở tập `benchmark`:
`inference.py::_predict_chunk` gọi `build_schema_question` trực tiếp rồi
tokenize với `truncation="only_first"`. Param `country` của xLAM có
description dài 883 ký tự (461 token) — cắt query về 0 vẫn không lọt 256 nên
tokenizer ném `Truncation error`. `custom_seen`/`custom_unseen` chạy qua được
vì CustomTools-VI không có description dài như vậy; chỉ glaive/xLAM mới có.

Lỗi thứ hai, im lặng: nếu train cắt ở 96 token mà inference không cắt thì
model gặp question dài hơn hẳn những gì nó từng học (train/serve skew).
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402

from src.models.crossencoder.data_collator import (  # noqa: E402
    DEFAULT_MAX_QUESTION_TOKENS,
    CollatorConfig,
    CrossEncoderCollator,
    build_schema_question,
    cap_question,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)


def _huge_param() -> dict:
    return {
        "name": "country",
        "description": "Quốc gia cần lọc tin tức. " * 60,
        "type": "string",
        "routing_type": "string",
    }


def test_pair_encoding_would_crash_without_the_cap(tokenizer):
    """Tái hiện đúng exception đã giết job."""
    raw = build_schema_question(_huge_param())

    with pytest.raises(Exception, match="Truncation error"):
        tokenizer("tìm tin tức Hoa Kỳ", raw, max_length=256, truncation="only_first")


def test_capped_question_encodes_fine(tokenizer):
    capped, n_tokens = cap_question(tokenizer, build_schema_question(_huge_param()))

    encoded = tokenizer("tìm tin tức Hoa Kỳ", capped, max_length=256, truncation="only_first")

    assert n_tokens <= DEFAULT_MAX_QUESTION_TOKENS
    assert len(encoded["input_ids"]) <= 256


def test_train_and_inference_cap_identically(tokenizer):
    """Cùng một param phải cho cùng một chuỗi question ở cả hai đường.

    Lệch nhau là train/serve skew: model học trên bản cắt, chạy thật trên bản
    khác — không crash, chỉ kém đi mà không ai biết.
    """
    param = _huge_param()
    collator = CrossEncoderCollator(
        CollatorConfig(max_question_tokens=DEFAULT_MAX_QUESTION_TOKENS), tokenizer=tokenizer
    )

    train_side, _ = collator._question_entry(param)
    infer_side, _ = cap_question(tokenizer, build_schema_question(param))

    assert train_side == infer_side


def test_cache_returns_the_same_entry(tokenizer):
    cache: dict = {}
    param = _huge_param()

    first = cap_question(tokenizer, build_schema_question(param), cache=cache)
    second = cap_question(tokenizer, build_schema_question(param), cache=cache)

    assert first == second
    assert len(cache) == 1


def test_short_question_is_untouched(tokenizer):
    param = {"name": "city", "description": "Thành phố.", "type": "string",
             "routing_type": "string"}
    raw = build_schema_question(param)

    assert cap_question(tokenizer, raw)[0] == raw


def test_inference_config_defaults_to_the_training_cap():
    from src.models.crossencoder.inference import ExtractionConfig

    assert ExtractionConfig().max_question_tokens == DEFAULT_MAX_QUESTION_TOKENS


def test_both_configs_declare_the_same_cap():
    """`crossencoder.yaml` (train) và `pipeline.yaml` (inference) phải khớp.

    Đây là ràng buộc không thể suy ra từ code — hai file độc lập nhau, và lệch
    một con số là train/serve skew im lặng.
    """
    import yaml
    from pathlib import Path

    train = yaml.safe_load(Path("configs/method2/crossencoder.yaml").read_text(encoding="utf-8"))
    infer = yaml.safe_load(Path("configs/method2/pipeline.yaml").read_text(encoding="utf-8"))

    assert (
        train["model"]["max_question_tokens"] == infer["extraction"]["max_question_tokens"]
    ), "max_question_tokens lệch giữa train và inference"
