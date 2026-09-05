"""Ngân sách token giữa query và schema question.

Hai lỗi thật, phát hiện lúc chạy Phase 3 trên Kaggle:

A. **Crash.** `truncation="only_first"` chỉ cắt query. Param có `description`
   dài (đo được 883 ký tự trong xLAM) cho question vượt `max_length`, cắt query
   về 0 vẫn không lọt → `Truncation error: Sequence to truncate too short to
   respect the provided max_length`. Nổ ở giữa epoch, sau khi smoke đã pass.

B. **Hỏng nhãn im lặng — nặng hơn.** `dataset._prepare` căn span theo
   `max_length` (query được 254 token), còn collator chỉ chừa
   `max_length - n_question - n_special`. Nhãn vượt biên bị
   `_collate_labels._clip` kẹp về vị trí hợp lệ nhưng SAI, nên span head học
   trên nhãn rác mà không có dấu hiệu gì.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.models.crossencoder.data_collator import (  # noqa: E402
    CollatorConfig,
    CrossEncoderCollator,
    build_schema_question,
)


@pytest.fixture(scope="module")
def collator():
    return CrossEncoderCollator(CollatorConfig(max_length=64, max_question_tokens=24))


def _param(desc: str, name: str = "target") -> dict:
    return {"name": name, "description": desc, "type": "string", "routing_type": "string"}


def test_question_longer_than_max_length_does_not_crash(collator):
    """Trước khi vá: tokenizer ném Truncation error ở đúng dòng này."""
    huge = _param("mô tả rất dài " * 200)

    item = collator.encode_one("tìm quán phở gần đây", huge)

    assert len(item["input_ids"]) <= collator.config.max_length


def test_question_is_capped_to_the_token_budget(collator):
    huge = _param("mô tả rất dài " * 200)

    text, n_tokens = collator._question_entry(huge)

    assert n_tokens <= collator.config.max_question_tokens
    assert len(text) < len(build_schema_question(huge))


def test_short_question_is_left_untouched(collator):
    param = _param("Giá trị mục tiêu cần tìm.")

    text, _ = collator._question_entry(param)

    assert text == build_schema_question(param)


def test_query_budget_shrinks_as_the_question_grows(collator):
    ngan = collator.query_token_budget(_param("ngắn."))
    dai = collator.query_token_budget(_param("mô tả dài hơn hẳn " * 5))

    assert dai < ngan
    assert ngan < collator.config.max_length


def test_budget_matches_what_the_tokenizer_actually_leaves_for_the_query(collator):
    """Ngân sách phải khớp số token query THẬT sau khi cắt.

    Lệch một token cũng đủ làm nhãn span cuối câu trỏ sai chỗ.
    """
    param = _param("mô tả vừa phải cho tham số này")
    query = "tìm nhà hàng " * 40  # cố tình dài hơn ngân sách để chắc chắn bị cắt

    item = collator.encode_one(query, param)
    n_query_tokens = sum(item["query_token_mask"])

    assert n_query_tokens == collator.query_token_budget(param)


def test_dataset_aligns_spans_against_the_collator_budget():
    """`_prepare` phải hỏi collator, không dùng thẳng `max_length`."""
    from pathlib import Path

    source = Path("src/models/crossencoder/dataset.py").read_text(encoding="utf-8")

    assert "self.collator.query_token_budget(row[\"param\"])" in source
    assert "budget + 2" in source, "aligner tự trừ 2 cho special token"
