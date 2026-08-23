"""Test cho bug §2.3c (boolean cue toàn query) và chính sách SKIP non-verbatim."""

import pytest

pytest.importorskip("torch")

from src.models.crossencoder.label_generator import (
    BOOLEAN_LABEL_FALSE,
    BOOLEAN_LABEL_TRUE,
    SKIP_LABEL,
    LabelGenerator,
    LabelGeneratorConfig,
    SkipLabel,
)

QUERY = "Tìm quán phở bò ở Hà Nội, giá dưới 50.000 đồng, giao có hỗ trợ hóa đơn, không cay"


@pytest.fixture
def generator(fake_tokenizer) -> LabelGenerator:
    return LabelGenerator(LabelGeneratorConfig(max_length=64), tokenizer=fake_tokenizer)


def test_two_boolean_params_in_one_query_get_their_own_gold_labels(generator) -> None:
    # Bug §2.3c: rule cue quét toàn query nên trả cùng một nhãn cho cả hai param.
    spicy = {"name": "spicy", "type": "boolean", "description": "Có chọn vị cay hay không"}
    invoice = {"name": "needs_invoice", "type": "boolean", "description": "Có hỗ trợ hóa đơn"}

    spicy_label = generator.generate(QUERY, spicy, False)
    invoice_label = generator.generate(QUERY, invoice, True)

    assert spicy_label["boolean_label"] == BOOLEAN_LABEL_FALSE
    assert invoice_label["boolean_label"] == BOOLEAN_LABEL_TRUE
    assert spicy_label["has_value"] == invoice_label["has_value"] == 1


def test_boolean_accepts_string_gold(generator) -> None:
    param = {"name": "spicy", "type": "boolean"}

    assert generator.generate(QUERY, param, "true")["boolean_label"] == BOOLEAN_LABEL_TRUE
    assert generator.generate(QUERY, param, "không")["boolean_label"] == BOOLEAN_LABEL_FALSE
    assert generator.generate(QUERY, param, "abc") == SKIP_LABEL


def test_boolean_cue_filter_is_opt_in(fake_tokenizer) -> None:
    strict = LabelGenerator(
        LabelGeneratorConfig(max_length=64, require_boolean_cue=True),
        tokenizer=fake_tokenizer,
    )
    param = {"name": "gift_wrap", "type": "boolean", "description": "Gói quà"}

    label = strict.generate("Đặt một cuốn sách giao tới Quận 1", param, True)

    assert isinstance(label, SkipLabel)
    assert label.reason == "boolean_no_cue"


def test_verbatim_string_span_points_at_the_value(generator) -> None:
    param = {"name": "location", "type": "string"}

    label = generator.generate(QUERY, param, "Hà Nội")

    assert label["has_value"] == 1
    assert QUERY[label["char_start"] : label["char_end"]] == "Hà Nội"
    assert label["span_start"] <= label["span_end"]


def test_non_verbatim_value_is_skipped_not_labelled_at_cls(generator) -> None:
    # Bản cũ trả has_value=1 với span (0,0) → dạy model trỏ vào [CLS].
    param = {"name": "max_price_vnd", "type": "integer"}

    label = generator.generate(QUERY, param, 50000)

    assert isinstance(label, SkipLabel)
    assert label.reason == "non_verbatim"


def test_optional_param_without_value_is_a_negative_sample(generator) -> None:
    param = {"name": "min_rating", "type": "number", "required": False}

    label = generator.generate(QUERY, param, None)

    assert label["has_value"] == 0


def test_required_param_without_value_is_skipped(generator) -> None:
    param = {"name": "dish", "type": "string", "required": True}

    label = generator.generate(QUERY, param, None)

    assert isinstance(label, SkipLabel)
    assert label.reason == "required_without_value"


def test_enum_matching_is_case_insensitive_and_skips_unknown(generator) -> None:
    param = {"name": "platform", "type": "string", "enum": ["Shopee", "Lazada"]}

    assert generator.generate(QUERY, param, "lazada")["enum_label"] == 1
    label = generator.generate(QUERY, param, "Tiki")
    assert isinstance(label, SkipLabel)
    assert label.reason == "enum_value_not_in_schema"


def test_array_and_object_marked_unsupported(generator) -> None:
    for ptype in ("array", "object"):
        label = generator.generate(QUERY, {"name": "x", "type": ptype}, ["a"])
        assert isinstance(label, SkipLabel)
        assert label.reason == "unsupported_type"


def test_relaxed_match_handles_case_and_spacing(generator) -> None:
    query = "Tìm quán ở  hà nội hôm nay"

    label = generator.generate(query, {"name": "location", "type": "string"}, "Hà Nội")

    assert label["has_value"] == 1
    assert query[label["char_start"] : label["char_end"]].lower() == "hà nội"
