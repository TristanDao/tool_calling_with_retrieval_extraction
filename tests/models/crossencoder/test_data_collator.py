"""Test cho bug §2.3b (padding/max_length) và schema-type routing."""

import pytest

torch = pytest.importorskip("torch")

from src.models.crossencoder.data_collator import (
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_NUMBER,
    SCHEMA_TYPE_STRING,
    CollatorConfig,
    CrossEncoderCollator,
    build_schema_question,
    iter_parameters,
    normalize_schema_type,
    raw_schema_type,
    resolve_param_type,
)

TOOL = {
    "name": "vi_order_food",
    "parameters": {
        "type": "object",
        "properties": {
            "dish": {"type": "string", "description": "Món ăn cần đặt"},
            "quantity": {"type": "integer", "description": "Số phần"},
            "platform": {
                "type": "string",
                "description": "Sàn",
                "enum": ["Shopee", "Lazada"],
            },
            "spicy": {"type": "boolean", "description": "Có cay không"},
            "toppings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["dish", "quantity"],
    },
}


def _collator(fake_tokenizer, **kwargs) -> CrossEncoderCollator:
    return CrossEncoderCollator(CollatorConfig(**kwargs), tokenizer=fake_tokenizer)


def test_enum_detected_from_enum_key_not_type_field() -> None:
    # JSON Schema viết enum là {"type": "string", "enum": [...]}, không phải "type": "enum".
    param = {"name": "platform", "type": "string", "enum": ["Shopee", "Lazada"]}
    assert resolve_param_type(param) == SCHEMA_TYPE_ENUM
    assert "Enum=Shopee|Lazada" in build_schema_question(
        {**param, "routing_type": SCHEMA_TYPE_ENUM}
    )


def test_iter_parameters_reads_json_schema_object() -> None:
    params = {p["name"]: p for p in iter_parameters(TOOL)}

    assert set(params) == {"dish", "quantity", "platform", "spicy", "toppings"}
    assert params["dish"]["required"] is True
    assert params["spicy"]["required"] is False
    assert params["quantity"]["routing_type"] == SCHEMA_TYPE_NUMBER
    assert params["quantity"]["value_type"] == "integer"
    assert params["platform"]["routing_type"] == SCHEMA_TYPE_ENUM
    assert params["toppings"]["routing_type"] == "array"


def test_iter_parameters_survives_glaive_malformed_required() -> None:
    # Glaive có sample ghi `required: true` thay vì list tên parameter.
    tool = {
        "name": "get_movie_details",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "required": True},
                "year": {"type": "integer"},
            },
            "required": True,
        },
    }

    params = {p["name"]: p for p in iter_parameters(tool)}

    assert params["title"]["required"] is True
    assert params["year"]["required"] is False


def test_iter_parameters_ignores_non_dict_parameters() -> None:
    assert list(iter_parameters({"name": "x", "parameters": None})) == []
    assert list(iter_parameters({"name": "x", "parameters": "oops"})) == []


def test_integer_routes_to_number_but_keeps_raw_type() -> None:
    assert normalize_schema_type("integer") == SCHEMA_TYPE_NUMBER
    assert raw_schema_type("integer") == "integer"
    assert normalize_schema_type(["string", "null"]) == SCHEMA_TYPE_STRING
    assert normalize_schema_type("List[str]") == "array"


def test_dynamic_padding_uses_batch_max_not_max_length(fake_tokenizer) -> None:
    collator = _collator(fake_tokenizer, max_length=256, padding="longest")
    params = list(iter_parameters(TOOL))
    batch = [
        collator.encode_one("Đặt phở", params[0]),
        collator.encode_one("Đặt hai phần bún bò Huế giao tới Cầu Giấy", params[1]),
    ]

    out = collator(batch)

    expected = max(len(b["input_ids"]) for b in batch)
    assert out["input_ids"].shape == (2, expected)
    assert expected < 256  # §2.3b: không còn pad tới max_length


def test_padding_max_length_still_available(fake_tokenizer) -> None:
    collator = _collator(fake_tokenizer, max_length=64, padding="max_length")
    param = next(iter_parameters(TOOL))
    batch = [collator.encode_one("Đặt phở", param)]

    out = collator(batch)

    assert out["input_ids"].shape == (1, 64)


def test_query_token_mask_covers_only_query_segment(fake_tokenizer) -> None:
    collator = _collator(fake_tokenizer)
    param = next(iter_parameters(TOOL))
    query = "Đặt hai phần phở"

    item = collator.encode_one(query, param)

    mask = item["query_token_mask"]
    n_query_tokens = len(query.split())
    assert sum(mask) == n_query_tokens
    assert mask[0] is False  # [CLS]
    assert all(mask[1 : n_query_tokens + 1])
    assert not any(mask[n_query_tokens + 1 :])  # [SEP] + schema question


def test_span_labels_are_clipped_into_padded_length(fake_tokenizer) -> None:
    collator = _collator(fake_tokenizer)
    param = next(iter_parameters(TOOL))
    labels = {
        "has_value": 1,
        "span_start": 1,
        "span_end": 999,
        "enum_label": 0,
        "boolean_label": 0,
        "schema_type": SCHEMA_TYPE_STRING,
    }

    out = collator([collator.encode_one("Đặt phở bò", param, labels)])

    max_len = out["input_ids"].shape[1]
    assert out["labels"]["span_end"].item() == max_len - 1
    assert out["labels"]["schema_type"] == [SCHEMA_TYPE_STRING]
