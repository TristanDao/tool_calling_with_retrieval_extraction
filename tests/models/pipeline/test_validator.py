"""Test validator: coerce theo schema, phục hồi required, đánh dấu incomplete."""

from dataclasses import dataclass

import pytest

pytest.importorskip("jsonschema")

from src.models.pipeline.validator import (
    STATUS_INCOMPLETE,
    STATUS_OK,
    ArgumentValidator,
)

TOOL = {
    "name": "vi_order_food",
    "parameters": {
        "type": "object",
        "properties": {
            "dish": {"type": "string", "description": "Món ăn"},
            "quantity": {"type": "integer", "description": "Số phần"},
            "max_price_vnd": {"type": "integer", "description": "Giá tối đa"},
            "spicy": {"type": "boolean", "description": "Có cay không"},
            "platform": {"type": "string", "enum": ["Shopee", "Tiki"]},
            "toppings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["dish", "quantity"],
        "additionalProperties": False,
    },
}


@dataclass
class FakePrediction:
    name: str
    value: object
    has_value_prob: float


@pytest.fixture
def validator() -> ArgumentValidator:
    return ArgumentValidator(fallback_threshold=0.3)


def test_coerces_string_number_to_schema_integer(validator):
    result = validator.validate({"dish": "phở", "quantity": "2", "max_price_vnd": "50.000 đồng"}, TOOL)

    assert result.arguments["quantity"] == 2
    assert result.arguments["max_price_vnd"] == 50000
    assert result.status == STATUS_OK
    assert set(result.coerced_keys) == {"quantity", "max_price_vnd"}


def test_drops_keys_not_in_schema(validator):
    # ArgA yêu cầu không được thừa key.
    result = validator.validate({"dish": "phở", "quantity": 1, "made_up": "x"}, TOOL)

    assert "made_up" not in result.arguments
    assert result.dropped_keys == ["made_up"]


def test_missing_required_marks_incomplete(validator):
    result = validator.validate({"dish": "phở"}, TOOL)

    assert result.status == STATUS_INCOMPLETE
    assert result.missing_required == ["quantity"]


def test_missing_required_recovered_at_lower_threshold(validator):
    # has_value=0.35 < 0.5 nên bị lọc ở bước extract, nhưng >= 0.3 nên validator lấy lại.
    predictions = [FakePrediction("quantity", "3", 0.35)]

    result = validator.validate({"dish": "phở"}, TOOL, predictions)

    assert result.status == STATUS_OK
    assert result.arguments["quantity"] == 3


def test_recovery_respects_fallback_threshold(validator):
    predictions = [FakePrediction("quantity", "3", 0.10)]

    result = validator.validate({"dish": "phở"}, TOOL, predictions)

    assert result.status == STATUS_INCOMPLETE
    assert "quantity" not in result.arguments


def test_enum_matching_is_case_insensitive(validator):
    result = validator.validate({"dish": "phở", "quantity": 1, "platform": "shopee"}, TOOL)

    assert result.arguments["platform"] == "Shopee"


def test_boolean_from_vietnamese_string(validator):
    result = validator.validate({"dish": "phở", "quantity": 1, "spicy": "không"}, TOOL)

    assert result.arguments["spicy"] is False


def test_unsupported_array_param_is_reported_not_guessed(validator):
    result = validator.validate({"dish": "phở", "quantity": 1, "toppings": "trứng"}, TOOL)

    assert "toppings" not in result.arguments
    assert result.unsupported_params == ["toppings"]


def test_unparseable_number_is_dropped_rather_than_kept_as_text(validator):
    result = validator.validate({"dish": "phở", "quantity": "rất nhiều"}, TOOL)

    assert "quantity" not in result.arguments
    assert result.status == STATUS_INCOMPLETE
