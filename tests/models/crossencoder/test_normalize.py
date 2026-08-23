"""Test normalizer §4 Phase 4 — chỉ chạy trên span, chỉ khi schema cho phép."""

from datetime import date

import pytest

from src.models.crossencoder.normalize import (
    NormalizerConfig,
    SpanNormalizer,
    normalize_text,
    parse_date,
    parse_number,
)

REF = date(2026, 8, 23)


@pytest.fixture
def normalizer() -> SpanNormalizer:
    return SpanNormalizer(NormalizerConfig(reference_date=REF))


def _param(name: str, value_type: str, **extra):
    return {"name": name, "value_type": value_type, "type": value_type, **extra}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("50.000", 50000),
        ("50,000", 50000),
        ("1.234,5", 1234.5),
        ("1,234.5", 1234.5),
        ("1,5", 1.5),
        ("1.5", 1.5),
        ("30", 30),
    ],
)
def test_parse_number_handles_vn_and_en_separators(text, expected) -> None:
    assert parse_number(text) == pytest.approx(expected)


def test_thousand_separator_on_money_param(normalizer) -> None:
    result = normalizer.normalize("50.000 đồng", _param("max_price_vnd", "integer"))

    assert result.value == 50000
    assert "separator" in result.applied


@pytest.mark.parametrize(
    "text,expected",
    [("50k", 50000), ("2 triệu", 2000000), ("3 tỷ", 3000000000), ("15 nghìn", 15000)],
)
def test_money_units(normalizer, text, expected) -> None:
    result = normalizer.normalize(text, _param("max_price_vnd", "integer"))

    assert result.value == expected
    assert any(a.startswith("money_unit") for a in result.applied)


def test_vn_decimal_on_number_param(normalizer) -> None:
    result = normalizer.normalize("1,5", _param("min_rating", "number"))

    assert result.value == pytest.approx(1.5)


def test_time_unit_converted_to_param_unit(normalizer) -> None:
    param = _param("max_delivery_minutes", "integer")

    assert normalizer.normalize("30 phút", param).value == 30
    assert normalizer.normalize("2 tiếng", param).value == 120
    assert normalizer.normalize("1 giờ", param).value == 60


def test_time_unit_for_hour_param(normalizer) -> None:
    result = normalizer.normalize("90 phút", _param("duration_hours", "number"))

    assert result.value == pytest.approx(1.5)


def test_integer_param_returns_int_not_float(normalizer) -> None:
    result = normalizer.normalize("2", _param("quantity", "integer"))

    assert result.value == 2
    assert isinstance(result.value, int)


def test_string_param_is_not_coerced_to_number(normalizer) -> None:
    # Identifier dạng string phải giữ nguyên, kể cả khi trông như số.
    result = normalizer.normalize("001234", _param("account_number", "string"))

    assert result.value == "001234"
    assert result.applied == []


def test_date_only_parsed_when_schema_declares_format(normalizer) -> None:
    with_format = _param("checkin_date", "string", format="date")
    without_format = _param("checkin_date", "string")

    assert normalizer.normalize("ngày 15/3", with_format).value == "2026-03-15"
    assert normalizer.normalize("ngày 15/3", without_format).value == "ngày 15/3"


def test_relative_date_uses_reference_date(normalizer) -> None:
    param = _param("start_date", "string", format="date")

    assert normalizer.normalize("mai", param).value == "2026-08-24"
    assert normalizer.normalize("hôm nay", param).value == "2026-08-23"


def test_parse_date_formats() -> None:
    assert parse_date("2026-03-15", REF) == "2026-03-15"
    assert parse_date("ngày 15 tháng 3", REF) == "2026-03-15"
    assert parse_date("15/03/2025", REF) == "2025-03-15"
    assert parse_date("không phải ngày", REF) is None


def test_unparseable_number_keeps_text_and_flags_failure(normalizer) -> None:
    result = normalizer.normalize("rất rẻ", _param("max_price_vnd", "integer"))

    assert result.ok is False
    assert result.value == "rất rẻ"


def test_disabled_normalizer_only_trims() -> None:
    normalizer = SpanNormalizer(NormalizerConfig(enabled=False))

    result = normalizer.normalize("  50.000 đồng ", _param("max_price_vnd", "integer"))

    assert result.value == "50.000 đồng"
    assert result.applied == []


def test_normalize_text_nfc_and_whitespace() -> None:
    assert normalize_text("  Hà   Nội,  ") == "Hà Nội"
