"""Biến thể strict của extraction/end-to-end (§11 method2_plan).

Bản normalized trả lời "model có lấy đúng thông tin không"; bản strict trả lời
"chuỗi sinh ra dùng được ngay không". Chênh lệch giữa hai bản là phần công mà
normalizer đang gánh — con số đó phải xuất hiện được trong báo cáo, nếu không
thì không tách được đóng góp của Phase 4.
"""

from __future__ import annotations

from src.evaluation.config import EvaluationConfig, NormalizationConfig
from src.evaluation.evaluator import evaluate_records
from src.evaluation.io import parse_prediction_record
from src.evaluation.normalization import ArgumentNormalizer


def test_strict_tat_moi_noi_long_tru_unicode() -> None:
    cfg = NormalizationConfig.strict()
    assert cfg.unicode_form == "NFC"  # vệ sinh mã hoá, không phải nới lỏng
    assert not cfg.collapse_whitespace
    assert not cfg.casefold_strings
    assert not cfg.coerce_numbers
    assert not cfg.coerce_booleans
    assert not cfg.normalize_dates
    assert not cfg.global_aliases and not cfg.parameter_aliases


def test_strict_bat_khac_biet_ma_normalized_bo_qua() -> None:
    """`"5000"` vs `5000` — normalized coi là bằng, strict thì không."""
    lenient = ArgumentNormalizer(NormalizationConfig())
    strict = ArgumentNormalizer(NormalizationConfig.strict())
    schema = {"type": "integer"}

    def norm(engine, value):
        return engine.normalize_value(value, schema, "convert_currency", "amount")

    assert norm(lenient, "5000") == norm(lenient, 5000)
    assert norm(strict, "5000") != norm(strict, 5000)


def test_report_co_ca_hai_ban_va_strict_khong_cao_hon(
    gold_records: list, prediction_records: list
) -> None:
    report, _ = evaluate_records(gold_records, prediction_records, EvaluationConfig())
    metrics = report["metrics"]
    for key in ("strict_extraction", "strict_end_to_end", "strict_oracle_extraction"):
        assert key in metrics, f"thiếu {key} trong report"

    lenient = metrics["extraction"]["normalized_arg_em_given_correct_tool"]
    strict = metrics["strict_extraction"]["normalized_arg_em_given_correct_tool"]
    if lenient is not None and strict is not None:
        # Strict là điều kiện chặt hơn hẳn nên không bao giờ được vượt lenient.
        assert strict <= lenient


def test_strict_oracle_none_khi_khong_truyen_oracle(
    gold_records: list, prediction_records: list
) -> None:
    report, _ = evaluate_records(gold_records, prediction_records, EvaluationConfig())
    assert report["metrics"]["strict_oracle_extraction"] is None
    assert report["metrics"]["oracle_extraction"] is None


def test_strict_oracle_duoc_tinh_khi_co_oracle(gold_records: list) -> None:
    oracle = [
        parse_prediction_record(
            {
                "id": record.id,
                "function_calls": [
                    {"name": call.name, "arguments": dict(call.arguments)}
                    for call in record.function_calls
                ],
            }
        )
        for record in gold_records
    ]
    report, _ = evaluate_records(gold_records, oracle, EvaluationConfig(), oracle)
    strict_oracle = report["metrics"]["strict_oracle_extraction"]
    assert strict_oracle is not None
    # Oracle trùng khít gold thì strict cũng phải tuyệt đối.
    assert strict_oracle["normalized_arg_em_given_correct_tool"] == 1.0
