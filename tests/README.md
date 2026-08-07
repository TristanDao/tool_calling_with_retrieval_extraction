# `tests/`

Unit tests cho data pipeline và evaluation framework.

- `test_schema_parser.py` — Test parse JSON Schema.
- `test_validator.py` — Test JSON Schema validator.
- `test_translate_guidelines.py` — Test rule dịch (bảo vệ identifier, JSON structure).
- `test_pipeline.py` — Test end-to-end pipeline.
- `test_metrics.py` — Test các metric (Recall@k, F1, EM, etc).
- `evaluation/` — Test adapters, normalization, component metrics, N-FCEM, schema validity, reports, robustness và paired statistics.

> Chạy: `pytest tests/` (đã cấu hình trong `pyproject.toml`).
