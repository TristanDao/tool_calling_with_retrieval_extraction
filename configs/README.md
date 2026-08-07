# `configs/`

Hydra **structured config** (Python `@dataclass`).

Cấu trúc:

- `data/` — config cho data pipeline (collect, translate, benchmark).
- `model/` — config cho model (biencoder, crossencoder, unsloth_baseline).
- `pipeline/` — config cho end-to-end pipeline (retrieval k, schema format).
- `baseline/` — config cho baselines (openai, gemini, local_llm).
- `eval/` — alias và unordered-array rules tùy chọn cho schema-aware normalization; structured defaults nằm ở `src/evaluation/config.py`.

Evaluation entry point: `python -m src.evaluation.cli`; resolved structured config được lưu trong mọi `report.json`.

> **Quy ước**: Mọi hyperparameter và đường dẫn phải qua config, **không hardcode** trong code.
