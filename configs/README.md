# `configs/`

Hydra **structured config** (Python `@dataclass`).

Cấu trúc:

- `data/` — config cho data pipeline (collect, translate, benchmark).
- `model/` — config cho model (biencoder, crossencoder, unsloth_baseline).
- `pipeline/` — config cho end-to-end pipeline (retrieval k, schema format).
- `baseline/` — config cho baselines (openai, gemini, local_llm).
- `eval/` — config cho evaluation (metrics, output paths).

Entry point: `configs/config.py` (sẽ thêm ở Phase 1).

> **Quy ước**: Mọi hyperparameter và đường dẫn phải qua config, **không hardcode** trong code.
