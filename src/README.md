# `src/`

Source code chính của project.

Cấu trúc:

- `data/` — Code xử lý data (collect, normalize, translate, build benchmark, stats, stress test).
- `models/`
  - `biencoder/` — Semantic Tool Retrieval (BGE-M3 + FlagEmbedding + MNRL).
  - `crossencoder/` — Schema-aware Parameter Extraction (BGE-M3 + custom head, format schema first).
  - `baselines/` — OpenAI FC, Gemini FC.
- `pipeline/` — End-to-end Tool Calling (retriever, extractor, validator, tool_caller).
- `evaluation/` — Metrics + comparison (retrieval, extraction, latency, cost, stress test).
- `utils/` — Logger, seed, IO, schema utils, hydra utils.

> **Ngoài scope**: `local_baseline/` (Qwen/Llama Unsloth) đã bỏ do timeline 3 tháng.

> Mỗi module Python sẽ có `__init__.py` và docstring mô tả ngắn ở đầu file (xem `AGENTS.md` section 8).
