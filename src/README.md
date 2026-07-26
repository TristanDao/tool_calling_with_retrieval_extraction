# `src/`

Source code chính của project.

Cấu trúc:

- `data/` — Code xử lý data (collect, normalize, translate, build benchmark, stats).
- `models/`
  - `biencoder/` — Semantic Tool Retrieval (BGE-M3 / multilingual-e5).
  - `crossencoder/` — Schema-aware Parameter Extraction.
  - `local_baseline/` — Qwen2.5/Llama-3.1 local (Unsloth + vLLM).
  - `baselines/` — OpenAI FC, Gemini FC.
- `pipeline/` — End-to-end Tool Calling (retriever, extractor, validator, tool_caller).
- `evaluation/` — Metrics + comparison (retrieval, extraction, latency, cost, throughput).
- `utils/` — Logger, seed, IO, schema utils, hydra utils.

> Mỗi module Python sẽ có `__init__.py` và docstring mô tả ngắn ở đầu file (xem `AGENTS.md` section 8).
