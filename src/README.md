# `src/`

Source code chính của project.

Cấu trúc:

- `data/` — Code xử lý data (collect, normalize, translate, rebuild benchmark, experiment preparation, stats, stress test).
- `models/`
  - `slm/` — Native Qwen3.5 rendering, assistant-only labels và Unsloth QLoRA/SFT.
  - `biencoder/` — Semantic Tool Retrieval (BGE-M3 + FlagEmbedding + MNRL).
  - `crossencoder/` — Schema-aware Parameter Extraction (BGE-M3 + custom heads, BERT-QA format).
  - `baselines/` — OpenAI FC, Gemini FC.
- `pipeline/` — End-to-end Tool Calling (retriever, extractor, validator, tool_caller).
- `evaluation/` — Metrics + comparison (retrieval, extraction, latency, cost, stress test).
- `utils/` — Logger, seed, IO, schema utils, hydra utils.

> Method 1 local SLM dùng `src/models/slm/`; API baselines nằm trong `src/models/baselines/`.

> Mỗi module Python sẽ có `__init__.py` và docstring mô tả ngắn ở đầu file (xem `AGENTS.md` section 8).
