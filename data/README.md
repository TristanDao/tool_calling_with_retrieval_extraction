# `data/`

Chỉ lưu data, **không chứa code xử lý** (code ở `src/data/`).

Cấu trúc:

- `raw/` — Dataset gốc EN (**Glaive + xLAM**, chỉ 2 nguồn chính). Tải bằng `src/data/collect.py`. Không commit vào git.
- `processed/` — Dataset đã chuẩn hóa schema, dedupe. Không commit.
  - `glaive_single_turn_raw.jsonl` — Glaive positive first-turn records, giữ format raw để dịch Bộ 1.
  - `glaive_single_turn_index.jsonl` — Mapping filtered index → raw source index.
  - `tools/`
  - `queries/`
  - `parameters/`
  - `stress_test/` — Dữ liệu phục vụ Phase 7 (RAG-MCP inspired stress test).
    - `anchors.jsonl` — 200 samples từ test.jsonl.
    - `augmented/` — Instances đã augment với distractor tools (random / same_domain × N).
- `benchmark_vi/` — Benchmark tiếng Việt cuối cùng. Không commit.
  - `tool_pool.json` — Tool pool lớn gộp từ Glaive + xLAM (dùng cho stress test).
  - `tool_schema/` — JSON Schema cho mỗi tool.
  - `train.jsonl`, `val.jsonl`, `test.jsonl` — splits.
- `translations/` — Log/quá trình dịch EN→VI.
  - `guidelines.md` — Quy tắc dịch (commit vào git).
  - `qwen_mt_logs/` — Raw output từ Qwen-MT API.
  - `qa_samples/` — Sample để QC.
- `statistics/` — Báo cáo thống kê dataset (số samples, distribution, etc).

> Tất cả folder con (trừ `translations/guidelines.md`) đã được gitignore. Khi cần share data, dùng DVC hoặc upload lên HuggingFace Hub.
