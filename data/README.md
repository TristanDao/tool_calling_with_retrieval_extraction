# `data/`

Chỉ lưu data, **không chứa code xử lý** (code ở `src/data/`).

Cấu trúc:

- `raw/` — Dataset gốc EN (Glaive, ToolBench, xLAM, ToolACE). Tải bằng `src/data/collect.py`. Không commit vào git.
- `processed/` — Dataset đã chuẩn hóa schema, dedupe. Không commit.
  - `tools/`
  - `queries/`
  - `parameters/`
- `benchmark_vi/` — Benchmark tiếng Việt cuối cùng. Không commit.
  - `tool_schema/` — JSON Schema cho mỗi tool.
  - `train.jsonl`, `val.jsonl`, `test.jsonl` — splits.
- `translations/` — Log/quá trình dịch EN→VI.
  - `guidelines.md` — Quy tắc dịch (commit vào git).
  - `qwen_mt_logs/` — Raw output từ Qwen-MT API.
  - `qa_samples/` — Sample để QC.
- `statistics/` — Báo cáo thống kê dataset (số samples, distribution, etc).

> Tất cả folder con (trừ `translations/guidelines.md`) đã được gitignore. Khi cần share data, dùng DVC hoặc upload lên HuggingFace Hub.
