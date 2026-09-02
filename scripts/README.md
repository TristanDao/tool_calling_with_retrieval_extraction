# `scripts/`

Shell scripts gọi các Python modules, dùng để chạy nhanh từng bước.

Cấu trúc:

- `data/` — Pipeline xử lý data.
  - `run_collect.sh` — Tải dataset EN (Glaive + xLAM).
  - `run_filter_glaive.sh` — Lọc Glaive single-turn.
  - `run_translate_glaive.sh`, `run_translate_xlam.sh` — Dịch EN→VI bằng Qwen-MT.
  - `run_qa.sh` — QC translation.
  - `run_cleanup.sh` — Inventory và archive generated pilot artifacts.
  - `run_benchmark.sh` — Build/freeze paired core revision và active VI export.
  - `prepare_experiments.sh` — Materialize E0/E1/E2/E3/E4 train-only artifacts.
  - `06_build_stress_test.sh` — Build stress test (Phase 7).
- `train/` — Huấn luyện.
  - `train_biencoder.sh` — Train Bi-Encoder (BGE-M3 + FlagEmbedding + MNRL).
  - `train_crossencoder.sh` — Train Cross-Encoder (BGE-M3 + custom head).
- `eval/` — Đánh giá.
  - `eval_biencoder.sh`
  - `eval_crossencoder.sh`
  - `eval_pipeline.sh`
  - `eval_openai.sh`
  - `eval_gemini.sh`
  - `stress_test.sh`
- `compare_all.sh` — So sánh tất cả.

`data/benchmark_core/<revision>/` là source canonical cho validation/test;
`data/benchmark_vi/` chỉ là active VI export. Experiment folders không chứa
copy validation/test.

Method 1 dùng `scripts/train/train_unsloth.sh` với
`data/experiments/e*/instruction/train_chat.jsonl`; dùng
`scripts/train/smoke_native_qwen.sh` trước khi train.

> Các wrapper giữ đường dẫn mặc định; hyperparameter pipeline nằm trong config hoặc CLI.

> **Ngoài scope**: `serve/` (vLLM) và `train_unsloth_baseline.sh` đã bỏ do timeline 3 tháng.
