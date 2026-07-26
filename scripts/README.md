# `scripts/`

Shell scripts gọi Python modules qua Hydra, dùng để chạy nhanh từng bước.

Cấu trúc:

- `data/` — Pipeline xử lý data.
  - `01_collect.sh` — Tải dataset EN (Glaive + xLAM).
  - `02_normalize.sh` — Chuẩn hóa schema.
  - `03_translate_qwenmt.sh` — Dịch EN→VI bằng Qwen-MT.
  - `04_qa_translation.sh` — QC dịch.
  - `05_build_benchmark.sh` — Sinh benchmark_vi.
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

> Mỗi script chỉ wrap Hydra command, không hardcode hyperparameter.

> **Ngoài scope**: `serve/` (vLLM) và `train_unsloth_baseline.sh` đã bỏ do timeline 3 tháng.
