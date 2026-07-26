# `scripts/`

Shell scripts gọi Python modules qua Hydra, dùng để chạy nhanh từng bước.

Cấu trúc:

- `data/` — Pipeline xử lý data.
  - `01_collect.sh` — Tải dataset EN.
  - `02_normalize.sh` — Chuẩn hóa schema.
  - `03_translate_qwenmt.sh` — Dịch EN→VI bằng Qwen-MT.
  - `04_qa_translation.sh` — QC dịch.
  - `05_build_benchmark.sh` — Sinh benchmark_vi.
- `train/` — Huấn luyện.
  - `train_biencoder.sh`
  - `train_crossencoder.sh`
  - `train_unsloth_baseline.sh`
- `serve/` — Serving.
  - `serve_vllm.sh`
- `eval/` — Đánh giá.
  - `eval_biencoder.sh`
  - `eval_crossencoder.sh`
  - `eval_pipeline.sh`
  - `eval_openai.sh`
  - `eval_gemini.sh`
  - `eval_local_llm.sh`
- `compare_all.sh` — So sánh tất cả.

> Mỗi script chỉ wrap Hydra command, không hardcode hyperparameter.
