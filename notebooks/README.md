# `notebooks/`

Jupyter Notebooks phục vụ EDA, huấn luyện mô hình (Colab) và đánh giá benchmark (Kaggle).

## Cấu trúc thư mục

- `training/`
  - `colab_E3.ipynb` — Notebook huấn luyện Unsloth QLoRA trên Google Colab T4 cho E3 (song ngữ), hỗ trợ push adapter lên Hugging Face Hub.
- `benchmarks/`
  - `kaggle_core_benchmark.ipynb` — Benchmark chạy đa GPU (2x T4) trên tập Core Benchmark (EN/VI test, 7,712 mẫu).
  - `kaggle_custom_benchmark.ipynb` — Benchmark chạy đa GPU trên tập `CustomTools-VI` (`test_seen` và `test_unseen`, 1,600 mẫu).
  - `kaggle_full_benchmark.ipynb` — Benchmark toàn diện tích hợp cả Core và CustomTools.
- `01_eda_raw_data.ipynb` — Khám phá dataset gốc (Glaive, xLAM).

> Lưu ý: Các file `.ipynb_checkpoints/` đã được cấu hình trong `.gitignore`. Hãy clean output trước khi commit notebook lớn lên Git.

