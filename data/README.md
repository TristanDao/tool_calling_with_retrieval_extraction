# Data Directory Architecture (`data/`)

Thư mục `data/` chứa toàn bộ dữ liệu dự án ở các giai đoạn pipeline khác nhau. **Không chứa code xử lý** (tất cả code xử lý nằm tại `src/data/` và `scripts/data/`).

---

## 📂 Cấu trúc thư mục

### 1. `benchmark_core/<revision>/` — FROZEN CORE BENCHMARK
- **Mục đích**: Benchmark canonical paired EN/VI dùng chung cho mọi method và experiment.
- **Thành phần**:
  - `en/{train,val,test}.jsonl` và `vi/{train,val,test}.jsonl`: Split 80/10/10 của revision `2026-09-02-full-dedup-seed42`, gồm 77.028 paired records và cả negative.
  - `manifest.json`, `split_manifest.json`, `metadata.json`: Hash, mapping split, counts và rejected records.
  - `tool_pool.json`: Tool pool gộp từ Glaive + xLAM kèm `feature_group`.

### 2. `benchmark_vi/` — ACTIVE VI EXPORT
- **Mục đích**: Convenience path trỏ tới `vi/{train,val,test}.jsonl` của frozen revision active.
- Không phải một split độc lập và không được rebuild theo từng experiment.

### 3. `custom_vi/` — 🇻🇳 CUSTOMTOOLS-VI BENCHMARK
- **Mục đích**: Bộ dữ liệu 8.000 mẫu function calling tiếng Việt tổng hợp, tập trung vào ngữ cảnh thực tế tại Việt Nam.
- **Thành phần**:
  - `tools.json`: Danh mục 40 tools thuộc 10 nhóm chức năng.
  - `dataset_card.md`: Mô tả dataset và thông số QA.
  - `qa_report.json`: Báo cáo kiểm định chất lượng và phân phối dữ liệu.
  - `{train,val_seen,val_unseen,test_seen,test_unseen}.jsonl`: Các tập dữ liệu tương ứng.

### 4. `translations/` — 🌏 DỮ LIỆU DỊCH QWEN-MT (TIẾNG VIỆT THÔ)
- **Mục đích**: Lưu kết quả dịch tiếng Việt từ API Qwen-MT trước khi build vào master benchmark.
- **Thành phần**:
  - `glaive_normalized_vi.jsonl`, `xlam_normalized_vi.jsonl`.
  - `guidelines.md`: Quy tắc dịch thuật.
  - `qwen_mt_logs/`, `qa_samples/`: Nhật ký dịch và file QA.

### 5. `normalized_en/` — 📝 DỮ LIỆU TIẾNG ANH ĐÃ CHUẨN HÓA
- **Mục đích**: Dữ liệu tiếng Anh sau khi lọc single-turn và chuẩn hóa JSON Schema chuẩn.
- **Thành phần**: `glaive_normalized.jsonl`, `xlam_normalized.jsonl`, `glaive_negative.jsonl`.

### 6. `raw/` — 📦 DỮ LIỆU GỐC TIẾNG ANH
- **Mục đích**: Dataset thô ban đầu tải về từ HuggingFace (chưa qua xử lý).
- **Thành phần**: `glaive_raw.jsonl`, `xlam_raw.jsonl`, `EDA_SUMMARY.md`.

### 7. `processed/` — ⚙️ NƠI XỬ LÝ TRUNG GIAN & STRESS TEST
- **Mục đích**: Lưu index mapping và dữ liệu thử nghiệm tải (Phase 7 - Stress test).
- **Thành phần**: `glaive_single_turn_index.jsonl`, `glaive_single_turn_raw.jsonl`, `stress_test/`.

### 8. `experiments/` — DỮ LIỆU TRAIN CHO TỪNG EXPERIMENT
- **Mục đích**: Các train artifacts được materialize tái lập từ canonical inputs để chạy Method 1; validation/test dùng shared evaluation files.
- **Tạo bằng**: `bash scripts/data/prepare_experiments.sh`.
- **Thành phần**: `e0`, `e1`, `e2`, `e3`, `e4`; mỗi thư mục có `manifest.json`, E0 không có train file, các E còn lại có `train.jsonl` và native `instruction/train_chat.jsonl`.
- **Quản lý**: Được gitignore vì có thể regenerate; manifest phải được lưu cùng log và checkpoint của experiment.

---

## 📌 Quy định & Hướng dẫn cho đồng đội (Team Members)

1. **Vị trí dữ liệu canonical**:
    - Để train/eval, dùng frozen revision trong **`data/benchmark_core/`**, active VI export hoặc **`data/custom_vi/`**.
    - Không xóa/rebuild revision theo từng experiment. Chỉ tạo revision mới khi input/split thay đổi.
2. **Quản lý Git**:
   - Tất cả dữ liệu trong `data/` (ngoại trừ tài liệu `.md` và `guidelines.md`) đều đã được đưa vào `.gitignore` để không làm nặng Git.
3. **Chạy script không bị lỗi thư mục**:
   - Mọi script trong `src/data/` đều tự động khởi tạo thư mục đầu ra nếu chưa có (`Path.mkdir(parents=True, exist_ok=True)`).
