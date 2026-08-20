# Data Directory Architecture (`data/`)

Thư mục `data/` chứa toàn bộ dữ liệu dự án ở các giai đoạn pipeline khác nhau. **Không chứa code xử lý** (tất cả code xử lý nằm tại `src/data/` và `scripts/data/`).

---

## 📂 Cấu trúc thư mục

### 1. `benchmark_vi/` — 🎯 FINAL MASTER BENCHMARK (TIẾNG VIỆT)
- **Mục đích**: Bộ dữ liệu tiếng Việt chuẩn dùng trực tiếp cho huấn luyện và đánh giá mô hình.
- **Thành phần**:
  - `train.jsonl`, `val.jsonl`, `test.jsonl`: Dữ liệu master split theo tỷ lệ 80/10/10. Snapshot hiện tại có 51.227 positive samples; đây là pilot/rebuild snapshot, chưa phải full benchmark cuối ~105k samples.
  - `tool_pool.json`: Tool pool gộp từ Glaive + xLAM kèm `feature_group`.
  - `instruction/`: Dữ liệu định dạng instruction-chat (`train_chat.jsonl`, `val_chat.jsonl`, `test_chat.jsonl`) cho Method 1 (SLM).

### 2. `custom_vi/` — 🇻🇳 CUSTOMTOOLS-VI BENCHMARK
- **Mục đích**: Bộ dữ liệu 8.000 mẫu function calling tiếng Việt tổng hợp, tập trung vào ngữ cảnh thực tế tại Việt Nam.
- **Thành phần**:
  - `tools.json`: Danh mục 40 tools thuộc 10 nhóm chức năng.
  - `dataset_card.md`: Mô tả dataset và thông số QA.
  - `qa_report.json`: Báo cáo kiểm định chất lượng và phân phối dữ liệu.
  - `{train,val_seen,val_unseen,test_seen,test_unseen}.jsonl`: Các tập dữ liệu tương ứng.

### 3. `translations/` — 🌏 DỮ LIỆU DỊCH QWEN-MT (TIẾNG VIỆT THÔ)
- **Mục đích**: Lưu kết quả dịch tiếng Việt từ API Qwen-MT trước khi build vào master benchmark.
- **Thành phần**:
  - `glaive_normalized_vi.jsonl`, `xlam_normalized_vi.jsonl`.
  - `guidelines.md`: Quy tắc dịch thuật.
  - `qwen_mt_logs/`, `qa_samples/`: Nhật ký dịch và file QA.

### 4. `normalized_en/` — 📝 DỮ LIỆU TIẾNG ANH ĐÃ CHUẨN HÓA
- **Mục đích**: Dữ liệu tiếng Anh sau khi lọc single-turn và chuẩn hóa JSON Schema chuẩn.
- **Thành phần**: `glaive_normalized.jsonl`, `xlam_normalized.jsonl`, `glaive_negative.jsonl`.

### 5. `raw/` — 📦 DỮ LIỆU GỐC TIẾNG ANH
- **Mục đích**: Dataset thô ban đầu tải về từ HuggingFace (chưa qua xử lý).
- **Thành phần**: `glaive_raw.jsonl`, `xlam_raw.jsonl`, `EDA_SUMMARY.md`.

### 6. `processed/` — ⚙️ NƠI XỬ LÝ TRUNG GIAN & STRESS TEST
- **Mục đích**: Lưu index mapping và dữ liệu thử nghiệm tải (Phase 7 - Stress test).
- **Thành phần**: `glaive_single_turn_index.jsonl`, `glaive_single_turn_raw.jsonl`, `stress_test/`.

---

## 📌 Quy định & Hướng dẫn cho đồng đội (Team Members)

1. **Vị trí dữ liệu Final**:
   - Để train / eval mô hình, hãy dùng dữ liệu trong **`data/benchmark_vi/`** hoặc **`data/custom_vi/`**.
   - Không xóa/rebuild `benchmark_vi/` theo từng experiment. Freeze canonical benchmark một lần và ghi manifest/snapshot cho từng experiment.
2. **Quản lý Git**:
   - Tất cả dữ liệu trong `data/` (ngoại trừ tài liệu `.md` và `guidelines.md`) đều đã được đưa vào `.gitignore` để không làm nặng Git.
3. **Chạy script không bị lỗi thư mục**:
   - Mọi script trong `src/data/` đều tự động khởi tạo thư mục đầu ra nếu chưa có (`Path.mkdir(parents=True, exist_ok=True)`).
