# Tool Calling tiếng Việt — So sánh 2 Phương pháp (SLM End-to-End vs Bi-Encoder + Cross-Encoder)

> **Đồ án / Luận văn tốt nghiệp UIT** — Nghiên cứu so sánh thực nghiệm toàn diện giữa Mô hình Ngôn ngữ Nhỏ (SLM End-to-End) và Kiến trúc Chuyên biệt Phân tách (Bi-Encoder Retrieval + Cross-Encoder Extraction) cho tác tử gọi công cụ tiếng Việt.  
> **Sinh viên thực hiện**: Đào Phước Thịnh, Hà Quang Đạt  
> **Giảng viên hướng dẫn**: TS. Đặng Văn Thìn  
> **Đơn vị**: Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM.

---

## 1. Tổng Quan & Kết Quả Đột Phá

Dự án phát triển và so sánh đối đầu **2 trường phái kiến trúc** Tool Calling cho tiếng Việt:

| Phương pháp | Kiến trúc & Công nghệ | Điểm mạnh chính |
|---|---|---|
| **Method 1: SLM End-to-End** | `unsloth/Qwen3.5-2B` / `4B` + Unsloth QLoRA SFT trên native XML tool calls | **Chuẩn vàng về độ chính xác (ArgA 87.00% seen, 86.38% unseen)**, năng lực tổng quát hóa zero-shot vượt trội |
| **Method 2: Bi-Encoder + Cross-Encoder** | `BAAI/bge-m3` (2-round CachedMNRL) + `xlm-roberta-base` (Hierarchical Heads) + Value Normalizer | **P50 55–108 ms trong stress test**, VRAM allocated ~3.21 GiB theo telemetry, 0% lỗi JSON theo định nghĩa structured-output |

### Bảng So Sánh Đối Đầu Nổi Bật (Head-to-Head Summary)

| Tập kiểm thử | Chỉ số (Metric) | Method 1: SLM Qwen3.5-2B (E4) | Method 2: Bi+Cross (BGE-M3 + XLM-R) | Chênh lệch (M1 - M2) |
|---|---|:---:|:---:|:---:|
| **Custom Seen** (800) | **ArgA / Exact Match** | **87.00%** | 67.75% | **+19.25%** |
| **Custom Unseen** (800) | **ArgA / Exact Match** | **86.38%** | 22.75% | **+63.63% (Áp đảo)** |
| **Core VI Test** (7,712) | **ArgA / Exact Match** | **69.75%** | 40.21% | **+29.54%** |
| **Độ trễ suy luận (P50)** | **Thời gian phản hồi** | ~970 ms | **55.13 – 107.68 ms** | **Hai protocol đo khác nhau; chỉ đối chiếu mô tả** |

📄 Chi tiết bài báo khoa học: xem [paper/paper_en.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_en.md) (tiếng Anh) và [paper/paper_vi.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_vi.md) (tiếng Việt).

---

## 2. Cấu Trúc Thư Mục

```
tool_calling_with_retrieval_extraction/
├── AGENTS.md                  # Bộ nhớ cross-session dài hạn cho AI agents
├── README.md                  # Tài liệu tổng quan dự án (file này)
├── pyproject.toml             # Khai báo dependency và build system
│
├── paper/                     # BÀI BÁO KHOA HỌC HOÀN CHỈNH
│   ├── README.md              # Giới thiệu và hướng dẫn xuất PDF/LaTeX
│   ├── paper_en.md            # Bản tiếng Anh chuẩn học thuật quốc tế
│   └── paper_vi.md            # Bản tiếng Việt chuẩn báo cáo luận văn UIT
│
├── docs/                      # TÀI LIỆU KỸ THUẬT & HƯỚNG DẪN
│   ├── README.md              # Mục lục toàn bộ tài liệu
│   ├── architecture.md        # Thiết kế kiến trúc 2 pipeline
│   ├── methodology.md         # Phương pháp luận nghiên cứu
│   ├── experimental_plan.md   # Kế hoạch thực nghiệm E0-E4 & stress test
│   ├── data_pipeline_and_splits.md # Đặc tả chuẩn dữ liệu và frozen revision
│   ├── paper_vi_method2_shared.md # Báo cáo thực nghiệm chi tiết Method 2 (Shared E4)
│   └── ... (các hướng dẫn Kaggle/Colab, dịch thuật, CustomTools)
│
├── data/                      # DỮ LIỆU & BENCHMARK
│   ├── benchmark_core/        # Frozen revision canonical (77k samples EN/VI)
│   ├── custom_vi/             # Tập benchmark đặc thù Việt Nam CustomTools-VI
│   └── experiments/           # Dữ liệu train native theo từng cấu hình E0-E4
│
├── results/                   # KẾT QUẢ THỰC NGHIỆM
│   ├── slm/                   # Metrics JSON, predictions và bảng tổng hợp SLM
│   │   ├── metrics/           # 20 file metrics JSON chuẩn hóa (E0-E4)
│   │   ├── predictions/       # Full predictions JSONL và scored JSON
│   │   └── summary_table.md   # Bảng tổng hợp số liệu chi tiết E0-E4
│   └── tables_figures/        # Bảng biểu và đồ thị báo cáo
│
├── src/                       # MÃ NGUỒN CỐT LÕI
│   ├── data/                  # Pipeline thu thập, dịch, chuẩn hóa, rebuild
│   ├── models/                # slm/, biencoder/, crossencoder/, baselines/
│   └── evaluation/            # Bộ công cụ đo lường (retrieval, extraction, latency)
│
└── notebooks/                 # JUPYTER NOTEBOOKS
    ├── 01_eda_raw_data.ipynb
    └── benchmarks/            # Notebooks chạy benchmark trên Kaggle
```

---

## 3. Hệ Thống Dữ Liệu & Benchmark

1. **Canonical Core Benchmark** (`data/benchmark_core/2026-09-02-full-dedup-seed42/`):
   - **77,028** bản ghi ghép cặp Anh–Việt (18,210 Glaive, 58,818 xLAM), **4,421** unique tools.
   - Tỷ lệ phân chia cố định: `61,615` train / `7,701` val / `7,712` test (seed 42).
2. **CustomTools-VI Benchmark** (`data/custom_vi/`):
   - **8,000** mẫu thuộc **40** công cụ thuần Việt trên 10 nhóm lĩnh vực thực tế (thương mại điện tử, vé xe, đồ ăn, hóa đơn, phạt nguội, gia sư,...).
   - Tỷ lệ 70/10/20: 5,600 train, 800 val, 1,600 test.
   - Phân chia nghiêm ngặt: 20 công cụ **Seen** và 20 công cụ **Unseen** (zero-shot evaluation).

---

## 4. Các Bước Chạy & Thực Nghiệm

### 4.1 Cài đặt môi trường
```bash
git clone https://github.com/TristanDao/tool_calling_with_retrieval_extraction.git
cd tool_calling_with_retrieval_extraction
pip install -e ".[dev,translate,train]"
```

### 4.2 Chuẩn bị dữ liệu thực nghiệm (E0 $\to$ E4)
```bash
# Tạo các train view native cho từng experiment
bash scripts/data/prepare_experiments.sh --overwrite
```

### 4.3 Trích xuất và tổng hợp kết quả đánh giá
```bash
# Trích xuất toàn bộ metrics từ zip Kaggle, lọc file thừa và cập nhật bảng tổng hợp
python3 scripts/eval/extract_and_filter_results.py
```

---

## 5. Tiến Độ Dự Án (Roadmap)

| Phase | Nội dung thực hiện | Trạng thái |
|---|---|---|
| 0 | Skeleton thư mục + tài liệu nền tảng | ✅ Hoàn thành |
| 1 | Data pipeline (dịch thuật, QA, frozen revision 77k) | ✅ Hoàn thành |
| 2 | Method 2: Bi-Encoder (BGE-M3 + 2 rounds CachedMNRL) | ✅ Hoàn thành |
| 3 | Method 2: Cross-Encoder (XLM-RoBERTa hierarchical heads) | ✅ Hoàn thành |
| 4 | Method 1: SLM Fine-tune (Qwen3.5-2B E0 $\to$ E4) | ✅ Hoàn thành |
| - | Viết Bài Báo Khoa Học (EN & VI) | ✅ Hoàn thành |
| - | Model Scaling Study (Qwen3.5-4B E3 & E4) | ✅ Hoàn thành |
| 6 | Tổng hợp so sánh đối đầu toàn diện M1 vs M2 | ✅ Hoàn thành |
| 7 | Stress Test với số lượng công cụ tăng dần ($N = 3 \to 1000$) | ✅ Hoàn thành |

---

## 6. License & Trích Dẫn

Dự án phát hành theo giấy phép **MIT License**.

```bibtex
@misc{thinh2026vietnamesetoolcalling,
  author = {Dao Phuoc Thinh and Ha Quang Dat and Dang Van Thin},
  title = {Vietnamese Tool Calling: A Comparative Study Between End-to-End Small Language Models and Specialized Bi-Encoder + Cross-Encoder Architecture},
  year = {2026},
  publisher = {GitHub},
  journal = {University of Information Technology (UIT), VNU-HCM}
}
```
