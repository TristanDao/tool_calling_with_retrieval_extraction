# 📚 Tài Liệu Kỹ Thuật & Nghiên Cứu (Documentation Index)

Thư mục này chứa toàn bộ các tài liệu kỹ thuật, kiến trúc, phương pháp luận và đặc tả benchmark cho đề tài:
**"Nghiên cứu phương pháp Tool Calling dựa trên truy hồi ngữ nghĩa và trích xuất tham số theo Tool Schema"**
*(Research on Tool Calling using Semantic Retrieval and Schema-aware Parameter Extraction)*.

---

## 1. Kiến Trúc & Phương Pháp Nghiên Cứu (Architecture & Methodology)

| File | Mô tả nội dung |
|---|---|
| [architecture.md](./architecture.md) | Sơ đồ kiến trúc tổng thể, luồng dữ liệu và thiết kế 2 pipeline (Method 1: SLM và Method 2: Bi-Encoder + Cross-Encoder). |
| [methodology.md](./methodology.md) | Cơ sở toán học, các hàm mất mát (Multiple Negatives Ranking, Span/Classification loss) và tiêu chuẩn đánh giá. |
| [experimental_plan.md](./experimental_plan.md) | Kế hoạch thực nghiệm tổng thể (controlled data budget, các nấc E0 $\to$ E4, stress test). |
| [method2.md](./method2.md) | Kiến trúc kỹ thuật và hướng dẫn vận hành chuyên sâu của Method 2 (BGE-M3 + XLM-RoBERTa + Vietnamese Value Normalizer). |
| [references.md](./references.md) | Danh mục tài liệu tham khảo và các công trình khoa học liên quan (BFCL, ToolLLM, RAG-MCP, Gorik, v.v.). |

---

## 2. Dữ Liệu & Benchmark (Data & Benchmarks)

| File | Mô tả nội dung |
|---|---|
| [data_pipeline_and_splits.md](./data_pipeline_and_splits.md) | **Tài liệu chuẩn tắc** về pipeline dữ liệu, frozen canonical revision (`2026-09-02-full-dedup-seed42`), cấu trúc split train/val/test và schema chuẩn. |
| [benchmark.md](./benchmark.md) | Đặc tả cấu trúc benchmark tiếng Việt: Core Benchmark (77k mẫu) và CustomTools-VI (8,000 mẫu, 40 công cụ, chia strict Seen/Unseen). |

---

## 3. Bài Báo Khoa Học & Kết Quả (Paper & Experimental Results)

- **Bài báo khoa học**:
  - [paper/paper_en.md](../paper/paper_en.md): Bản thảo bài báo tiếng Anh chuẩn format hội nghị quốc tế.
  - [paper/paper_vi.md](../paper/paper_vi.md): Bản thảo bài báo tiếng Việt.
  - [latex/](../latex/): Mã nguồn dự án LaTeX hoàn chỉnh sẵn sàng biên dịch PDF.
- **Kết quả thực nghiệm**:
  - [results/slm/summary_table.md](../results/slm/summary_table.md): Bảng tổng hợp số liệu chi tiết của Method 1 (SLM Qwen3.5-2B/4B E0 $\to$ E4) và đối đầu.
  - [results/](../results/): Toàn bộ metrics JSON, predictions và biểu đồ so sánh.
- **Khóa luận tốt nghiệp UIT**:
  - [thesis/README.md](../thesis/README.md): Báo cáo toàn văn Khóa luận tốt nghiệp nộp Phòng Đào tạo UIT.
