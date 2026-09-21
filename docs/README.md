# 📚 Tài Liệu Kỹ Thuật & Nghiên Cứu (Documentation Index)

Thư mục này chứa toàn bộ các tài liệu kỹ thuật, kiến trúc, phương pháp luận và hướng dẫn thực nghiệm cho đề tài:
**"Tool Calling tiếng Việt — So sánh 2 phương pháp: SLM End-to-End vs Bi-Encoder + Cross-Encoder"**.

---

## 1. Kiến Trúc & Phương Pháp Nghiên Cứu (Architecture & Methodology)

| File | Mô tả nội dung |
|---|---|
| [architecture.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/architecture.md) | Sơ đồ kiến trúc tổng thể, luồng dữ liệu và thiết kế 2 pipeline (Method 1: SLM và Method 2: Bi+Cross Encoder). |
| [methodology.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/methodology.md) | Phương pháp luận nghiên cứu, các giả thuyết khoa học, thiết lập baseline và tiêu chuẩn đánh giá. |
| [experimental_plan.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/experimental_plan.md) | Kế hoạch thực nghiệm tổng thể (controlled data budget, các nấc E0 $\to$ E4, stress test). |
| [references.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/references.md) | Danh mục tài liệu tham khảo và các bài báo khoa học liên quan (Ersoy et al., BFCL, RAG-MCP, ToolLLM,...). |

---

## 2. Dữ Liệu & Benchmark (Data & Benchmarks)

| File | Mô tả nội dung |
|---|---|
| [data_pipeline_and_splits.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/data_pipeline_and_splits.md) | **Tài liệu chuẩn tắc** về pipeline dữ liệu, frozen revision (`2026-09-02-full-dedup-seed42`), cấu trúc split train/val/test và quy cách dữ liệu. |
| [benchmark.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/benchmark.md) | Đặc tả cấu trúc benchmark tiếng Việt, định nghĩa ground truth, phân loại function call đơn/đa lệnh. |
| [custom_vi_dataset_plan.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/custom_vi_dataset_plan.md) | Thiết kế chi tiết tập dữ liệu đặc thù Việt Nam `CustomTools-VI` (8,000 mẫu, 40 công cụ, 10 nhóm domain, chia strict seen/unseen). |
| [translation_guidelines.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/translation_guidelines.md) | Quy chuẩn dịch thuật EN $\to$ VI giữ nguyên identifier, cấu trúc JSON và thuật ngữ kỹ thuật. |

---

## 3. Báo Cáo Kết Quả Thực Nghiệm & Kiến Trúc Chi Tiết (Reports & System Details)

| File | Mô tả nội dung |
|---|---|
| [method2.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/method2.md) | Hướng dẫn vận hành và kiến trúc kỹ thuật toàn diện của Method 2 (Bi-Encoder BGE-M3 + Cross-Encoder XLM-RoBERTa). |
| [paper_vi_method2_shared.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/paper_vi_method2_shared.md) | Báo cáo thực nghiệm chuyên sâu **Method 2**: huấn luyện Shared E4, đánh giá Core Benchmark và CustomTools-VI, phân tích Oracle, Normalizer ablation. |
| [../reports/stress_report_vi.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/reports/stress_report_vi.md) | Báo cáo thử nghiệm ứng suất độ bền (Stress Test $N = 3 \to 1000$) đối đầu trực diện giữa Method 1 và Method 2. |
| [../results/slm/summary_table.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/results/slm/summary_table.md) | Báo cáo tổng hợp kết quả **Method 1 (SLM Qwen3.5-2B/4B E0 $\to$ E4)** và so sánh đối đầu. |
| [../paper/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/) | Bài báo khoa học: bản tiếng Anh ([paper_en.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_en.md)) và bản tiếng Việt ([paper_vi.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_vi.md)). |

---

## 4. Hướng Dẫn Huấn Luyện & Đánh Giá (Execution Guides)

| File | Môi trường / Mục đích |
|---|---|
| [kaggle_execution_guide.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/kaggle_execution_guide.md) | Hướng dẫn toàn diện trên Kaggle Dual GPU T4: Huấn luyện DDP Unsloth (E1 $\to$ E4) có stage resume và đánh giá suy luận (E0 / post-training). |
| [colab_e1_e4_training_guide.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/colab_e1_e4_training_guide.md) | Hướng dẫn huấn luyện đơn GPU trên Google Colab Pro (A100 / L4 / T4) đồng bộ qua Google Drive. |

---

## 5. Tài Liệu Tham Khảo Mở Rộng & Biểu Mẫu (References & Templates)

| Đường dẫn | Mô tả nội dung |
|---|---|
| [references/tool_calling_Arabic_ocr.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/references/tool_calling_Arabic_ocr.md) | Bản OCR toàn văn bài báo *Ersoy et al. (ArabicNLP 2025)* dùng làm tài liệu tham khảo đối sánh. |
| [../thesis/templates/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/templates/) | Các biểu mẫu Word chính thức của Trường ĐH Công nghệ Thông tin (`BieuMau.docx`, `phuluc2_hinhthuctrinhbay.docx`). |
