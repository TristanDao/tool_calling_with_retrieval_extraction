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

## 3. Báo Cáo Kết Quả Thực Nghiệm (Experiment Reports)

| File | Mô tả nội dung |
|---|---|
| [bao_cao_method2.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/bao_cao_method2.md) | Báo cáo chi tiết kết quả thực nghiệm **Method 2 (Bi-Encoder BGE-M3 + Cross-Encoder XLM-RoBERTa-base)**: hai round retrieval, hierarchical extraction, normalizer ablation và random stress test. |
| [../results/slm/summary_table.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/results/slm/summary_table.md) | Báo cáo tổng hợp kết quả **Method 1 (SLM Qwen3.5-2B E0 $\to$ E4)** và bảng so sánh đối đầu trực diện M1 vs M2. |
| [../paper/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/) | Bài báo khoa học hoàn chỉnh: bản tiếng Anh ([paper_en.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_en.md)) và bản tiếng Việt ([paper_vi.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_vi.md)). |

---

## 4. Hướng Dẫn Huấn Luyện & Đánh Giá (Execution Guides)

| File | Môi trường / Mục đích |
|---|---|
| [kaggle_e1_e4_training_guide.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/kaggle_e1_e4_training_guide.md) | Hướng dẫn huấn luyện DDP trên Kaggle Dual GPU T4 cho E1 $\to$ E4 với Unsloth, quản lý checkpoint resume theo stage. |
| [kaggle_e0_evaluation_guide.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/kaggle_e0_evaluation_guide.md) | Hướng dẫn chạy đánh giá suy luận (evaluation inference) đa GPU trên Kaggle. |
| [colab_e1_e4_training_guide.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/colab_e1_e4_training_guide.md) | Hướng dẫn huấn luyện đơn GPU trên Google Colab Pro (A100 / L4 / T4) đồng bộ qua Google Drive. |
