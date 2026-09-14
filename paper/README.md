# 📄 Bài Báo Khoa Học / Luận Văn Tốt Nghiệp (Academic Papers)

Thư mục này chứa hai phiên bản đầy đủ của bài báo nghiên cứu khoa học phục vụ báo cáo đồ án / luận văn tốt nghiệp UIT và nộp kỷ yếu hội nghị:

1. **Bản tiếng Anh (English Version)**: [`paper_en.md`](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_en.md)
   - *Title*: **Vietnamese Tool Calling: A Comparative Study Between End-to-End Small Language Models and Specialized Bi-Encoder + Cross-Encoder Architecture**
   - Định dạng chuẩn bài báo quốc tế (IEEE / ACL / NeurIPS style Markdown).

2. **Bản tiếng Việt (Vietnamese Version)**: [`paper_vi.md`](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_vi.md)
   - *Tựa đề*: **Gọi Công Cụ (Tool Calling) Tiếng Việt: Nghiên Cứu So Sánh Giữa Mô Hình Ngôn Ngữ Nhỏ End-to-End Và Kiến Trúc Chuyên Biệt Bi-Encoder + Cross-Encoder**
   - Định dạng chuẩn báo cáo khoa học / luận văn đại học UIT - ĐHQG-HCM.

---

## Cấu Trúc & Nội Dung Chính

- **Abstract / Tóm tắt**: Tóm lược vấn đề, giải pháp, kết quả nổi bật (SLM E4 đạt 87% Seen / 86.38% Unseen; Method 2 đạt 58-92 ms nhanh gấp 10-15 lần).
- **1. Introduction / Giới thiệu**: Bối cảnh tác tử AI, rào cản tài nguyên, độ trễ và sự thiếu hụt nghiên cứu tiếng Việt.
- **2. Related Work / Nghiên cứu liên quan**: Toolformer, Gorilla, BFCL, ToolLLM, Ersoy et al. (2025).
- **3. Dataset & Benchmark**: Thiết kế bộ dữ liệu Canonical Core (77,028 cặp, 4,421 tools) và CustomTools-VI (8,000 mẫu, 40 tools, 10 nhóm domain, chia strict seen/unseen).
- **4. Methodology / Phương pháp luận**:
  - *Method 1 (SLM)*: Qwen3.5 (2B/4B), Unsloth QLoRA, XML native tool call, response-only loss, 5 cấu hình E0 $\to$ E4.
  - *Method 2 (Bi+Cross)*: BGE-M3 2-round CachedMNRL retrieval + XLM-RoBERTa-base hierarchical extraction heads + Value Normalizer.
- **5. Evaluation / Đánh giá thực nghiệm**: Bộ tiêu chuẩn ArgA / EM, Tool Acc, Non-FC Recall, Syntax Error, Latency.
- **6. Results & Discussion / Kết quả & Phân tích chuyên sâu**:
  - Bảng 1: CustomTools-VI (Seen vs Unseen).
  - Bảng 2: Canonical Core Benchmark (VI Test vs EN Test).
  - Bảng 3: So sánh đối đầu Method 1 vs Method 2 (Chất lượng vs Tốc độ).
  - Phân tích chuyển giao ngôn ngữ (Cross-lingual transfer), hiện tượng over-triggering khi thiếu dữ liệu âm tính, năng lực zero-shot trên unseen tools.
- **7. Model Scaling Study (2B vs 4B)**: Khung phân tích cho mô hình 4B (E3 & E4).
- **8. Limitations & Future Work**: Đồ thị suy giảm độ chính xác khi số lượng công cụ tăng (Stress Test), kiến trúc lai ghép Hybrid.

---

## Hướng Dẫn Xuất Ra PDF Hoặc LaTeX

Bạn có thể dễ dàng chuyển đổi các file markdown này sang PDF hoặc LaTeX bằng `pandoc`:

```bash
# Xuất bản tiếng Anh sang PDF
pandoc paper/paper_en.md -o paper/paper_en.pdf --pdf-engine=xelatex

# Xuất bản tiếng Việt sang PDF (cần font tiếng Việt như DejaVu Sans hoặc Times New Roman)
pandoc paper/paper_vi.md -o paper/paper_vi.pdf --pdf-engine=xelatex -V mainfont="DejaVu Sans"
```
