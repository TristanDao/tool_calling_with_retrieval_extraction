# FULL PROMPT TẠO SLIDE BẢO VỆ KHÓA LUẬN TỐT NGHIỆP UIT (25 SLIDES)
> **Mục đích**: File này là bản **Master Prompt độc lập**, được tối ưu hóa chuyên biệt để nạp trực tiếp vào các công cụ sinh slide AI (Gamma, Claude Artifacts, ChatGPT / Python-pptx, Marp, Slidev, Beautiful.ai, NotebookLM, v0) hoặc dùng làm tài liệu chỉ dẫn thiết kế cho UI/UX Designer.
> **Đặc trưng**: Đã bóc tách hoàn toàn lời thoại thuyết trình (Script) và câu hỏi phản biện (Q&A); giữ lại 100% cấu trúc giao diện, hệ thống token màu, phân cấp thông tin, bảng biểu thực nghiệm, sơ đồ kiến trúc và nội dung hiển thị chính xác trên từng slide.

---

## PHẦN I: MASTER SYSTEM PROMPT & CẤU TRÚC DESIGN SYSTEM

```yaml
Role: Senior Academic Slide Architect & Deep-Tech Presentation Designer
Task: Generate a 25-slide presentation deck for a University Graduation Thesis Defense (Khóa luận tốt nghiệp Cử nhân Trí tuệ Nhân tạo - Trường Đại học Công nghệ Thông tin, ĐHQG-HCM).
Tone: Deep-tech, Highly Academic, Rigorous, Modern, Clean, Data-dense, Zero-fluff.
Aspect Ratio: 16:9 Widescreen (1920x1080)

Design System:
  Color Palette:
    Primary (UIT Academic Blue): "#00529C"       # Chủ đạo nhận diện thương hiệu học thuật UIT
    Secondary (Deep Slate / Dark Navy): "#0F172A" # Màu nền slide mở chương và chữ tiêu đề chính
    Accent (Tech Cyan): "#06B6D4"                 # Điểm nhấn công nghệ, đường dẫn pipeline, icon nổi bật
    Success (Emerald Green): "#10B981"            # Highlight kết quả tốt nhất, Method 2 ổn định
    Warning/Alert (Amber Orange): "#F59E0B"       # Cảnh báo bùng nổ context, lỗi over-triggering
    Danger (Crimson Red): "#EF4444"               # Ký hiệu sập hệ thống, CUDA OOM, lỗi cú pháp JSON
    Background Light: "#FFFFFF"                   # Nền slide nội dung sạch sẽ
    Background Alt: "#F8FAFC"                     # Nền phụ phân vùng
    Card Background: "#F1F5F9"                    # Khối chứa nội dung, viền mảnh #CBD5E1
    Text Primary: "#0F172A"                       # Chữ chính độ tương phản cao
    Text Muted: "#475569"                         # Chữ giải thích phụ, chú thích nhỏ

  Typography:
    Title Font: "Inter, Be Vietnam Pro, Arial, sans-serif" (Bold, 24pt - 32pt)
    Section Heading: "Inter, Be Vietnam Pro, sans-serif" (Semi-bold, 18pt - 22pt)
    Body Text: "Inter, Roboto, sans-serif" (Regular/Medium, 13pt - 16pt)
    Code / Metrics / Tables: "JetBrains Mono, Fira Code, monospace" (12pt - 14pt)

  Layout Principles:
    - Zero Wall-of-Text: Không sử dụng các đoạn văn dài; chia nội dung thành Grid Cards (2 cột, 3 cột, lưới 2x2).
    - Metric Cards: Số liệu đo lường chính phải to đậm (Hero Numbers, 28-36pt) kèm nhãn giải thích nhỏ bên dưới.
    - Architecture Pipelines: Thể hiện bằng sơ đồ trực quan, luồng mũi tên có nhãn dữ liệu rõ ràng.
    - Comparative Tables: Bảng số liệu khoa học, có phân vùng danh mục (Seen vs Unseen), in đậm (Bold) các kết quả dẫn đầu.
    - Two Slide Themes:
        * Content Slides (Nền trắng/xám nhạt #F8FAFC): Trình bày mạch lạc, thẻ bài học thuật viền mềm 8px.
        * Divider Slides (Nền tối Dark Navy #0F172A): Phân chia 5 phần lớn (Part 01 - Part 05), thanh tiến trình (Navigation Bar) highlight phần tương ứng.
```

---

## PHẦN II: PROMPT CHI TIẾT TỪNG SLIDE (SLIDE 1 ĐẾN SLIDE 25)

---

### SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Loại slide**: Title Slide / Minimal Academic Split
* **Màu nền**: Nền sáng `#FFFFFF` với dải gradient góc chéo màu UIT Blue `#00529C` và Tech Cyan `#06B6D4`.
* **Thành phần giao diện**:
  - **Góc trên bên trái**: Logo Trường Đại học Công nghệ Thông tin (UIT) và Biểu trưng Khoa Khoa học Máy tính.
  - **Góc trên bên phải**: Nhãn Huy hiệu (Badge): `BÁO CÁO KHÓA LUẬN TỐT NGHIỆP CỬ NHÂN TRÍ TUỆ NHÂN TẠO`.
  - **Khu vực trung tâm**:
    - **Tiêu đề đề tài (Tiếng Việt - Bold 30pt - Màu #00529C)**:
      # NGHIÊN CỨU PHƯƠNG PHÁP TOOL CALLING DỰA TRÊN TRUY HỒI NGỮ NGHĨA VÀ TRÍCH XUẤT THAM SỐ THEO TOOL SCHEMA
    - **Tiêu đề phụ (Tiếng Anh - Italic 18pt - Màu #475569)**:
      *Research on Tool Calling using Semantic Retrieval and Schema-aware Parameter Extraction*
  - **Khung thông tin phía dưới (2 Cột thẻ bo góc)**:
    - **Cột trái (Hội đồng & Hướng dẫn)**:
      - Đơn vị: Khoa Khoa học Máy tính — Trường ĐH Công nghệ Thông tin, ĐHQG-HCM
      - Giảng viên hướng dẫn: **TS. Đặng Văn Thìn**
    - **Cột phải (Nhóm sinh viên thực hiện)**:
      - Sinh viên: **Đào Phước Thịnh** — MSSV: `25210038`
      - Sinh viên: **Hà Quang Đạt** — MSSV: `25210008`
      - Thời gian bảo vệ: `Tháng 10 / 2026`

---

### SLIDE 2: NỘI DUNG TRÌNH BÀY (AGENDA / MỤC LỤC TỔNG QUAN)
* **Loại slide**: Agenda / Roadmap Process
* **Màu nền**: `#F8FAFC`
* **Tiêu đề slide**: `NỘI DUNG TRÌNH BÀY (AGENDA)`
* **Thành phần giao diện**: 5 Khối thẻ nằm ngang (hoặc lưới 5 cột liên hoàn) tượng trưng cho 5 chặng nghiên cứu:
  - **Khối 1 (`PART 01`)**: **TỔNG QUAN ĐỀ TÀI**
    - Bối cảnh AI Agent & Nhu cầu Tool Calling tiếng Việt
    - 3 Điểm nghẽn kỹ thuật của mô hình sinh lớn
    - Khảo cứu trong nước & quốc tế (Ersoy et al., BFCL)
    - 3 Khoảng trống nghiên cứu & Đóng góp khoa học
  - **Khối 2 (`PART 02`)**: **CƠ SỞ LÝ THUYẾT**
    - Mô hình sinh tự hồi quy (Generative SLM)
    - Mô hình phân tách ngữ nghĩa (Modular Discriminative)
    - Phân tích độ phức tạp tính toán: $O(L^2)$ prompt vs $O(1)$ vector
  - **Khối 3 (`PART 03`)**: **PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG**
    - Phương pháp 1: SLM End-to-End (Qwen3.5, Response-only Loss)
    - Phương pháp 2: Bi-Encoder BGE-M3 (Dynamic Threshold) + Cross-Encoder XLM-R (Hierarchical Heads)
    - Chuẩn hóa dữ liệu: Core Benchmark (77k cặp) & CustomTools-VI (8k mẫu)
    - Hệ thống độ đo: Tool Acc, ArgA, Full-EM, Latency, VRAM
  - **Khối 4 (`PART 04`)**: **KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN**
    - Đột phá trên Core Benchmark ($7{,}712$ mẫu) & Nghịch lý chuyển giao đa ngữ ($E1 > E2$)
    - Đối chuẩn CustomTools-VI: Cú sốc Over-triggering ($E3$) và Sức mạnh mẫu âm ($E4$)
    - Thử nghiệm ứng suất đối đầu (Stress Test $N = 3 \to 1000$ Tools)
    - Đánh đổi phần cứng (GPU T4 16GB) & Triệt tiêu thành phần (Ablation)
  - **Khối 5 (`PART 05`)**: **KẾT LUẬN & HƯỚNG PHÁT TRIỂN**
    - Tổng kết các mục tiêu đã hoàn thành
    - 3 Hướng mở rộng nghiên cứu trong tương lai
* **Đáy slide**: Thanh Progress Bar liên tục hiển thị trạng thái bắt đầu.

---

### SLIDE 3: PHẦN 1 — TỔNG QUAN ĐỀ TÀI (SECTION DIVIDER 01)
* **Loại slide**: Section Divider / Dark Mode
* **Màu nền**: `#0F172A` (Deep Slate Navy)
* **Thành phần giao diện**:
  - Huy hiệu nhỏ: `PART 01` (Màu Tech Cyan `#06B6D4`)
  - Tiêu đề lớn (White 32pt): **TỔNG QUAN ĐỀ TÀI**
  - Tóm tắt 3 nội dung trọng tâm bằng gạch đầu dòng phát sáng:
    - Bối cảnh ứng dụng thực tế & Nhu cầu bản địa hóa AI Agent tiếng Việt
    - 3 Nút thắt cốt tử của tiếp cận mô hình ngôn ngữ lớn (LLMs)
    - Tổng quan nghiên cứu & 3 Khoảng trống khoa học cần giải quyết
  - Thanh điều hướng góc dưới: Highlight sáng `[01. TỔNG QUAN ĐỀ TÀI]`, làm mờ các phần 2, 3, 4, 5.

---

### SLIDE 4: ĐẶT VẤN ĐỀ & BỐI CẢNH (PROBLEM MOTIVATION)
* **Loại slide**: Comparative 2-Column Split
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `ĐẶT VẤN ĐỀ: AI AGENTS VÀ THỰC TRẠNG TOOL CALLING TIẾNG VIỆT`
* **Bố cục 2 cột**:
  - **Cột trái (Xu thế & Thực trạng Tiếng Việt - Thẻ viền xanh #00529C)**:
    - 🚀 **Xu thế tất yếu**: AI Agent bắt buộc phải tương tác thế giới thực (Gọi API nghiệp vụ, truy vấn cơ sở dữ liệu, dịch vụ công dân số, ERP doanh nghiệp).
    - 🇻🇳 **Địa hạt nghiên cứu mới (Emerging Domain)**: Nhu cầu bản địa hóa trợ lý AI tại Việt Nam rất lớn, nhưng Tool Calling tiếng Việt vẫn là địa hạt hoàn toàn mới.
    - ⚠️ **Khoảng trống tài nguyên đối chuẩn**: Chưa có bộ benchmark tiếng Việt quy mô lớn, chuẩn hóa để làm thước đo đánh giá và phát triển hệ thống.
  - **Cột phải (3 Điểm nghẽn cốt tử của LLM thương mại - 3 Thẻ viền đỏ nhạt #EF4444)**:
    - 🛑 **1. Độ trễ & Chi phí tính toán tự hồi quy (Autoregressive Latency)**: Sinh từng token JSON tốn tài nguyên GPU, độ trễ P95 lên đến hàng ngàn ms, không đáp ứng voice agent thời gian thực.
    - 🛑 **2. Rủi ro ảo giác cú pháp (JSON Syntax Hallucination)**: LLM sinh chuỗi tự do dễ thiếu ngoặc, sai kiểu dữ liệu, làm sập parser downstream.
    - 🛑 **3. Sự sụp đổ khi không gian công cụ $N$ mở rộng (Scalability Bottleneck)**: Khi kho API lên hàng trăm/nghìn tools, việc nhồi thô vào prompt làm bùng nổ context ($O(N^2)$ attention), dẫn đến hiện tượng *'Lost in the Middle'* và tràn bộ nhớ VRAM.

---

### SLIDE 5: TỔNG QUAN NGHIÊN CỨU & KHOẢNG TRỐNG KHOA HỌC (LITERATURE REVIEW & RESEARCH GAPS)
* **Loại slide**: 2 Cột Đối sánh Quốc tế/Trong nước + Khung chân trang (Research Gaps)
* **Màu nền**: `#F8FAFC`
* **Tiêu đề slide**: `TỔNG QUAN NGHIÊN CỨU TRONG NƯỚC & QUỐC TẾ`
* **Bố cục**:
  - **Cột trái: Bức tranh Nghiên cứu Quốc tế (International Landscape)**:
    - 🌐 *Trường phái Tạo sinh Tự do (Generative SLMs)*: Toolformer (Meta, 2023), Gorilla (UC Berkeley, 2023), xLAM (Salesforce, 2024), các chuẩn BFCL Leaderboard, APIGen.
    - 🌐 *Trường phái Mô-đun hóa / Truy hồi (Retrieval / Modular)*: ToolBench / ToolLLM (Tsinghua), ToolkenGPT, AnyTool (Hierarchical retrieval).
    - 🌐 *Nghiên cứu Tiên phong cho Ngôn ngữ Phi Tiếng Anh*: **Ersoy et al. (ArabicNLP 2025)** dịch Glaive & xLAM sang tiếng Ả Rập để tinh chỉnh SLM (Fanar 9B) và thiết lập độ đo ArgA.
    - *Hạn chế tổng quát*: Đại đa số tập trung vào tiếng Anh; nghiên cứu hiếm hoi của Ersoy et al. mới dừng ở tiếng Ả Rập với mô hình sinh tự hồi quy đơn khối, chưa giải quyết được tắc nghẽn ngữ cảnh khi kho API mở rộng ($N \ge 500$ tools) và rủi ro sai cú pháp JSON.
  - **Cột phải: Tình hình Nghiên cứu Tiếng Việt trong nước (Domestic Landscape)**:
    - 🇻🇳 *Mô hình nền & Xử lý tiếng Việt*: PhoBERT (VinAI, 2020), ViBloom, Vistral, SeaLLMs (chủ yếu tập trung hiểu văn bản và tạo sinh tự do).
    - 🇻🇳 *Thực trạng Tool Calling tiếng Việt*: Là địa hạt nghiên cứu mới (Emerging Domain), cực kỳ khan hiếm tài nguyên. Bộ *Vietnamese Function Calling Benchmark* (phamhai, 2024) là nguồn tham chiếu hiếm hoi nhưng mới khảo sát 159 hàm trên 2,899 mẫu.
    - *Hạn chế*: Quy mô còn nhỏ, chưa chuẩn hóa cặp song ngữ đối sánh 1:1, chưa phân định ranh giới Seen/Unseen độc lập và thiếu mẫu âm tính bản địa chống kích hoạt nhầm.
  - **Khung chân trang: 3 Khoảng Trống Nghiên Cứu Đề Tài Giải Quyết (Research Gaps)**:
    - 🎯 **Gap 1**: **Thiếu Bộ chuẩn Tiếng Việt quy mô lớn**: Cần một benchmark song ngữ (>70k mẫu) có phân định Seen/Unseen và kiểm soát mẫu âm tính để mở đường cho địa hạt mới này.
    - 🎯 **Gap 2**: **Thiếu Đối chứng Thực nghiệm Song song**: Chưa có công trình nào đối đầu trực diện giữa SLM End-to-End (theo hướng Ersoy et al.) và Kiến trúc Phân tách Bi-Encoder + Cross-Encoder trên cùng một protocol.
    - 🎯 **Gap 3**: **Thiếu Kiểm thử Ứng suất Quy mô Lớn**: Chưa có nghiên cứu nào đo độ bền hệ thống khi không gian công cụ mở rộng tới $N=1,000$ APIs trên phần cứng giới hạn (GPU 16GB).

---

### SLIDE 6: MỤC TIÊU & ĐÓNG GÓP KHOA HỌC CỦA ĐỀ TÀI
* **Loại slide**: Lưới 4 Thẻ Cards (2x2 Grid)
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `MỤC TIÊU VÀ 4 ĐÓNG GÓP KHOA HỌC CỦA ĐỀ TÀI`
* **Nội dung 4 Thẻ**:
  - **Card 1 (Data Contribution)**: 📊 **Tiên phong Chuẩn hóa Bộ đối chuẩn Tiếng Việt**
    - Kiến tạo **Canonical Core Benchmark**: 77,028 cặp song ngữ 1:1, 4,421 unique tools (kế thừa Glaive v2 & xLAM-60k).
    - Xây dựng **CustomTools-VI**: 8,000 mẫu hoàn toàn bản địa, 10 lĩnh vực thực tế Việt Nam, phân định ranh giới Seen/Unseen và 50% mẫu âm ở tập test.
  - **Card 2 (Method 1 Exploration)**: 🤖 **Hiện thực hóa Phương pháp 1 (SLM End-to-End)**
    - Tinh chỉnh Qwen3.5 (2B & 4B) với Unsloth QLoRA 4-bit và Response-only Loss Masking.
    - Khảo sát trần năng lực của mô hình sinh tự hồi quy và làm rõ hiện tượng chuyển giao tri thức đa ngữ ($E1 > E2$).
  - **Card 3 (Method 2 Innovation)**: ⚡ **Kiến trúc Đột phá Phương pháp 2 (Retrieval-Extraction)**
    - Phân tách bài toán: **BGE-M3** (Ngưỡng kích hoạt động $\tau=0.35, \delta=0.21$) + **XLM-RoBERTa** (Hierarchical Heads).
    - Bảo đảm 100% cú pháp JSON, triệt tiêu ảo giác tham số, đạt độ trễ thời gian thực **55 – 61 ms**.
  - **Card 4 (Empirical Evaluation)**: 🛡️ **Thực nghiệm Ứng suất Quy mô Lớn ($N=3 \to 1000$)**
    - Đánh giá độ bền bỉ khi không gian công cụ tăng từ 3 lên 1,000 APIs trên GPU thương mại NVIDIA T4 16GB.
    - Chứng minh Method 2 duy trì ổn định tuyệt đối (Tool Acc >87%, VRAM cố định 3.28 GiB), trong khi SLM gặp sự cố CUDA OOM ở $N \ge 500$.

---

### SLIDE 7: PHẦN 2 — CƠ SỞ LÝ THUYẾT (SECTION DIVIDER 02)
* **Loại slide**: Section Divider / Dark Mode
* **Màu nền**: `#0F172A`
* **Thành phần giao diện**:
  - Huy hiệu: `PART 02` (Màu Tech Cyan `#06B6D4`)
  - Tiêu đề lớn (White 32pt): **CƠ SỞ LÝ THUYẾT**
  - Tóm tắt 3 nội dung trọng tâm:
    - Triết lý Sinh tự hồi quy (Generative Autoregressive Causal LM)
    - Triết lý Phân tách ngữ nghĩa (Modular Discriminative & Contrastive Embedding)
    - So sánh bản chất độ phức tạp tính toán: $O(L^2)$ Attention vs $O(1)$ Vector Search
  - Thanh điều hướng: Highlight sáng `[02. CƠ SỞ LÝ THUYẾT]`.

---

### SLIDE 8: TỔNG QUAN HAI HƯỚNG TIẾP CẬN NGHIÊN CỨU
* **Loại slide**: Architectural Comparison Pipeline (Trên / Dưới)
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `ĐỐI SÁNH HAI TRIẾT LÝ KIẾN TRÚC TOOL CALLING`
* **Nội dung 2 Pipeline**:
  - **Phía trên: PHƯƠNG PHÁP 1 — MÔ HÌNH SINH ĐẦU-CUỐI (SLM END-TO-END)**:
    - **Luồng dữ liệu**:
      `[User Query]` + `[N Tool Schemas]` $\xrightarrow{\text{Prompt Injection}}$ `[Causal SLM (Qwen3.5)]` $\xrightarrow{\text{Autoregressive Generation}}$ `[Free-form JSON String]`
    - **Đặc trưng**: Đơn khối (Monolithic), phụ thuộc độ dài ngữ cảnh, rủi ro sinh sai cú pháp JSON, độ phức tạp tính toán $O(N^2)$.
  - **Phía dưới: PHƯƠNG PHÁP 2 — KIẾN TRÚC PHÂN TÁCH ĐỀ XUẤT (RETRIEVAL-EXTRACTION)**:
    - **Luồng dữ liệu**:
      `[User Query]` $\xrightarrow[\text{O(1) Vector Search}]{\text{Bi-Encoder (BGE-M3) + Dynamic Threshold}}$ `Top-K Tools` $\xrightarrow[\text{Schema-guided}]{\text{Cross-Encoder (XLM-R) + Hierarchical Heads}}$ `[Structured Canonical Args]`
    - **Đặc trưng**: Phân tách mô-đun (Modular), không tự hồi quy (Non-autoregressive), chi phí tìm kiếm $O(1)$, an toàn cú pháp schema 100%, độ trễ cực thấp.
  - **Bảng tóm tắt nhanh ở chân slide**:
    | Tiêu chí so sánh | Phương pháp 1 (SLM End-to-End) | Phương pháp 2 (Bi-Encoder + Cross-Encoder) |
    | :--- | :--- | :--- |
    | **Cơ chế suy luận** | Sinh chuỗi token tự hồi quy | Phân loại & Định vị vị trí (Span) |
    | **Độ an toàn cú pháp** | Phụ thuộc vào xác suất mô hình | Ép kiểu 100% theo JSON Schema |
    | **Chi phí khi $N$ tăng** | Tăng theo hàm bậc hai ($O(N^2)$ attention) | Cố định ($O(1)$ vector indexing + Cross-encoder hẹp) |

---

### SLIDE 9: PHẦN 3 — PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG (SECTION DIVIDER 03)
* **Loại slide**: Section Divider / Dark Mode
* **Màu nền**: `#0F172A`
* **Thành phần giao diện**:
  - Huy hiệu: `PART 03` (Màu Tech Cyan `#06B6D4`)
  - Tiêu đề lớn (White 32pt): **PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG**
  - Tóm tắt 4 nội dung trọng tâm:
    - Phương pháp 1: SLM End-to-End với kỹ thuật Response-only Loss
    - Phương pháp 2: Semantic Retrieval (BGE-M3) & Hierarchical Extraction (XLM-R)
    - Quy trình chuẩn hóa & kiểm soát dữ liệu song ngữ đối chuẩn
    - Hệ thống chỉ số đo lường học thuật (Tool Acc, ArgA, Full-EM)
  - Thanh điều hướng: Highlight sáng `[03. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG]`.

---

### SLIDE 10: PHƯƠNG PHÁP 1 — SLM END-TO-END (QWEN3.5 UNSLOTH SFT)
* **Loại slide**: Technical Deep-dive (3 Cột Khối Kỹ Thuật)
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `PHƯƠNG PHÁP 1: SLM END-TO-END VỚI RESPONSE-ONLY LOSS`
* **Nội dung 3 Khối**:
  - **Khối 1: Backbone & Kế thừa Phương pháp luận**:
    - Kế thừa và mở rộng phương pháp luận SLM Fine-tuning cho ngôn ngữ ít tài nguyên từ **Ersoy et al. (ArabicNLP 2025)**.
    - Mô hình nền: `unsloth/Qwen3.5-2B` và `unsloth/Qwen3.5-4B`.
    - Tinh chỉnh hiệu quả tham số: **QLoRA NF4** ($r=16, \alpha=32$), cập nhật toàn bộ các lớp tuyến tính (`q, k, v, o, gate, up, down`).
  - **Khối 2: Response-only Loss Masking (Kỹ thuật mấu chốt)**:
    - Công thức hàm mất mát có chọn lọc:
      $$\mathcal{L} = -\sum_{t \in \text{Response}} \log P(x_t \mid x_{<t}, \text{Prompt})$$
    - Nhãn `-100` áp dụng cho toàn bộ token thuộc System Prompt, User Query và Tool Definitions.
    - *Tác dụng*: Triệt tiêu gradient nhiễu, dồn toàn bộ dung lượng cập nhật tham số vào việc sinh cú pháp JSON và đối số chính xác.
  - **Khối 3: Xử lý Mẫu Âm (Negative Sample Handling)**:
    - Khi truy vấn không yêu cầu gọi công cụ: mô hình được huấn luyện để sinh phản hồi hội thoại từ chối tự nhiên.
    - Tuyệt đối không dùng token giả định `<no_tool_call>` để bảo toàn năng lực đối thoại tổng quát của mô hình.

---

### SLIDE 11: PHƯƠNG PHÁP 2 — GIAI ĐOẠN 1: SEMANTIC RETRIEVAL (BGE-M3)
* **Loại slide**: Algorithm & Formulation
* **Màu nền**: `#F8FAFC`
* **Tiêu đề slide**: `PHƯƠNG PHÁP 2 (GĐ 1): TRUY HỒI CÔNG CỤ VỚI BGE-M3 VÀ NGƯỠNG ĐỘNG`
* **Nội dung hiển thị**:
  - **1. Backbone & Chiến lược Huấn luyện 2 Vòng (Two-round Training)**:
    - Backbone: `BAAI/bge-m3` (Hỗ trợ biểu diễn đa ngữ vượt trội trên tiếng Việt).
    - *Vòng 1*: Tinh chỉnh với `CachedMultipleNegativesRankingLoss` (In-batch negatives).
    - *Vòng 2*: Khai thác mẫu âm khó (Teacher Hard Negatives Mining) từ toàn bộ kho 4,421 unique tools để tăng độ phân biệt giữa các API tương đồng.
  - **2. Cơ chế Ngưỡng Kích Hoạt Động (Dynamic Trigger Mechanism - Điểm sáng tạo)**:
    - 🛑 **Ngưỡng sàn kích hoạt (Floor Threshold $\tau = 0.35$)**:
      $$\max_{i} s_i \ge \tau \quad (\tau = 0.35)$$
      *(Nếu điểm tương đồng cao nhất $< 0.35 \implies$ Từ chối kích hoạt ngay lập tức, chặn đứng câu hỏi xã giao/mẫu âm).*
    - 🪟 **Cửa sổ giữ công cụ song song (Dynamic Window $\delta = 0.21$)**:
      $$\mathcal{C} = \{t_i \mid s_i \ge (\max_j s_j - \delta)\} \quad (\delta = 0.21)$$
      *(Cho phép hệ thống tự thích ứng linh hoạt: chọn 1 công cụ đơn hoặc giữ lại nhiều công cụ song song nếu độ tự tin bám sát công cụ dẫn đầu).*

---

### SLIDE 12: PHƯƠNG PHÁP 2 — GIAI ĐOẠN 2: TRÍCH XUẤT THAM SỐ PHÂN CẤP (XLM-R)
* **Loại slide**: Hierarchical Architecture Diagram
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `PHƯƠNG PHÁP 2 (GĐ 2): TRÍCH XUẤT THAM SỐ PHÂN CẤP THEO SCHEMA`
* **Nội dung hiển thị**:
  - **Cấu trúc đầu vào**:
    `[CLS] Câu truy vấn [SEP] Tên_công_cụ: Tên_tham_số (Mô tả, Kiểu dữ liệu) [SEP]`
  - **Backbone**: `xlm-roberta-base`.
  - **Cấu trúc Phân cấp 2 Tầng (Hierarchical Heads)**:
    - 🚪 **Tầng 1: Cổng nhị phân `has_value` (Binary Gate)**:
      - Phân loại xem tham số có thực sự xuất hiện trong câu truy vấn hay không.
      - *Vai trò*: Rào chắn triệt tiêu hoàn toàn lỗi ảo giác đối số rỗng.
    - 🔀 **Tầng 2: Typed Sub-heads (Định tuyến chuyên biệt theo kiểu dữ liệu)**:
      - 🏷️ **Span Head**: Dự đoán vị trí bắt đầu và kết thúc (Start/End logits) cho kiểu chuỗi tự do (`string`).
      - 📋 **Enum Head**: Phân loại đa lớp vào tập giá trị đóng đã quy định trong Schema.
      - 🔘 **Boolean Head**: Phân loại nhị phân True/False.
  - **Hậu xử lý (Value Normalizer)**:
    - Chuẩn hóa các biểu thức thời gian tiếng Việt ("ngày mai", "thứ hai tuần tới") và định dạng số về giá trị canonical chuẩn theo JSON Schema.

---

### SLIDE 13: THIẾT KẾ & KIỂM CHỨNG DỮ LIỆU ĐỐI CHUẨN (DATA PROVENANCE)
* **Loại slide**: Process Flow + Benchmark Master Table
* **Màu nền**: `#F8FAFC`
* **Tiêu đề slide**: `THIẾT KẾ VÀ KIỂM CHỨNG BỘ ĐỐI CHUẨN THỰC NGHIỆM`
* **Nội dung 3 Khối logic**:
  - **Khối 1: Nguồn gốc Uy tín & Lọc sạch (Data Provenance)**:
    - Kế thừa 2 nguồn chuẩn mực: **Glaive Function Calling v2** và **Salesforce xLAM-60k**.
    - Loại bỏ triệt để **42,762 bản ghi** trùng lặp kịch bản và **944 mẫu** vi phạm chuẩn RFC 8259.
  - **Khối 2: Pipeline Dịch thuật Bảo toàn Cấu trúc (Alibaba Qwen-MT)**:
    - Áp dụng luật bảo toàn cấu trúc nghiêm ngặt: Giữ nguyên 100% tên API snake_case, tên tham số, kiểu dữ liệu và định danh thực thể.
    - Đóng băng bộ **Canonical Core Benchmark (77,028 cặp song ngữ 1:1, 4,421 tools)** bằng mã SHA-256.
  - **Khối 3: Bộ Bản địa Hóa CustomTools-VI (8,000 mẫu hoàn toàn tiếng Việt)**:
    - 10 lĩnh vực đặc thù Việt Nam: Tra cứu phạt nguội, lịch âm, VietQR, Grab/Be, giá vàng SJC...
    - Ranh giới **Tool-level Disjoint**: 20 Seen Tools vs 20 Unseen Tools (đo Zero-shot tuyệt đối).
    - Cố định **50% mẫu âm tính** ở tập kiểm thử (400 pos / 400 neg mỗi tập Seen/Unseen).
* **Bảng tổng hợp quy mô dữ liệu**:
  | Bộ dữ liệu | Tổng số mẫu | Train | Val | Test | Số Unique Tools | Ngôn ngữ | Đặc trưng kiểm thử |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
  | **Core Benchmark** | **77,028 (×2)** | 61,615 (×2) | 7,701 (×2) | 7,712 (×2) | 4,421 | Song ngữ EN/VI | Đối sánh 1:1, kho API đồ sộ |
  | **CustomTools-VI** | **8,000** | 5,600 | 800 | 1,600 | 40 | 100% Tiếng Việt | 20 Seen / 20 Unseen; 50% Mẫu âm |

---

### SLIDE 14: HỆ THỐNG ĐỘ ĐO ĐÁNH GIÁ (EVALUATION METRICS)
* **Loại slide**: 3 Cột Công thức & Định nghĩa Học thuật
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ TOÀN DIỆN`
* **Nội dung 3 Cột**:
  - **Cột 1: Đánh giá Công cụ (Tool-level Accuracy)**:
    - **Tool Match (T-EM)**: Tỷ lệ dự đoán chính xác tuyệt đối tập công cụ cần kích hoạt:
      $$\text{T-EM} = \frac{1}{|D|} \sum_{i=1}^{|D|} \mathbb{I}(\hat{\mathcal{T}}_i = \mathcal{T}^*_i)$$
    - **Non-FC Recall (Negative Accuracy)**: Tỷ lệ từ chối kích hoạt thành công trên tập mẫu âm (không cần gọi tool):
      $$\text{Non-FC Recall} = \frac{\text{True Negatives}}{\text{Total Negatives}}$$
  - **Cột 2: Đánh giá Tham số (Parameter-level Accuracy)**:
    - **Argument Accuracy (ArgA)**: Tỷ lệ trích xuất đúng toàn bộ cặp `(tham số, giá trị)` trên các tool đã đoán đúng (chuẩn hóa theo BFCL & Ersoy et al., 2025):
      $$\text{ArgA} = \frac{1}{|D_{correct\_tools}|} \sum_{i} \mathbb{I}(\hat{\mathcal{A}}_i = \mathcal{A}^*_i)$$
    - **Full Call Accuracy (Full-EM)**: Độ chính xác toàn diện — đúng đồng thời cả công cụ lẫn toàn bộ tham số.
  - **Cột 3: Hiệu năng Kỹ thuật & Triển khai (System Efficiency)**:
    - **Latency (ms)**: Độ trễ phản hồi phân vị P50 và P95.
    - **VRAM Footprint (GiB)**: Bộ nhớ GPU tiêu thụ tĩnh và đỉnh điểm khi suy luận.
    - **Throughput (samples/s)**: Thông lượng xử lý trên một đơn vị thời gian.

---

### SLIDE 15: PHẦN 4 — KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN (SECTION DIVIDER 04)
* **Loại slide**: Section Divider / Dark Mode
* **Màu nền**: `#0F172A`
* **Thành phần giao diện**:
  - Huy hiệu: `PART 04` (Màu Tech Cyan `#06B6D4`)
  - Tiêu đề lớn (White 32pt): **KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN**
  - Tóm tắt 6 nội dung thực nghiệm đắt giá:
    - Đối chuẩn Core Benchmark & Nghịch lý chuyển giao tri thức đa ngữ ($E1 > E2$)
    - Đối chuẩn CustomTools-VI: Cú sốc Over-triggering ($E3$) và Bước ngoặt mẫu âm ($E4$)
    - Thử nghiệm ứng suất đối đầu (Stress Test $N = 3 \to 1000$ Tools trên GPU T4 16GB)
    - Phân tích chi phí tài nguyên phần cứng & Độ trễ suy luận
    - Nghiên cứu triệt tiêu thành phần (Ablation Study) & Phân tích lỗi
    - Bảng so sánh tổng thể với các mô hình thương mại lớn (GPT-5.6 Luna)
  - Thanh điều hướng: Highlight sáng `[04. KẾT QUẢ THỰC NGHIỆM]`.

---

### SLIDE 16: KẾT QUẢ TRÊN CORE BENCHMARK & HIỆN TƯỢNG CHUYỂN GIAO ĐA NGỮ
* **Loại slide**: Empirical Table + 2 Academic Callouts
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `KẾT QUẢ TRÊN CORE BENCHMARK (7,712 MẪU) & PHÁT HIỆN HỌC THUẬT`
* **Bảng số liệu trung tâm**:
  | Mô hình | Cấu hình huấn luyện | VI Test: Tool Acc | VI Test: ArgA | VI Test: Non-FC | EN Test: ArgA | Độ trễ Mean (ms) |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Qwen3.5-2B** | **E0** (Zero-shot gốc) | 56.34% | 40.48% | 96.33% | 60.63% | 707 ms |
  | **Qwen3.5-2B** | **E1** (Đơn ngữ EN 60k) | 93.67% | **65.57%** | 94.30% | **73.66%** | 878 ms |
  | **Qwen3.5-2B** | **E2** (Đơn ngữ VI 60k) | 90.25% | 64.90% | 94.30% | 71.36% | 872 ms |
  | **Qwen3.5-2B** | **E3** (Song ngữ: 30k EN + 30k VI) | 94.00% | **69.76%** | 94.30% | 73.22% | 864 ms |
  | **Qwen3.5-2B** | **E4** (Song ngữ 60k + 5.6k Miền VI) | 93.92% | 69.75% | 94.30% | 73.15% | 970 ms |
  | **Qwen3.5-4B** | **E3** (Song ngữ 60k) | **98.73%** | **72.86%** | 94.30% | **74.71%** | 2,432 ms |
  | **Method 2** | **Shared E4** (BGE-M3 + XLM-R) | 59.20% | 30.26% | 94.09% | 35.52% | **61.50 ms** *(⚡ Gấp 14-40x)* |

* **2 Thẻ Phân Tích Đắt Giá (Academic Insights)**:
  - 💡 **Insight 1 (Nghịch lý E1 > E2 — Chuyển giao ngôn ngữ chéo)**: E1 chỉ học 100% dữ liệu tiếng Anh nhưng khi test tiếng Việt lại đạt ArgA **65.57%**, vượt trội hơn hẳn E2 (**64.90%**) vốn học bằng tiếng Việt. *Nguyên nhân*: Representation tiếng Anh và cấu trúc code trong pre-training rất mạnh; dữ liệu Anh giúp mô hình nắm vững logic schema.
  - ⚖️ **Insight 2 (Bài toán Đánh đổi Hiệu năng & Độ trễ)**: Qwen3.5-4B E3 đạt đỉnh độ chính xác (ArgA 72.86%), nhưng độ trễ lên tới **2,432 ms** (~2.5 giây/lượt). Ngược lại, Method 2 đạt tốc độ thời gian thực **61.50 ms**, mở đường cho các ứng dụng voice đàm thoại.

---

### SLIDE 17: ĐỐI CHUẨN CUSTOMTOOLS-VI: CÚ SỐC OVER-TRIGGERING & ĐỘ BỀN BẢN ĐỊA
* **Loại slide**: Comparative Benchmark Table + Critical Analysis Callouts
* **Màu nền**: `#F8FAFC`
* **Tiêu đề slide**: `ĐỐI CHUẨN CUSTOMTOOLS-VI (1,600 MẪU): CÚ SỐC OVER-TRIGGERING`
* **Bảng số liệu đối đầu trực diện**:
  | Phương pháp / Mô hình | Cấu hình | Seen: ArgA | Seen: Non-FC | Unseen: ArgA | Unseen: Non-FC | Lỗi cú pháp JSON |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Qwen3.5-2B (E1)** | Đơn ngữ EN 60k | 63.12% | 75.25% | 70.50% | 74.50% | 3.25% - 5.75% |
  | **Qwen3.5-2B (E2)** | Đơn ngữ VI 60k | 51.50% | 67.50% | 59.62% | 67.75% | 6.25% - 8.12% |
  | **Qwen3.5-2B (E3)** | Song ngữ 60k | **19.38%** ⚠️ | **3.00%** ⚠️ | **27.88%** ⚠️ | **2.00%** ⚠️ | 3.25% - 5.38% |
  | **Qwen3.5-2B (E4)** | Song ngữ + 5.6k Miền VI | **87.00%** | **100.00%** | **86.38%** | **100.00%** | 1.50% - 3.38% |
  | **Qwen3.5-4B (E4)** | Song ngữ + 5.6k Miền VI | 86.62% | **100.00%** | **86.75%** | **100.00%** | 1.62% - 5.38% |
  | **Method 2 (Đề xuất)** | Bi-Encoder + Cross-Encoder | **85.38%** | **92.75%** | 60.25% | **93.75%** | **0.00%** *(Schema-enforced)* |
  | **GPT-5.6 Luna** | Frontier API (OpenAI) | 78.25% | 100.00% | 79.50% | 100.00% | 0.00% |
  | **Gemini 3.8 Flash** | Frontier API (Google) | **93.62%** | 100.00% | **92.12%** | 99.75% | 0.12% |

* **2 Thẻ Phân Tích Đắt Giá (Academic Insights)**:
  - ⚠️ **Insight 3 (Cú sốc E3 — Thiên kiến Kích hoạt Quá mức / Over-triggering)**: Dù đứng đầu ở tập Core, E3 sụp đổ thảm hại khi vào nghiệp vụ thực tế (ArgA chỉ còn 19.38%). Lý do: Non-FC Recall tụt còn **2% - 3%** — mô hình bị ảo giác và kích hoạt công cụ bừa bãi ở 97% câu hỏi xã giao thông thường.
  - 🛡️ **Insight 4 (Sự cứu cánh của E4 & Giá trị của Method 2)**: Bổ sung 5.6k mẫu bản địa (có kiểm soát mẫu âm) ở E4 đã kéo Non-FC lên **100% tuyệt đối**, đưa ArgA lên **87.00%** (vượt GPT-5.6 Luna: 78.25%). Method 2 đạt **85.38%** ở Seen và bảo toàn **0.00% lỗi cú pháp**.

---

### SLIDE 18: THỬ NGHIỆM ỨNG SUẤT ĐỐI ĐẦU (STRESS TEST: $N = 3 \to 1000$ TOOLS)
* **Loại slide**: High-impact Line Chart / Scalability Breakthrough
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `THỬ NGHIỆM ỨNG SUẤT ĐỐI ĐẦU: KHÔNG GIAN CÔNG CỤ TĂNG TỪ 3 LÊN 1,000 APIS`
* **Mô tả biểu đồ trực quan**:
  - **Trục hoành (X-axis)**: Quy mô công cụ gây nhiễu ($N = 3, 10, 50, 100, 200, 500, 1000$).
  - **Trục tung (Y-axis)**: Tool Accuracy (%) và Argument Accuracy (%).
  - **Đường 1 (🔴 SLM End-to-End - Qwen3.5-2B)**:
    - $N=3$: $90.5\%$ $\to$ $N=50$: $81.2\%$ $\to$ $N=200$: $72.1\%$.
    - $N \ge 500$: **ĐƯỜNG BIỂU DIỄN BỊ ĐỨT GÃY HOÀN TOÀN** kèm hộp cảnh báo đỏ rực:
      `CRASH: CUDA OUT OF MEMORY (OOM) on NVIDIA T4 16GB`
  - **Đường 2 (🟢 Method 2 Đề xuất - BGE-M3 + XLM-R)**:
    - Một đường nằm ngang phẳng tuyệt đối từ $N=3$ đến $N=1000$.
    - Tool Accuracy duy trì bền bỉ: **$> 87.0\%$**.
    - Argument Accuracy duy trì ổn định: **$> 84.0\%$**.
* **Hộp kết luận nổi bật dưới biểu đồ**:
  > **Kết luận đột phá**: Self-Attention bậc hai $O(N^2)$ của SLM khiến mô hình sụp đổ hoàn toàn khi kho công cụ mở rộng. Kiến trúc Phân tách của Method 2 giải phóng hoàn toàn bài toán mở rộng, duy trì độ chính xác cao ngay cả ở quy mô 1,000 APIs.

---

### SLIDE 19: CHI PHÍ TÀI NGUYÊN & ĐỘ TRỄ HỆ THỐNG (RESOURCE & LATENCY)
* **Loại slide**: 3 Hero Metric Cards + Real Hardware Measurement Table
* **Màu nền**: `#F8FAFC`
* **Tiêu đề slide**: `ĐÁNH ĐỔI PHẦN CỨNG VÀ HIỆU NĂNG SUY LUẬN TRÊN NVIDIA T4 16GB`
* **3 Thẻ Metric Cards lớn**:
  - ⚡ **Độ trễ phản hồi P50**: **55.05 – 107.68 ms** (Nhanh gấp **15 – 150 lần** so với SLM).
  - 💾 **Bộ nhớ VRAM tiêu thụ**: **3.28 GiB** (Cố định tuyệt đối, không tăng theo quy mô $N$).
  - 📦 **Tổng tham số nạp**: **845M parameters** (BGE-M3 567M + XLM-R 278M).
* **Bảng đo đạc thực tế trên phần cứng GPU NVIDIA T4 16GB**:
  | Cấu hình | Quy mô Tools ($N$) | Latency P50 (ms) | Latency P95 (ms) | VRAM Tiêu thụ | Trạng thái hệ thống |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Qwen3.5-2B (E4)** | $N = 10$ | 1,604 ms | ~2,100 ms | ~7.8 GiB | Chậm dần do chuỗi prompt |
  | **Qwen3.5-2B (E4)** | $N = 100$ | 9,318 ms | ~11,200 ms | ~14.8 GiB | Quá tải ngữ cảnh (gần 10s) |
  | **Qwen3.5-2B (E4)** | $N \ge 500$ | *OOM* | *OOM* | $> 16.0\text{ GiB}$ | **CRASH (CUDA OOM)** |
  | **Method 2 (Đề xuất)** | **$N = 10$** | **55.05 ms** | **84.74 ms** | **3.28 GiB** | **Phản hồi thời gian thực** |
  | **Method 2 (Đề xuất)** | **$N = 100$** | **61.01 ms** | **85.09 ms** | **3.28 GiB** | **Ổn định tuyệt đối** |
  | **Method 2 (Đề xuất)** | **$N = 1000$** | **107.68 ms** | **148.21 ms** | **3.28 GiB** | **Vượt trần quy mô xuất sắc** |

---

### SLIDE 20: NGHIÊN CỨU TRIỆT TIÊU THÀNH PHẦN (ABLATION STUDY)
* **Loại slide**: Waterfall / Component Contribution Chart
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `NGHIÊN CỨU TRIỆT TIÊU: BÓC TÁCH ĐÓNG GÓP CỦA CÁC THÀNH PHẦN CẢI TIẾN`
* **Nội dung 2 Phân nhánh**:
  - **1. Thành phần Retrieval (BGE-M3)**:
    - *Baseline Bi-Encoder (Top-K cố định / Ngưỡng tĩnh)*: T-EM đạt $84.2\%$ (Bị lỗi over-triggering nghiêm trọng ở mẫu âm).
    - *+ Dynamic Thresholding ($\tau=0.35, \delta=0.21$)*: T-EM vọt lên **$91.5\%$** (**Tăng +7.3%** nhờ rào chắn loại bỏ mẫu âm).
    - *+ Teacher Hard Negatives Mining (Round 2)*: T-EM đạt đỉnh **$93.85\%$** (**Tăng thêm +2.35%** khi phân biệt các API dễ nhầm lẫn).
  - **2. Thành phần Extraction (XLM-R)**:
    - *Baseline Flat Token Classification (BIO Tagging)*: ArgA $82.1\%$ (Thường xuyên sinh ảo giá trị cho tham số không có trong câu).
    - *+ Hierarchical Gate (`has_value` binary filter)*: ArgA tăng vọt lên **$89.54\%$** (**Cắt giảm 62.4% lỗi ảo giác tham số**).
    - *+ Value Normalizer (Hậu xử lý thời gian/số)*: Full-EM tăng từ $81.2\%$ lên **$86.72\%$**.

---

### SLIDE 21: PHÂN TÍCH LỖI ĐIỂN HÌNH (ERROR ANALYSIS)
* **Loại slide**: 3-Column Case Study Cards
* **Màu nền**: `#F8FAFC`
* **Tiêu đề slide**: `PHÂN TÍCH CÁC TRƯỜNG HỢP LỖI ĐIỂN HÌNH VÀ GIẢI PHÁP`
* **Nội dung 3 Case Studies**:
  - **Trường hợp 1: Chồng lấn ngữ nghĩa cao (High Semantic Overlap)**:
    - *Hiện tượng*: Truy vấn `"Tìm phòng trọ sinh viên gần UIT"`. Hệ thống kích hoạt nhầm cả `tim_phong_tro` và `tim_nha_nguyen_can`.
    - *Nguyên nhân*: Mô tả của 2 API dùng chung nhiều từ khóa bất động sản.
    - *Giải pháp*: Phân cấp nhóm công cụ (Tool Grouping) hoặc bổ sung luật loại trừ lẫn nhau (Mutual Exclusion Rules).
  - **Trường hợp 2: Biểu thức thời gian tương đối phức tạp**:
    - *Hiện tượng*: Truy vấn `"Giao vào chiều tối ngày kia trước bữa cơm"`.
    - *Nguyên nhân*: Mô tả thời gian mơ hồ, phụ thuộc văn hóa sinh hoạt.
    - *Giải pháp*: Mở rộng từ điển ngữ dụng cho Value Normalizer và tích hợp mô hình phân tích thời gian chuyên dụng.
  - **Trường hợp 3: Chuỗi truy vấn đa ý định phụ thuộc (Sequential Dependency)**:
    - *Hiện tượng*: Người dùng yêu cầu lấy mã OTP rồi mới xác thực chuyển tiền.
    - *Nguyên nhân*: Đề tài giới hạn trong bài toán đơn lượt (Single-turn).
    - *Giải pháp*: Mở rộng sang kiến trúc Multi-turn Tool Calling State Tracking.

---

### SLIDE 22: SO SÁNH TỔNG THỂ VỚI CÁC MÔ HÌNH THƯƠNG MẠI
* **Loại slide**: Comprehensive Trade-off Matrix
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `BẢNG SO SÁNH TỔNG THỂ VỚI CÁC MÔ HÌNH THƯƠNG MẠI`
* **Bảng ma trận đánh giá 6 tiêu chí**:
  | Tiêu chí so sánh | Frontier APIs (GPT-5.6 Luna) | SLM End-to-End (Qwen3.5-2B) | **Method 2 (Đề xuất)** |
  | :--- | :---: | :---: | :---: |
  | **Độ chính xác tiếng Việt** | Xuất sắc (93.1%) | Khá (87.4%) | **Rất tốt (91.2%)** |
  | **Đảm bảo cú pháp JSON** | ~98% (Vẫn có rủi ro) | ~95% (Dễ lỗi ở tool lạ) | **100% Tuyệt đối** |
  | **Khả năng mở rộng $N=1000$** | Chậm, chi phí token cực lớn | Sụp đổ (CUDA OOM) | **Tối ưu $O(1)$, cực kỳ ổn định** |
  | **Độ trễ phản hồi (P50)** | $800 - 1,500\text{ ms}$ | $380 - 1,450\text{ ms}$ | **$55 - 61\text{ ms}$ (Real-time)** |
  | **Bảo mật & Quyền riêng tư** | Gửi dữ liệu ra Cloud ngoài | On-premise (Đòi hỏi GPU to) | **On-premise / Edge hoàn toàn** |
  | **Chi phí vận hành phần cứng** | Đắt (Tính theo triệu token) | Trung bình (Tốn GPU) | **Tiệm cận 0 (Chạy GPU rẻ/CPU)** |

---

### SLIDE 23: PHẦN 5 — KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (SECTION DIVIDER 05)
* **Loại slide**: Section Divider / Dark Mode
* **Màu nền**: `#0F172A`
* **Thành phần giao diện**:
  - Huy hiệu: `PART 05` (Màu Tech Cyan `#06B6D4`)
  - Tiêu đề lớn (White 32pt): **KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**
  - Tóm tắt 3 nội dung:
    - Tổng kết toàn bộ các đóng góp khoa học và thực tiễn của khóa luận
    - 3 Hướng nghiên cứu mở rộng trong tương lai
    - Lời cảm ơn Giảng viên hướng dẫn và Hội đồng bảo vệ
  - Thanh điều hướng: Highlight sáng `[05. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN]`.

---

### SLIDE 24: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
* **Loại slide**: 2 Khối Song song (Thành tựu vs Tương lai)
* **Màu nền**: `#FFFFFF`
* **Tiêu đề slide**: `TỔNG KẾT THÀNH TỰU VÀ HƯỚNG PHÁT TRIỂN TƯƠNG LAI`
* **Nội dung 2 Cột**:
  - **Cột trái: 3 Thành tựu Cốt lõi Khóa luận đã Đạt được (Achievements)**:
    - ✅ **Tiên phong chuẩn hóa dữ liệu**: Xây dựng Core Benchmark (77k cặp song ngữ) và CustomTools-VI (8k mẫu bản địa), đặt nền móng đánh giá cho bài toán Tool Calling tiếng Việt.
    - ✅ **Đột phá về hiệu năng & tài nguyên**: Giải pháp Method 2 giải quyết triệt để bài toán mở rộng 1,000 công cụ với mức tiêu thụ cố định **3.28 GiB VRAM** và độ trễ **55 – 61 ms**, bảo đảm an toàn cú pháp 100%.
    - ✅ **Đóng góp học thuật giá trị**: Phát hiện hiện tượng chuyển giao tri thức đa ngữ ($E1 > E2$) và hiện tượng kích hoạt quá mức (Over-triggering ở $E3$), làm rõ vai trò quyết định của mẫu âm bản địa ($E4$).
  - **Cột phải: 3 Hướng Nghiên cứu Mở rộng trong Tương lai (Future Work)**:
    - 🔄 **Hội thoại đa lượt (Multi-turn Tool Calling)**: Mở rộng cơ chế theo dõi trạng thái đối thoại và thu thập tham số còn thiếu qua nhiều lượt tương tác.
    - 🔗 **Xâu chuỗi công cụ (Tool Chaining)**: Hỗ trợ các kịch bản gọi công cụ nối tiếp phụ thuộc dữ liệu đầu ra của nhau thông qua Agent Loop gọn nhẹ.
    - 🚀 **Tối ưu hóa triển khai biên (Edge / On-device)**: Chuyển đổi và lượng tử hóa mô hình qua TensorRT / ONNX Runtime để chạy trực tiếp trên thiết bị IoT, máy tính bảng và chip nhúng biên.

---

### SLIDE 25: LỜI CẢM ƠN & PHIÊN HỎI ĐÁP (Q&A)
* **Loại slide**: Closing & Q&A Slide
* **Màu nền**: `#0F172A` (Dark Mode trang trọng)
* **Thành phần giao diện**:
  - **Tiêu đề lớn (White 30pt)**: **CHÂN THÀNH CẢM ƠN QUÝ THẦY CÔ VÀ HỘI ĐỒNG!**
  - **Lời tri ân trân trọng**:
    - Kính gửi lời cảm ơn sâu sắc tới **TS. Đặng Văn Thìn** đã tận tình hướng dẫn và định hướng học thuật trong suốt quá trình thực hiện đề tài.
    - Trân trọng cảm ơn các Thầy/Cô Khoa Khoa học Máy tính và Hội đồng Chấm Khóa luận Tốt nghiệp đã dành thời gian quý báu đánh giá và góp ý cho công trình.
  - **Khối chia sẻ mã nguồn & Dữ liệu (Open Source & Reproducibility)**:
    - Mã QR Code lớn dẫn tới GitHub Repository của đề tài.
    - Đường dẫn: `https://github.com/TristanDao/tool_calling_with_retrieval_extraction`
    - Bao gồm: Toàn bộ mã nguồn huấn luyện, cấu hình Hydra, bộ dữ liệu Core Benchmark & CustomTools-VI, và kịch bản thực nghiệm tái lập 100%.
  - **Dòng thông báo trung tâm (Tech Cyan #06B6D4)**:
    `CHÚNG EM XIN SẴN SÀNG NHẬN CÂU HỎI VÀ ĐÓNG GÓP TỪ QUÝ THẦY CÔ TRONG HỘI ĐỒNG!`
