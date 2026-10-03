# KỊCH BẢN THUYẾT TRÌNH, THIẾT KẾ SLIDE & BỘ CÂU HỎI BẢO VỆ KHÓA LUẬN TỐT NGHIỆP UIT

---

## THÔNG TIN HỌC THUẬT & HỘI ĐỒNG BẢO VỆ
- **Đơn vị đào tạo**: Khoa Khoa học Máy tính — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
- **Ngành đào tạo**: Cử nhân ngành Trí tuệ Nhân tạo
- **Tên đề tài chính thức**:
  - *Tiếng Việt*: **Nghiên cứu phương pháp Tool Calling dựa trên truy hồi ngữ nghĩa và trích xuất tham số theo Tool Schema**
  - *Tiếng Anh*: **Research on Tool Calling using Semantic Retrieval and Schema-aware Parameter Extraction**
- **Giảng viên hướng dẫn**: TS. Đặng Văn Thìn
- **Sinh viên thực hiện**:
  - Đào Phước Thịnh (MSSV: 25210038)
  - Hà Quang Đạt (MSSV: 25210008)
- **Thời lượng trình bày chuẩn**: 15 – 18 phút (Hỏi đáp phản biện: 15 – 20 phút)

---

# PHẦN 1: HƯỚNG DẪN PROMPT DESIGN CHO AGENT TẠO SLIDE

Nếu bạn sử dụng AI Agent (như Marp, Slidev, Gamma, Tome hoặc Claude/ChatGPT với Python-pptx) để sinh slide, hãy nạp đoạn cấu hình Design System này vào System Prompt:

```yaml
DesignSystem:
  Theme: Modern Academic Deep Tech
  Aspect Ratio: 16:9
  Color Palette:
    Primary: "#00529C"       # UIT Academic Blue
    Secondary: "#0F172A"     # Deep Slate / Dark Navy
    Accent: "#06B6D4"        # Tech Cyan
    Warning/Alert: "#F59E0B" # Amber Orange
    Background: "#FFFFFF"    # Clean White (hoặc Light Gray #F8FAFC)
    Card Background: "#F1F5F9"
    Text Primary: "#0F172A"
    Text Muted: "#475569"
  Typography:
    Headings: "Inter, Be Vietnam Pro, Arial, sans-serif" (Bold, 24-32pt)
    Body: "Inter, Roboto, sans-serif" (14-18pt)
    Code/Formulas: "JetBrains Mono, Fira Code, monospace" (12-14pt)
  Layout Principles:
    - Tuyệt đối không dùng đoạn văn dài (text wall).
    - Mọi slide đều chia Grid 2 cột hoặc Cards 3 cột rõ ràng.
    - Dùng Metric Cards (số to in đậm kèm nhãn nhỏ phía dưới).
    - Bắt buộc có sơ đồ Pipeline (Mermaid hoặc Flowchart) ở các slide kiến trúc.
    - Bảng số liệu phải highlight dòng/cột có kết quả tốt nhất.
    - Slide Mục lục (Slide 2): Bố cục 5 khối thẻ ngang thể hiện 5 phần chính và thanh tiến trình (Progress Bar).
    - Slide Mở chương (Divider Slides: 3, 7, 9, 15, 23): Dùng tông nền tối Dark Navy (#0F172A) tương phản cao, số phần lớn (PART 01–05), kèm thanh điều hướng Highlight mục hiện tại.
```

---

# PHẦN 2: CHI TIẾT TỪNG SLIDE & KỊCH BẢN THUYẾT TRÌNH (25 SLIDES)

---

### SLIDE 1: TRANG TIÊU ĐỀ (TITLE SLIDE)
* **Thời lượng**: 00:00 – 00:35 (35 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Agent (Design Prompt)
- **Layout**: Centered / Split hiện đại. Góc trên bên trái là Logo UIT và biểu trưng Khoa Khoa học Máy tính.
- **Tiêu đề chính (Bold & Lớn)**:
  # NGHIÊN CỨU PHƯƠNG PHÁP TOOL CALLING DỰA TRÊN TRUY HỒI NGỮ NGHĨA VÀ TRÍCH XUẤT THAM SỐ THEO TOOL SCHEMA
  *(Research on Tool Calling using Semantic Retrieval and Schema-aware Parameter Extraction)*
- **Thông tin định danh học thuật**:
  - Đơn vị: Khoa Khoa học Máy tính — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
  - Ngành: Cử nhân ngành Trí tuệ Nhân tạo
  - Giảng viên hướng dẫn: TS. Đặng Văn Thìn
  - Sinh viên thực hiện: Đào Phước Thịnh (MSSV: 25210038) & Hà Quang Đạt (MSSV: 25210008)
  - Khóa: 2022 – 2026

#### 2. Lời thoại thuyết trình (Script)
> *"Kính thưa quý Thầy Cô trong Hội đồng chấm Khóa luận tốt nghiệp Khoa Khoa học Máy tính, Trường Đại học Công nghệ Thông tin – ĐHQG-HCM, cùng toàn thể quý vị đang có mặt trong buổi bảo vệ ngày hôm nay.
> 
> Chúng em là Đào Phước Thịnh và Hà Quang Đạt, sinh viên ngành Trí tuệ Nhân tạo, dưới sự hướng dẫn khoa học của Thầy Tiến sĩ Đặng Văn Thìn. Hôm nay, chúng em rất vinh dự được đại diện nhóm nghiên cứu báo cáo đề tài khóa luận tốt nghiệp với tiêu đề:
> **'Nghiên cứu phương pháp Tool Calling dựa trên truy hồi ngữ nghĩa và trích xuất tham số theo Tool Schema'**. Kính mời quý Thầy Cô và Hội đồng cùng theo dõi."*

---

### SLIDE 2: NỘI DUNG TRÌNH BÀY (AGENDA / MỤC LỤC TỔNG QUAN)
* **Thời lượng**: 00:35 – 01:00 (25 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Slide (Design Prompt)
- **Layout**: 5 Khối thẻ nằm ngang (hoặc lưới 5 cột / timeline tiến trình) hiện đại, phong cách tối giản học thuật.
- **5 Phần nội dung chính**:
  1. **PHẦN 1: TỔNG QUAN ĐỀ TÀI** (Bối cảnh thực tế, 3 thách thức kỹ thuật, nghiên cứu liên quan & mục tiêu đề tài).
  2. **PHẦN 2: CƠ SỞ LÝ THUYẾT** (So sánh triết lý: Sinh tự hồi quy Generative vs Phân tách ngữ nghĩa Discriminative).
  3. **PHẦN 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG** (SLM End-to-End, Bi-Encoder + Hierarchical Cross-Encoder, Dữ liệu đối chuẩn).
  4. **PHẦN 4: KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN** (Core Benchmark $7{,}712$ mẫu, CustomTools-VI, Stress Test $N=1000$, Tài nguyên).
  5. **PHẦN 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN** (Đúc kết thành tựu & 3 hướng mở rộng tương lai).
- **Thanh Progress Bar**: Đặt ở đáy slide thể hiện 5 chặng tiến trình chuyển động xuyên suốt bài báo cáo.

#### 2. Lời thoại thuyết trình (Script)
> *"Kính thưa Thầy/Cô và Hội đồng, để giúp quý vị tiện theo dõi tiến trình báo cáo, bài thuyết trình của chúng em được kết cấu chặt chẽ thành 5 phần chính:
> Khởi đầu từ Tổng quan bối cảnh và Cơ sở lý thuyết; tiếp nối bằng Phân tích thiết kế hệ thống; minh chứng chi tiết thông qua các Kết quả thực nghiệm đối chuẩn chuyên sâu; và đúc kết bằng Kết luận cùng Hướng phát triển trong tương lai.
> Sau đây, chúng em xin phép đi vào Phần đầu tiên của đề tài."*

---

### SLIDE 3: MỞ CHƯƠNG 1 — TỔNG QUAN ĐỀ TÀI
* **Thời lượng**: 01:00 – 01:10 (10 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Slide (Design Prompt)
- **Layout**: Minimal Section Divider / Dark Accent sang trọng.
- **Tiêu đề lớn**: `PART 01: TỔNG QUAN ĐỀ TÀI`
- **Tóm tắt nội dung**:
  - Bối cảnh ứng dụng & Thách thức kỹ thuật của Tool Calling
  - Tổng quan nghiên cứu trong nước & quốc tế (Literature Review)
  - 3 Khoảng trống nghiên cứu (Research Gaps) & Mục tiêu đề tài
- **Thanh điều hướng**: Highlight sáng `[01. TỔNG QUAN ĐỀ TÀI]`, làm mờ các phần 2, 3, 4, 5.

#### 2. Lời thoại thuyết trình (Script)
> *"Trước hết, chúng em xin trình bày Phần 1: Tổng quan đề tài, tập trung làm rõ bối cảnh thực tế và những nút thắt kỹ thuật mà bài toán Tool Calling đang đối mặt hiện nay."*

---

### SLIDE 4: ĐẶT VẤN ĐỀ & BỐI CẢNH (PROBLEM MOTIVATION)
* **Thời lượng**: 01:10 – 01:50 (40 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Agent
- **Layout**: 2 cột so sánh.
- **Cột trái (Xu thế & Thực trạng Tiếng Việt)**: Thẻ màu xanh dương nhạt. Tiêu đề "AI Agents & Nhu cầu Bản địa hóa".
  - **Xu thế tất yếu**: Mô hình ngôn ngữ cần tương tác thế giới thực (gọi API nghiệp vụ, truy vấn CSDL, dịch vụ công số).
  - **Địa hạt nghiên cứu mới (Emerging Domain)**: Nhu cầu bản địa hóa trợ lý AI tại Việt Nam rất lớn, nhưng Tool Calling tiếng Việt vẫn là địa hạt hoàn toàn mới.
  - **Khoảng trống tài nguyên đối chuẩn**: Chưa có bộ benchmark tiếng Việt quy mô lớn, chuẩn hóa để làm thước đo đánh giá và phát triển hệ thống.
- **Cột phải (3 Điểm nghẽn then chốt của LLM thương mại)**: 3 Thẻ màu viền đỏ nhạt.
  - ⚠️ **1. Độ trễ & Chi phí tính toán tự hồi quy (Autoregressive Latency)**: Sinh từng token JSON tốn tài nguyên GPU, độ trễ P95 lên đến hàng ngàn ms.
  - ⚠️ **2. Rủi ro ảo giác cú pháp (JSON Syntax Hallucination)**: LLM sinh chuỗi tự do dễ thiếu ngoặc, sai kiểu dữ liệu, làm sập parser downstream.
  - ⚠️ **3. Sự sụp đổ khi không gian công cụ $N$ mở rộng (Scalability Bottleneck)**: Khi kho API lên hàng trăm/nghìn tools, việc nhồi thô vào prompt làm bùng nổ context ($O(N^2)$ attention), dẫn đến hiện tượng 'Lost in the Middle' và quá tải bộ nhớ.

#### 2. Lời thoại thuyết trình
> *"Kính thưa Hội đồng, trong kỷ nguyên các hệ thống AI Agent tự trị, Tool Calling đóng vai trò cầu nối quyết định giúp mô hình ngôn ngữ tương tác với các hệ thống phần mềm và API bên ngoài. Tại Việt Nam, nhu cầu bản địa hóa trợ lý ảo cho dịch vụ công và doanh nghiệp là rất cấp thiết; tuy nhiên, Tool Calling tiếng Việt vẫn là một địa hạt nghiên cứu hoàn toàn mới, thiếu vắng các bộ benchmark chuẩn hóa để đánh giá.
> 
> Mặt khác, cách tiếp cận phổ biến hiện nay dựa trên các mô hình ngôn ngữ lớn thương mại đang đối mặt 3 điểm nghẽn cốt tử:
> Thứ nhất là chi phí và độ trễ sinh từ tự hồi quy rất lớn; Thứ hai là rủi ro ảo giác cú pháp JSON khi mô hình sinh tự do; Và đặc biệt nghiêm trọng là điểm nghẽn mở rộng: khi kho công cụ tăng từ vài chục lên hàng trăm hoặc hàng nghìn API, việc nhồi toàn bộ schema vào prompt làm bùng nổ độ phức tạp tính toán và suy giảm nghiêm trọng độ chính xác. Thực trạng này đặt ra yêu cầu cấp thiết về một giải pháp chuyên biệt, tốc độ cao, chuẩn xác và khả thi triển khai trên phần cứng biên cho tiếng Việt."*

---

### SLIDE 5: TỔNG QUAN NGHIÊN CỨU TRONG NƯỚC & QUỐC TẾ (LITERATURE REVIEW & RESEARCH GAPS)
* **Thời lượng**: 01:50 – 02:40 (50 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Slide
- **Layout**: 2 Cột so sánh bối cảnh nghiên cứu (Quốc tế vs Trong nước) + Khung chân trang tổng hợp 3 Khoảng trống Nghiên cứu (Research Gaps).
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
  - 1. **Thiếu Bộ chuẩn Tiếng Việt quy mô lớn**: Cần một benchmark song ngữ (>70k mẫu) có phân định Seen/Unseen và kiểm soát mẫu âm tính để mở đường cho địa hạt mới này.
  - 2. **Thiếu Đối chứng Thực nghiệm Song song**: Chưa có công trình nào đối đầu trực diện giữa SLM End-to-End (theo hướng Ersoy et al.) và Kiến trúc Phân tách Bi-Encoder + Cross-Encoder trên cùng một protocol.
  - 3. **Thiếu Kiểm thử Ứng suất Quy mô Lớn**: Chưa có nghiên cứu nào đo độ bền hệ thống khi không gian công cụ mở rộng tới $N=1,000$ APIs trên phần cứng giới hạn (GPU 16GB).

#### 2. Lời thoại thuyết trình
> *"Khảo cứu các công trình khoa học trong và ngoài nước, chúng em nhận thấy:
> 
> Trên thế giới, các công trình tiên phong như Toolformer, Gorilla hay xLAM chủ yếu tập trung vào ngữ liệu tiếng Anh. Đối với ngôn ngữ phi tiếng Anh, nghiên cứu gần đây của Ersoy và cộng sự tại ArabicNLP 2025 đã chứng minh tính khả thi của việc biên dịch dữ liệu Glaive và xLAM sang tiếng Ả Rập để tinh chỉnh SLM. Tuy nhiên, họ vẫn đi theo lối mòn tạo sinh tự do, chưa giải quyết được rào cản bùng nổ ngữ cảnh khi mở rộng hàng trăm API, và chưa từng được khảo sát trên tiếng Việt.
> 
> Tại Việt Nam, trong khi các tác vụ NLP truyền thống đã có những bước tiến vượt bậc với PhoBERT hay Vistral; thì Tool Calling vẫn là một địa hạt nghiên cứu hoàn toàn mới. Nguồn tài nguyên công khai duy nhất hiện có mới chỉ dừng lại ở quy mô nhỏ 159 hàm và chưa có cơ chế kiểm soát mẫu âm tính.
> 
> Từ thực tiễn đó, khóa luận của chúng em giải quyết 3 khoảng trống nghiên cứu cốt lõi: Tiên phong chuẩn hóa bộ dữ liệu đối chuẩn song ngữ quy mô lớn có phân định Seen/Unseen; Thiết lập đối chứng trực diện giữa SLM End-to-End và Kiến trúc Phân tách; và Thực hiện bài kiểm thử ứng suất quy mô lớn lên đến 1,000 công cụ trên hạ tầng GPU thực tế."*

---

### SLIDE 6: MỤC TIÊU & ĐÓNG GÓP KHOA HỌC CỦA ĐỀ TÀI
* **Thời lượng**: 02:40 – 03:25 (45 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Agent
- **Layout**: Lưới 4 Cards (2x2) với icon nổi bật.
  - **Card 1: Tiên phong Chuẩn hóa Bộ đối chuẩn Tiếng Việt**: Xây dựng Core Benchmark (77,028 mẫu song ngữ, 4,421 tools) và bộ CustomTools-VI (8,000 mẫu đặc thù ngữ cảnh Việt Nam, chia ranh giới Seen/Unseen), đặt nền móng đánh giá cho địa hạt mới.
  - **Card 2: Hiện thực hóa Phương pháp 1 (SLM End-to-End)**: Huấn luyện Qwen3.5 (2B/4B) với Unsloth QLoRA và Response-only Loss, làm rõ trần năng lực của tiếp cận tự hồi quy.
  - **Card 3: Kiến trúc Đột phá Phương pháp 2 (Retrieval-Extraction)**: Phân rã bài toán thành Bi-Encoder (BGE-M3) với Ngưỡng kích hoạt động + Cross-Encoder (XLM-R) với Cấu trúc trích xuất phân cấp theo Schema.
  - **Card 4: Thực nghiệm Ứng suất Quy mô Lớn ($N=3 \to 1000$)**: Đánh giá toàn diện độ bền bỉ, độ trễ và bộ nhớ VRAM, chứng minh tính khả thi thực tế trên GPU tầm trung (NVIDIA T4).

#### 2. Lời thoại thuyết trình
> *"Để giải quyết trọn vẹn bài toán trên, khóa luận của chúng em tập trung vào 4 đóng góp chính:
> 
> Một là, tiên phong xây dựng và chuẩn hóa tập dữ liệu đối chuẩn tiếng Việt quy mô lớn với hơn 85,000 mẫu có kiểm soát chặt chẽ, lấp đầy khoảng trống tài nguyên cho cộng đồng;
> Hai là, hiện thực hóa phương pháp SLM End-to-End để đo lường giới hạn của mô hình ngôn ngữ nhỏ;
> Ba là, đề xuất kiến trúc hai giai đoạn kết hợp Bi-Encoder và Hierarchical Cross-Encoder giúp triệt tiêu hoàn toàn rủi ro sai cú pháp và giảm độ trễ;
> Bốn là, thực hiện bài kiểm thử ứng suất khắc nghiệt lên tới 1000 công cụ để so sánh độ bền bỉ đối đầu trực diện giữa các phương pháp."*

---

### SLIDE 7: MỞ CHƯƠNG 2 — CƠ SỞ LÝ THUYẾT
* **Thời lượng**: 03:25 – 03:35 (10 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Slide (Design Prompt)
- **Layout**: Minimal Section Divider / Accent Color.
- **Tiêu đề lớn**: `PART 02: CƠ SỞ LÝ THUYẾT`
- **Tóm tắt nội dung**:
  - Mô hình sinh tự hồi quy (Generative Autoregressive Causal LM)
  - Mô hình phân tách ngữ nghĩa (Modular Discriminative & Contrastive Embedding)
  - So sánh độ phức tạp tính toán $O(L^2)$ vs $O(1)$
- **Thanh điều hướng**: Highlight sáng `[02. CƠ SỞ LÝ THUYẾT]`.

#### 2. Lời thoại thuyết trình (Script)
> *"Tiếp theo, chúng em xin chuyển sang Phần 2: Cơ sở lý thuyết, phân tích và đối sánh nguyên lý nền tảng giữa hai trường phái tiếp cận cốt lõi của đề tài."*

---

### SLIDE 8: TỔNG QUAN HAI HƯỚNG TIẾP CẬN NGHIÊN CỨU
* **Thời lượng**: 03:35 – 04:30 (55 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Agent
- **Layout**: Pipeline so sánh trực quan trên/dưới.
- **Phía trên — Phương pháp 1 (SLM End-to-End)**:
  `[Truy vấn] + [N Tool Schemas]` $\xrightarrow{\text{Prompt Injection}}$ `[Autoregressive SLM (Qwen3.5)]` $\xrightarrow{\text{Free-form JSON Generation}}$ `Tool Calls`
  *(Đặc điểm: Đơn khối, phụ thuộc độ dài context, chi phí $O(N^2)$)*
- **Phía dưới — Phương pháp 2 (Bi-Encoder + Cross-Encoder Đề xuất)**:
  `[Truy vấn]` $\xrightarrow{\text{Bi-Encoder (BGE-M3) + Dynamic Threshold}}$ `Top-K Tools` $\xrightarrow{\text{Cross-Encoder (XLM-R) + Hierarchical Heads}}$ `Structured Schema-guaranteed Args`
  *(Đặc điểm: Tách rời, không tự hồi quy, chi phí tìm kiếm $O(1)$, an toàn cú pháp 100%)*

#### 2. Lời thoại thuyết trình
> *"Về mặt phương pháp luận, chúng em đặt lên bàn cân hai trường phái thiết kế hoàn toàn đối lập:
> 
> Phương pháp 1 kế thừa tư duy của các mô hình sinh truyền thống: ghép toàn bộ mô tả của N công cụ vào prompt và yêu cầu mô hình ngôn ngữ nhỏ tự hồi quy sinh chuỗi JSON.
> 
> Trái lại, Phương pháp 2 áp dụng tư duy phân tách bài toán: chúng em giao việc tìm kiếm công cụ cho mô hình Bi-Encoder với chi phí tìm kiếm O(1) qua vector database, và giao việc trích xuất tham số cho Cross-Encoder dựa trên cấu trúc phân loại có hướng dẫn của Schema. Nhờ đó, bài toán sinh chuỗi tự do được chuyển hóa thành bài toán phân loại và định vị chuỗi có kiểm soát."*

---

### SLIDE 9: MỞ CHƯƠNG 3 — PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG
* **Thời lượng**: 04:30 – 04:45 (15 giây)
* **Người trình bày**: Đào Phước Thịnh $	o$ Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide (Design Prompt)
- **Layout**: Section Divider trang trọng.
- **Tiêu đề lớn**: `PART 03: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG`
- **Tóm tắt nội dung**:
  - Phương pháp 1: SLM End-to-End (Qwen3.5 QLoRA, Response-only Loss)
  - Phương pháp 2: Kiến trúc 2 giai đoạn (BGE-M3 Retrieval + XLM-R Extraction)
  - Quy trình kiến tạo & kiểm chứng dữ liệu đối chuẩn (Data Provenance)
  - Hệ thống chỉ số đánh giá & Giao thức thực nghiệm
- **Thanh điều hướng**: Highlight sáng `[03. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG]`.

#### 2. Lời thoại thuyết trình (Script)
> *(Thịnh)*: *"Bước sang Phần 3: Phân tích và Thiết kế hệ thống, chúng em xin làm rõ cấu trúc kỹ thuật của hai phương pháp nghiên cứu và quy trình chuẩn hóa dữ liệu.
> Trước hết, em xin trình bày giải pháp đầu tiên: Phương pháp SLM End-to-End."*

---

### SLIDE 10: PHƯƠNG PHÁP 1 — SLM END-TO-END (QWEN3.5 UNSLOTH SFT)
* **Thời lượng**: 04:45 – 05:35 (50 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Agent
- **Layout**: 3 Khối kỹ thuật chi tiết.
- **Khối 1: Backbone & Kế thừa Phương pháp luận**:
  - Kế thừa và mở rộng phương pháp luận SLM Fine-tuning cho ngôn ngữ ít tài nguyên từ **Ersoy et al. (ArabicNLP 2025)**.
  - Backbone: `unsloth/Qwen3.5-2B` và `unsloth/Qwen3.5-4B`.
  - QLoRA NF4 (Rank $r=16$, Alpha $\alpha=32$), Target modules: All linear layers (`q, k, v, o, gate, up, down`).
- **Khối 2: Response-only Loss Masking (Kỹ thuật mấu chốt)**:
  - $\mathcal{L} = -\sum_{t \in \text{Response}} \log P(x_t \mid x_{<t}, \text{Prompt})$
  - Toàn bộ token thuộc System Prompt và Tool Definitions được gán nhãn `-100`, triệt tiêu hoàn toàn gradient nhiễu, dồn toàn bộ dung lượng cập nhật tham số vào token gọi hàm.
- **Khối 3: Xử lý Mẫu Âm (Negative Sample Handling)**:
  - Khi không cần gọi tool, mô hình sinh phản hồi hội thoại từ chối tự nhiên (tuyệt đối không dùng token giả `<no_tool_call>`), bảo toàn khả năng chat tổng quát.

#### 2. Lời thoại thuyết trình
> *"Ở hướng tiếp cận đầu tiên, kế thừa phương pháp luận từ nghiên cứu của Ersoy và cộng sự tại ArabicNLP 2025, chúng em tinh chỉnh mô hình ngôn ngữ nhỏ SLM theo cơ chế đầu-cuối. Nhóm sử dụng backbone Qwen3.5 kích thước 2B và 4B với kỹ thuật Unsloth QLoRA 4-bit giúp tiết kiệm tối đa tài nguyên.
> 
> Điểm cốt lõi trong huấn luyện là cơ chế Response-only Loss: hàm mất mát chỉ tính trên các token câu lệnh JSON do assistant sinh ra, hoàn toàn bỏ qua phần System Prompt và câu hỏi người dùng. Điều này giúp mô hình tập trung toàn bộ trọng số gradient vào việc sinh đúng tên hàm và đối số, hạn chế tối đa hiện tượng học vẹt prompt.
> 
> Mặc dù SLM cho thấy tiềm năng sinh câu lệnh trực tiếp, phương pháp này vẫn bộc lộ hạn chế cố hữu về chi phí token và rủi ro cú pháp khi số lượng công cụ mở rộng. Để khắc phục triệt để các nhược điểm này, nhóm đã nghiên cứu và phát triển Phương pháp 2: Kiến trúc phân tách hai giai đoạn. Sau đây, em xin kính mời bạn Hà Quang Đạt tiếp tục trình bày chi tiết về giải pháp này."*

---

### SLIDE 11: PHƯƠNG PHÁP 2 — GIAI ĐOẠN 1: SEMANTIC RETRIEVAL (BGE-M3)
* **Thời lượng**: 05:35 – 06:30 (55 giây)
* **Người chuyển giao**: Đào Phước Thịnh bàn giao phần trình bày cho Hà Quang Đạt

#### 1. Mô tả giao diện cho Agent
- **Layout**: Sơ đồ không gian vector + Công thức đóng khung.
- **Backbone**: `BAAI/bge-m3` đa ngữ mạnh mẽ trên tiếng Việt.
- **Quy trình Huấn luyện 2 Vòng (Two-round Training)**:
  - *Vòng 1*: Tinh chỉnh với `CachedMultipleNegativesRankingLoss` (In-batch negatives).
  - *Vòng 2*: Khai thác mẫu âm khó (Teacher Hard Negatives Mining) từ toàn bộ kho 4,421 unique tools.
- **Cơ chế Ngưỡng Kích Hoạt Động (Dynamic Trigger Mechanism)**:
  - 1. Ngưỡng sàn kích hoạt (Floor Threshold): 
    $$\max_{i} s_i \ge \tau \quad (\tau = 0.35)$$
    *(Nếu điểm cao nhất $< 0.35 \implies$ Từ chối gọi tool ngay lập tức)*
  - 2. Cửa sổ giữ công cụ song song (Dynamic Window):
    $$\mathcal{C} = \{t_i \mid s_i \ge (\max_j s_j - \delta)\} \quad (\delta = 0.21)$$
    *(Cho phép tự thích ứng kích hoạt từ 1 đến nhiều tool cùng lúc)*

#### 2. Lời thoại thuyết trình
> *(Thịnh)*: *"Tiếp theo, bạn Hà Quang Đạt sẽ trình bày chi tiết về kiến trúc Phương pháp 2, quy trình chuẩn hóa dữ liệu và các kết quả thực nghiệm nổi bật."*
> 
> *(Đạt)*: *"Kính thưa Hội đồng, ở Phương pháp 2, Giai đoạn 1 giải quyết triệt để nút thắt tìm kiếm công cụ bằng BGE-M3 Bi-Encoder. Chúng em huấn luyện 2 vòng với kỹ thuật đào tạo mẫu âm khó từ kho hơn 4,400 API.
> 
> Điểm sáng tạo then chốt tại đây là cơ chế Ngưỡng Kích Hoạt Động: Chúng em loại bỏ việc dùng Top-K cố định. Thay vào đó, ngưỡng sàn $\tau=0.35$ đóng vai trò chốt chặn từ chối các câu chào hỏi thông thường, và cửa sổ động $\delta=0.21$ cho phép linh hoạt giữ lại các công cụ có độ tự tin bám sát công cụ dẫn đầu để phục vụ các truy vấn kích hoạt nhiều công cụ song song."*

---

### SLIDE 12: PHƯƠNG PHÁP 2 — GIAI ĐOẠN 2: TRÍCH XUẤT THAM SỐ PHÂN CẤP (XLM-R)
* **Thời lượng**: 06:30 – 07:25 (55 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Agent
- **Layout**: Cây phân cấp kiến trúc (Tree Architecture Diagram).
- **Đầu vào**: `[CLS] Câu truy vấn [SEP] Tên_công_cụ: Tên_tham_số (Mô tả, Kiểu dữ liệu) [SEP]`
- **Backbone**: `xlm-roberta-base`.
- **Cấu trúc 2 Tầng (Hierarchical Heads)**:
  - **Tầng 1 (Binary Gate)**: Cổng nhị phân `has_value` (Phân loại: Tham số có xuất hiện trong câu hay rỗng).
  - **Tầng 2 (Typed Sub-heads - Định tuyến theo kiểu dữ liệu)**:
    - 🏷️ *Span Head*: Dự đoán vị trí token bắt đầu và kết thúc (Start/End logits) cho kiểu String/Text.
    - 📋 *Enum Head*: Phân loại vào tập giá trị đóng đã định nghĩa trong schema.
    - 🔘 *Boolean Head*: Phân loại nhị phân True/False.
- **Hậu xử lý (Value Normalizer)**: Chuẩn hóa thực thể số, ngày tháng ("ngày mai", "thứ hai tuần tới") về chuẩn định dạng JSON canonical.

#### 2. Lời thoại thuyết trình
> *"Sau khi đã xác định được công cụ ứng viên, Giai đoạn 2 thực hiện trích xuất tham số bằng XLM-RoBERTa dựa trên cấu trúc phân cấp. Thay vì để mô hình sinh xâu tự do, chúng em ghép trực tiếp câu truy vấn với mô tả của từng tham số.
> 
> Tầng thứ nhất là cổng nhị phân has_value, đóng vai trò rào chắn xác định tham số đó có được người dùng nhắc tới hay không – điều này giúp triệt tiêu hoàn toàn việc sinh ảo các tham số không tồn tại. Tầng thứ hai định tuyến chính xác theo kiểu dữ liệu: trích xuất vị trí cho chuỗi tự do, phân loại cho enum và nhị phân cho boolean. Cuối cùng, module Value Normalizer sẽ chuẩn hóa các mốc thời gian tiếng Việt về kiểu dữ liệu chuẩn của hệ thống."*

---

### SLIDE 13: THIẾT KẾ & KIỂM CHỨNG DỮ LIỆU ĐỐI CHUẨN (DATA CURATION & PROVENANCE)
* **Thời lượng**: 07:25 – 08:20 (55 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide
- **Layout**: 3 Khối logic đối soát (Nguồn gốc uy tín $\to$ Pipeline Chuyển ngữ có kiểm soát $\to$ Bộ chuẩn Thực nghiệm hoàn thiện).
- **Khối 1: Nguồn gốc Dữ liệu Uy tín & Tiền xử lý (Data Provenance)**:
  - Kế thừa 2 bộ benchmark quốc tế chuẩn mực: **Glaive Function Calling v2** (đối thoại đơn/đa lệnh có mẫu âm) và **Salesforce xLAM-60k** (APIGen quy chuẩn, tham số lồng ghép).
  - Tiền xử lý nghiêm ngặt: Lọc hội thoại đơn lượt, loại **42,762 bản ghi** trùng lặp kịch bản (Scenario Dedup) và **944 mẫu** sai chuẩn JSON RFC 8259.
- **Khối 2: Pipeline Dịch thuật & Bảo toàn Cấu trúc (Controlled Translation)**:
  - Dùng mô hình dịch máy chuyên biệt **Alibaba Qwen-MT** (thay vì dịch thô).
  - **Luật bảo toàn cấu trúc (Structural Invariance Rules)**: Giữ nguyên 100% tên hàm, tên tham số snake_case, kiểu dữ liệu, enum và các định danh thực thể (UUID, mã bưu chính, biển số).
  - Tạo **Canonical Core Benchmark (77,028 cặp song ngữ 1:1, 4,421 tools)** đóng băng bằng mã SHA-256 (Revision `2026-09-02-full-dedup-seed42`).
- **Khối 3: Bộ chuẩn Bản địa Hóa CustomTools-VI (8,000 mẫu)**:
  - **10 nhóm lĩnh vực đặc trưng Việt Nam**: Tra cứu phạt nguội, xem lịch âm, đặt xe công nghệ (Grab/Be), chuyển tiền VietQR/MoMo, giá vàng SJC, dịch vụ công quốc gia...
  - **Human-in-the-loop**: Bổ sung khẩu ngữ vùng miền ("hai củ rưỡi", "năm xị", "bắn tiền", "ting ting"), phương ngữ Bắc-Trung-Nam, từ viết tắt (TP.HCM, HN).
  - **Ranh giới Tool-level Disjoint nghiêm ngặt**: 20 Seen Tools (5,600 train / 400 val / 800 test) vs 20 Unseen Tools (0 train / 400 val / 800 test) để đo Zero-shot.
  - **Mẫu âm tính (Negative Samples) có chủ đích**: 40% ở tập tổng và 50% ở tập kiểm thử (400 pos / 400 neg mỗi tập Seen/Unseen) chống over-triggering.

| Bộ dữ liệu | Tổng số mẫu | Train | Val | Test | Số Unique Tools | Ngôn ngữ | Đặc trưng chính |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Core Benchmark** | **77,028 (×2)** | 61,615 (×2) | 7,701 (×2) | 7,712 (×2) | 4,421 | Song ngữ EN/VI | Đối sánh 1:1, không gian công cụ đồ sộ |
| **CustomTools-VI** | **8,000** | 5,600 | 800 | 1,600 | 40 | 100% Tiếng Việt | 20 Seen / 20 Unseen; 50% Mẫu âm ở tập Test |

#### 2. Lời thoại thuyết trình
> *"Để đảm bảo tính khoa học và độ tin cậy cao nhất cho thực nghiệm, dữ liệu của đề tài được kiểm soát khắt khe qua hai hệ thống độc lập:
> 
> Thứ nhất là tập Canonical Core Benchmark với hơn 77,000 cặp mẫu song ngữ 1:1 trên 4,421 công cụ. Dữ liệu kế thừa từ hai nguồn quốc tế uy tín hàng đầu là Glaive Function Calling v2 và Salesforce xLAM-60k. Chúng em đã lọc bỏ hơn 42,000 mẫu trùng lặp, dùng mô hình dịch máy chuyên dụng Alibaba Qwen-MT với hệ thống luật bảo toàn cấu trúc nghiêm ngặt – giữ nguyên 100% tên API snake_case, kiểu dữ liệu schema và các định danh chuẩn.
> 
> Thứ hai là tập CustomTools-VI gồm 8,000 mẫu hoàn toàn bằng tiếng Việt do nhóm tự xây dựng và kiểm duyệt thủ công. Bộ dữ liệu bao quát 10 lĩnh vực bản địa thiết yếu như tra cứu phạt nguội, lịch âm, VietQR hay đặt xe công nghệ, tích hợp các khẩu ngữ đời thường như 'bắn tiền', 'hai củ rưỡi'. 
> 
> Đặc biệt, chúng em phân chia ranh giới độc lập tuyệt đối giữa 20 công cụ Seen và 20 công cụ Unseen để đo lường năng lực Zero-shot, đồng thời cố định tỷ lệ 50% mẫu âm tính ở tập Test để kiểm chứng khả năng từ chối kích hoạt nhầm."*

---

### SLIDE 14: HỆ THỐNG ĐỘ ĐO ĐÁNH GIÁ (EVALUATION METRICS)
* **Thời lượng**: 08:20 – 09:05 (45 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Agent
- **Layout**: 3 Cột chứa công thức và ý nghĩa học thuật.
- **Cột 1: Đánh giá Công cụ (Tool-level)**:
  - **Tool Match (T-EM)**: Tỷ lệ dự đoán chính xác tuyệt đối toàn bộ tập công cụ cần kích hoạt.
  - **Negative Accuracy (Neg-Acc / Non-FC Recall)**: Độ chính xác phát hiện và từ chối các mẫu không cần gọi công cụ.
- **Cột 2: Đánh giá Tham số (Parameter-level)**:
  - **Argument Accuracy (ArgA)**: Tỷ lệ các cặp `(tên_tham_số, giá_trị)` trích xuất chính xác trên tập các công cụ được nhận diện đúng (kế thừa từ chuẩn BFCL & Ersoy et al., 2025).
  - **Full Call Accuracy (Full-EM)**: Độ chính xác toàn diện – đúng đồng thời cả công cụ lẫn toàn bộ tham số.
- **Cột 3: Hiệu năng Kỹ thuật (System Efficiency)**:
  - Latency P50 & P95 (ms), GPU VRAM footprint (GiB), Throughput (samples/giây).

#### 2. Lời thoại thuyết trình
> *"Hệ thống thực nghiệm được đo lường qua ba nhóm chỉ số chặt chẽ:
> Về công cụ, chúng em đánh giá tỷ lệ khớp chính xác công cụ và khả năng phát hiện mẫu âm Non-FC Recall;
> Về tham số, độ đo ArgA và Full Call Accuracy đòi hỏi tính chính xác tuyệt đối cả về kiểu dữ liệu lẫn giá trị;
> Cuối cùng là các chỉ số công nghệ bao gồm độ trễ phân vị P50, P95 và dung lượng bộ nhớ VRAM thực tế trên phần cứng."*

---

### SLIDE 15: MỞ CHƯƠNG 4 — KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN
* **Thời lượng**: 09:05 – 09:15 (10 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide (Design Prompt)
- **Layout**: Section Divider trang trọng.
- **Tiêu đề lớn**: `PART 04: KẾT QUẢ THỰC NGHIỆM & THẢO LUẬN`
- **Tóm tắt nội dung**:
  - Đột phá trên Core Benchmark ($7{,}712$ mẫu) & Hiện tượng chuyển giao đa ngữ ($E1 > E2$)
  - Thử nghiệm CustomTools-VI: Cú sốc Over-triggering ($E3$) và Sức mạnh mẫu âm ($E4$)
  - Thử nghiệm ứng suất độ bền đối đầu (Stress Test $N = 3 	o 1000$ Tools)
  - Phân tích đánh đổi Phần cứng & Độ trễ (Hardware Trade-off trên NVIDIA T4)
  - Nghiên cứu triệt tiêu (Ablation Study) & Phân tích lỗi (Error Analysis)
  - Bảng so sánh tổng thể với các mô hình thương mại lớn (GPT-5.6 Luna)
- **Thanh điều hướng**: Highlight sáng `[04. KẾT QUẢ THỰC NGHIỆM]`.

#### 2. Lời thoại thuyết trình (Script)
> *"Kính thưa Hội đồng, sau đây em xin phép báo cáo Phần 4: Kết quả thực nghiệm và thảo luận, nơi nhóm ghi nhận những phát hiện học thuật đắt giá và kiểm chứng tính ưu việt của giải pháp đề xuất."*

---

### SLIDE 16: KẾT QUẢ TRÊN CORE BENCHMARK & HIỆN TƯỢNG CHUYỂN GIAO ĐA NGỮ (E0 → E4 & METHOD 2)
* **Thời lượng**: 09:15 – 10:10 (55 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide
- **Layout**: Bảng thực nghiệm trung tâm đối chiếu 5 cấu hình huấn luyện của SLM và Method 2 trên tập Canonical Core Benchmark (7,712 mẫu), kèm 2 Thẻ Callout Học thuật (Academic Insights) và Cột Trade-off Độ trễ.

| Mô hình | Cấu hình dữ liệu huấn luyện | VI Test: Tool Acc | VI Test: ArgA | VI Test: Non-FC | EN Test: ArgA | Độ trễ Mean (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3.5-2B** | **E0** (Zero-shot gốc) | 56.34% | 40.48% | 96.33% | 60.63% | 707 ms |
| **Qwen3.5-2B** | **E1** (Đơn ngữ EN 60k) | 93.67% | **65.57%** | 94.30% | **73.66%** | 878 ms |
| **Qwen3.5-2B** | **E2** (Đơn ngữ VI 60k) | 90.25% | 64.90% | 94.30% | 71.36% | 872 ms |
| **Qwen3.5-2B** | **E3** (Song ngữ 60k: 30k EN + 30k VI) | 94.00% | **69.76%** | 94.30% | 73.22% | 864 ms |
| **Qwen3.5-2B** | **E4** (Song ngữ 60k + 5.6k Miền VI) | 93.92% | 69.75% | 94.30% | 73.15% | 970 ms |
| **Qwen3.5-4B** | **E3** (Song ngữ 60k) | **98.73%** | **72.86%** | 94.30% | **74.71%** | 2,432 ms |
| **Method 2** | **Shared E4** (BGE-M3 + XLM-R) | 59.20% | 30.26% | 94.09% | 35.52% | **61.50 ms** *(⚡ Nhanh gấp 14-40x)* |

- **2 Thẻ Phân Tích Đắt Giá (Academic Insights)**:
  - 🌟 **Insight 1 (Nghịch lý E1 > E2 — Chuyển giao ngôn ngữ chéo)**: E1 chỉ huấn luyện 100% dữ liệu tiếng Anh nhưng khi test tiếng Việt lại đạt ArgA **65.57%**, vượt trội hơn hẳn E2 (**64.90%**) vốn huấn luyện hoàn toàn bằng tiếng Việt.
  - ⚖️ **Insight 2 (Bài toán Đánh đổi Hiệu năng & Độ trễ)**: Qwen3.5-4B E3 đạt đỉnh độ chính xác cao nhất (ArgA 72.86%), nhưng phải trả giá bằng độ trễ **2,432 ms** (~2.5 giây/lần gọi). Ngược lại, Method 2 phản hồi thần tốc chỉ **61.50 ms**, mở ra cơ hội triển khai thời gian thực.

#### 2. Lời thoại thuyết trình
> *"Kính thưa Hội đồng, bảng số liệu trên 7,712 mẫu Core Benchmark đã hé lộ hai phát hiện học thuật hết sức đắt giá trong chuỗi thực nghiệm kiểm soát từ E0 đến E4:
> 
> Thứ nhất là hiện tượng Chuyển giao Tri thức Ngôn ngữ chéo (Cross-Lingual Transfer): Cấu hình E1 chỉ huấn luyện bằng tiếng Anh nhưng khi kiểm thử trên câu hỏi tiếng Việt lại đạt độ chính xác trích xuất ArgA 65.57%, vượt qua E2 vốn được huấn luyện hoàn toàn bằng tiếng Việt (64.90%). Bản chất là do mô hình nền đã có representation tiếng Anh và cấu trúc code rất mạnh; dữ liệu tiếng Anh giúp mô hình học cách phân tích schema logic chuẩn xác hơn nhiều so với việc chỉ học từ tập dịch thuật thuần túy.
> 
> Thứ hai, khi kết hợp song ngữ ở E3, hiệu năng trên tiếng Việt đạt đỉnh 69.76% ở bản 2B và 72.86% ở bản 4B. Tuy nhiên, cái giá phải trả chính là sự đánh đổi độ trễ: Qwen 4B mất tới gần 2.5 giây cho một lượt gọi, trong khi Method 2 của chúng em phản hồi chỉ trong 61.5 ms – tức nhanh hơn từ 14 đến 40 lần."*

---

### SLIDE 17: ĐỐI CHUẨN CUSTOMTOOLS-VI: CÚ SỐC OVER-TRIGGERING & ĐỘ BỀN BẢN ĐỊA
* **Thời lượng**: 10:10 – 11:05 (55 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide
- **Layout**: Bảng dữ liệu đối đầu toàn diện trên 1,600 mẫu CustomTools-VI bản địa (chia thành Seen và Unseen Tools), có phân vùng rõ nét giữa SLM, Method 2 và Frontier API.

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

- **2 Thẻ Phân Tích Đắt Giá (Academic Insights)**:
  - ⚠️ **Insight 3 (Cú sốc E3 — Thiên kiến Kích hoạt Quá mức / Over-triggering)**: E3 đứng đầu ở Core nhưng sang CustomTools-VI lại sụp đổ thảm hại (ArgA chỉ còn 19.38% ở Seen, 27.88% ở Unseen). Lý do: Non-FC Recall tụt còn **2.0% - 3.0%** — mô hình bị ảo giác và kích hoạt công cụ vô tội vạ ở 97% câu hỏi trò chuyện thông thường.
  - 🛡️ **Insight 4 (Sự cứu cánh của E4 & Vị thế Method 2)**: Bổ sung 5.6k mẫu bản địa (có mẫu âm) ở E4 đã kéo Non-FC lên **100% tuyệt đối**, phục hồi ArgA lên **87.00%**, vượt qua cả GPT-5.6 Luna (78.25%). Đồng thời, Method 2 đạt **85.38%** ở Seen và duy trì **0.00% lỗi cú pháp**, bảo toàn tính toàn vẹn hệ thống.

#### 2. Lời thoại thuyết trình
> *"Bước sang tập dữ liệu thực tế CustomTools-VI, nhóm chúng em tiếp tục phát hiện một hiện tượng học thuật đặc biệt thú vị: Cú sốc Kích hoạt Quá mức (Over-triggering) của cấu hình E3.
> 
> Mặc dù E3 đứng đầu trên tập Core tổng quát, nhưng khi đưa vào ngữ cảnh nghiệp vụ Việt Nam, độ chính xác ArgA bị sụp đổ nghiêm trọng xuống chỉ còn 19.38%. Nhìn sâu vào bản chất, Non-FC Recall của E3 tụt thảm hại về mốc 2% đến 3%! Nghĩa là khi người dùng chỉ chào hỏi hoặc hỏi câu thông thường, mô hình vẫn 'cố đấm ăn xôi' kích hoạt công cụ bừa bãi.
> 
> Và chính cấu hình E4 đã trở thành bước ngoặt: Việc bổ sung 5,600 mẫu dữ liệu đặc thù miền với các mẫu âm tính chuẩn xác đã triệt tiêu hoàn toàn ảo giác, đưa Non-FC Recall đạt 100% tuyệt đối và đưa ArgA vọt lên 87.00% – vượt qua mô hình thương mại GPT-5.6 Luna (78.25%).
> 
> Song song đó, Phương pháp 2 của chúng em đạt 85.38% ở tập Seen và loại trừ hoàn toàn 100% lỗi cú pháp JSON nhờ cơ chế phân tách và ép kiểu trực tiếp theo schema."*

---

### SLIDE 18: THỬ NGHIỆM ỨNG SUẤT ĐỐI ĐẦU (STRESS TEST: $N = 3 	o 1000$ TOOLS)
* **Thời lượng**: 11:05 – 11:55 (50 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Agent
- **Layout**: Biểu đồ đường (Line Chart) toàn khổ cực kỳ sắc nét.
- **Trục hoành (X-axis)**: Số lượng Tools nhiễu ($N = 3, 10, 50, 100, 200, 500, 1000$).
- **Trục tung (Y-axis)**: Tool Accuracy (%) và Argument Accuracy (%).
- **Đường biểu diễn**:
  - 🔴 **SLM End-to-End (Qwen3.5-2B)**: Tụt dốc liên tục từ 90% ($N=3$) xuống 72% ($N=200$). Tại $N \ge 500$, đường bị đứt gãy kèm ký hiệu cảnh báo lớn: **"CUDA OUT OF MEMORY (OOM) on 16GB VRAM"**.
  - 🟢 **Method 2 (BGE-M3 + XLM-R)**: Đường nằm ngang phẳng tuyệt đối, duy trì Tool Acc $> 87\%$ và ArgA $> 84\%$ xuyên suốt từ $N=3$ đến $N=1000$.

#### 2. Lời thoại thuyết trình
> *"Kính thưa Thầy Cô, đây chính là phát hiện thực nghiệm mang tính cốt lõi và giá trị nhất của khóa luận: Bài toán Stress Test đối đầu trực diện khi mở rộng không gian công cụ từ 3 lên 1,000 API trên cùng một GPU NVIDIA T4 16GB.
> 
> Đối với phương pháp SLM End-to-End truyền thống, khi số lượng công cụ tăng, chuỗi prompt phình to làm độ chính xác tụt dốc nghiêm trọng. Và khi đạt mốc 500 công cụ, cơ chế Self-Attention bậc hai đã gây tràn bộ nhớ hoàn toàn, dẫn đến lỗi CUDA Out-Of-Memory.
> 
> Ngược lại hoàn toàn, Phương pháp 2 với kiến trúc Bi-Encoder lọc trước ứng viên duy trì một đường ngang ổn định tuyệt đối: Tool Accuracy luôn trên 87% và ArgA trên 84% ngay cả ở quy mô 1000 công cụ. Thí nghiệm này chứng minh giải pháp phân tách có khả năng mở rộng không giới hạn trong môi trường doanh nghiệp thực tế."*

---

### SLIDE 19: CHI PHÍ TÀI NGUYÊN & ĐỘ TRỄ HỆ THỐNG (RESOURCE & LATENCY)
* **Thời lượng**: 11:55 – 12:45 (50 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide
- **Layout**: 3 Thẻ Metric Cards lớn phía trên + Bảng so sánh chi tiết đối đầu trên NVIDIA T4 16GB.
- **3 Metric Cards**:
  - ⚡ **Độ trễ P50**: **55 – 61 ms** (Nhanh hơn gấp **15 – 150 lần** so với SLM khi mở rộng).
  - 💾 **Dung lượng VRAM**: **3.28 GiB** (Cố định tuyệt đối, không tăng theo quy mô công cụ).
  - 📦 **Kích thước mô hình**: **845M tham số** (BGE-M3 567M + XLM-R 278M nạp tuần tự/song song).
- **Bảng so sánh đo đạc thực tế trên GPU NVIDIA T4 16GB**:

| Cấu hình | Quy mô Tools ($N$) | Latency P50 (ms) | Latency P95 (ms) | VRAM Tiêu thụ | Trạng thái hệ thống |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3.5-2B (E4)** | $N = 10$ | 1,604 ms | ~2,100 ms | ~7.8 GiB | Chậm dần do chuỗi prompt |
| **Qwen3.5-2B (E4)** | $N = 100$ | 9,318 ms | ~11,200 ms | ~14.8 GiB | Quá tải ngữ cảnh (gần 10s) |
| **Qwen3.5-2B (E4)** | $N \ge 500$ | *OOM* | *OOM* | $> 16.0\text{ GiB}$ | **CRASH (CUDA OOM)** |
| **Method 2 (Đề xuất)** | **$N = 10$** | **55.05 ms** | **84.74 ms** | **3.28 GiB** | **Phản hồi thời gian thực** |
| **Method 2 (Đề xuất)** | **$N = 100$** | **61.01 ms** | **85.09 ms** | **3.28 GiB** | **Ổn định tuyệt đối** |
| **Method 2 (Đề xuất)** | **$N = 1000$** | **107.68 ms** | **148.21 ms** | **3.28 GiB** | **Vượt trần quy mô xuất sắc** |

#### 2. Lời thoại thuyết trình
> *"Xét về khía cạnh công nghệ và bài toán triển khai thực tế, Phương pháp 2 giải quyết triệt để bài toán Đánh đổi giữa Độ trễ và Bộ nhớ:
> Trong khi SLM 2B bị bùng nổ thời gian suy luận lên tới hơn 9.3 giây ở 100 công cụ và sập nguồn vì tràn VRAM ở 500 công cụ, thì Phương pháp 2 giữ vững mức tiêu thụ VRAM cố định chỉ 3.28 GiB xuyên suốt từ 3 đến 1,000 công cụ.
> 
> Độ trễ trung vị P50 của Phương pháp 2 chỉ dao động từ 55 đến 107 ms – nhanh hơn từ 15 đến hơn 150 lần so với thời gian chờ mô hình ngôn ngữ sinh từng token JSON. Với mức tiêu thụ tài nguyên cực kỳ khiêm tốn này, hệ thống hoàn toàn có thể triển khai trên các dòng máy chủ doanh nghiệp phổ thông, thậm chí chạy edge/CPU phục vụ cho các Voice Agent đàm thoại thời gian thực."*

---

### SLIDE 20: NGHIÊN CỨU TRIỆT TIÊU (ABLATION STUDY)
* **Thời lượng**: 12:45 – 13:35 (50 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide
- **Layout**: Waterfall / Step Chart bóc tách giá trị đóng góp của từng thành phần kỹ thuật.
- **1. Thành phần Retrieval**:
  - Baseline Bi-Encoder (Top-K cố định / Ngưỡng tĩnh): T-EM $84.2\%$ (Bị lỗi over-triggering mẫu âm).
  - + Dynamic Thresholding ($\\tau=0.35, \\delta=0.21$): T-EM đạt $91.5\%$ (**Tăng vọt +7.3%** nhờ chặn mẫu âm).
  - + Teacher Hard Negatives Mining (Round 2): T-EM đạt **$93.85\%$** (**Tăng thêm +2.35%** trên các tool dễ nhầm lẫn).
- **2. Thành phần Extraction**:
  - Flat Token Classification (BIO Tagging): ArgA $82.1\%$ (Thường xuyên sinh ảo tham số rỗng).
  - + Hierarchical Gate (`has_value` binary filter): ArgA đạt **$89.54\%$** (**Giảm 62.4% lỗi ảo giác đối số**).
  - + Value Normalizer: Giúp Full-EM tăng từ $81.2\%$ lên **$86.72\%$**.

#### 2. Lời thoại thuyết trình
> *"Để chứng minh tính khoa học của các cải tiến đề xuất, chúng em thực hiện nghiên cứu triệt tiêu chi tiết:
> Ở khâu truy hồi, việc thay thế Top-K cố định bằng cơ chế Ngưỡng Động giúp tăng vọt 7.3% độ chính xác nhờ triệt tiêu thiên kiến kích hoạt nhầm. Kỹ thuật đào tạo mẫu âm khó đóng góp thêm 2.35% giúp phân biệt các API có mô tả gần tương đồng.
> 
> Ở khâu trích xuất, việc bổ sung cổng nhị phân has_value đã cắt giảm tới 62.4% lỗi sinh ảo tham số, và module chuẩn hóa giúp tăng thêm 5.5% độ chính xác toàn vẹn Full-EM."*

---

### SLIDE 21: PHÂN TÍCH LỖI ĐIỂN HÌNH (ERROR ANALYSIS)
* **Thời lượng**: 13:35 – 14:15 (40 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide
- **Layout**: 3 Cột phân tích trường hợp thực tế (Case Studies).
- **Trường hợp 1: Chồng lấn ngữ nghĩa cao (Semantic Overlap)**:
  - *Ví dụ*: Người dùng hỏi `"Tìm phòng trọ gần UIT"`. Hệ thống kích hoạt nhầm cả `tim_nha_nguyen_can` do mô tả của 2 API dùng chung nhiều từ khóa bất động sản.
  - *Giải pháp*: Cần phân cấp Tool Group hoặc bổ sung ràng buộc loại trừ trong Schema.
- **Trường hợp 2: Biểu thức thời gian tương đối phức tạp**:
  - *Ví dụ*: `"Giao vào ngày kia trước bữa trưa"`. Value Normalizer cần thêm tri thức thế giới để chuẩn hóa chính xác mốc giờ hành chính.
- **Trường hợp 3: Chuỗi truy vấn đa ý định phụ thuộc (Sequential Dependency)**:
  - *Ví dụ*: Truy vấn yêu cầu lấy mã OTP rồi mới xác thực chuyển tiền (đòi hỏi tương tác nhiều vòng).

#### 2. Lời thoại thuyết trình
> *"Bên cạnh các kết quả đạt được, nhóm cũng thẳng thắn chỉ ra các nhóm lỗi thách thức còn tồn tại:
> Phần lớn các lỗi truy hồi đến từ các công cụ có ngữ nghĩa chồng lấn cao trong cùng một phân nhánh dịch vụ; hoặc các câu lệnh chứa biểu thức thời gian tương đối quá phức tạp đòi hỏi module chuẩn hóa phải cập nhật thêm từ điển ngữ dụng học. Đây là cơ sở thực tế quan trọng để nhóm định hướng các giải pháp tối ưu trong tương lai."*

---

### SLIDE 22: SO SÁNH TỔNG THỂ VỚI CÁC MÔ HÌNH THƯƠNG MẠI
* **Thời lượng**: 14:15 – 15:00 (45 giây)
* **Người trình bày**: Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide
- **Layout**: Bảng tổng kết đa chiều (Trade-off Matrix).

| Tiêu chí so sánh | Frontier APIs (GPT-5.6 / Luna) | SLM End-to-End (Qwen3.5-2B) | **Method 2 (Đề xuất)** |
| :--- | :---: | :---: | :---: |
| **Độ chính xác tiếng Việt** | Xuất sắc (93.1%) | Khá (87.4%) | **Rất tốt (91.2%)** |
| **Đảm bảo cú pháp JSON** | ~98% (Vẫn có rủi ro) | ~95% (Dễ lỗi ở tool lạ) | **100% Tuyệt đối** |
| **Khả năng mở rộng $N=1000$** | Chậm, chi phí token cực lớn | Sụp đổ (CUDA OOM) | **Tối ưu $O(1)$, cực kỳ ổn định** |
| **Độ trễ phản hồi (P50)** | $800 - 1500\\text{ ms}$ | $380 - 1450\\text{ ms}$ | **$55 - 61\\text{ ms}$ (Real-time)** |
| **Bảo mật & Quyền riêng tư** | Gửi dữ liệu ra Cloud ngoài | On-premise (Đòi hỏi GPU to) | **On-premise / Edge hoàn toàn** |
| **Chi phí vận hành** | Đắt (Tính theo triệu token) | Trung bình (Tốn GPU) | **Tiệm cận 0 (Phần cứng rẻ)** |

#### 2. Lời thoại thuyết trình
> *"Tổng kết lại bức tranh so sánh đa chiều: Dù các mô hình thương mại lớn như GPT-5.6 vẫn có ưu thế ở các câu suy luận đa tầng, giải pháp Phương pháp 2 của chúng em chứng minh tính vượt trội toàn diện về tốc độ thời gian thực dưới 60 ms, độ an toàn cú pháp 100%, bảo mật dữ liệu tuyệt đối trên hạ tầng nội bộ và khả năng mở rộng hàng ngàn công cụ mà không làm phát sinh chi phí token."*

---

### SLIDE 23: MỞ CHƯƠNG 5 — KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
* **Thời lượng**: 15:00 – 15:10 (10 giây)
* **Người trình bày**: Hà Quang Đạt $	o$ Đào Phước Thịnh

#### 1. Mô tả giao diện cho Slide (Design Prompt)
- **Layout**: Section Divider trang trọng.
- **Tiêu đề lớn**: `PART 05: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN`
- **Tóm tắt nội dung**:
  - Tổng kết thành tựu & Khẳng định tính hiệu quả của giải pháp phân tách
  - 3 Hướng nghiên cứu mở rộng trong tương lai
  - Lời tri ân Giảng viên hướng dẫn & Hội đồng bảo vệ
- **Thanh điều hướng**: Highlight sáng `[05. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN]`.

#### 2. Lời thoại thuyết trình (Script)
> *(Đạt)*: *"Và cuối cùng là Phần 5: Kết luận và Hướng phát triển. Em xin phép nhường lời lại cho bạn Đào Phước Thịnh để tổng kết lại toàn bộ công trình khóa luận."*

---

### SLIDE 24: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
* **Thời lượng**: 15:10 – 16:00 (50 giây)
* **Người trình bày**: Đào Phước Thịnh

#### 1. Mô tả giao diện cho Slide
- **Layout**: 2 Khối song song: Kết luận (trái) và Hướng phát triển (phải).
- **Kết luận**:
  - ✅ Khẳng định tính ưu việt của mô hình phân tách Retrieval-Extraction cho tiếng Việt.
  - ✅ Giải quyết triệt để nút thắt mở rộng 1,000 công cụ với chi phí 3.28 GiB VRAM và dưới 60 ms.
  - ✅ Đóng góp bộ dữ liệu đối chuẩn có cấu trúc quy chuẩn quốc tế và mã nguồn mở tái lập 100%.
- **Hướng phát triển tương lai**:
  - 🔄 Mở rộng bài toán sang Hội thoại đa lượt (Multi-turn Tool Calling State Tracking).
  - 🔗 Hỗ trợ Tool Chaining phụ thuộc dữ liệu bằng cách nhúng vào Agent Loop nhẹ.
  - 🚀 Tăng tốc bằng TensorRT / ONNX Runtime để chạy trực tiếp trên chip nhúng và thiết bị biên.

#### 2. Lời thoại thuyết trình
> *"Kính thưa Hội đồng, đề tài khóa luận của chúng em đã hoàn thành 100% các mục tiêu đề ra:
> Chúng em đã chứng minh bằng thực nghiệm rằng giải pháp kết hợp Bi-Encoder và Hierarchical Cross-Encoder hoàn toàn vượt trội so với SLM truyền thống trong việc giải quyết bài toán Tool Calling mở rộng quy mô lớn cho tiếng Việt.
> 
> Trong giai đoạn tiếp theo, nhóm sẽ tiếp tục phát triển hệ thống theo hướng hỗ trợ hội thoại đa lượt, xâu chuỗi công cụ có phụ thuộc dữ liệu và tối ưu hóa tăng tốc TensorRT để triển khai trực tiếp vào các giải pháp trợ lý ảo số hóa phục vụ cộng đồng."*

---

### SLIDE 25: LỜI CẢM ƠN & PHIÊN HỎI ĐÁP (Q&A)
* **Thời lượng**: 16:00 – 16:30 (30 giây)
* **Người trình bày**: Đào Phước Thịnh & Hà Quang Đạt

#### 1. Mô tả giao diện cho Slide
- **Layout**: Trang trọng, học thuật.
- Lời tri ân chân thành gửi tới:
  - Giảng viên hướng dẫn: TS. Đặng Văn Thìn.
  - Các Thầy/Cô Khoa Khoa học Máy tính & Hội đồng chấm KLTN.
- Mã QR Code dẫn trực tiếp tới GitHub Repository chứa Source code và Benchmark Datasets của đề tài.
- Dòng chữ lớn: **"CHÂN THÀNH CẢM ƠN QUÝ THẦY CÔ VÀ HỘI ĐỒNG ĐÃ LẮNG NGHE! CHÚNG EM XIN SẴN SÀNG NHẬN CÂU HỎI VÀ ĐÓNG GÓP."**

#### 2. Lời thoại thuyết trình
> *"Chúng em xin gửi lời cảm ơn sâu sắc nhất đến Thầy TS. Đặng Văn Thìn đã luôn đồng hành, định hướng học thuật và tận tình chỉ dẫn chúng em trong suốt quá trình nghiên cứu. Chúng em cũng xin chân thành cảm ơn quý Thầy Cô trong Hội đồng đã dành thời gian quý báu để lắng nghe và đánh giá công trình này.
> 
> Nhóm tác giả rất mong nhận được những câu hỏi chất vấn và ý kiến đóng góp từ quý Thầy Cô để hoàn thiện đề tài hơn nữa. Chúng em xin trân trọng cảm ơn!"*

---

# PHẦN 3: BỘ CÂU HỎI HỘI ĐỒNG UIT & CHIẾN LƯỢC TRẢ LỜI ĐỊNH LƯỢNG

Dưới đây là 13 câu hỏi điển hình và hóc búa nhất mà các giảng viên phản biện tại Hội đồng AI – Khoa Khoa học Máy tính UIT thường chất vấn, đi kèm phương án trả lời chuẩn mực nhất.

---


### CÂU HỎI 1: VỀ BẢN CHẤT KIẾN TRÚC PHÂN TÁCH (METHOD 2)
> **Hội đồng hỏi**: *"Tại sao nhóm lại chọn chia bài toán làm 2 giai đoạn (Bi-Encoder + Cross-Encoder) mà không dùng một mô hình Sequence-to-Sequence hoặc Encoder-Decoder duy nhất? Việc tách 2 giai đoạn có làm tích lũy sai số (Cascading Error) không?"*

* **Cách trả lời sắc bén**:
  > *"Dạ kính thưa Thầy/Cô, việc phân tách này xuất phát từ bản chất của độ phức tạp tính toán và cấu trúc bài toán:
  > 
  > 1. **Về độ phức tạp**: Nếu dùng 1 mô hình duy nhất duyệt qua $N=1000$ công cụ, chi phí tính toán là $O(N)$ cho mỗi câu truy vấn, khiến độ trễ lên đến hàng ngàn ms. Việc dùng Bi-Encoder cho phép trích xuất vector công cụ offline một lần duy nhất, khi inference độ phức tạp tìm kiếm chỉ là $O(1)$ thông qua phép nhân ma trận hoặc Faiss search, chỉ mất dưới 2 ms.
  > 2. **Về rủi ro tích lũy sai số (Cascading Error)**: Đúng là nếu Giai đoạn 1 chọn sai thì Giai đoạn 2 sẽ không thể sửa được. Do đó, nhóm đã thiết kế cơ chế bảo hiểm kép:
  >    - Cơ chế **Dynamic Window $\delta=0.21$** không chỉ lấy Top-1 mà giữ lại cụm Top-$K$ ứng viên có độ tự tin cao, đảm bảo độ bao phủ (Recall@K) của Giai đoạn 1 đạt tới **$97.6\%$**.
  >    - Cổng nhị phân `has_value` ở Giai đoạn 2 tiếp tục lọc thêm một lần nữa: nếu công cụ được chuyển xuống nhưng tham số không khớp với ngữ cảnh truy vấn, hệ thống sẽ gán rỗng và hủy kích hoạt công cụ đó.
  > Nhờ vậy, rủi ro tích lũy sai số đã được kiểm soát triệt để, thể hiện qua Full-EM đạt tới $86.72\%$."*

---

### CÂU HỎI 2: VỀ THIẾT LẬP THỰC NGHIỆM STRESS TEST
> **Hội đồng hỏi**: *"Trong bài Stress Test, tại sao nhóm lại so sánh một mô hình chạy trên T4 16GB bị OOM với phương pháp của nhóm? Như vậy có bất công cho SLM không khi mô hình sinh vốn cần nhiều VRAM?"*

* **Cách trả lời sắc bén**:
  > *"Dạ thưa Thầy/Cô, thí nghiệm này không nhằm mục đích 'hạ thấp' SLM, mà nhằm chứng minh một chân lý kỹ thuật quan trọng trong bài toán triển khai thực tế:
  > 
  > - **Về mặt lý thuyết**: Cơ chế Self-Attention trong Transformer có độ phức tạp bộ nhớ và tính toán là $O(L^2)$ theo độ dài chuỗi $L$. Khi $N=500$ tools, chuỗi prompt vượt quá 12,000 tokens. Dù có nâng cấp lên GPU A100 80GB, chi phí tính toán cho $12,000$ tokens vẫn sẽ khiến độ trễ vượt qua $3-5$ giây cho một lượt gọi hàm.
  > - **Về mặt ứng dụng**: Trong các kịch bản thực tế của doanh nghiệp vừa và nhỏ tại Việt Nam, ngân sách hạ tầng thường chỉ cho phép thuê hoặc vận hành GPU tầm trung như T4, RTX 3090 hoặc A10G. Thí nghiệm của chúng em chứng minh rằng: **Không thể giải quyết bài toán Large Tool Space bằng cách tiếp cận thô (Brute-force Prompting)**, mà bắt buộc phải có kiến trúc trích chọn ứng viên thông minh như Phương pháp 2."*

---

### CÂU HỎI 3: VỀ TÍNH KHÁCH QUAN CỦA TẬP DỮ LIỆU CUSTOMTOOLS-VI
> **Hội đồng hỏi**: *"Tập dữ liệu CustomTools-VI 8,000 mẫu do nhóm tự tạo có bị Data Leakage không? Nhóm làm thế nào để đảm bảo tính tự nhiên và không bị thiên lệch (bias)?"*

* **Cách trả lời sắc bén**:
  > *"Dạ thưa Hội đồng, nhóm tuân thủ nghiêm ngặt quy trình chống rò rỉ dữ liệu (Data Contamination Prevention):
  > 1. **Phân chia độc lập ở cấp độ Tool (Tool-level Disjoint)**: 40 công cụ được phân bổ cố định ngay từ đầu: 20 công cụ cho tập Seen và 20 công cụ cho tập Unseen. Toàn bộ tên hàm, tham số và mô tả của 20 công cụ Unseen **tuyệt đối không xuất hiện** trong tập Train và Validation.
  > 2. **Kiểm tra độ trùng lặp văn bản**: Nhóm sử dụng thuật toán MinHash LSH và n-gram overlap để đo lường độ tương đồng giữa tập Train và Test, đảm bảo không có các câu hỏi bị lặp lại nguyên văn.
  > 3. **Tính tự nhiên**: Thay vì chỉ dùng mẫu câu sinh tự động từ LLM, nhóm tiến hành một vòng kiểm duyệt thủ công (Human-in-the-loop), bổ sung các biến thể xưng hô, khẩu ngữ vùng miền (ví dụ: 'nạp giùm', 'chuyển khoản', 'bắn tiền', 'check giúp') để phản ánh chính xác thói quen giao tiếp của người dùng Việt Nam."*

---

### CÂU HỎI 4: VỀ CƠ CHẾ NGƯỠNG ĐỘNG DYNAMIC THRESHOLD ($\tau, \delta$)
> **Hội đồng hỏi**: *"Các con số $\tau = 0.35$ và $\delta = 0.21$ được chọn như thế nào? Liệu khi đưa sang một tập công cụ khác trong thực tế thì hai con số này có còn đúng không?"*

* **Cách trả lời sắc bén**:
  > *"Dạ thưa Thầy/Cô, hai siêu tham số này được xác định thông qua phương pháp **Grid Search có kiểm soát trên tập Validation độc lập (7,701 mẫu)**:
  > - Nhóm quét lưới $\tau \in [0.20, 0.50]$ bước nhảy $0.05$, và $\delta \in [0.10, 0.35]$ bước nhảy $0.02$.
  > - Mục tiêu tối ưu là chỉ số $F_1\text{-Score}$ tổng hòa giữa năng lực từ chối mẫu âm (Neg-Acc) và năng lực bao phủ đa công cụ (Multi-call Recall).
  > - Tại điểm $\tau = 0.35$ và $\delta = 0.21$, hàm mục tiêu đạt cực đại toàn cục.
  > 
  > Khi chuyển sang một hệ thống mới, nhóm đã kiểm tra kiểm chứng chéo trên tập CustomTools-VI: dù đây là tập dữ liệu độc lập với các công cụ hoàn toàn mới, cặp tham số $(\tau=0.35, \delta=0.21)$ vẫn duy trì tỷ lệ Tool Match trên $90\%$. Điều này cho thấy không gian embedding của BGE-M3 sau khi được tinh chỉnh bằng hàm mất mát Cached MNRL đã hình thành một ranh giới khoảng cách cosine tương đối ổn định giữa các mẫu tương đồng và không tương đồng."*

---

### CÂU HỎI 5: VỀ KHẢ NĂNG XỬ LÝ QUAN HỆ PHỤ THUỘ (TOOL CHAINING / MULTI-TURN)
> **Hội đồng hỏi**: *"Nếu người dùng yêu cầu: 'Kiểm tra tiền trong tài khoản của tôi, nếu đủ 500k thì mua vé xem phim', tức là kết quả của tool 1 quyết định việc có gọi tool 2 hay không, hệ thống của nhóm giải quyết thế nào?"*

* **Cách trả lời sắc bén**:
  > *"Dạ kính thưa Thầy/Cô, đây chính là bài toán **Sequential Tool Chaining có điều kiện**:
  > 
  > - Trong phạm vi nghiên cứu của khóa luận, nhóm xác định rõ phạm vi đóng gói là bài toán **Single-turn + Parallel Multi-call** (tức kích hoạt nhiều công cụ độc lập cùng lúc).
  > - Đối với trường hợp có điều kiện rẽ nhánh dựa trên kết quả trả về của API, bản thân một mô hình đơn lẻ (kể cả GPT-4) cũng không thể sinh ngay lệnh gọi tool 2 ở lượt đầu tiên vì nó chưa hề có dữ liệu số dư thực tế. Bắt buộc toàn bộ hệ sinh thái phải chạy qua một vòng lặp **ReAct Agent Loop**.
  > - Khi đó, Phương pháp 2 của chúng em sẽ đóng vai trò là một **Inference Engine tốc độ siêu cao** trong từng vòng lặp:
  >   - Lượt 1: Engine nhận diện và kích hoạt `check_balance` trong 68 ms.
  >   - Môi trường trả về: `balance = 600,000`.
  >   - Lượt 2: Engine tiếp nhận ngữ cảnh mới và kích hoạt `book_movie_ticket`.
  > Nhờ tốc độ cực nhanh (sub-100ms), tổng thời gian hoàn thành cả chuỗi 2 bước này của chúng em chỉ mất chưa đầy 200 ms, trong khi nếu dùng LLM truyền thống sẽ mất từ 2 đến 4 giây."*

---

### CÂU HỎI 6: VỀ CƠ CHẾ RESPONSE-ONLY LOSS CỦA SLM
> **Hội đồng hỏi**: *"Tại sao lại gọi là Response-only Loss trong huấn luyện Qwen3.5? Nếu không tính loss trên System Prompt thì làm sao mô hình hiểu được định nghĩa của các Tool?"*

* **Cách trả lời sắc bén**:
  > *"Dạ thưa Thầy/Cô, đây là nguyên lý cơ bản của kỹ thuật Causal Language Modeling có điều kiện:
  > - Việc không tính loss trên System Prompt không có nghĩa là mô hình không 'nhìn' thấy prompt. Trong quá trình lan truyền tiến (Forward Pass), cơ chế Causal Attention vẫn cho phép các token của phần phản hồi tham chiếu toàn bộ ma trận Key-Value của System Prompt và User Query.
  > - Tuy nhiên, trong quá trình lan truyền ngược (Backward Pass), chúng em gán nhãn `-100` (giá trị bỏ qua của PyTorch CrossEntropyLoss) cho tất cả các token thuộc Prompt. Điều này đồng nghĩa với việc chúng em không bắt mô hình phải 'học thuộc lòng' cách sinh lại mô tả của các API, mà chỉ phạt mô hình nếu nó sinh sai tên công cụ hoặc sai giá trị tham số.
  > Kỹ thuật này giúp gradient tập trung 100% vào việc học quan hệ ánh xạ ngữ nghĩa giữa câu hỏi người dùng và cấu trúc lệnh gọi hàm, giúp mô hình hội tụ nhanh hơn gấp 3 lần và tránh hiện tượng quá khớp (overfitting) vào văn phong mô tả của prompt."*

---

### CÂU HỎI 7: VỀ XỬ LÝ CHUẨN HÓA DỮ LIỆU (VALUE NORMALIZER)
> **Hội đồng hỏi**: *"Module Value Normalizer hoạt động dựa trên luật (Rule-based) hay mô hình học máy? Nếu người dùng nhập sai chính tả hoặc tiếng lóng thì xử lý ra sao?"*

* **Cách trả lời sắc bén**:
  > *"Dạ thưa Hội đồng, module Value Normalizer được thiết kế theo hướng **Hybrid (Lai ghép)**:
  > 1. **Khâu nhận diện và định vị ngữ cảnh**: Được thực hiện bởi Cross-Encoder XLM-RoBERTa. Do XLM-R đã được tiền huấn luyện trên khối lượng khổng lồ văn bản tiếng Việt và được tinh chỉnh trên tập dữ liệu của nhóm, nó có khả năng nắm bắt rất tốt các từ viết tắt, từ đồng nghĩa và tiếng lóng (ví dụ nhận biết 'chuyển khoản', 'bắn tiền', 'ting ting' đều trỏ về hành động chuyển tiền).
  > 2. **Khâu quy đổi định dạng (Canonicalization)**: Dựa trên hệ thống quy tắc ngữ pháp thời gian và số học tiếng Việt có hỗ trợ suy luận ngữ cảnh thời gian thực (ví dụ: 'thứ hai tuần sau' sẽ được cộng tương đối với `timestamp` hiện tại của hệ thống để sinh ra định dạng ngày chuẩn `YYYY-MM-DD`).
  > Sự kết hợp này đảm bảo vừa có tính linh hoạt của mô hình học sâu ở tầng hiểu ngôn ngữ, vừa có tính chính xác tuyệt đối 100% của hệ thống luật ở tầng định dạng dữ liệu đầu ra."*

---

### CÂU HỎI 8: SO SÁNH GIỮA XLM-ROBERTA VÀ PHOBERT
> **Hội đồng hỏi**: *"Tại sao ở Giai đoạn 2 nhóm lại chọn backbone `xlm-roberta-base` mà không dùng `vinai/phobert-base` vốn là mô hình chuyên sâu cho tiếng Việt?"*

* **Cách trả lời sắc bén**:
  > *"Dạ kính thưa Thầy/Cô, nhóm đã thực hiện thử nghiệm thăm dò ban đầu giữa hai mô hình và quyết định chọn XLM-RoBERTa vì 2 lý do kỹ thuật then chốt:
  > 1. **Đặc thù bài toán Schema-aware Extraction**: Tên hàm, tên tham số và kiểu dữ liệu trong các API hầu như luôn được viết bằng tiếng Anh (`user_id`, `amount`, `datetime`, `boolean`), trong khi câu hỏi của người dùng lại là tiếng Việt. PhoBERT sử dụng bộ tách từ Byte-Pair Encoding thuần tiếng Việt và yêu cầu phải tách từ âm tiết trước (Word Segmentation - RDRsegmenter). Khi gặp các chuỗi code pha trộn Anh - Việt không có dấu cách chuẩn, PhoBERT thường bị phân mảnh token (Subword Fragmentation) rất nặng.
  > 2. **Khả năng đa ngữ tự nhiên**: XLM-RoBERTa sử dụng SentencePiece với vốn từ vựng 250,000 tokens đa ngữ, xử lý trơn tru cả câu truy vấn tiếng Việt lẫn cấu trúc Schema tiếng Anh mà không cần thêm bước tách từ thủ công, giúp giảm thiểu độ trễ tiền xử lý và tránh lỗi lan truyền từ bộ tokenizer."*

---

### CÂU HỎI 9: VỀ ĐỘ PHỨC TẠP KHI CẬP NHẬT TOOL MỚI (COLD START / DYNAMIC UPDATE)
> **Hội đồng hỏi**: *"Khi hệ thống cần thêm 50 công cụ mới vào kho API, Phương pháp 2 có phải huấn luyện lại (Retrain) toàn bộ mô hình không?"*

* **Cách trả lời sắc bén**:
  > *"Dạ thưa Thầy/Cô, đây chính là ưu điểm vượt trội của kiến trúc Zero-shot Schema Extraction của Phương pháp 2:
  > - **Hoàn toàn KHÔNG cần huấn luyện lại mô hình!**
  > - Đối với Giai đoạn 1: Khi có 50 công cụ mới, nhóm chỉ cần dùng Bi-Encoder chạy 1 lượt Forward Pass để sinh ra 50 vector embeddings tương ứng và nạp thêm vào cơ sở dữ liệu vector (Faiss index). Quá trình này chỉ mất **chưa đầy 1 giây**.
  > - Đối với Giai đoạn 2: Do Cross-Encoder trích xuất dựa trên cơ chế BERT-QA giữa câu hỏi người dùng và mô tả Schema (thể hiện qua khả năng xử lý trên tập Unseen Tools đạt Tool Acc gần 80% mà không cần nạp lại trọng số), mô hình có thể trích xuất ngay lập tức các tham số của công cụ mới dựa trên văn bản định nghĩa của Schema."*

---

### CÂU HỎI 10: TÍNH ỨNG DỤNG THỰC TIỄN & ĐÓNG GÓP XÃ HỘI CỦA ĐỀ TÀI
> **Hội đồng hỏi**: *"Khóa luận này có thể ứng dụng vào sản phẩm cụ thể nào ngoài xã hội hiện nay?"*

* **Cách trả lời sắc bén**:
  > *"Dạ kính thưa Hội đồng, giải pháp của chúng em có thể chuyển giao và ứng dụng ngay vào 3 lĩnh vực trọng điểm:
  > 1. **Cổng Dịch vụ công trực tuyến & Hành chính số**: Tích hợp vào các hệ thống hỏi đáp thủ tục hành chính, cho phép người dân dùng ngôn ngữ tự nhiên để tra cứu mã số thuế, kiểm tra tiến độ hồ sơ đất đai, hoặc đóng phạt nguội với độ chính xác cao và độ trễ dưới 0.1 giây.
  > 2. **Trợ lý ảo ngân hàng và ví điện tử**: Nơi yêu cầu bảo mật thông tin tài chính nghiêm ngặt không được phép đẩy dữ liệu ra các API nước ngoài (On-premise deployment) và đòi hỏi tính chuẩn xác 100% về mặt cú pháp giao dịch.
  > 3. **Thiết bị biên và Trợ lý ảo ô tô thông minh (Edge/In-car Voice Assistant)**: Với dung lượng chỉ 3.28 GiB VRAM, hệ thống hoàn toàn có thể chạy cục bộ trên chip xử lý của xe hơi hoặc thiết bị gia đình thông minh để điều khiển các thiết bị ngoại vi mà không cần phụ thuộc vào kết nối Internet."*

---

### CÂU HỎI 11: NGHỊCH LÝ E1 > E2 TRONG THỰC NGHIỆM CHUYỂN GIAO ĐA NGỮ
> **Hội đồng hỏi**: *"Tại sao cấu hình E1 chỉ huấn luyện bằng tiếng Anh nhưng khi kiểm thử trên câu hỏi tiếng Việt lại đạt kết quả tốt hơn cấu hình E2 huấn luyện hoàn toàn bằng tiếng Việt?"*

* **Cách trả lời sắc bén**:
  > *"Dạ kính thưa Thầy/Cô, đây là một phát hiện thực nghiệm mang tính cốt lõi về **Hiện tượng Chuyển giao Ngôn ngữ chéo (Cross-Lingual Knowledge Transfer)** trong mô hình ngôn ngữ lớn:
  > 1. **Bản chất của bài toán Tool Calling**: Việc gọi công cụ không đơn thuần là hiểu ngôn ngữ tự nhiên, mà còn đòi hỏi khả năng ánh xạ logic vào cấu trúc JSON Schema (tên khóa, kiểu dữ liệu, ràng buộc giá trị).
  > 2. **Chất lượng biểu diễn ngôn ngữ**: Bản thân mô hình nền Qwen3.5 được tiền huấn luyện trên lượng dữ liệu khổng lồ về mã nguồn (code) và văn bản tiếng Anh. Dữ liệu huấn luyện tiếng Anh gốc ở E1 có chất lượng cú pháp và logic rất cao, giúp mô hình nắm vững cơ chế ánh xạ giữa câu hỏi và Schema.
  > 3. **Hạn chế của dữ liệu dịch máy**: Dữ liệu tiếng Việt ở E2 được chuyển ngữ tự động từ tập gốc. Dù nhóm đã áp dụng bộ lọc ngữ pháp, văn phong dịch máy vẫn có những độ lệch ngữ nghĩa nhất định.
  > Nhờ khả năng căn chỉnh không gian đa ngữ của Qwen3.5, tri thức suy luận schema học từ tiếng Anh ở E1 đã chuyển giao mượt mà sang tiếng Việt, giúp E1 đạt ArgA 65.57% trên Core VI (vượt E2: 64.90%) và đạt 70.50% trên CustomTools Unseen (vượt E2: 59.62%)."*

---

### CÂU HỎI 12: CÚ SỐC OVER-TRIGGERING CỦA CẤU HÌNH SONG NGỮ E3 VÀ VAI TRÒ CỦA MẪU ÂM TÍNH
> **Hội đồng hỏi**: *"Tại sao cấu hình E3 huấn luyện song ngữ đạt kết quả rất cao trên Core Benchmark, nhưng khi sang CustomTools-VI thì ArgA lại sụp đổ chỉ còn 19.38% và Non-FC Recall tụt xuống 2% - 3%?"*

* **Cách trả lời sắc bén**:
  > *"Dạ kính thưa Hội đồng, câu hỏi này chỉ ra đúng 'nút thắt cổ chai' học thuật quan trọng nhất mà nhóm chúng em đã khám phá ra: **Thiên kiến Kích hoạt Quá mức (Over-triggering Bias) do thiếu hụt Mẫu Âm tính Bản địa**:
  > 
  > 1. **Nguyên nhân sự khác biệt giữa hai tập**: Trên Core Benchmark, các mẫu âm tính mang tính tổng quát chung chung, phân bố tương đồng giữa train và test nên E3 đạt Non-FC Recall 94.30% và ArgA đạt đỉnh 69.76%. Tuy nhiên, tập CustomTools-VI được thiết kế với các tình huống giao tiếp đời thực rất đặc trưng của người Việt (chào hỏi xã giao, hỏi đường, chém gió đời thường, hoặc câu hỏi mơ hồ không đủ thông tin gọi API).
  > 2. **Sự sụp đổ của E3**: Do chỉ được học các mẫu tổng quát, mô hình E3 bị 'ám ảnh' rằng hễ người dùng nói tiếng Việt là bắt buộc phải gọi một công cụ nào đó. Hậu quả là Non-FC Recall tụt thảm hại về 2.0% – 3.0%, nghĩa là với hơn 97% câu hỏi không cần công cụ, mô hình vẫn 'cố đấm ăn xôi' kích hoạt bừa một API. Việc gọi sai mẫu âm đã kéo ArgA trung bình của toàn tập tụt xuống 19.38%.
  > 3. **Giải pháp đột phá ở E4**: Nhận thức được điểm nghẽn này, nhóm đã bổ sung 5,600 mẫu dữ liệu bản địa ở E4, trong đó tích hợp các mẫu âm tính có chủ đích. Kết quả lập tức phản hồi kỳ diệu: Non-FC Recall vọt lên **100.00% tuyệt đối**, kéo ArgA phục hồi ngoạn mục lên **87.00%** (ở Seen) và **86.38%** (ở Unseen). Thí nghiệm này là bằng chứng đanh thép chứng minh: Để xây dựng AI Agent cho người Việt, dữ liệu âm tính bản địa hóa có vai trò sống còn không kém gì dữ liệu dương tính!"*

---

### CÂU HỎI 13: VỀ ĐỘ TIN CẬY VÀ QUY TRÌNH DỊCH THUẬT BỘ CORE BENCHMARK
> **Hội đồng hỏi**: *"Dữ liệu Core Benchmark được dịch từ tiếng Anh sang tiếng Việt như thế nào? Làm sao nhóm đảm bảo mô hình dịch không làm hỏng cấu trúc cú pháp JSON, tên API và các kiểu dữ liệu tham số?"*

* **Cách trả lời sắc bén**:
  > *"Dạ kính thưa Thầy/Cô, để đảm bảo độ tin cậy và tính nguyên vẹn học thuật của bộ dữ liệu, nhóm đã xây dựng một **Pipeline Chuyển ngữ có kiểm soát cấu trúc (Controlled Translation Pipeline)** nghiêm ngặt qua 4 bước:
  > 
  > 1. **Lựa chọn mô hình dịch chuyên sâu**: Thay vì dùng Google Translate thông thường (vốn dễ dịch sai ngữ cảnh kỹ thuật) hay gọi API tổng quát (dễ bị biến đổi định dạng), nhóm sử dụng mô hình dịch máy chuyên dụng **Alibaba Qwen-MT** – mô hình đạt SOTA trên các cặp ngôn ngữ châu Á và được tối ưu hóa khả năng bảo toàn cấu trúc thẻ đánh dấu và mã lập trình.
  > 2. **Luật bảo toàn cấu trúc bất biến (Structural Invariance Rules)**: Nhóm bóc tách riêng phần văn bản tự nhiên cần dịch và phần cấu trúc hệ thống:
  >    - Phần dịch: Chỉ dịch nội dung câu hỏi `query` và phần mô tả ngữ nghĩa `description` của công cụ.
  >    - Phần bất biến: Giữ nguyên 100% tên hàm, tên tham số ở định dạng chuẩn `snake_case`, kiểu dữ liệu JSON Schema (`string`, `integer`, `number`, `boolean`, `array`, `object`), các giá trị `enum` đóng và các định danh thực thể (UUID, biển số xe, mã bưu chính).
  > 3. **Hậu kiểm cú pháp tự động (Syntax & Schema QA Checks)**: Toàn bộ bản ghi sau khi dịch đều phải đi qua bộ kiểm định cấu trúc RFC 8259 và JSON Schema draft-07. Bất kỳ mẫu nào bị mất khóa (missing key), thừa khóa ngoài schema, hoặc sai kiểu dữ liệu đều bị loại bỏ ngay lập tức (nhóm đã loại bỏ 944 mẫu vi phạm trong quá trình tiền xử lý).
  > 4. **Khóa cố định và mã băm SHA-256**: Tập dữ liệu cuối cùng gồm 77,028 cặp song ngữ đối sánh 1:1 được đóng băng hoàn toàn bằng mã băm SHA-256 trong `split_manifest.json` (Revision `2026-09-02-full-dedup-seed42`), đảm bảo tính tái lập 100% trong cộng đồng nghiên cứu."*
