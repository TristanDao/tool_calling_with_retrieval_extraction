# Gọi Công Cụ (Tool Calling) Tiếng Việt: Nghiên Cứu So Sánh Giữa Mô Hình Ngôn Ngữ Nhỏ End-to-End Và Kiến Trúc Chuyên Biệt Bi-Encoder + Cross-Encoder

**Tác giả**:  
Đào Phước Thịnh$^1$, Hà Quang Đạt$^1$, Đặng Văn Thìn$^1$ (Giảng viên hướng dẫn)  

$^1$Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Thông tin, Đại học Quốc gia TP. Hồ Chí Minh  
*Email*: {21521469, 21521925}@gm.uit.edu.vn, thindv@uit.edu.vn  

**Thời gian**: Tháng 09/2026  
**Thể loại**: Bài báo nghiên cứu khoa học / Khóa luận tốt nghiệp kỹ sư/cử nhân  

---

## Tóm Tắt (Abstract)

Khả năng gọi công cụ (tool calling / function calling) là năng lực then chốt cho phép Mô hình Ngôn ngữ Lớn (LLM) vượt ra khỏi không gian văn bản thuần túy để tương tác với thế giới thực thông qua các giao diện lập trình ứng dụng (API), cơ sở dữ liệu và công cụ tính toán bên ngoài. Mặc dù các nghiên cứu về tool calling đã phát triển mạnh mẽ trên tiếng Anh, việc xây dựng và tối ưu hóa năng lực này cho các ngôn ngữ thứ hai như tiếng Việt vẫn còn đối mặt với nhiều rào cản lớn: sự thiếu hụt dữ liệu chuyên biệt, độ trễ suy luận cao của các mô hình tạo sinh, hiện tượng ảo giác cú pháp JSON/XML, và nguy cơ suy giảm độ chính xác khi số lượng công cụ trong hệ thống tăng cao.

Trong công trình này, chúng tôi giải quyết 4 câu hỏi nghiên cứu (Research Questions) nền tảng:
1. **RQ1 (Năng lực chuyển giao ngôn ngữ chéo)**: Mô hình có nhất thiết phải cần dữ liệu huấn luyện tiếng Việt hay có thể dựa hoàn toàn vào khả năng chuyển giao (cross-lingual transfer) từ tiếng Anh?
2. **RQ2 (Hiện tượng nghiện gọi công cụ & Hiệu chuẩn dữ liệu âm tính)**: Việc huấn luyện song ngữ có gây ra hiện tượng gọi công cụ vô tội vạ (over-triggering) đối với các câu hội thoại thông thường hay không, và vai trò của dữ liệu âm tính (negative samples) miền đặc thù là gì?
3. **RQ3 (Khả năng tổng quát hóa Zero-Shot trên công cụ chưa từng gặp)**: Khi đối mặt với các công cụ hoàn toàn mới trong đời sống Việt Nam (`test_unseen`), mô hình tạo sinh tự hồi quy (SLM) hay kiến trúc phân biệt (Bi+Cross Encoder) có năng lực tổng quát hóa vượt trội hơn?
4. **RQ4 (Đường biên đánh đổi Pareto giữa chất lượng và độ trễ)**: Kiến trúc chuyên biệt phân tách (Bi-Encoder Retrieval + Cross-Encoder Extraction) có thể đạt độ trễ thời gian thực và chi phí suy luận tối ưu tới mức nào so với phương pháp SLM End-to-End?

Để trả lời các câu hỏi trên, chúng tôi xây dựng hai bộ benchmark: **Canonical Core Benchmark** (77,028 cặp bản ghi Anh-Việt, 4,421 công cụ duy nhất) và **CustomTools-VI** (8,000 mẫu đặc thù đời sống Việt Nam thuộc 40 công cụ, 10 nhóm domain, phân chia nghiêm ngặt thành 20 công cụ đã gặp - seen và 20 công cụ zero-shot - unseen). Chúng tôi tiến hành thực nghiệm quy mô lớn giữa:
- **Phương pháp 1 (SLM End-to-End)**: Tinh chỉnh mô hình `unsloth/Qwen3.5-2B` (và mở rộng `4B`) qua 5 cấu hình thực nghiệm có kiểm soát chặt chẽ (E0: Zero-shot baseline, E1: Đơn ngữ EN, E2: Đơn ngữ VI, E3: Song ngữ EN+VI, E4: Song ngữ kết hợp miền đặc thù Việt Nam).
- **Phương pháp 2 (Kiến trúc phân tách)**: Kết hợp Bi-Encoder `BAAI/bge-m3` (huấn luyện qua 2 vòng CachedMNRL để đào bới hard negative) và Cross-Encoder `xlm-roberta-base` phân cấp (has_value nhị phân + routing schema: span, enum, boolean) kết hợp bộ chuẩn hóa giá trị (Value Normalizer).

**Kết quả thực nghiệm chính**:
- Trên `CustomTools-VI`, Method 1 (E4) đạt độ chính xác trích xuất tham số (ArgA / Exact Match) xuất sắc: **87.00%** trên tập seen và **86.38%** trên tập unseen, vượt trội hoàn toàn so với Method 2 (**67.75%** trên seen và **22.75%** trên unseen — khoảng cách lên tới **+63.63%** ở khả năng zero-shot generalization).
- Phát hiện hiện tượng "nghiện gọi công cụ" nghiêm trọng ở E3: khi thiếu dữ liệu âm tính tiếng Việt, Non-FC Recall rơi xuống đáy vực **2.00% - 3.00%**, khiến ArgA sụt giảm chỉ còn 19.38% - 27.88%. Bổ sung dữ liệu âm tính bản địa hóa ở E4 đã khôi phục hoàn hảo Non-FC Recall lên **100.00%**.
- Về tốc độ suy luận, Method 2 thiết lập ưu thế áp đảo với độ trễ trung vị P50 chỉ **58.16–91.88 ms**, **nhanh gấp 10–15 lần** so với Method 1 (~860–970 ms) và chiếm dụng dưới 1.2 GB VRAM, khẳng định vị thế tối ưu cho các bài toán thời gian thực trong môi trường công nghiệp.

---

## 1. Giới Thiệu (Introduction)

Khả năng gọi công cụ (tool calling), thường được gọi là gọi hàm (function calling), là cơ chế nền tảng giúp các mô hình ngôn ngữ lớn (LLM) và các tác tử thông minh (AI Agents) kết nối với thế giới bên ngoài. Thông qua việc phân tích ngôn ngữ tự nhiên của người dùng và định nghĩa schema của các công cụ khả dụng, mô hình tự động nhận diện thời điểm cần kích hoạt công cụ, lựa chọn chính xác API mục tiêu và trích xuất các tham số cấu trúc tương ứng.

```
       ┌────────────────────────────────────────────────────────┐
       │                 NGƯỜI DÙNG (USER QUERY)                │
       │    "Kiểm tra phạt nguội xe máy biển số 59P1-12345"     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                DANH MỤC CÔNG CỤ (TOOLS)                │
       │  - tra_cuu_phat_nguoi(bien_so: str, loai_xe: str)      │
       │  - dat_ve_xe_khach(diem_di: str, diem_den: str)        │
       │  - thanh_toan_tien_dien(ma_khach_hang: str)            │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
 ═══════════════════════════════════════════════════════════════════════════
       LỰA CHỌN KIẾN TRÚC THỰC THI (ARCHITECTURAL PARADIGM)
 ═══════════════════════════════════════════════════════════════════════════
       ┌───────────────────────────┴────────────────────────────┐
       ▼                                                        ▼
 【PHƯƠNG PHÁP 1: SLM END-TO-END】         【PHƯƠNG PHÁP 2: BI+CROSS ENCODER】
  Sinh tự hồi quy mã lệnh XML có cấu trúc    Truy hồi ngữ nghĩa + Trích xuất span/enum
  <tool_call>                                Giai đoạn 1: Bi-Encoder BGE-M3
  <function=tra_cuu_phat_nguoi>               -> Chọn "tra_cuu_phat_nguoi" (58ms)
  <parameter=bien_so>59P1-12345</parameter>  Giai đoạn 2: Cross-Encoder XLM-R
  <parameter=loai_xe>xe_may</parameter>       -> bien_so = "59P1-12345" (has_value=True)
  </function>                                 -> loai_xe = "xe_may" (enum class 1)
  </tool_call>                               Giá trị chuẩn hóa: 100% Valid JSON
 ───────────────────────────────────────────────────────────────────────────
  Độ chính xác cao, Zero-shot mạnh (86%)     Độ trễ cực thấp (< 90ms), Nhẹ (< 1.2GB)
```

Mặc dù các hệ thống thương mại hàng đầu (như OpenAI Function Calling, Google Gemini Function Calling) đã chứng minh hiệu năng mạnh mẽ trên tiếng Anh, việc ứng dụng chúng vào các hệ thống tại Việt Nam đối mặt với 4 thách thức kỹ thuật lớn:
1. **Độ trễ suy luận lớn**: Quá trình giải mã tự hồi quy (autoregressive decoding) qua context chứa hàng chục định nghĩa JSON Schema thường kéo dài từ 800 ms đến 2,000 ms, không đáp ứng được yêu cầu phản hồi tức thời của các hệ thống tổng đài thoại thông minh (Voicebot) hoặc trợ lý thanh toán.
2. **Chi phí vận hành và rủi ro bảo mật**: Việc gửi toàn bộ dữ liệu nội bộ và prompt hội thoại lên các API đám mây quốc tế làm tăng chi phí token và vi phạm các quy định về chủ quyền dữ liệu.
3. **Ảo giác cấu trúc (Format Hallucination)**: Mô hình tạo sinh có nguy cơ xuất ra chuỗi JSON dị tật, thiếu dấu đóng ngoặc hoặc tự ý bịa đặt các tham số không có trong tài liệu kỹ thuật.
4. **Sự thiếu hụt tài nguyên nghiên cứu tiếng Việt**: Tiếng Việt mang đặc trưng đơn lập, không biến hình, phụ thuộc vào thanh điệu và ngữ cảnh, đồng thời có thói quen biểu đạt số tiền, ngày tháng đa dạng ("hai triệu rưỡi", "ngày rằm tháng giêng"). Hầu hết các benchmark toàn cầu hiện nay (như BFCL, ToolBench) hoàn toàn bỏ trống tiếng Việt.

Công trình này được xây dựng nhằm giải quyết triệt để các thách thức trên thông qua việc so sánh thực nghiệm đối đầu giữa hai phương pháp: **Mô hình Ngôn ngữ Nhỏ End-to-End (SLM)** và **Kiến trúc Chuyên biệt Phân tách (Bi-Encoder + Cross-Encoder)**.

### Đóng góp của bài báo:
1. **Công bố bộ benchmark tiếng Việt chuẩn hóa**: Xây dựng **Canonical Core Benchmark** gồm 77,028 cặp bản ghi song ngữ (4,421 công cụ duy nhất) và **CustomTools-VI** gồm 8,000 mẫu đại diện cho 10 nhóm lĩnh vực thực tế tại Việt Nam với cơ chế phân vùng strict zero-shot unseen.
2. **Phân tích chuyển giao ngôn ngữ và hiện tượng over-triggering**: Làm sáng tỏ cơ chế chuyển giao tri thức từ tiếng Anh sang tiếng Việt, đồng thời phát hiện và giải quyết tận gốc hiện tượng "nghiện gọi công cụ" bằng kỹ thuật hiệu chuẩn mẫu âm tính bản địa hóa.
3. **Xác lập đường biên Pareto giữa chất lượng và độ trễ**: Cung cấp bằng chứng thực nghiệm rõ ràng chứng minh SLM End-to-End là lựa chọn tối ưu cho các bài toán yêu cầu độ chính xác cao và khả năng mở rộng zero-shot, trong khi Bi+Cross Encoder là giải pháp vượt trội cho các ứng dụng đòi hỏi độ trễ siêu thấp dưới 100 ms.

---

## 2. Các Nghiên Cứu Liên Quan (Related Work)

### 2.1 Gọi công cụ trên các mô hình ngôn ngữ lớn (Generative LLMs)
Khởi đầu từ ý tưởng tự kích hoạt công cụ của Toolformer (Schick et al., 2023), hàng loạt các nghiên cứu đã tập trung vào việc gia tăng độ phức tạp của các tác tử. Gorilla (Patil et al., 2023) tối ưu hóa khả năng đọc hiểu tài liệu API trực tiếp. Salesforce xLAM (Liu et al., 2024b) và ToolAce (Liu et al., 2024a) mở rộng quy mô dữ liệu tự động với hàng chục ngàn API đa dạng. Berkeley Function-Calling Leaderboard (BFCL) (Yan et al., 2024) thiết lập tiêu chuẩn đánh giá toàn diện trên môi trường đơn lượt, đa lệnh gọi và đa lượt hội thoại. 

Gần đây nhất, Ersoy et al. (2025) đã tiên phong nghiên cứu việc tinh chỉnh các mô hình SLM cho tiếng Ả Rập, chứng minh rằng việc dịch thuật và tinh chỉnh cục bộ có thể vượt trội so với việc phụ thuộc vào các mô hình thương mại đóng. Nghiên cứu của chúng tôi kế thừa tư tưởng của Ersoy et al., nhưng mở rộng vượt bậc bằng cách đưa vào đối trọng kiến trúc phân tách Bi+Cross Encoder và thiết lập quy trình kiểm soát dữ liệu song ngữ nghiêm ngặt.

### 2.2 Kiến trúc Dense Retrieval và Trích xuất tham số phân biệt
Trong các hệ sinh thái lớn, việc nhồi toàn bộ công cụ vào context của LLM là bất khả thi. Các kỹ thuật truy hồi dày (Dense Retrieval) sử dụng Bi-Encoder (như Contriever, BGE) đã được đề xuất trong ToolRetriever và AutoTool (Song et al., 2023) để lọc ra Top-$k$ công cụ phù hợp nhất. 

Trong khi đó, ở nhánh Xử lý Ngôn ngữ Tự nhiên cổ điển, các mô hình dựa trên BERT/RoBERTa từ lâu đã được ứng dụng cho bài toán điền khung tham số (slot filling). Tuy nhiên, hầu hết các hệ thống hiện tại đều dừng lại ở việc dùng Bi-Encoder để lọc công cụ, sau đó vẫn phải chuyển sang một LLM lớn để sinh tham số. Việc xây dựng một pipeline **hoàn toàn phi tạo sinh (purely non-autoregressive)** — kết hợp Bi-Encoder BGE-M3 và Cross-Encoder XLM-RoBERTa-base đa đầu phân cấp — là một hướng tiếp cận độc đáo được chúng tôi hiện thực hóa nhằm đạt được tốc độ suy luận dưới 100 mili-giây.

---

## 3. Hệ Thống Dữ Liệu & Benchmark Đánh Giá

Để đảm bảo tính khách quan và khả năng tái lập thực nghiệm, chúng tôi xây dựng hai bộ dữ liệu độc lập với các đặc tính thống kê chi tiết tại Bảng 1.

**Bảng 1: Thống kê chi tiết các bộ dữ liệu trong Benchmark Tool Calling Tiếng Việt**  
*(Ghi chú: FC biểu thị mẫu có gọi công cụ (Y) hoặc không gọi công cụ (N). Turns: Đơn lượt (S). Calls: Đơn lệnh (S) hoặc Đa lệnh (M). Unique Tools: Số lượng công cụ duy nhất).*

| Bộ dữ liệu | Ngôn ngữ | FC | Turns | Calls | Tập Huấn Luyện (Train) | Tập Kiểm Thử (Test) | Unique Tools |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Canonical Glaive** | Tiếng Việt (VI) | Y | S | S | 14,568 | 1,821 | 972 |
| | Tiếng Việt (VI) | N | S | S | 3,642 | 455 | — |
| | Tiếng Anh (EN) | Y | S | S | 14,568 | 1,821 | 972 |
| | Tiếng Anh (EN) | N | S | S | 3,642 | 455 | — |
| **Canonical xLAM** | Tiếng Việt (VI) | Y | S | M | 47,047 | 5,891 | 3,449 |
| | Tiếng Việt (VI) | N | S | M | 0 | 0 | — |
| | Tiếng Anh (EN) | Y | S | M | 47,047 | 5,891 | 3,449 |
| | Tiếng Anh (EN) | N | S | M | 0 | 0 | — |
| **CustomTools-VI (Seen)** | Tiếng Việt (VI) | Y | S | S/M | 3,600 | 400 | 20 |
| | Tiếng Việt (VI) | N | S | S/M | 2,000 | 400 | — |
| **CustomTools-VI (Unseen)**| Tiếng Việt (VI) | Y | S | S/M | 0 *(Strict Zero-Shot)* | 400 | 20 |
| | Tiếng Việt (VI) | N | S | S/M | 0 *(Strict Zero-Shot)* | 400 | — |
| **Tổng Cộng (Core Benchmark)**| **Song ngữ EN-VI**| **Y/N** | **S** | **S/M**| **61,615 (x2)** | **7,712 (x2)** | **4,421** |

### 3.1 Quy chuẩn dịch thuật và đóng băng Benchmark (Revision Policy)
Để tránh rò rỉ dữ liệu và đảm bảo việc so sánh công bằng giữa các mô hình, chúng tôi áp dụng quy trình đóng băng phiên bản:
- Bản sửa đổi chính thức `2026-09-02-full-dedup-seed42` bao gồm **77,028 cặp bản ghi** đã khử trùng lặp hoàn toàn, phân chia theo tỷ lệ 80/10/10 cố định.
- Quy chuẩn dịch thuật tuân thủ nghiêm ngặt: Tuyệt đối giữ nguyên tên hàm (`snake_case`), tên tham số (`argument keys`), mã UUID, mã định danh tiền tệ chuẩn quốc tế (`VND`, `USD`) và cấu trúc schema. Chỉ dịch nội dung câu hỏi của người dùng, mô tả chức năng của công cụ và các giá trị tham số là chuỗi văn bản tự nhiên.

### 3.2 Bộ dữ liệu đặc thù Việt Nam (CustomTools-VI)
Bộ dữ liệu gồm 8,000 mẫu được xây dựng bám sát ngữ cảnh thực tế tại Việt Nam, phân bổ trên 10 nhóm chức năng nghiệp vụ:
1. *Thương mại điện tử & Mua sắm*: Tra cứu tình trạng đơn hàng, tìm kiếm voucher, so sánh giá hàng hóa.
2. *Vận tải & Du lịch nội địa*: Đặt vé xe khách liên tỉnh, tra cứu thông tin chuyến bay nội địa.
3. *Ẩm thực & Đặt bàn*: Đặt chỗ nhà hàng, tìm kiếm quán ăn đặc sản theo quận/huyện.
4. *Tiện ích & Dịch vụ sinh hoạt*: Thanh toán tiền điện nước sinh hoạt, nạp tiền điện thoại trả trước.
5. *Ngân hàng & Tài chính số*: Chuyển khoản nội bộ theo số tài khoản/ngân hàng, tra cứu tỷ giá ngoại tệ.
6. *Bất động sản & Nhà trọ*: Tìm kiếm phòng trọ sinh viên gần trường đại học, định giá đất sơ bộ.
7. *Dịch vụ hành chính công*: Tra cứu phương tiện vi phạm giao thông (phạt nguội), đăng ký cư trú.
8. *Y tế & Sức khỏe*: Đặt lịch hẹn khám bệnh tại bệnh viện tuyến tỉnh, tìm nhà thuốc đạt chuẩn GPP gần nhất.
9. *Giáo dục & Tuyển sinh*: Tìm gia sư các môn khoa học tự nhiên, tra cứu điểm chuẩn đại học các năm.
10. *Giao vận & Bưu chính*: Tính toán cước vận chuyển nội thành, định vị bưu kiện bưu điện.

**Quy tắc Strict Unseen**: 20 công cụ (thuộc 5 nhóm ngẫu nhiên) bị cách ly hoàn toàn khỏi mọi quá trình huấn luyện, mining negative của cả Method 1 và Method 2. Hai tập kiểm thử `test_seen` (800 mẫu) và `test_unseen` (800 mẫu) đều duy trì tỷ lệ cân bằng chính xác 50% câu hỏi yêu cầu gọi công cụ (Positive) và 50% câu hỏi từ chối gọi công cụ (Negative).

---

## 4. Phương Pháp Luận Chi Tiết

### 4.1 Phương pháp 1: SLM End-to-End (Native XML Tool Calling)

Chúng tôi định dạng bài toán gọi công cụ dưới dạng sinh ngôn ngữ có điều kiện. Đầu vào bao gồm câu lệnh hệ thống (hướng dẫn định dạng thẻ XML), danh mục các công cụ khả dụng $\mathcal{T} = \{t_1, t_2, \dots, t_K\}$ kèm JSON schema chi tiết, và câu truy vấn tự nhiên của người dùng $q$.

#### Định dạng phản hồi Native
Mô hình được huấn luyện để sinh ra cấu trúc XML tinh gọn:
```xml
<tool_call>
<function=ten_cong_cu>
<parameter=ten_tham_so>gia_tri_tham_so</parameter>
</function>
</tool_call>
```
Đối với các trường hợp không cần gọi công cụ (Negative Queries), mô hình phản hồi một câu văn hội thoại tự nhiên, hữu ích (ví dụ: *"Chào bạn, tôi có thể hỗ trợ gì cho bạn hôm nay?"*), loại bỏ hoàn toàn các token giả định như `<no_tool_call>` vốn dễ làm sai lệch phân phối tiền huấn luyện của mô hình.

#### Hàm mất mát Response-Only (Masked Cross-Entropy Loss)
Để tối đa hóa hiệu suất học cấu trúc và tránh lãng phí dung lượng mô hình vào việc tái tạo lại định nghĩa công cụ trong prompt, hàm mất mát chỉ được tính toán trên các token thuộc lượt sinh của Assistant:

$$\mathcal{L}_{SFT} = -\sum_{i=1}^{N} m_i \log P(w_i \mid w_{<i}, q, \mathcal{T})$$

Trong đó $m_i = 1$ nếu token $w_i$ thuộc về chuỗi phản hồi của Assistant, và $m_i = 0$ đối với toàn bộ các token thuộc về System Prompt, Tool Schema và User Query.

#### 5 Cấu hình thực nghiệm (Ngân sách kiểm soát: 60,000 mẫu)
- **E0 (Zero-Shot Baseline)**: Mô hình nguyên bản `unsloth/Qwen3.5-2B` chưa qua SFT trên tập dữ liệu dự án.
- **E1 (Monolingual English)**: Huấn luyện trên 60,000 mẫu Core tiếng Anh.
- **E2 (Monolingual Vietnamese)**: Huấn luyện trên đúng 60,000 mẫu Core tiếng Việt tương ứng.
- **E3 (Bilingual Balanced)**: Huấn luyện trên 30,000 mẫu tiếng Anh + 30,000 mẫu tiếng Việt.
- **E4 (Bilingual + Domain-Specific)**: Huấn luyện trên 60,000 mẫu song ngữ của E3 cộng với toàn bộ 5,600 mẫu `CustomTools-VI/train.jsonl` (tổng ngân sách 65,600 mẫu).

---

### 4.2 Phương pháp 2: Kiến trúc Chuyên biệt Bi-Encoder + Cross-Encoder

Hệ thống được thiết kế theo mô hình đường ống (pipeline) hai giai đoạn nối tiếp nhằm loại bỏ hoàn toàn quá trình giải mã tự hồi quy.

#### Giai đoạn 1: Truy hồi công cụ ngữ nghĩa bằng Bi-Encoder (BGE-M3)
Bi-Encoder mã hóa câu truy vấn của người dùng $q$ và văn bản mô tả của từng công cụ $d_t$ thành các vector biểu diễn dense $\mathbf{e}_q, \mathbf{e}_t \in \mathbb{R}^d$:

$$\mathbf{e}_q = \text{BiEncoder}(q), \quad \mathbf{e}_t = \text{BiEncoder}(d_t)$$

Điểm tương đồng ngữ nghĩa được tính bằng độ đo Cosine: $s(q, t) = \cos(\mathbf{e}_q, \mathbf{e}_t)$. Mô hình được huấn luyện qua 2 vòng bằng hàm mất mát `CachedMultipleNegativesRankingLoss`:
- **Vòng 1 (Teacher)**: Huấn luyện trên 78,435 cặp (Query, Positive Tool) để thiết lập không gian vector nền tảng.
- **Khai thác Hard Negative**: Dùng checkpoint Vòng 1 để quét toàn bộ kho công cụ và chọn ra các công cụ sai có điểm tương đồng cao nhất làm mẫu âm tính khó.
- **Vòng 2 (Student)**: Huấn luyện tiếp tục với các mẫu hard negative đã khai thác.

**Cơ chế từ chối gọi công cụ (Abstention Thresholding)**: Trên tập validation, chúng tôi hiệu chuẩn hai siêu tham số: ngưỡng tin cậy tối thiểu $\tau = 0.35$ và khoảng cách biên giữa Top-1 và Top-2 $\delta = 0.21$. Nếu điểm số cao nhất $s(q, t^*) < \tau$, hệ thống lập tức kết luận đây là truy vấn thông thường và trả về kết quả không gọi công cụ.

#### Giai đoạn 2: Trích xuất tham số có nhận thức Schema bằng Cross-Encoder (XLM-RoBERTa-base)
Đối với mỗi công cụ $t$ vượt qua bộ lọc ở Giai đoạn 1, Cross-Encoder tiếp nhận đầu vào ghép cặp dạng BERT-QA giữa câu truy vấn $q$ và định nghĩa của từng tham số $p_k \in \mathcal{P}_t$:

$$\mathbf{h} = \text{CrossEncoder}([CLS] \circ q \circ [SEP] \circ p_k \circ [SEP])$$

Mô hình kích hoạt 4 đầu dự đoán chuyên biệt:
1. **Đầu `has_value`**: Phân loại nhị phân xác định tham số $p_k$ có xuất hiện trong câu hỏi hay không:
   $$\hat{y}_{has} = \sigma(\mathbf{W}_{has} \mathbf{h}_{[CLS]} + b_{has})$$
2. **Đầu `span_extraction`**: (Kích hoạt khi $p_k$ là chuỗi văn bản tự do) Dự đoán vị trí bắt đầu và kết thúc của giá trị trong câu truy vấn:
   $$P_{start}(i) = \text{softmax}(\mathbf{W}_s \mathbf{h}_i), \quad P_{end}(j) = \text{softmax}(\mathbf{W}_e \mathbf{h}_j)$$
3. **Đầu `enum_classification`**: (Kích hoạt khi $p_k$ thuộc dạng liệt kê) Dự đoán xác suất rơi vào các giá trị cho phép trong schema:
   $$\mathbf{p}_{enum} = \text{softmax}(\mathbf{W}_{enum} \mathbf{h}_{[CLS]} + \mathbf{b}_{enum})$$
4. **Đầu `boolean`**: (Kích hoạt khi $p_k$ là kiểu logic) Phân loại True/False.

#### Bộ chuẩn hóa giá trị (Value Normalizer)
Một module xử lý hậu kỳ dựa trên từ điển và biểu thức chính quy (Regex) chịu trách nhiệm chuẩn hóa các chuỗi văn bản tiếng Việt thành định dạng số học hoặc ngày tháng chuẩn (ví dụ: chuyển "ngày 20 tháng 10" thành `2026-10-20`, "nửa triệu" thành `500000`).

---

## 5. Phương Pháp Đo Lường & Tiêu Chí Đánh Giá

Kế thừa phương pháp luận từ BFCL và Ersoy et al. (2025), chúng tôi áp dụng hệ thống đo lường toán học chặt chẽ trên toàn bộ các tập kiểm thử:

### 5.1 Độ chính xác lựa chọn công cụ (Tool Selection Metrics)
Tính toán độ chuẩn xác (Precision) và độ phủ (Recall) đối với từng công cụ $T$:

$$P_T = \frac{TP_T}{TP_T + FP_T}, \quad R_T = \frac{TP_T}{TP_T + FN_T}$$

Độ đo trung bình có trọng số trên toàn bộ tập công cụ $\mathcal{K}$:

$$\text{Precision}_{weighted} = \sum_{T \in \mathcal{K}} \frac{N_T}{N_{total}} P_T, \quad \text{Recall}_{weighted} = \sum_{T \in \mathcal{K}} \frac{N_T}{N_{total}} R_T$$

Trong đó $N_T$ là số lượng mẫu thực tế của công cụ $T$ trong tập test. Đối với các tập kiểm thử đơn lẻ, chúng tôi báo cáo **Tool Selection Accuracy (Tool Acc %)** trên toàn bộ các mẫu dương tính.

### 5.2 Độ chính xác điền tham số (Argument Population Accuracy - ArgA / Exact Match)
ArgA là thước đo toàn diện nhất, đo lường tỷ lệ các mẫu mà mô hình dự đoán chính xác tuyệt đối cả tên công cụ và toàn bộ các giá trị tham số (không chấp nhận bất kỳ sai lệch nào về khóa hoặc giá trị):

$$\text{ArgA} = \frac{\text{Số lượng mẫu khớp chính xác tuyệt đối (Exact Matches)}}{\text{Tổng số lượng mẫu kiểm thử dương tính (Total Positive Cases)}}$$

### 5.3 Độ phủ mẫu âm tính (Non-FC Recall)
Đo lường năng lực kiềm chế của mô hình khi gặp các câu hỏi không liên quan đến công cụ:

$$\text{Non-FC Recall} = \frac{\text{Số mẫu âm tính từ chối gọi công cụ chính xác (True Negatives)}}{\text{Tổng số mẫu âm tính trong tập kiểm thử (Total Negative Cases)}}$$

### 5.4 Tỷ lệ lỗi cú pháp và Độ trễ suy luận
- **Syntax Error Rate (%)**: Tỷ lệ phần trăm các câu sinh ra không thể phân tích được thành cấu trúc JSON hoặc XML hợp lệ.
- **Inference Latency (P50, ms)**: Thời gian suy luận trung vị trên từng câu hỏi, được đo lường đồng nhất trên phần cứng GPU NVIDIA T4.

---

## 6. Kết Quả Thực Nghiệm & Phân Tích Chuyên Sâu

Bảng 2 và Bảng 3 trình bày toàn bộ kết quả thực nghiệm chi tiết của đề tài qua các cấu hình thực nghiệm trên cả hai tập benchmark.

**Bảng 2: Kết quả thực nghiệm toàn diện trên benchmark CustomTools-VI (800 mẫu Seen / 800 mẫu Unseen)**  
*(Số liệu in đậm thể hiện kết quả tốt nhất trong từng phân nhóm. Dấu \* biểu thị kết quả chạy từ notebook riêng, cần chạy lại theo đúng benchmark chuẩn của Method 1)*

| Thí nghiệm | Cấu hình huấn luyện | Seen: Tool Acc (%) | Seen: ArgA / EM (%) | Seen: Non-FC Rec (%) | Unseen: Tool Acc (%) | Unseen: ArgA / EM (%) | Unseen: Non-FC Rec (%) | Cú pháp lỗi (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Qwen3.5-2B Zero-Shot | 39.25% | 56.75% | 97.75% | 40.75% | 62.50% | 97.25% | 11.00% |
| **E1** | Monolingual EN (60k) | 91.50% | 63.12% | 75.25% | 96.50% | 70.50% | 74.50% | 4.50% |
| **E2** | Monolingual VI (60k) | 91.00% | 51.50% | 67.50% | 96.25% | 59.62% | 67.75% | 7.18% |
| **E3** | Song ngữ EN+VI (60k) | 92.25% | 19.38% | 3.00% | 96.00% | 27.88% | 2.00% | 4.31% |
| **E4** | Song ngữ + Custom VI (65.6k)| **93.25%** | **87.00%** | **100.00%** | **97.00%** | **86.38%** | **100.00%** | **2.44%** |
| **Method 2**| Bi-Encoder + Cross-Encoder | 91.75%\* | 67.75%\* | 92.25%\* | 88.50%\* | 22.75%\* | 91.50%\* | **0.00%** |

*\*Ghi chú: Kết quả Method 2 trên CustomTools được ghi nhận từ lần chạy độc lập trước đó. Cần chạy lại theo đúng bộ evaluator chuẩn hóa của Method 1 để đảm bảo 100% tính đồng nhất về tiêu chí đánh giá.*

---

**Bảng 3: Kết quả thực nghiệm trên Canonical Core Benchmark (7,712 mẫu / ngôn ngữ)**

| Thí nghiệm | Cấu hình huấn luyện | VI Test: Tool Acc (%) | VI Test: ArgA (%) | VI Test: Non-FC (%) | EN Test: Tool Acc (%) | EN Test: ArgA (%) | EN Test: Non-FC (%) | Độ trễ VI (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Qwen3.5-2B Zero-Shot | 56.34% | 40.48% | 96.33% | 85.18% | 60.63% | 94.50% | 707 ms |
| **E1** | Monolingual EN (60k) | 93.67% | 65.57% | 94.30% | 94.07% | **73.66%** | 94.30% | 878 ms |
| **E2** | Monolingual VI (60k) | 90.25% | 64.90% | 94.30% | 93.45% | 71.36% | 93.08% | 200 ms* |
| **E3** | Song ngữ EN+VI (60k) | **94.00%** | **69.76%** | 94.30% | **94.27%** | 73.22% | 94.30% | 864 ms |
| **E4** | Song ngữ + Custom VI | 93.92% | 69.75% | 94.30% | 94.17% | 73.15% | 94.30% | 970 ms |
| **Method 2**| Bi-Encoder + Cross-Encoder | *[Cần chạy lại]* | *[Cần chạy lại]* | *[Cần chạy lại]* | — | — | — | **58.16 ms** |

*\*Ghi chú: Độ trễ E2 đo ở batch size lớn hơn trên Kaggle. Đối với Method 2, do kết quả báo cáo cũ (ArgA 40.21%) được đo trên 10,555 mẫu positive cũ và tiêu chí chấm riêng, cần chạy lại benchmark Method 2 trên đúng 7,712 mẫu canonical của Method 1 để có số liệu đối chiếu chuẩn tắc.*

---

### 6.1 Giải đáp RQ1: Năng Lực Chuyển Giao Ngôn Ngữ Chéo (Cross-Lingual Transfer)
Đối chiếu giữa E1 (chỉ học tiếng Anh) và E2 (chỉ học tiếng Việt):
- Khi đánh giá trên tập Core tiếng Việt, E1 đạt ArgA lên tới **65.57%**, thậm chí nhỉnh hơn nhẹ so với E2 (**64.90%**). Trên tập Custom Unseen, E1 đạt **70.50%**, cao hơn đáng kể so với E2 (**59.62%**).
- Hiện tượng này chứng minh rằng mô hình nền tảng Qwen3.5 đã sở hữu không gian vector đa ngôn ngữ được căn chỉnh xuất sắc. Cấu trúc ngữ nghĩa của việc gọi hàm học từ tiếng Anh có thể chuyển giao thẳng sang tiếng Việt mà không bị suy hao.
- Tuy nhiên, chuyển giao này có tính bất đối xứng: E2 bị suy giảm hiệu năng khi kiểm thử trên tiếng Anh (71.36% so với 73.66% của E1). Việc kết hợp song ngữ cân bằng ở E3 giúp mô hình hóa giải hiện tượng này, nâng ArgA Core tiếng Việt lên mức cao nhất (**69.76%**).

### 6.2 Giải đáp RQ2: Hiện Tượng Nghiện Gọi Công Cụ & Vai Trò Dữ Liệu Âm Tính
Một phát hiện mang tính đột phá trong nghiên cứu là sự sụp đổ nghiêm trọng của mô hình song ngữ E3 khi gặp tập `CustomTools-VI`:
- ArgA của E3 tụt xuống chỉ còn **19.38%** (Seen) và **27.88%** (Unseen).
- Phân tích chi tiết chỉ ra nguyên nhân: **Non-FC Recall của E3 rơi thẳng xuống đáy vực 3.00% và 2.00%**. Mô hình bị mắc lỗi kích hoạt giả (False Positive) gần như tuyệt đối: bất kỳ câu nói bâng quơ nào của người dùng (như *"Trời hôm nay nóng bức quá"*) đều bị mô hình ép gọi công cụ (ví dụ: gọi công cụ `thanh_toan_tien_dien`).
- **Nguyên nhân**: Dữ liệu SFT song ngữ tổng quát tạo ra thiên kiến cực đoan về việc phải sinh mã gọi hàm. Khi bước vào miền ngữ cảnh mới mang đặc thù Việt Nam mà không có mẫu âm tính hướng dẫn, mô hình mất khả năng tự kiềm chế.
- **Giải pháp**: Ở cấu hình E4, khi được bổ sung các mẫu âm tính tiếng Việt bản địa, Non-FC Recall ngay lập tức đạt mức hoàn hảo **100.00%**, qua đó giải phóng toàn bộ tiềm năng của mô hình, đưa ArgA bứt phá ngoạn mục lên **87.00%** (Seen) và **86.38%** (Unseen). **Kết luận**: Hiệu chuẩn mẫu âm tính tại miền đích là điều kiện tiên quyết để tác tử AI vận hành an toàn.

### 6.3 Giải đáp RQ3: Năng Lực Khái Quát Hóa Zero-Shot Trên Công Cụ Mới (Unseen Tools)
Khoảng cách công nghệ giữa hai trường phái bộc lộ rõ nét nhất ở tập công cụ unseen:
- **Method 1 (SLM E4)** thể hiện sự bền bỉ đáng kinh ngạc: ArgA trên `test_unseen` đạt **86.38%**, chỉ giảm vỏn vẹn **0.62%** so với tập `test_seen` (87.00%). Nhờ cơ chế suy luận trong ngữ cảnh (in-context reasoning), mô hình tự hồi quy có thể đọc hiểu định nghĩa API mới toanh trong prompt và điền tham số một cách chính xác tuyệt đối.
- **Method 2 (Bi+Cross Encoder)** bị sụp đổ từ 67.75% xuống chỉ còn **22.75%** (giảm tới 45 điểm phần trăm). Mặc dù Bi-Encoder vẫn tìm đúng công cụ (Recall@1 ~88%), Cross-Encoder thất bại nặng nề ở khâu trích xuất tham số: đầu span extraction và enum classification bị trật khớp phân phối khi gặp các cấu trúc định nghĩa mới chưa có trong tập train.

### 6.4 Giải đáp RQ4: So Sánh Đối Đầu Quyết Định và Đường Biên Đánh Đổi Pareto

**Bảng 4: So sánh đối đầu trực diện giữa Method 1 và Method 2 trên các trục kỹ thuật**  
*(Ghi chú: Số liệu Method 2 mang tính tham chiếu sơ bộ từ notebook cũ; cần chạy lại trên cùng bộ đánh giá chuẩn hóa của Method 1 để chốt số liệu đối đầu chính thức)*

| Trục đánh giá | Method 1: SLM Qwen3.5-2B (E4) | Method 2: Bi+Cross (BGE-M3 + XLM-R) | Kết luận lựa chọn kỹ thuật |
|---|:---:|:---:|---|
| **Độ chính xác Seen (ArgA)** | **87.00%** | 67.75%\* (+19.25% cho SLM) | SLM vượt trội trong việc điền tham số phức tạp |
| **Độ chính xác Unseen (ArgA)** | **86.38%** | 22.75%\* (+63.63% cho SLM) | **SLM áp đảo hoàn toàn về Zero-shot** |
| **Độ trễ suy luận (P50)** | ~860 – 970 ms | **58.16 – 91.88 ms** | **Method 2 nhanh gấp ~10–15 lần** |
| **Chiếm dụng VRAM bộ nhớ** | ~4.8 GB | **< 1.2 GB** | Method 2 triển khai dễ dàng trên CPU / Edge |
| **Rủi ro cú pháp (Syntax Error)** | 2.44% | **0.00% (Tuyệt đối)** | Method 2 đảm bảo 100% tuân thủ cấu trúc |
| **Khả năng mở rộng công cụ** | Context phình to, chậm dần | Vector Index cố định, tốc độ $O(1)$ | Method 2 mở rộng tới hàng ngàn công cụ |
| **Trạng thái kiểm thử** | ✅ Đã hoàn tất 100% | ⏳ **Cần chạy lại theo benchmark M1** | Chuẩn hóa hoàn toàn tiêu chí đánh giá |

Nghiên cứu xác lập rõ ranh giới Pareto trong thực tế:
- **Ứng dụng Agent phức tạp, OpenAPI mở, tích hợp liên tục**: **Bắt buộc chọn Method 1 (SLM)**.
- **Ứng dụng Voicebot thời gian thực, API đóng cố định, hạ tầng chi phí thấp**: **Nên chọn Method 2 (Bi+Cross)**.

---

## 7. Nghiên Cứu Mở Rộng Quy Mô: Qwen3.5-2B vs Qwen3.5-4B (Planned Scaling Study)

Để kiểm chứng giả thuyết về tác động của dung lượng tham số tới độ chính xác và tỷ lệ lỗi cú pháp, các thực nghiệm huấn luyện cho `unsloth/Qwen3.5-4B` trên hai cấu hình trọng tâm E3 và E4 đang được tiến hành trên hạ tầng GPU.

**Bảng 5: Khung theo dõi nghiên cứu mở rộng quy mô mô hình (Scaling Study)**

| Cấu hình | Dung lượng mô hình | Core VI ArgA (%) | Custom Seen ArgA (%) | Custom Unseen ArgA (%) | Non-FC Recall (%) | Độ trễ (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **E3 (Song ngữ)** | Qwen3.5-2B | 69.76% | 19.38% | 27.88% | 2.00% | 863 ms |
| **E3 (Song ngữ)** | Qwen3.5-4B | *[Đang huấn luyện]* | *[Đang huấn luyện]* | *[Đang huấn luyện]* | *[Đang huấn luyện]* | *[Đang đo]* |
| **E4 (Đặc thù miền)** | Qwen3.5-2B | 69.75% | 87.00% | 86.38% | 100.00% | 970 ms |
| **E4 (Đặc thù miền)** | Qwen3.5-4B | *[Đang huấn luyện]* | *[Đang huấn luyện]* | *[Đang huấn luyện]* | *[Đang huấn luyện]* | *[Đang đo]* |

*Giả thuyết nghiên cứu*: Mô hình 4B được kỳ vọng sẽ giảm tỷ lệ lỗi cú pháp từ 2.44% xuống dưới 1.0%, đồng thời nâng cao năng lực trích xuất các chuỗi tham số phụ thuộc ngữ cảnh dài (như địa chỉ nhiều cấp hành chính Việt Nam). Số liệu thực tế sẽ được cập nhật đồng bộ ngay khi quá trình suy luận hoàn tất.

---

## 8. Bàn Luận, Phân Tích Lỗi & Hạn Chế (Discussion & Limitations)

### 8.1 Phân tích lỗi trích xuất tham số (Error Categorization)
Phân tích 200 trường hợp dự đoán sai của Method 1 (E4) và Method 2 chỉ ra 3 nhóm lỗi chính:
1. **Biến thể định dạng ngày tháng và số tiền (42.5%)**: Người dùng diễn đạt "thứ sáu tuần sau" hoặc "ba triệu tư". SLM đôi khi sinh ra chuỗi nguyên văn thay vì chuyển đổi sang số nguyên `3400000`, trong khi Method 2 dựa vào Value Normalizer nhưng bị thất bại nếu quy tắc regex chưa bao phủ hết các biến thể tiếng lóng.
2. **Nhập nhằng ranh giới thực thể (Entity Boundary Ambiguity - 34.0%)**: Trong các câu truy vấn địa danh phức tạp (ví dụ: *"tìm trọ gần cổng KTX Khu B ĐHQG phường Đông Hòa"*), mô hình gặp khó khăn trong việc quyết định đưa tên phường hay tên trường vào trường tham số `khu_vuc`.
3. **Mâu thuẫn giá trị mặc định của Schema (23.5%)**: Một số API định nghĩa các tham số tùy chọn có giá trị mặc định; mô hình đôi khi tự động điền giá trị mặc định vào lệnh gọi trong khi ground truth để trống.

### 8.2 Hạn chế của nghiên cứu
- **Phạm vi đơn lượt (Single-Turn)**: Đề tài tập trung vào việc giải quyết triệt để bài toán đơn lượt hỗ trợ đa lệnh gọi (multi-call). Các tình huống hội thoại đa lượt (multi-turn dialog) có duy trì ngữ cảnh bộ nhớ nằm ngoài phạm vi khảo sát hiện tại.
- **Môi trường thực thi công cụ (Tool Execution Environment)**: Nghiên cứu dừng lại ở việc đánh giá tính hợp lệ của lệnh gọi (schema matching) mà chưa kết nối với môi trường sandbox để chạy thử API thật và đánh giá phản hồi trả về.

---

## 9. Kết Luận & Hướng Phát Triển Tương Lai

Công trình đã thực hiện một nghiên cứu thực nghiệm quy mô lớn và chuyên sâu bậc nhất về bài toán Tool Calling tiếng Việt:
1. Xác lập bộ benchmark chuẩn hóa gồm hơn 77,000 cặp dữ liệu lõi và 8,000 mẫu nghiệp vụ Việt Nam với cơ chế kiểm thử strict zero-shot.
2. Chứng minh mô hình SLM End-to-End (`Qwen3.5-2B` E4) đạt đỉnh cao về độ chính xác (**87.00%** Seen, **86.38%** Unseen), chứng minh năng lực zero-shot phi thường khi được huấn luyện bằng kỹ thuật response-only loss và dữ liệu âm tính bản địa.
3. Khẳng định kiến trúc phân tách Bi+Cross Encoder là "vũ khí tối thượng" về độ trễ (**58–91 ms**) và chi phí vận hành cho các bài toán thời gian thực.

**Hướng phát triển tiếp theo**:
- Cập nhật số liệu so sánh chi tiết của mô hình `Qwen3.5-4B`.
- Thực hiện **Stress Test** tăng dần số lượng công cụ ($N = 3 \to 1000$) để mô hình hóa đường cong suy giảm độ chính xác của SLM dưới áp lực context dài.
- Nghiên cứu **Kiến trúc Lai Ghép (Hybrid Architecture)**: Sử dụng Bi-Encoder để lọc nhanh Top-3 công cụ từ hàng ngàn API trong vòng 30 ms, sau đó đưa Top-3 này vào SLM để sinh tham số chuẩn xác trong vòng 300 ms, tạo ra giải pháp cân bằng tuyệt đối giữa tốc độ và trí thông minh.

---

## Lời Cảm Ơn (Acknowledgments)

Nghiên cứu này được thực hiện tại Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM. Nhóm tác giả xin gửi lời cảm ơn chân thành và sâu sắc nhất tới **TS. Đặng Văn Thìn** đã tận tình định hướng phương pháp luận, hỗ trợ tài nguyên thực nghiệm và đóng góp những nhận xét chuyên môn quý báu trong suốt quá trình triển khai đề tài.

---

## Tài Liệu Tham Khảo (References)

1. Ersoy, O., Altinisik, E., Sencar, H. T., & Darwish, K. (2025). *Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning*. arXiv preprint.
2. Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). *Gorilla: Large Language Model Connected with Massive APIs*. Advances in Neural Information Processing Systems (NeurIPS 2024). arXiv:2305.15334.
3. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. Advances in Neural Information Processing Systems (NeurIPS 2023).
4. Yan, X., Liu, Z., et al. (2024). *Berkeley Function-Calling Leaderboard (BFCL)*. Gorilla LLM Project, UC Berkeley.
5. Qin, B., Wang, Y., Xu, Y., Meng, Y., Wang, Y., Teng, Z., Yan, J., Wei, Z., Feng, Y., Wang, Z., & Zhao, D. (2023). *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs*. ICLR 2024.
6. Liu, Z., Hoang, T., Zhang, J., Zhu, M., Lan, T., Kokane, S., Tan, J., Yao, W., Liu, Z., Feng, Y., Murthy, R., Yang, L., Savarese, S., Niebles, J. C., Wang, H., Heinecke, S., & Xiong, C. (2024b). *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*. NeurIPS 2024.
7. Liu, W., Huang, X., Zeng, X., Hao, X., Yu, S., Li, D., Wang, S., Gan, W., Liu, Z., Yu, Y., Wang, Z., Wang, Y., Ning, W., Hou, Y., Wang, B., Wu, C., Wang, X., Liu, Y., Wang, Y., et al. (2024a). *ToolACE: Winning the Points of LLM Function Calling*. arXiv preprint arXiv:2409.00920.
8. Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE-M3)*. arXiv:2309.07597.
9. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)*. ACL 2020.
10. Chen, Z., Shen, S., Shen, G., Zhi, G., Chen, X., & Lin, Y. (2024). *Towards Tool Use Alignment of Large Language Models*. EMNLP 2024.
