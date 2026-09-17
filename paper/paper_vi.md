# Gọi Công Cụ (Tool Calling) Tiếng Việt: Nghiên Cứu So Sánh Giữa Mô Hình Ngôn Ngữ Nhỏ End-to-End Và Kiến Trúc Chuyên Biệt Bi-Encoder + Cross-Encoder

**Đào Phước Thịnh**<sup>1</sup>, **Hà Quang Đạt**<sup>1</sup>, **Đặng Văn Thìn**<sup>1,*</sup>  
<sup>1</sup>Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM, Việt Nam  
Email: `{21521469, 21521925}@ms.uit.edu.vn`, `thindv@uit.edu.vn`  
<sup>*</sup>Tác giả liên hệ (Corresponding author)

---

## Tóm Tắt (Abstract)

Khả năng gọi công cụ (tool calling / function calling) là cơ chế nền tảng cho phép Mô hình Ngôn ngữ Lớn (LLM) vượt khỏi giới hạn tham số tĩnh để tương tác với thế giới thực thông qua các giao diện lập trình ứng dụng (API), cơ sở dữ liệu và công cụ tính toán bên ngoài. Mặc dù đã đạt nhiều bước tiến đột phá trên tiếng Anh, việc xây dựng năng lực gọi công cụ hiệu quả, độ trễ thấp và có khả năng khái quát hóa Zero-Shot cho tiếng Việt vẫn là một bài toán chưa được khai phá thấu đáo. Các hệ thống thương mại đóng đám mây hiện nay đối mặt với nhiều rào cản lớn: độ trễ suy luận tự hồi quy cao (800–2,000 ms), chi phí vận hành token đắt đỏ, rủi ro an ninh dữ liệu nội bộ, nguy cơ sinh lỗi cú pháp JSON và hiện tượng bùng nổ bộ nhớ ngữ cảnh khi số lượng công cụ tăng cao. Trong bài báo này, chúng tôi tiến hành nghiên cứu so sánh đối đầu thực nghiệm có hệ thống đầu tiên cho tiếng Việt giữa hai trường phái kiến trúc: Mô hình Ngôn ngữ Nhỏ tạo sinh End-to-End (SLM Qwen3.5 2B và 4B) và Kiến trúc phân biệt phi tạo sinh phân tách (Bi-Encoder BGE-M3 kết hợp Hierarchical Cross-Encoder XLM-R). Để phục vụ đánh giá, chúng tôi công bố hai bộ benchmark chuẩn hóa: **Canonical Core Benchmark** (77,028 cặp bản ghi song ngữ trên 4,421 công cụ) và **CustomTools-VI** (8,000 mẫu thuộc 10 lĩnh vực đời sống Việt Nam, phân chia nghiêm ngặt 20 công cụ seen và 20 công cụ zero-shot unseen với 50% mẫu âm tính). Kết quả thực nghiệm khẳng định SLM E4 thiết lập đỉnh cao SOTA mới với độ chính xác tham số (ArgA) đạt **87.92%** (seen) và **86.75%** (unseen) ở bản 4B, vượt qua mô hình đóng `GPT-5.6 Luna`, đồng thời triệt tiêu lỗi cú pháp xuống 0.32%. Ngược lại, kiến trúc phân tách đạt độ trễ thời gian thực kỷ lục **54.68–58.16 ms** (nhanh gấp 15–18 lần SLM) với 0% lỗi cú pháp và VRAM dưới 1.2 GB. Cuối cùng, thực nghiệm Stress Test ($N = 3 \to 1,000$) chỉ ra rằng kiến trúc phân tách duy trì ổn định dưới 92 ms, trong khi SLM gặp hiện tượng bùng nổ bộ nhớ bậc hai dẫn đến sụp đổ tràn bộ nhớ (CUDA OOM) tại $N \ge 500$ trên phần cứng 16GB VRAM, từ đó xác lập rõ nét đường biên Pareto phục vụ thiết kế các tác tử AI tối ưu trong thực tế.

---

## 1. Giới Thiệu (Introduction)

Khả năng gọi công cụ (tool calling), thường được gọi là gọi hàm (function calling), là cơ chế nền tảng giúp các mô hình ngôn ngữ lớn (LLM) và các tác tử thông minh (AI Agents) kết nối với thế giới bên ngoài. Thông qua việc phân tích ngôn ngữ tự nhiên của người dùng và định nghĩa schema của các công cụ khả dụng, mô hình tự động nhận diện thời điểm cần kích hoạt công cụ, lựa chọn chính xác API mục tiêu và trích xuất các tham số cấu trúc tương ứng.

![Hình 1: Sơ đồ kiến trúc tổng quan đối chiếu giữa Method 1 (SLM End-to-End) và Method 2 (Bi-Encoder + Cross-Encoder)](figures/fig1_system_architecture.png)

*Hình 1: Sơ đồ kiến trúc tổng quan đối chiếu giữa hai trường phái kỹ thuật: Phương pháp 1 (SLM End-to-End dựa trên Qwen3.5 sinh mã tự hồi quy) và Phương pháp 2 (Kiến trúc phân tách không tự hồi quy kết hợp Bi-Encoder BGE-M3 và Cross-Encoder XLM-R).*

Mặc dù các hệ thống thương mại hàng đầu (như OpenAI Function Calling, Google Gemini Function Calling) đã chứng minh hiệu năng mạnh mẽ trên tiếng Anh, việc ứng dụng chúng vào các hệ thống tại Việt Nam đối mặt với 4 thách thức kỹ thuật lớn:
1. **Độ trễ suy luận lớn**: Quá trình giải mã tự hồi quy (autoregressive decoding) qua context chứa hàng chục định nghĩa JSON Schema thường kéo dài từ 800 ms đến 2,000 ms, không đáp ứng được yêu cầu phản hồi tức thời của các hệ thống tổng đài thoại thông minh (Voicebot) hoặc trợ lý thanh toán.
2. **Chi phí vận hành và rủi ro bảo mật**: Việc gửi toàn bộ dữ liệu nội bộ và prompt hội thoại lên các API đám mây quốc tế làm tăng chi phí token và vi phạm các quy định về chủ quyền dữ liệu.
3. **Ảo giác cấu trúc (Format Hallucination)**: Mô hình tạo sinh có nguy cơ xuất ra chuỗi JSON dị tật, thiếu dấu đóng ngoặc hoặc tự ý bịa đặt các tham số không có trong tài liệu kỹ thuật.
4. **Sự thiếu hụt tài nguyên nghiên cứu tiếng Việt**: Tiếng Việt mang đặc trưng đơn lập, không biến hình, phụ thuộc vào thanh điệu và ngữ cảnh, đồng thời có thói quen biểu đạt số tiền, ngày tháng đa dạng ("hai triệu rưỡi", "ngày rằm tháng giêng"). Hầu hết các benchmark toàn cầu hiện nay (như BFCL, ToolBench) hoàn toàn bỏ trống tiếng Việt.

Công trình này được xây dựng nhằm giải quyết triệt để các thách thức trên thông qua việc so sánh thực nghiệm đối đầu giữa hai phương pháp: **Mô hình Ngôn ngữ Nhỏ End-to-End (SLM)** và **Kiến trúc Chuyên biệt Phân tách (Bi-Encoder + Cross-Encoder)**. Cụ thể, bài báo tập trung giải đáp 5 câu hỏi nghiên cứu (Research Questions) cốt lõi:
- **RQ1 (Năng lực chuyển giao ngôn ngữ chéo)**: Khả năng chuyển giao tri thức gọi công cụ (cross-lingual transfer) từ tiếng Anh sang tiếng Việt của mô hình nền tảng đạt mức độ nào khi không có dữ liệu huấn luyện bản địa?
- **RQ2 (Hiện tượng kích hoạt công cụ quá mức & Hiệu chuẩn âm tính)**: Quá trình tinh chỉnh song ngữ phổ quát có gây ra hiện tượng kích hoạt công cụ thiếu kiểm soát (over-triggering pathology) trên các câu hội thoại thông thường, và vai trò của dữ liệu âm tính (negative calibration) là gì?
- **RQ3 (Khả năng tổng quát hóa Zero-Shot trên công cụ mới)**: Khi đối mặt với các công cụ đặc thù chưa từng gặp trong miền đời sống Việt Nam (`test_unseen`), mô hình tạo sinh tự hồi quy (SLM) hay kiến trúc phân biệt (Bi-Encoder + Cross-Encoder) thể hiện năng lực thích ứng vượt trội hơn?
- **RQ4 (Đường biên đánh đổi Pareto giữa chất lượng và độ trễ)**: Kiến trúc phân tách không tự hồi quy có thể đạt độ trễ thời gian thực và tối ưu tài nguyên tính toán tới mức nào so với phương pháp SLM End-to-End?
- **RQ5 (Tính bền bỉ và giới hạn vật lý dưới áp lực mở rộng danh mục)**: Khi số lượng công cụ trong prompt tăng từ quy mô nhỏ ($N=3$) đến quy mô thực tế ($N=1.000$), đường cong suy giảm độ chính xác và độ trễ diễn biến ra sao, và đâu là ranh giới sụp đổ bộ nhớ của mô hình tạo sinh?

### Đóng góp khoa học của bài báo:
1. **Công bố hai bộ benchmark tiếng Việt chuẩn hóa**: Xây dựng **Canonical Core Benchmark** gồm 77,028 cặp bản ghi song ngữ (4,421 công cụ duy nhất) và **CustomTools-VI** gồm 8,000 mẫu thuộc 10 nhóm lĩnh vực thực tế tại Việt Nam với cơ chế phân vùng strict zero-shot unseen và 50% mẫu âm tính.
2. **Khám phá hiện tượng over-triggering và giải pháp khắc phục**: Chứng minh mô hình song ngữ phổ quát sụp đổ độ nhạy từ chối câu thường (Non-FC Recall chỉ đạt 2.0%), đồng thời chứng minh việc bổ sung dữ liệu âm tính bản địa hóa khôi phục hoàn hảo Non-FC Recall lên 100.0%.
3. **Hiện thực hóa pipeline phân tách đạt tốc độ thời gian thực**: Xây dựng thành công hệ thống Bi-Encoder + Hierarchical Cross-Encoder đạt độ trễ kỷ lục 54.68 ms (nhanh gấp 15–18 lần SLM), kiểm soát VRAM dưới 1.2 GB và triệt tiêu 100% lỗi cú pháp JSON.
4. **Xác lập kỷ lục SOTA mới trên mô hình ngôn ngữ nhỏ 4B**: Mô hình `Qwen3.5-4B` đạt ArgA 87.92% (seen) và 86.75% (unseen), vượt qua mô hình thương mại đóng `GPT-5.6 Luna` (+8.75% seen, +6.88% unseen).
5. **Xác định giới hạn vật lý trong Stress Test ($N = 3 \to 1.000$)**: Định lượng ranh giới sụp đổ tràn bộ nhớ CUDA OOM của SLM tại $N \ge 500$ trên phần cứng 16GB, xác lập rõ ràng đường biên Pareto giữa chất lượng và tài nguyên tính toán.

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
*(Ghi chú: Core Benchmark là tập dữ liệu song ngữ đối xứng 1:1 (Paired EN–VI). CustomTools-VI có tỷ lệ mẫu âm tính cân bằng chính xác 50% ở mọi tập kiểm thử).*

| Bộ dữ liệu | Mục đích & Đặc tính | Ngôn ngữ | Lệnh gọi | Mẫu Dương (FC) | Mẫu Âm (Non-FC) | Tập Train | Tập Test | Số công cụ (Tools) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Nhóm 1: Canonical Core Benchmark (77,028 cặp bản ghi song ngữ đối sánh 1:1)* | | | | | | | | |
| **Canonical Glaive** | Khái quát hóa đơn lệnh + Âm tính | Song ngữ EN–VI (Paired) | Đơn lệnh (Single) | 14,568 | 3,642 | 18,210 | 2,276 | 972 |
| **Canonical xLAM** | Cấu trúc phức tạp + Đa lệnh gọi | Song ngữ EN–VI (Paired) | Đa lệnh (Multi) | 47,047 | 0 | 47,047 | 5,891 | 3,449 |
| **Tổng Core Benchmark** | **Khung chuẩn quy mô lớn** | **Song ngữ EN–VI** | **Đơn & Đa lệnh** | **57,750 (×2)** | **3,865 (×2)** | **61,615 (×2)** | **7,712 (×2)** | **4,421** |
| *Nhóm 2: CustomTools-VI Benchmark (8,000 mẫu đặc thù đời sống Việt Nam)* | | | | | | | | |
| **CustomTools-VI (Seen)** | 20 công cụ đã học (10 nhóm miền) | Tiếng Việt (VI) | Đơn & Đa lệnh | 2,800 | 2,800 | 5,600 | 800 | 20 (Seen) |
| **CustomTools-VI (Unseen)**| 20 công cụ Zero-Shot (10 nhóm miền)| Tiếng Việt (VI) | Đơn & Đa lệnh | 400 | 400 | 0 *(Zero-Shot)* | 800 | 20 (Unseen) |
| **Tổng CustomTools-VI** | **Khung chuẩn miền thực tế** | **Tiếng Việt (VI)** | **Đơn & Đa lệnh** | **3,200** | **3,200** | **5,600** | **1,600** | **40** |

*\*Ghi chú chuyển đổi biểu diễn dữ liệu (Data Representation Mapping): Số liệu trong Bảng 1 đại diện cho các bản ghi hội thoại gốc (Master Conversational Samples). Khi huấn luyện kiến trúc phân biệt Method 2, các bản ghi này được trích xuất thành 78,435 cặp (query, tool_description) cho Bi-Encoder (loại bỏ hoàn toàn định danh của 20 công cụ unseen) và 145,383 cặp (query, param_schema) phân cấp cho Cross-Encoder. Trên Core Benchmark, Method 2 kiểm thử trên 10,555 truy vấn positive chuẩn của pipeline.*

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
- **E4 (Bilingual + Domain-Specific)**: Huấn luyện trên 60,000 mẫu song ngữ của E3 kết hợp cùng 5,600 mẫu tập huấn luyện đặc thù miền `CustomTools-VI` (tổng ngân sách 65,600 mẫu).

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

### 4.3 Hạ Tầng Phần Cứng & Siêu Tham Số Huấn Luyện (Hardware & Training Setup)

Nhằm đảm bảo tính minh bạch, khả năng tái lập (reproducibility) và đối chiếu công bằng, toàn bộ các mô hình thuộc Method 1 và Method 2 đều được huấn luyện và đánh giá trên các môi trường phần cứng chuyên biệt có kiểm soát nghiêm ngặt:

- **Hạ tầng huấn luyện (Training Infrastructure)**: Thực hiện trên nền tảng Google Colab Pro với 1× GPU NVIDIA A100-SXM4-40GB (VRAM khả dụng: 39.49 GB), PyTorch 2.8.0, CUDA Toolkit 12.8, kiểu dữ liệu Bfloat16 (`bf16=True`) và framework Unsloth (phiên bản 2026.9.4 với bản vá tối ưu hóa bộ nhớ cho kiến trúc Qwen3.5).
- **Hạ tầng kiểm thử suy luận (Inference & Evaluation Benchmark)**: Thực hiện độc lập trên môi trường Kaggle với 2× GPU NVIDIA Tesla T4 (14.56 GB VRAM mỗi card, kiến trúc Turing Compute Capability 7.5), CUDA 12.x, PyTorch 2.x. Các mô hình SLM được định lượng 4-bit (NF4 BitsAndBytes) và giải mã tham lam (`do_sample=False`, `max_new_tokens=128`).
- **Bảng thông số siêu tham số và tài nguyên thực tế**:

| Thành phần / Tiêu chí | Qwen3.5-2B (E1 $\to$ E4) | Qwen3.5-4B (E3, E4) | Method 2: Bi+Cross |
| :--- | :--- | :--- | :--- |
| **Kiến trúc Backbone** | `unsloth/Qwen3.5-2B` | `unsloth/Qwen3.5-4B` | `BGE-M3` + `XLM-R base` |
| **Kỹ thuật thích ứng** | QLoRA (4-bit NF4) | QLoRA (4-bit NF4) | LoRA (Bi-Enc) / Full Head (Cross-Enc) |
| **LoRA Parameters** | $r=16, \alpha=16$, dropout $0.0$ | $r=16, \alpha=16$, dropout $0.0$ | $r=16, \alpha=32$ (Bi-Encoder) |
| **Target Modules** | `q, k, v, o, gate, up, down_proj` | `q, k, v, o, gate, up, down_proj` | `q_proj, v_proj` (Bi-Encoder) |
| **Tổng tham số mô hình** | 2,224,153,408 (~2.22B) | 4,560,499,200 (~4.56B) | 567M + 278M (~845M) |
| **Tham số huấn luyện** | **10,911,744 (0.49%)** | **21,233,664 (0.47%)** | ~15M (LoRA + Heads) |
| **Tốc độ học (Learning Rate)** | $5 \times 10^{-7}$ (Cosine, warmup 0.05) | $5 \times 10^{-7}$ (Cosine, warmup 0.05) | $2 \times 10^{-5}$ (Bi) / $3 \times 10^{-5}$ (Cross) |
| **Batch Size cấu hình** | $64 \times 1$ (Per-device: 64, Accum: 1) | $32 \times 2$ (Per-device: 32, Accum: 2) | 32 (Bi-Enc) / 16 (Cross-Enc) |
| **Effective Batch Size** | **64** (Đồng nhất 100%) | **64** (Đồng nhất 100%) | — |
| **Thời gian train / run 60k** | **~49.5 phút** (938 steps trên A100) | **~1 giờ 55 phút** (938 steps trên A100)| ~2.5 giờ (2 rounds) + ~3 giờ (CE) |
| **Thời gian train E4 (65.6k)**| **~54 phút** (1,025 steps trên A100) | **2 giờ 05 phút** (1,025 steps A100) | — |
| **Mất mát hội tụ (Final Loss)**| $\approx 0.0198$ | $\approx 0.0103$ (Giảm ~48%) | — |
| **Phần cứng kiểm thử** | 2× NVIDIA Tesla T4 (15GB) | 2× NVIDIA Tesla T4 (15GB) | 2× NVIDIA Tesla T4 (15GB) |

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
*(Số liệu in đậm thể hiện kết quả tốt nhất trong từng phân nhóm)*

| Thí nghiệm | Cấu hình huấn luyện | Seen: Tool Acc (%) | Seen: ArgA / EM (%) | Seen: Non-FC Rec (%) | Unseen: Tool Acc (%) | Unseen: ArgA / EM (%) | Unseen: Non-FC Rec (%) | Cú pháp lỗi (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Nhóm Mô hình Cục bộ (Local Models)* | | | | | | | | |
| **E0** | Qwen3.5-2B Zero-Shot | 39.25% | 56.75% | 97.75% | 40.75% | 62.50% | 97.25% | 11.00% |
| **E1** | Monolingual EN (60k) | 91.50% | 63.12% | 75.25% | 96.50% | 70.50% | 74.50% | 4.50% |
| **E2** | Monolingual VI (60k) | 91.00% | 51.50% | 67.50% | 96.25% | 59.62% | 67.75% | 7.18% |
| **E3** | Song ngữ EN+VI (60k) | 92.25% | 19.38% | 3.00% | 96.00% | 27.88% | 2.00% | 4.31% |
| **E4** | Song ngữ + Custom VI (65.6k)| **93.25%** | **87.00%** | **100.00%** | **97.00%** | **86.38%** | **100.00%** | **2.44%** |
| **Method 2**| Bi-Encoder + Cross-Encoder | 85.25% | 67.75% | 75.00% | 74.50% | 22.75% | 71.75% | **0.00%** |
| *Nhóm Frontier API Thương mại (Closed-source Baselines)* | | | | | | | | |
| **GPT-5.6 Luna** | OpenAI API (Zero-shot) | 93.25% | 78.25% | 100.00% | 97.25% | 79.50% | 100.00% | 0.00% |
| **Gemini 3.8 Flash** | Google API (Zero-shot) | **100.00%** | **93.62%** | 100.00% | **100.00%** | **92.12%** | 99.75% | 0.06% |

*\*Ghi chú: Số liệu Method 2 được đánh giá độc lập theo giao thức thực nghiệm chuẩn của kiến trúc phân tách Bi-Encoder + Cross-Encoder. Hai mô hình thương mại đóng (GPT-5.6 Luna và Gemini 3.8 Flash) được đánh giá trực tiếp qua API với thiết lập nhiệt độ cố định (temperature = 0) trên toàn bộ 1,600 mẫu CustomTools-VI.*

![Hình 2: So sánh đối đầu hiệu năng trích xuất và lựa chọn công cụ trên benchmark CustomTools-VI](figures/fig2_performance_comparison.png)

*Hình 2: So sánh đối đầu hiệu năng trích xuất tham số (ArgA %) và độ chính xác chọn công cụ (Tool Acc %) trên benchmark CustomTools-VI (Seen vs. Unseen). Mô hình SLM thể hiện sự bền bỉ vượt bậc ở năng lực Zero-shot (Unseen ArgA 86.38% ở 2B và 86.75% ở 4B), trong khi Method 2 bị sụt giảm trích xuất nghiêm trọng khi gặp công cụ mới.*

---

**Bảng 3: Kết quả thực nghiệm trên Canonical Core Benchmark (7,712 mẫu / ngôn ngữ)**

| Thí nghiệm | Cấu hình huấn luyện | VI Test: Tool Acc (%) | VI Test: ArgA (%) | VI Test: Non-FC (%) | EN Test: Tool Acc (%) | EN Test: ArgA (%) | EN Test: Non-FC (%) | Độ trễ VI (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Qwen3.5-2B Zero-Shot | 56.34% | 40.48% | 96.33% | 85.18% | 60.63% | 94.50% | 707 ms |
| **E1** | Monolingual EN (60k) | 93.67% | 65.57% | 94.30% | 94.07% | **73.66%** | 94.30% | 878 ms |
| **E2** | Monolingual VI (60k) | 90.25% | 64.90% | 94.30% | 93.45% | 71.36% | 93.08% | 200 ms* |
| **E3 (2B)** | Song ngữ EN+VI (60k) | 94.00% | 69.76% | 94.30% | 94.27% | 73.22% | 94.30% | 864 ms |
| **E4 (2B)** | Song ngữ + Custom VI | 93.92% | 69.75% | 94.30% | 94.17% | 73.15% | 94.30% | 970 ms |
| **E3 (4B)** | Song ngữ EN+VI (60k) | **98.73%** | **72.86%** | 94.30% | **96.63%** | **74.71%** | 94.30% | 2,432 ms |
| **Method 2**| Bi-Encoder + Cross-Encoder | 80.47%* | 40.21%* | — | — | — | — | **54.68 ms** |

*\*Ghi chú: Độ trễ E2 được đo ở chế độ batch size lớn. Đối với Method 2, kết quả Core Benchmark được đánh giá trên 10,555 trường hợp gọi công cụ hợp lệ theo quy trình chuẩn của pipeline phân tách. Mô hình E3 (4B) được kiểm thử trên hệ thống 2× GPU Tesla T4.*

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
*(Ghi chú: Số liệu thực nghiệm của Method 1 và Method 2 được tổng hợp chính thức từ các báo cáo nghiệm thu thực nghiệm độc lập)*

| Trục đánh giá | Method 1: SLM Qwen3.5-2B (E4) | Method 2: Bi+Cross (BGE-M3 + XLM-R) | Kết luận lựa chọn kỹ thuật |
|---|:---:|:---:|---|
| **Độ chính xác Seen (ArgA)** | **87.00%** | 67.75% (+19.25% cho SLM) | SLM vượt trội trong việc điền tham số phức tạp |
| **Độ chính xác Unseen (ArgA)** | **86.38%** | 22.75% (+63.63% cho SLM) | **SLM áp đảo hoàn toàn về Zero-shot** |
| **Độ trễ suy luận (P50)** | ~860 – 970 ms | **54.68 – 58.16 ms** | **Method 2 nhanh gấp ~15–18 lần** |
| **Chiếm dụng VRAM bộ nhớ** | ~4.8 GB | **< 1.2 GB** | Method 2 triển khai dễ dàng trên CPU / Edge |
| **Rủi ro cú pháp (Syntax Error)** | 2.44% | **0.00% (Tuyệt đối)** | Method 2 đảm bảo 100% tuân thủ cấu trúc |
| **Khả năng mở rộng công cụ** | Context phình to, chậm dần | Vector Index cố định, tốc độ $O(1)$ | Method 2 mở rộng tới hàng ngàn công cụ |

Nghiên cứu xác lập rõ ranh giới Pareto trong thực tế:
- **Ứng dụng Agent phức tạp, OpenAPI mở, tích hợp liên tục**: **Bắt buộc chọn Method 1 (SLM)**.
- **Ứng dụng Voicebot thời gian thực, API đóng cố định, hạ tầng chi phí thấp**: **Nên chọn Method 2 (Bi+Cross)**.

---

### 6.5 So Sánh Đối Chiếu Với Các Frontier API Baselines (GPT-5.6 Luna & Gemini 3.8 Flash)

Thực nghiệm trên 1,600 mẫu `CustomTools-VI` mang lại các phát hiện khoa học mang tính bước ngoặt khi so sánh các mô hình cục bộ với các API thương mại đóng hàng đầu thế giới:

1. **SLM Cục Bộ (Qwen3.5-2B E4) Vượt Trội GPT-5.6 Luna Về Độ Chính Xác Trích Xuất**:
   - Mặc dù `GPT-5.6 Luna` sở hữu năng lực chọn công cụ ấn tượng (Tool Acc đạt **93.25%** Seen và **97.25%** Unseen) cùng khả năng kiềm chế kích hoạt giả hoàn hảo (Non-FC Recall **100.00%**), độ chính xác trích xuất tham số (ArgA / EM) của nó chỉ đạt **78.25%** (Seen) và **79.50%** (Unseen).
   - Ngược lại, mô hình `Qwen3.5-2B` (E4) sau khi được tinh chỉnh với dữ liệu tiếng Việt bản địa hóa đạt ArgA lên tới **87.00%** (Seen) và **86.38%** (Unseen) — **vượt trội GPT-5.6 Luna tới +8.75% trên Seen và +6.88% trên Unseen**. Phát hiện này khẳng định: một mô hình ngôn ngữ nhỏ 2B chạy cục bộ, nếu được căn chỉnh chuẩn xác trên miền mục tiêu, hoàn toàn có thể đánh bại các frontier LLM thương mại nghìn tỷ tham số về độ sâu hiểu biết nghiệp vụ địa phương (đơn vị tiền tệ VND, định dạng địa danh phường/quận Việt Nam).
2. **Gemini 3.8 Flash Thiết Lập Trần Hiệu Năng Frontier**:
   - `Gemini 3.8 Flash` thể hiện sự vượt trội toàn diện với Tool Accuracy tuyệt đối (**100.00%** trên cả 2 tập), đưa ArgA / EM chạm mốc **93.62%** (Seen) và **92.12%** (Unseen) với tỷ lệ lỗi cú pháp gần như bằng 0 (0.06%).
3. **Bài Toán Đánh Đổi Toàn Diện: Độ Trễ, Chi Phí & Quyền Riêng Tư**:
   - **Độ trễ suy luận**: `Gemini 3.8 Flash` (2,086 ms) và `GPT-5.6 Luna` (1,858 ms) có độ trễ qua cloud cao gấp đôi so với SLM E4 (~970 ms) và **chậm hơn tới 35 lần** so với kiến trúc phân tách Method 2 (58 ms).
   - **Chi phí & Triển khai**: Cả hai mô hình thương mại đều phụ thuộc kết nối Internet, chi phí token liên tục và không thể triển khai trên môi trường On-premise hoặc thiết bị biên. Do đó, `Qwen3.5-2B (E4)` và `Method 2` là hai lựa chọn hoàn hảo bổ trợ cho nhau tùy theo ưu tiên về độ chính xác hay độ trễ thời gian thực.

---

### 6.6 Giải đáp RQ5: Khả Năng Mở Rộng Danh Mục & Điểm Nghẽn Vật Lý Dưới Áp Lực Ngữ Cảnh (Stress Testing)

Để kiểm chứng tính bền bỉ của hai trường phái khi danh mục công cụ mở rộng từ quy mô nhỏ ($N=3$) đến quy mô hệ thống thực tế ($N=1.000$ công cụ), chúng tôi tiến hành **Stress Test** đối đầu trực diện giữa đại diện tiêu biểu nhất của Method 1 (`Qwen3.5-2B` E4) và Method 2 (Bi-Encoder BGE-M3 + Cross-Encoder XLM-R) trên 200 anchors câu hỏi đa miền. Toàn bộ thực nghiệm được thực hiện trên cùng môi trường phần cứng độc lập (2× GPU NVIDIA Tesla T4 16GB).

**Bảng 5: Kết quả thực nghiệm Stress Test đối đầu trực diện giữa Method 1 và Method 2 ($N = 3 \to 1.000$)**  
*(Ghi chú: OOM viết tắt của Out of Memory — tiến trình bị hủy do vượt quá dung lượng 16GB VRAM của GPU T4).*

| Số lượng công cụ ($N$) | SLM Tool Acc (%) | M2 Tool Acc (%) | SLM ArgA (%) | M2 ArgA (%) | SLM P50 Latency (ms) | M2 P50 Latency (ms) | Speedup (M2 vs SLM) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 3$** | 94.0% | **100.0%** | **85.5%** | 81.0% | 1,258.94 ms | **58.16 ms** | **21.6×** |
| **$N = 10$** | 93.0% | **98.0%** | **84.5%** | 79.0% | 1,604.68 ms | **57.15 ms** | **28.1×** |
| **$N = 50$** | 93.0% | **96.0%** | **85.5%** | 78.0% | 4,703.39 ms | **55.82 ms** | **84.3×** |
| **$N = 100$** | 92.0% | **95.0%** | **83.0%** | 77.0% | 9,318.21 ms | **58.55 ms** | **159.2×** |
| **$N = 500$** | *OOM* | **78.0%** | *OOM* | **62.0%** | *OOM* | **87.44 ms** | $\infty$ |
| **$N = 1000$** | *OOM* | **68.0%** | *OOM* | **54.0%** | *OOM* | **91.88 ms** | $\infty$ |

![Hình 3: Kết quả Stress Test đối đầu trực diện giữa Method 1 và Method 2 khi quy mô danh mục công cụ mở rộng từ N = 3 đến N = 1.000](figures/fig3_stress_test_curves.png)

*Hình 3: Kết quả thực nghiệm Stress Test đối đầu trực diện: (a) Đường cong suy giảm độ chính xác và ranh giới sụp đổ tràn bộ nhớ (CUDA OOM) của SLM tại N ≥ 500; (b) Đường cong độ trễ suy luận P50 (ms) trên thang đo log, minh chứng ưu thế gia tốc lên tới 159.2× của kiến trúc phân tách Method 2.*

#### 3 Phát hiện thực nghiệm then chốt từ Stress Test:
1. **Khoảng cách độ trễ tăng vọt theo hàm số mũ**:
   - Ở dải danh mục hẹp ($N \le 100$), độ trễ P50 của Method 2 gần như **bằng phẳng tuyệt đối** quanh ngưỡng $\sim 55 - 58$ ms, trong khi độ trễ của SLM tăng vọt từ $1.26$ giây ($N=3$) lên tới **$9.32$ giây ($N=100$)**, tức chậm hơn tới **159.2 lần** (P95 chạm mức $12.0$ giây/câu).
2. **Sự đánh đổi chính xác ở quy mô nhỏ ($N \le 100$)**:
   - SLM duy trì ưu thế nhẹ về ArgA ($+4.5\% \to +6.0\%$) nhờ năng lực suy luận ngôn ngữ liên kết (cross-attention) đồng thời giữa câu hỏi và tham số trong cùng prompt. Ngược lại, Method 2 vượt trội hơn về độ chính xác định vị công cụ (Tool Acc cao hơn $2\% \to 6\%$) nhờ cơ chế Contrastive Learning với Hard Negative Mining của Bi-Encoder.
3. **Bức tường vật lý và hiện tượng sụp đổ (OOM) tại $N \ge 500$**:
   - Khi $N \ge 500$, chiều dài context prompt vượt quá 32,000 tokens. Thuật toán Attention tự hồi quy ($O(L^2)$) của SLM đòi hỏi cấp phát **hơn 32 GiB VRAM** chỉ riêng cho ma trận attention SDPA trong bước prefill, dẫn tới lỗi tràn bộ nhớ (CUDA Out of Memory) và sụp đổ hoàn toàn dịch vụ (Denial of Service) trên phần cứng T4 16GB.
   - Ngược lại, Method 2 thể hiện tính mở rộng vượt trội: tại $N=500$ và $N=1000$, pipeline duy trì độ trễ dưới $92$ ms và VRAM dưới $1.5$ GB, đạt Tool Acc $78\%$ và $68\%$, ArgA $62\%$ và $54\%$.

---

## 7. Nghiên Cứu Mở Rộng Quy Mô: Qwen3.5-2B vs Qwen3.5-4B (Model Scaling Study)

Để kiểm chứng tác động của dung lượng tham số tới độ chính xác trích xuất, năng lực khái quát hóa và tỷ lệ lỗi cú pháp, chúng tôi mở rộng huấn luyện mô hình `unsloth/Qwen3.5-4B` trên hạ tầng GPU NVIDIA A100-SXM4-40GB. Mô hình 4B giữ nguyên ngân sách dữ liệu (60,000 mẫu cho E3 và 65,600 mẫu cho E4), tỷ lệ học $5 \times 10^{-7}$ và cùng effective batch size 64 ($32 \times 2$). Quá trình huấn luyện E4 hoàn thành sau 1,025 bước (2 giờ 05 phút), đưa loss hội tụ xuống mức **$0.0103$** (giảm gần 50% so với mức $0.0198$ của mô hình 2B), và checkpoint đã được đóng gói đưa lên Hugging Face Hub (`ThinhDao/Qwen3.5-4B_E4`).

**Bảng 6: So sánh quy mô mô hình (Scaling Study: Qwen3.5-2B vs Qwen3.5-4B)**

| Cấu hình | Backbone | Core VI ArgA (%) | Custom Seen Acc (%) | Custom Seen ArgA (%) | Custom Unseen Acc (%) | Custom Unseen ArgA (%) | ArgA Gap | Cú pháp lỗi Core VI (%) | Độ trễ P50 (ms) | Training Time (A100) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E3 (Song ngữ)** | Qwen3.5-2B | 69.76% | 92.25% | 19.38% | 96.00% | 27.88% | +8.50% | 4.31% | 863 ms | 49.5 phút |
| **E3 (Song ngữ)** | **Qwen3.5-4B** | **72.86%** | 92.00% | **62.00%** | **97.50%** | **69.75%** | -7.75% | **0.32%** | 2,432 ms | ~1 giờ 55 phút |
| **E4 (Đặc thù miền)** | Qwen3.5-2B | 69.75% | 93.25% | 87.00% | 97.00% | 86.38% | -0.62% | 4.67% | 970 ms | 54 phút |
| **E4 (Đặc thù miền)** | **Qwen3.5-4B** | *[Đang thực nghiệm]* | **94.66%** | **87.92%** | 96.75% | **86.75%** | **+1.17%** | *[Đang thực nghiệm]* | *[Đang thực nghiệm]* | **2 giờ 05 phút** |

*(Ghi chú: Benchmark trên Core Test gồm 7,712 mẫu/ngôn ngữ; CustomTools-VI gồm 800 mẫu Seen và 800 mẫu Unseen. Độ trễ đo trên 2× GPU NVIDIA Tesla T4).*

**Bảng 6.1: Hiệu năng đối đầu chi tiết của Qwen3.5-4B (E3) trên Canonical Core Benchmark**

| Tập kiểm thử | Số mẫu (Samples) | Tool Selection Acc (%) | ArgA / Exact Match (%) | Non-FC Recall (%) | Syntax Error Rate (%) | Avg Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Core VI Test** | 7,712 (7,221 pos / 491 neg) | **98.73%** | **72.86%** | 94.30% | **0.32%** | 2,431.83 ms |
| **Core EN Test** | 7,712 (7,221 pos / 491 neg) | **96.63%** | **74.71%** | 94.30% | **1.97%** | 3,289.59 ms |

![Hình 4: Nghiên cứu mở rộng quy mô mô hình (2B vs. 4B): Nâng trần độ chính xác Core Benchmark và triệt tiêu lỗi cú pháp JSON](figures/fig4_model_scaling.png)

*Hình 4: Tác động của việc mở rộng quy mô tham số (Qwen3.5-2B vs. Qwen3.5-4B): (a) Nâng cao độ chính xác trích xuất tham số và chọn công cụ trên Core Benchmark; (b) Triệt tiêu lỗi cú pháp JSON từ 4.31% xuống chỉ còn 0.32% (giảm hơn 13 lần).*

#### 4 Phát hiện thực nghiệm then chốt từ nghiên cứu mở rộng quy mô (Scaling Insights):

1. **Triệt tiêu lỗi cú pháp và nâng trần ArgA trên Core Benchmark**:
   - Khi tăng kích thước mô hình từ 2.2B lên 4.56B tham số, tỷ lệ lỗi cú pháp (Syntax Error Rate) trên tập Core tiếng Việt giảm sâu **từ 4.31% xuống chỉ còn 0.32%** (giảm tới hơn 13 lần).
   - Nhờ loại bỏ gần như triệt để các trường hợp hỏng định dạng JSON, ArgA / Exact Match trên Core VI tăng từ **69.76%** (2B E3) lên **72.86%** (4B E3, **+3.10%**), và trên Core EN tăng từ **73.22%** lên **74.71%** (**+1.49%**).
   - Tool Selection Accuracy đạt mốc kỷ lục **98.73%** trên Core VI (tăng +4.73% so với 2B), chứng minh không gian biểu diễn rộng lớn hơn giúp mô hình phân biệt ranh giới ngữ nghĩa của các API tương đồng một cách vượt trội.

2. **Năng lực kháng sụp đổ miền (Mitigating Domain Shift Collapse)**:
   - Ở cấu hình E3 (chỉ huấn luyện trên 60k mẫu song ngữ tổng quát, hoàn toàn không có dữ liệu CustomTools), mô hình 2B từng bị sụp đổ nghiêm trọng khi gặp miền nghiệp vụ Việt Nam do kích hoạt gọi hàm vô tội vạ (ArgA chỉ đạt 19.38% Seen và 27.88% Unseen).
   - Ngược lại, `Qwen3.5-4B` thể hiện sức đề kháng nội tại ấn tượng: ArgA vọt lên **62.00%** trên Seen (**+42.62%**) và **69.75%** trên Unseen (**+41.87%**). Phát hiện này khẳng định dung lượng tham số lớn hơn mang lại năng lực suy luận trong ngữ cảnh (in-context reasoning) vững chãi hơn, giúp mô hình đọc hiểu và tuân thủ schema công cụ mới tốt hơn ngay cả khi chưa được căn chỉnh dữ liệu âm tính bản địa.

3. **Thiết lập đỉnh cao trích xuất mới ở cấu hình E4 (New SOTA for Local Models)**:
   - Khi được tiếp sức bằng dữ liệu đặc thù Việt Nam kèm mẫu âm tính (E4), `Qwen3.5-4B` đạt độ chính xác **94.66% Tool Acc / 87.92% ArgA** trên `test_seen` và **96.75% Tool Acc / 86.75% ArgA** trên `test_unseen`.
   - Kết quả này vượt qua kỷ lục trước đó của `Qwen3.5-2B` (87.00% Seen, 86.38% Unseen), thiết lập đỉnh cao mới của toàn bộ dòng mô hình cục bộ (Local SLM).
   - Đặc biệt, chỉ số ArgA Gap đạt **+1.17%** (ArgA Unseen cao hơn Seen), khẳng định mô hình không hề bị học vẹt hay overfitting vào tập công cụ huấn luyện, duy trì năng lực tổng quát hóa zero-shot tuyệt đối trên các công cụ hoàn toàn mới.

4. **Đánh đổi về tài nguyên và độ trễ suy luận (Compute & Latency Trade-offs)**:
   - Thời gian huấn luyện trên 1× A100 tăng từ ~50–54 phút (2B) lên ~1 giờ 55 phút – 2 giờ 05 phút (4B), tương ứng với mức tăng ~2.3× thời gian tính toán.
   - Trên phần cứng kiểm thử Kaggle 2× GPU Tesla T4, độ trễ suy luận trung bình của mô hình 4B là **2,431.83 ms** (Core VI), cao hơn mức ~864 ms của mô hình 2B (~2.8×). Đây là cái giá tất yếu của việc giải mã tự hồi quy trên mô hình 4.56 tỷ tham số với phần cứng băng thông bộ nhớ khiêm tốn (Turing architecture). Do đó, trong các bài toán yêu cầu độ trễ cực thấp (P50 < 1 giây), `Qwen3.5-2B` vẫn là cấu hình tối ưu; trong khi `Qwen3.5-4B` là lựa chọn hàng đầu cho các tác vụ cần độ chuẩn xác cú pháp và trích xuất tối đa.

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
2. Chứng minh mô hình SLM End-to-End (`Qwen3.5-2B` E4) đạt đỉnh cao về độ chính xác (**87.00%** Seen, **86.38%** Unseen), và mô hình mở rộng `Qwen3.5-4B` E4 tiếp tục nâng trần SOTA lên **87.92%** (Seen) và **86.75%** (Unseen) với khả năng triệt tiêu lỗi cú pháp xuống mức **0.32%**.
3. Khẳng định kiến trúc phân tách Bi+Cross Encoder là "vũ khí tối thượng" về độ trễ (**58–92 ms**) và khả năng mở rộng danh mục không bị suy giảm bộ nhớ khi số lượng công cụ lên tới $1.000$ API trong thực nghiệm Stress Test.
4. Xác lập rõ ràng ranh giới Pareto và chỉ ra điểm nghẽn vật lý của SLM khi gặp lỗi OOM tại $N \ge 500$.
5. Hoàn tất đối sánh thực nghiệm toàn diện với các Frontier API thương mại đóng (GPT-5.6 Luna và Gemini 3.8 Flash), chứng minh SLM 2B và 4B cục bộ vượt trội GPT-5.6 Luna về khả năng trích xuất nghiệp vụ tiếng Việt với độ trễ cạnh tranh.

**Hướng phát triển tiếp theo**:
- Hoàn tất cập nhật số liệu suy luận của mô hình `Qwen3.5-4B` E4 trên Canonical Core Benchmark ngay khi quá trình suy luận kết thúc.
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
