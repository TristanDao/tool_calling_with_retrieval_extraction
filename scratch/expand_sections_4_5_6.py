import sys

NEW_SEC_44 = r"""### 4.4. Cơ chế chuẩn hóa giá trị thực thể tiếng Việt (Vietnamese Value Normalizer)

Trong kiến trúc phân tách Method 2, đầu trích xuất đoạn văn bản (`span_head`) của Cross-Encoder chỉ có nhiệm vụ định vị và trích xuất chuỗi ký tự bề mặt (surface text string) xuất hiện trực tiếp trong câu truy vấn của người dùng. Tuy nhiên, các giao diện lập trình ứng dụng (API) đòi hỏi các giá trị tham số đầu vào phải tuân thủ nghiêm ngặt các kiểu dữ liệu nguyên thủy được định nghĩa trong JSON Schema (`integer`, `number`, `boolean`, hoặc định dạng chuỗi chuẩn hóa như `date-time`). Nếu đưa trực tiếp chuỗi ký tự thô vào lời gọi hàm (chẳng hạn gán chuỗi `"hai triệu rưỡi"` cho một tham số yêu cầu kiểu số thực `float`), hệ thống phía máy chủ sẽ từ chối thực thi do lỗi định dạng kiểu dữ liệu.

Nhằm thu hẹp khoảng cách biểu diễn giữa ngôn ngữ tự nhiên đời thường và các đặc tả kỹ thuật nghiêm ngặt của API, chúng tôi thiết kế một module Chuẩn hóa Giá trị tiếng Việt (Vietnamese Value Normalizer) hoạt động theo cơ chế tiền định (deterministic rule-based pipeline) bốn giai đoạn nối tiếp:

Giai đoạn đầu tiên thực hiện phân đoạn từ vựng và ánh xạ từ lóng bản địa (Lexical Tokenization and Slang Mapping). Module tiến hành rà soát các từ vựng chỉ lượng mang tính khẩu ngữ đặc thù trong đời sống xã hội Việt Nam để chuyển đổi về các lượng từ chuẩn tắc. Cụ thể, các định danh tiền tệ khẩu ngữ như "củ", "chai" được chuẩn hóa thành đơn vị triệu đồng ($10^6$); "lít", "xị", "loét" được ánh xạ về đơn vị trăm nghìn đồng ($10^5$); "cành", "k" tương ứng với đơn vị nghìn đồng ($10^3$); và các cách diễn đạt tương đối như "rưỡi" được tính toán theo nửa đơn vị của hàng chữ số liền trước (chẳng hạn "hai củ rưỡi" được quy đổi tương đương với "2.500.000").

Giai đoạn thứ hai là bộ giải mã và phân tích cú pháp số chữ tiếng Việt (Vietnamese Written Number Parser). Module triển khai một máy trạng thái hữu hạn (Deterministic Finite Automaton) có khả năng phân tích đệ quy các chuỗi số chữ phức hợp gồm nhiều lớp đơn vị (tỷ, triệu, nghìn, trăm, mươi, lăm, mốt). Thuật toán tích lũy giá trị dựa trên thứ bậc hàng số, xử lý trơn tru các biến thể phát âm ngữ âm địa phương (chẳng hạn "mười lăm" so với "hai mươi nhăm", "bốn mươi" so với "bốn chục") và ánh xạ chuỗi văn bản thành giá trị số học nguyên thủy trong hệ thập phân.

Giai đoạn thứ ba phụ trách chuẩn hóa các thực thể định dạng có cấu trúc thông qua biểu thức chính quy (Regex Pattern Canonicalization). Đối với các tham số thời gian, module phân tích các cụm từ chỉ ngày tháng theo lịch dương và lịch âm phổ biến, tự động tính toán mốc thời gian tuyệt đối dựa trên thời điểm neo của hệ thống (system anchor time) và định dạng lại theo tiêu chuẩn quốc tế ISO-8601 (`YYYY-MM-DD`). Đối với các định danh phương tiện giao thông và mã giao dịch hành chính, hệ thống tự động loại bỏ khoảng trắng ngẫu nhiên, chuẩn hóa dấu gạch ngang phân tách và chuyển đổi toàn bộ ký tự sang dạng chữ hoa chuẩn mực (ví dụ: chuyển đổi `"59 x1 999.88"` thành `"59X1-999.88"`).

Giai đoạn cuối cùng là khâu áp đặt kiểu dữ liệu theo lược đồ (Schema-driven Type Casting). Căn cứ vào định nghĩa kiểu thuộc tính trong `tools[].parameters.properties[key].type`, module tự động ép kiểu giá trị đã chuẩn hóa về kiểu dữ liệu đích tương ứng: chuyển đổi về kiểu số nguyên `int`, số thực `float`, giá trị logic `bool`, hoặc giữ nguyên chuỗi ký tự đã làm sạch `string`. Nhờ đó, 100% các giá trị tham số do pipeline trích xuất đều đảm bảo tính tương thích tuyệt đối với các thư viện xác thực dữ liệu chuẩn (như Pydantic hay JSON Schema Validator) trước khi được chuyển tiếp tới môi trường thực thi."""

NEW_SEC_51 = r"""### 5.1. Thiết lập thực nghiệm và hạ tầng phần cứng

Nhằm đảm bảo tính khách quan, khả năng tái lập hoàn toàn (reproducibility) và phản ánh chính xác hiệu năng của các mô hình trong điều kiện triển khai thực tế, toàn bộ quá trình huấn luyện và kiểm thử được thực hiện trên một hệ thống hạ tầng và quy trình đánh giá chuẩn hóa nghiêm ngặt:

Về hạ tầng huấn luyện, các mô hình SLM tạo sinh thuộc Phương pháp 1 (`Qwen3.5-2B` và `Qwen3.5-4B`) được huấn luyện trên hạ tầng điện toán đám mây Google Colab Pro với một GPU đơn lẻ NVIDIA A100-SXM4-40GB VRAM, hệ thống bộ nhớ RAM 83.5 GB và băng thông kết nối PCIe Gen4 tốc độ cao. Môi trường phần mềm vận hành trên hệ điều hành Ubuntu 22.04 LTS, nền tảng tính toán song song CUDA 12.4, thư viện PyTorch 2.5 và framework tối ưu hóa tham số Unsloth kết hợp kỹ thuật QLoRA 4-bit. Đối với Phương pháp 2, cả hai mô hình Bi-Encoder (`BGE-M3`) và Cross-Encoder (`xlm-roberta-base`) được huấn luyện trên GPU NVIDIA Tesla T4 16GB (Kaggle Cloud Platform) nhằm chứng minh tính khả thi của kiến trúc phân tách trên các phần cứng phổ thông giá rẻ.

Về quy chuẩn kiểm thử và đo lường độ trễ, toàn bộ các bài đánh giá thời gian thực (wall-clock latency) đều được thực thi trên môi trường đồng nhất sử dụng cụm GPU NVIDIA Tesla T4 16GB. Để mô phỏng sát thực tế tương tác thời gian thực của người dùng với các tác tử AI cá nhân, kích thước lô suy luận (batch size) được cố định nghiêm ngặt ở mức $B = 1$. Trước khi ghi nhận thời gian xử lý chính thức, hệ thống luôn thực hiện 10 lượt suy luận khởi động ấm (warm-up runs) để tải toàn bộ trọng số mô hình vào bộ nhớ VRAM và kích hoạt các kernel tính toán CUDA, loại bỏ hoàn toàn các sai số do độ trễ khởi tạo phần cứng ban đầu. Độ trễ suy luận của từng mẫu được đo bằng đồng hồ bấm giờ độ chính xác cao của hệ điều hành, ghi nhận cả giá trị trung vị P50 và phân vị thứ 95 (P95) trên toàn bộ tập dữ liệu kiểm thử. Nhằm bảo đảm tính nhất quán giữa các lần chạy, hạt giống ngẫu nhiên (random seed) được cố định ở giá trị 42 cho toàn bộ các thư viện Python, PyTorch và CUDA."""

NEW_SEC_55 = r"""### 5.5. So sánh đối đầu với các Frontier API thương mại đóng (GPT-5.6 Luna, Gemini 3.8 Flash)

Để định vị chính xác vị thế công nghệ của các mô hình đề xuất so với mặt bằng chung của ngành công nghiệp trí tuệ nhân tạo toàn cầu, chúng tôi tiến hành đánh giá đối đầu trực diện giữa các giải pháp nội bộ và hai dịch vụ thương mại hàng đầu thế giới: `GPT-5.6 Luna` (phiên bản tác nhân tối ưu hóa gọi hàm mới nhất của OpenAI) và `Gemini 3.8 Flash` (mô hình suy luận tốc độ cao thế hệ mới của Google). Cả hai mô hình thương mại đều được cấp cùng một prompt hệ thống, cùng danh mục công cụ JSON Schema và được gọi thông qua các API chính thức với tham số nhiệt độ $\text{temperature} = 0.0$ nhằm đảm bảo tính tiền định cao nhất.

Kết quả thực nghiệm trên bộ dữ liệu bản địa `CustomTools-VI` tại Bảng 5.1 mang lại những phát hiện học thuật đầy bất ngờ và giàu ý nghĩa thực tiễn:

Trước hết, trên khía cạnh độ chính xác trích xuất tuyệt đối (ArgA), các mô hình SLM nội bộ kích thước nhỏ gọn của đề tài đã thiết lập chiến thắng thuyết phục trước đại diện thương mại đóng `GPT-5.6 Luna`. Cụ thể, mô hình `Qwen3.5-4B (E4)` đạt ArgA **87.92%** trên tập seen và **86.75%** trên tập unseen, vượt trội hơn `GPT-5.6 Luna` (đạt 78.25% seen và 79.50% unseen) với khoảng cách lần lượt là **+9.67%** và **+7.25%**. Ngay cả phiên bản nhỏ hơn là `Qwen3.5-2B (E4)` với vỏn vẹn 2.2 tỷ tham số cũng vượt qua `GPT-5.6 Luna` từ +8.75% (seen) đến +6.88% (unseen). 

Phân tích sâu về mặt biểu diễn ngôn ngữ chỉ ra rằng: mặc dù các Frontier API như GPT-5.6 sở hữu năng lực suy luận tổng quát đa lĩnh vực vượt trội nhờ kích thước mô hình khổng lồ (hàng trăm tỷ tham số), chúng lại thiếu vắng sự thích nghi sâu sắc với các đặc trưng ngữ dụng và phương thức diễn đạt đời thường của người Việt. Khi người dùng sử dụng các cách nói tắt, từ lóng định lượng tiền tệ bản địa ("hai củ rưỡi", "năm xị") hoặc các cấu trúc địa danh phức tạp, GPT-5.6 thường có xu hướng hiểu theo nghĩa đen hoặc sinh ra các giá trị phỏng đoán không khớp với ngữ cảnh thực tế của Việt Nam. Trái lại, các mô hình SLM nội bộ nhờ được tinh chỉnh trực tiếp trên nguồn dữ liệu bản địa hóa có kiểm soát đã thẩm thấu trọn vẹn các quy ước ngôn ngữ này, từ đó đạt độ chuẩn xác tham số vượt trội.

Mặc dù `Gemini 3.8 Flash` vẫn duy trì vị thế dẫn đầu tuyệt đối về độ chính xác tham số (~92%–93%) nhờ khả năng xử lý ngữ cảnh đa ngôn ngữ xuất sắc, việc triển khai các mô hình SLM nội bộ hoặc kiến trúc phân biệt Method 2 lại mang lại những ưu thế chiến lược quyết định đối với các tổ chức và doanh nghiệp trong nước:
- **Bảo mật và toàn vẹn dữ liệu nội bộ (Data Privacy & Compliance)**: Việc triển khai cục bộ (on-premise) đảm bảo 100% dữ liệu truy vấn của người dùng và bí mật kinh doanh không bị gửi qua Internet sang các máy chủ đặt tại nước ngoài, đáp ứng đầy đủ các quy định nghiêm ngặt của Luật An ninh mạng Việt Nam và các tiêu chuẩn bảo vệ dữ liệu cá nhân (Nghị định 13/2023/NĐ-CP).
- **Tối ưu hóa chi phí vận hành (Operational Cost)**: Chi phí gọi API của các dịch vụ thương mại dao động từ 0.30 USD đến 0.60 USD cho mỗi 1.000 lượt truy vấn (với các prompt chứa nhiều định nghĩa công cụ). Ngược lại, chi phí năng lượng điện toán để vận hành mô hình phân tách Method 2 trên máy chủ CPU/GPU nội bộ chỉ vào khoảng 0.00003 USD trên 1.000 lượt (tiết kiệm hơn 10.000 lần), loại bỏ hoàn toàn gánh nặng ngân sách định kỳ cho doanh nghiệp.
- **Tính khả dụng và kiểm soát độ trễ (Deterministic Latency & Availability)**: Các dịch vụ đám mây công cộng luôn tiềm ẩn rủi ro về độ trễ mạng quốc tế (network jitter), giới hạn hạn ngạch gọi hàm (rate limits) và nguy cơ gián đoạn dịch vụ ngoài tầm kiểm soát. Giải pháp nội bộ mang lại độ trễ cục bộ ổn định chỉ từ 55 ms đến 900 ms, bảo đảm tính sẵn sàng phục vụ 24/7 cho các hệ thống công nghiệp trọng yếu."""

NEW_CASE_STUDIES = r"""Để làm sáng tỏ cơ chế sai lệch của từng nhóm lỗi, Bảng 6.3 tổng hợp các trường hợp nghiên cứu điển hình (Case Studies) thu thập từ các lần chạy thực nghiệm, đối chiếu trực tiếp giữa câu truy vấn gốc của người dùng, nhãn chuẩn (Gold Function Call) và kết quả dự đoán thực tế của các mô hình.

**Bảng 6.3: Các ví dụ thực tế (Case Studies) minh họa 4 dạng lỗi trích xuất tham số điển hình**

| Dạng lỗi | Câu truy vấn thực tế | Nhãn chuẩn (Gold Call) | Kết quả dự đoán (Predicted Call) | Phân tích nguyên nhân học thuật |
| :--- | :--- | :--- | :--- | :--- |
| **Nhập nhằng ranh giới thực thể** *(Entity Boundary Ambiguity)* | "Đặt lịch khám tại Bệnh viện Đại học Y Dược cơ sở 2 vào sáng thứ Hai." | `book_clinic(hospital="Bệnh viện Đại học Y Dược cơ sở 2", date="2026-09-21")` | *Cross-Encoder:* `book_clinic(hospital="Đại học Y Dược", date="2026-09-21")` | Thuật toán tách từ phụ phân rã danh từ riêng thành các mảnh token; đầu phân loại span cắt cụt tiền tố hành chính "Bệnh viện" và hậu tố định danh "cơ sở 2". |
| **Biểu đạt phi quy chuẩn / Từ lóng** *(Slang & Implicit Formatting)* | "Chuyển gấp cho mẹ hai củ rưỡi vào số tài khoản 1903338888." | `transfer_money(amount=2500000, account="1903338888")` | *SLM 2B (E1):* `transfer_money(amount=2.5, account="1903338888")` | Mô hình thiếu tri thức bản địa hóa về tiếng lóng tiền tệ ("củ rưỡi"), dẫn đến trích xuất giá trị literal $2.5$ thay vì đơn vị chuẩn $2{,}500{,}000$ đồng. |
| **Ảo giác tham số ngầm định** *(Implicit Default Hallucination)* | "Tìm quán cà phê yên tĩnh có wifi gần hồ Con Rùa." | `search_coffee(location="hồ Con Rùa", amenities=["wifi", "yên tĩnh"])` | *SLM 4B (E3):* `search_coffee(location="hồ Con Rùa", amenities=["wifi"], price_range="bình dân")` | Mô hình tạo sinh tự hồi quy tự ý bổ sung tham số tùy chọn `price_range="bình dân"` dựa trên phân phối xác suất tiên nghiệm của tập dữ liệu huấn luyện dù truy vấn không đề cập. |
| **Sai lệch thứ tự đa lệnh gọi** *(Call Order Misalignment)* | "Kiểm tra thời tiết Đà Nẵng và đặt vé xe đi Nha Trang vào ngày mai." | `[check_weather(city="Đà Nẵng"), book_bus(destination="Nha Trang")]` | *Method 2:* `[book_bus(destination="Nha Trang"), check_weather(city="Đà Nẵng")]` | Bi-Encoder gán điểm tương đồng cosine cho công cụ đặt xe nhỉnh hơn công cụ thời tiết, dẫn tới đảo ngược vị trí hai lệnh gọi dù đã nhận diện đúng 100% công cụ và tham số. |

"""

with open("/home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update DANH MỤC BẢNG BIỂU to add Bảng 6.3
old_bang_62_line = "| **Bảng 6.2** | Đối chứng chẩn đoán Oracle giữa lỗi truy hồi (Retrieval) và lỗi trích xuất tham số (Extraction) | 60 |"
new_bang_lines = old_bang_62_line + "\n| **Bảng 6.3** | Các ví dụ thực tế (Case Studies) minh họa 4 dạng lỗi trích xuất tham số điển hình | 62 |"

if old_bang_62_line in content:
    content = content.replace(old_bang_62_line, new_bang_lines, 1)
    print("Updated DANH MỤC BẢNG BIỂU with Bảng 6.3")
else:
    print("Warning: old_bang_62_line not found in TOC!")

# 2. Replace Section 4.4
sec44_start = "### 4.4. Cơ chế chuẩn hóa giá trị thực thể tiếng Việt (Vietnamese Value Normalizer)"
sec44_end = "## Chương 5. THỰC NGHIỆM, ĐÁNH GIÁ VÀ BÀN LUẬN KẾT QUẢ"

p1 = content.find(sec44_start)
p2 = content.find(sec44_end)
if p1 != -1 and p2 != -1:
    content = content[:p1] + NEW_SEC_44 + "\n\n---\n\n<div style=\"page-break-after: always;\"></div>\n\n" + content[p2:]
    print("Replaced Section 4.4 successfully")
else:
    print(f"Error finding Sec 4.4: p1={p1}, p2={p2}")

# 3. Replace Section 5.1
sec51_start = "### 5.1. Thiết lập thực nghiệm và hạ tầng phần cứng"
sec51_end = "### 5.2. Hệ thống độ đo đánh giá (Evaluation Metrics)"

p1 = content.find(sec51_start)
p2 = content.find(sec51_end)
if p1 != -1 and p2 != -1:
    content = content[:p1] + NEW_SEC_51 + "\n\n" + content[p2:]
    print("Replaced Section 5.1 successfully")
else:
    print(f"Error finding Sec 5.1: p1={p1}, p2={p2}")

# 4. Replace Section 5.5
sec55_start = "### 5.5. So sánh đối đầu với các Frontier API thương mại đóng"
sec55_end = "## Chương 6. PHÂN TÍCH LỖI VÀ THẢO LUẬN GIỚI HẠN"

p1 = content.find(sec55_start)
p2 = content.find(sec55_end)
if p1 != -1 and p2 != -1:
    content = content[:p1] + NEW_SEC_55 + "\n\n---\n\n<div style=\"page-break-after: always;\"></div>\n\n" + content[p2:]
    print("Replaced Section 5.5 successfully")
else:
    print(f"Error finding Sec 5.5: p1={p1}, p2={p2}")

# 5. Insert Bảng 6.3 into Section 6.1 right after the fourth error type description
sec62_marker = "### 6.2. Phân tích nguyên nhân suy giảm zero-shot của Method 2"
p_sec62 = content.find(sec62_marker)
if p_sec62 != -1:
    content = content[:p_sec62] + NEW_CASE_STUDIES + content[p_sec62:]
    print("Inserted Bảng 6.3 into Section 6.1 successfully")
else:
    print(f"Error finding Sec 6.2 marker!")

with open("/home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md", "w", encoding="utf-8") as f:
    f.write(content)

print("All expansions written successfully!")
