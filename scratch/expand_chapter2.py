import sys

NEW_CH2 = r"""## Chương 2. CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN

### 2.1. Cơ chế gọi công cụ (Tool Calling / Function Calling) trong Mô hình Ngôn ngữ

Khái niệm trao quyền tương tác công cụ cho mô hình ngôn ngữ lớn (Tool-Augmented Language Models) đánh dấu bước chuyển dịch quan trọng từ các hệ thống đối thoại thông thường sang các tác nhân thông minh (AI Agents) có khả năng tác động đến thế giới thực. Ý tưởng nền tảng này được khởi xướng bởi công trình Toolformer (Schick et al., 2023), trong đó mô hình học cách tự chèn các thẻ gọi giao diện lập trình ứng dụng (API calls) thông qua cơ chế tự giám sát dựa trên hàm mất mát giảm thiểu sai số dự đoán văn bản kế tiếp. Tiếp nối hướng đi này, các nghiên cứu như Gorilla (Patil et al., 2023) và bộ chuẩn Berkeley Function-Calling Leaderboard (BFCL) (Yan et al., 2024) đã chính thức chuẩn hóa bài toán gọi công cụ dưới dạng sinh cấu trúc tuân thủ nghiêm ngặt đặc tả JSON Schema.

Về mặt toán học, bài toán gọi công cụ trong mô hình ngôn ngữ tự hồi quy có thể được hình thức hóa như sau. Giả sử hệ thống tiếp nhận một câu truy vấn của người dùng biểu diễn dưới dạng chuỗi các token $X = (x_1, x_2, \dots, x_{|X|})$ cùng với một danh mục gồm $N$ công cụ khả dụng $\mathcal{T} = \{t_1, t_2, \dots, t_N\}$. Mỗi công cụ $t_i = (n_i, d_i, \mathcal{S}_i)$ được định nghĩa bởi tên định danh duy nhất $n_i$, đoạn văn bản mô tả chức năng $d_i$ viết bằng ngôn ngữ tự nhiên, và lược đồ cấu trúc tham số $\mathcal{S}_i$ tuân thủ chuẩn JSON Schema. Mục tiêu của hệ thống là ánh xạ cặp dữ liệu $(X, \mathcal{T})$ thành một chuỗi các hành động $\mathcal{Y} = [(t^*_1, \mathcal{A}_1), (t^*_2, \mathcal{A}_2), \dots, (t^*_K, \mathcal{A}_K)]$, trong đó $K \ge 0$ là số lượng lệnh gọi cần thực thi ($K=0$ biểu thị truy vấn thông thường không cần gọi công cụ, $K \ge 1$ biểu thị kịch bản đơn lệnh hoặc đa lệnh gọi), $t^*_k \in \mathcal{T}$ là công cụ được kích hoạt, và $\mathcal{A}_k = \{(p_{k,j}, v_{k,j})\}_{j=1}^{M_k}$ là tập hợp các cặp khóa - giá trị tham số tương ứng với lược đồ $\mathcal{S}_{t^*_k}$.

Trong các hệ thống thương mại dựa trên LLM tạo sinh (chẳng hạn như OpenAI Function Calling hay Google Gemini), toàn bộ tập danh mục công cụ $\mathcal{T}$ được nhúng trực tiếp vào ngữ cảnh đầu vào (context prompt) của mô hình. Mô hình tối ưu hóa hàm phân phối xác suất có điều kiện của chuỗi token sinh ra:
$$P(\mathcal{Y} \mid X, \mathcal{T}) = \prod_{t=1}^{|\mathcal{Y}|} P(y_t \mid y_{<t}, X, \mathcal{T})$$
Phương thức tiếp cận tạo sinh nguyên khối này bộc lộ những hạn chế vật lý rõ rệt khi triển khai trong thực tế. Thứ nhất, việc đưa toàn bộ đặc tả schema của $N$ công cụ vào ngữ cảnh làm bùng nổ độ dài đầu vào (prompt bloat). Khi $N$ tăng từ vài đơn vị lên hàng trăm hoặc hàng nghìn công cụ, số lượng token đầu vào tăng tuyến tính, dẫn đến chi phí tính toán của cơ chế tự chú ý (Self-Attention) tăng theo cấp số bậc hai $\mathcal{O}(L^2)$ và làm cạn kiệt cửa sổ ngữ cảnh hữu dụng. Thứ hai, sự xuất hiện của quá nhiều định nghĩa công cụ gây nhiễu loạn khả năng tập trung của mô hình (hiện tượng kim đáy bể - Needle-in-a-Haystack), khiến xác suất trích xuất sai tên hàm hoặc sinh giá trị ảo giác (hallucinated arguments) gia tăng đột biến.

### 2.2. Các công trình và bộ chuẩn đánh giá Tool Calling tiêu biểu trên thế giới

Sự bùng nổ của các tác nhân AI đã thúc đẩy cộng đồng nghiên cứu xây dựng nhiều tập dữ liệu quy mô lớn nhằm huấn luyện và đánh giá năng lực tương tác công cụ. Điển hình trong số đó là tập ngữ liệu Glaive Function Calling v2, cung cấp hàng chục nghìn lượt đối thoại tổng hợp phản ánh đa dạng nhu cầu của người dùng trong đời sống hàng ngày, kết hợp hài hòa giữa các truy vấn kích hoạt API và các phản hồi trò chuyện thông thường. Tập dữ liệu này đóng vai trò bản lề trong việc định hình cấu trúc dữ liệu đối thoại có công cụ cho các mô hình mã nguồn mở.

Tiếp nối sự phát triển đó, Salesforce đã giới thiệu họ mô hình xLAM (Large Action Models) cùng bộ dữ liệu huấn luyện xLAM-60k chất lượng cao. Khác với các bộ dữ liệu trước đây chủ yếu tập trung vào đối thoại đơn lệnh, xLAM được chuẩn hóa nghiêm ngặt về mặt cấu trúc, hỗ trợ đầy đủ các kịch bản gọi đa hàm đồng thời (parallel function calls) với các ràng buộc kiểu dữ liệu phức tạp bao gồm chuỗi ký tự, giá trị số nguyên, số thực, mảng danh sách và kiểu liệt kê (enum). Đối với khía cạnh đánh giá độc lập, bộ chuẩn Berkeley Function-Calling Leaderboard (BFCL) (Yan et al., 2024) và ToolBench (Qin et al., 2024) hiện là các thước đo tiêu chuẩn vàng, cung cấp các kịch bản thử nghiệm đa dạng từ đơn giản đến phức tạp trong môi trường đa tác vụ.

Tuy nhiên, phần lớn các công trình nêu trên đều tập trung tuyệt đối vào tiếng Anh. Đối với các ngôn ngữ có nguồn tài nguyên xử lý hạn chế, nghiên cứu tiên phong của Ersoy et al. (2025) tại hội nghị ArabicNLP 2025 đã chứng minh tính khả thi của việc biên dịch có kiểm soát các tập ngữ liệu Glaive và xLAM sang tiếng Ả Rập, kết hợp tinh chỉnh các mô hình ngôn ngữ nhỏ (SLM) nhằm đạt hiệu năng cạnh tranh trực tiếp với GPT-4o-mini. Kế thừa có chọn lọc tư tưởng của Ersoy et al., đề tài khóa luận của chúng tôi mở rộng bài toán này sang tiếng Việt, nhưng tiến xa hơn về mặt đóng góp học thuật: chúng tôi không chỉ khảo sát phương pháp tinh chỉnh nội bộ SLM tạo sinh (Method 1) mà còn đề xuất một kiến trúc phân tách độc lập mang tính đột phá (Bi-Encoder kết hợp Cross-Encoder - Method 2), đồng thời thiết lập môi trường kiểm thử áp lực cực hạn khi quy mô công cụ mở rộng từ 3 lên 1.000 công cụ để phân định giới hạn vật lý của cả hai trường phái.

### 2.3. Mô hình Ngôn ngữ Nhỏ (SLMs) và Kỹ thuật Tinh chỉnh Tham số Hiệu quả (PEFT/QLoRA)

Trong bối cảnh bài toán triển khai thực tế đòi hỏi tối ưu hóa tài nguyên phần cứng và đảm bảo tính riêng tư của dữ liệu doanh nghiệp, các Mô hình Ngôn ngữ Nhỏ (Small Language Models - SLMs) với số lượng tham số dao động từ 1 tỷ đến 4 tỷ đang trở thành xu hướng tiếp cận chủ đạo. Dòng mô hình Qwen3.5 (cụ thể là phiên bản 2B và 4B tham số) do Alibaba Cloud phát triển thể hiện năng lực vượt trội nhờ được tiền huấn luyện trên khối lượng dữ liệu khổng lồ vượt quá 18 nghìn tỷ token đa ngôn ngữ.

Kiến trúc nội tại của Qwen3.5 tích hợp ba cải tiến cốt lõi trong mô hình Transformer hiện đại. Đầu tiên là cơ chế chú ý truy vấn nhóm Grouped-Query Attention (GQA), trong đó các đầu khóa (Key) và giá trị (Value) được chia sẻ giữa nhiều đầu truy vấn (Query), giúp cắt giảm đáng kể dung lượng bộ nhớ đệm KV Cache trong quá trình sinh tự hồi quy mà không làm suy giảm năng lực biểu diễn ngữ cảnh. Thứ hai là cơ chế mã hóa vị trí quay Rotary Position Embedding (RoPE), áp dụng phép quay vector trực tiếp lên không gian biểu diễn ẩn nhằm bảo toàn trọn vẹn thông tin khoảng cách tương đối giữa các vị trí token:
$$\mathbf{R}_{\Theta, m}^d = \text{diag}\left(\mathbf{R}_{\theta_1, m}, \mathbf{R}_{\theta_2, m}, \dots, \mathbf{R}_{\theta_{d/2}, m}\right)$$
Thứ ba là hàm kích hoạt phi tuyến tính SwiGLU (Swish Gated Linear Unit) trong các tầng mạng truyền thẳng (Feed-Forward Networks), mang lại độ mượt mà cao hơn cho bề mặt hàm mất mát và đẩy nhanh tốc độ hội tụ:
$$\text{SwiGLU}(x) = \left(x W_{gate} \cdot \sigma(x W_{gate})\right) \otimes (x W_{up})$$

Để tinh chỉnh các mô hình SLM này trên hạ tầng máy chủ phổ thông có tài nguyên bộ nhớ đồ họa hạn chế, kỹ thuật QLoRA (Quantized Low-Rank Adaptation) (Dettmers et al., 2024) được áp dụng. QLoRA kết hợp ba đột phá công nghệ: lượng tử hóa 4-bit NormalFloat (NF4) tối ưu hóa theo phân phối chuẩn của trọng số nơ-ron, lượng tử hóa kép (Double Quantization) để nén các hằng số lượng tử hóa giúp tiết kiệm trung bình 0,37 bit trên mỗi tham số, và cơ chế quản lý bộ nhớ đệm trang (Paged Optimizers) giúp ngăn chặn triệt để lỗi tràn bộ nhớ VRAM khi xử lý các chuỗi ngữ cảnh dài. Trong quá trình huấn luyện, toàn bộ ma trận trọng số gốc $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ được cố định ở định dạng lượng tử hóa 4-bit NF4, và gradient chỉ lan truyền qua hai ma trận thích ứng hạng thấp khả vi $A \in \mathbb{R}^{r \times d_{in}}$ và $B \in \mathbb{R}^{d_{out} \times r}$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
với hạng ma trận $r \ll \min(d_{in}, d_{out})$ và hệ số co giãn siêu tham số $\alpha$. Cơ chế này cho phép tinh chỉnh toàn diện khả năng gọi công cụ của mô hình với chưa đầy 1% tham số khả huấn, giảm yêu cầu phần cứng từ các cụm GPU công nghiệp xuống mức vận hành ổn định trên một GPU đơn lẻ.

### 2.4. Kiến trúc Phân tách: Truy hồi Ngữ nghĩa Dày (Bi-Encoder) và Trích xuất Tham số (Cross-Encoder)

Nhằm giải quyết triệt để vấn đề quá tải ngữ cảnh và độ trễ sinh từ tự hồi quy của mô hình tạo sinh, hướng tiếp cận phân tách (Method 2) chia tách bài toán Tool Calling thành hai khối thành phần chuyên biệt và xử lý tuần tự theo mô hình suy luận phân biệt (Discriminative Pipeline):

```
                                      [ Danh mục Công cụ N APIs ]
                                                   │
                                                   ▼
[ Câu truy vấn Người dùng ] ──────► ┌─────────────────────────────┐
                                    │     Bi-Encoder Retrieval    │ ──► [ Top-K Công cụ Phù hợp ]
                                    │           (BGE-M3)          │            │
                                    └─────────────────────────────┘            │
                                                   ┌───────────────────────────┘
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │   Cross-Encoder Extraction  │ ──► [ Cấu trúc Lệnh Gọi Hoàn chỉnh ]
                                    │        (XLM-RoBERTa)        │     {"name": ..., "arguments": {...}}
                                    └─────────────────────────────┘
```

#### 2.4.1. Giai đoạn 1: Truy hồi công cụ ngữ nghĩa dày (Bi-Encoder Retrieval)

Giai đoạn đầu tiên có nhiệm vụ định vị chính xác tập hợp các công cụ $t^* \in \mathcal{T}$ có khả năng đáp ứng truy vấn của người dùng từ một kho công cụ quy mô lớn. Chúng tôi sử dụng mạng nơ-ron hai nhánh tương đồng (Siamese Network) dựa trên nền tảng mô hình BGE-M3 (Chen et al., 2024). Mạng mã hóa chuỗi truy vấn người dùng $q$ và văn bản mô tả của từng công cụ $t$ thành các vector nhúng ngữ nghĩa dày đặc $u, v \in \mathbb{R}^d$ thông qua phép gom trung bình có trọng số (Mean Pooling):
$$u = \text{BiEncoder}(q), \quad v = \text{BiEncoder}(t)$$
Mức độ phù hợp giữa truy vấn và công cụ được xác định qua hàm khoảng cách Cosine Similarity:
$$s(q, t) = \frac{u^\top v}{\|u\|_2 \|v\|_2}$$

Để tối ưu hóa không gian biểu diễn ngữ nghĩa của các công cụ tiếng Việt, mô hình được huấn luyện bằng hàm mất mát Xếp hạng Đa mẫu Âm tính (Multiple Negatives Ranking Loss - MNRL) kết hợp kỹ thuật đệm gradient CachedMNRL (Gao et al., 2021). Hàm mất mát cho một lô huấn luyện gồm $B$ cặp mẫu dương $(q_i, t_i^+)$ được định nghĩa như sau:
$$\mathcal{L}_{\text{MNRL}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\left(s(q_i, t_i^+) / \tau\right)}{\exp\left(s(q_i, t_i^+) / \tau\right) + \sum_{j \neq i} \exp\left(s(q_i, t_j^+) / \tau\right) + \sum_{k=1}^{M} \exp\left(s(q_i, n_{i,k}^-) / \tau\right)}$$
trong đó $\tau$ là siêu tham số nhiệt độ (temperature scaling), $t_j^+$ là các mẫu âm ngẫu nhiên nội lô (in-batch negatives), và $n_{i,k}^-$ là các mẫu âm tính khó (hard negatives) được khai phá độc lập thông qua mô hình giáo viên ở vòng huấn luyện đầu tiên. Chiến lược này buộc mô hình phải phân biệt ranh giới ngữ nghĩa cực kỳ tinh tế giữa các công cụ có chức năng tương đồng trong cùng một lĩnh vực nghiệp vụ.

#### 2.4.2. Giai đoạn 2: Trích xuất tham số theo lược đồ (Cross-Encoder Extraction)

Sau khi Top-$K$ công cụ ứng viên được truy xuất, giai đoạn thứ hai tiến hành trích xuất giá trị cụ thể cho từng thuộc tính tham số được định nghĩa trong lược đồ $\mathcal{S}$. Bài toán được mô hình hóa theo cấu trúc Hỏi - Đáp trích xuất (Extractive Question Answering), sử dụng mô hình nền tảng đa ngôn ngữ XLM-RoBERTa (Conneau et al., 2020). Đối với mỗi tham số $p_j$ của công cụ được chọn, chuỗi đầu vào được cấu tạo bằng cách ghép nối câu truy vấn của người dùng đóng vai trò ngữ cảnh (Context) và mô tả thuộc tính của tham số đóng vai trò câu hỏi (Question):
$$\mathbf{X}_{\text{input}} = \text{[CLS]} \circ \text{Query} \circ \text{[SEP]} \circ \text{Parameter Prompt}(p_j, \mathcal{S}) \circ \text{[SEP]}$$

Biểu diễn ngữ cảnh tương tác sâu giữa truy vấn và câu hỏi tại đầu ra của mô hình được định tuyến qua kiến trúc các đầu phân loại phân cấp (Hierarchical Multi-task Heads):
1. **Đầu phân loại nhị phân hiện diện (`has_value_head`)**: Áp dụng hàm kích hoạt Sigmoid lên biểu diễn ẩn của token `[CLS]` để xác định xem tham số $p_j$ có xuất hiện và cần được gán giá trị trong truy vấn hay không:
   $$\hat{y}_{\text{exist}} = \sigma\left(\mathbf{w}_{\text{exist}}^\top \mathbf{h}_{\text{[CLS]}} + b_{\text{exist}}\right)$$
   Mất mát được tính bằng hàm entropy chéo nhị phân $\mathcal{L}_{\text{exist}} = \text{BCE}(\hat{y}_{\text{exist}}, y_{\text{exist}})$.
2. **Đầu trích xuất đoạn văn bản (`span_head`)**: Dự đoán phân phối xác suất của vị trí bắt đầu $i$ và kết thúc $j$ trên chuỗi token truy vấn đối với các tham số dạng chuỗi tự do, số thực hoặc số nguyên:
   $$P_{\text{start}}(i) = \frac{\exp(\mathbf{w}_s^\top \mathbf{h}_i)}{\sum_k \exp(\mathbf{w}_s^\top \mathbf{h}_k)}, \quad P_{\text{end}}(j) = \frac{\exp(\mathbf{w}_e^\top \mathbf{h}_j)}{\sum_k \exp(\mathbf{w}_e^\top \mathbf{h}_k)}$$
   Hàm mất mát tương ứng là tổng entropy chéo phân loại: $\mathcal{L}_{\text{span}} = \text{CE}(P_{\text{start}}, y_s) + \text{CE}(P_{\text{end}}, y_e)$.
3. **Đầu phân loại giá trị rời rạc (`enum_head` và `bool_head`)**: Dự đoán trực tiếp các nhãn thuộc tập giá trị hữu hạn đối với các tham số dạng kiểu liệt kê hoặc giá trị đúng/sai, loại bỏ hoàn toàn nguy cơ sinh lỗi chính tả hay không khớp danh mục.

Tổng hàm mất mát liên hợp của mô hình Cross-Encoder là sự kết hợp có trọng số của các mục tiêu:
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{exist}} + \mathbb{I}(y_{\text{exist}}=1) \cdot \left[ \lambda_2 \mathcal{L}_{\text{span}} + \lambda_3 \mathcal{L}_{\text{enum}} + \lambda_4 \mathcal{L}_{\text{bool}} \right]$$
Kiến trúc này cho phép suy luận song song độc lập cho từng tham số, đảm bảo định dạng đầu ra luôn tuân thủ 100% ràng buộc cú pháp của lược đồ JSON Schema.

### 2.5. Những thách thức đặc thù của ngôn ngữ tiếng Việt trong bài toán Tool Calling

Xử lý ngôn ngữ tự nhiên tiếng Việt đặt ra những rào cản mang tính đặc thù cao so với các ngôn ngữ phương Tây thuộc ngữ hệ Ấn-Âu, đòi hỏi các giải pháp mô hình hóa phải được tùy biến chuyên sâu:

Khía cạnh thứ nhất xuất phát từ đặc trưng loại hình học: tiếng Việt là một ngôn ngữ đơn lập không biến hình từ (Isolating Language), với đơn vị cơ sở là các âm tiết có thanh điệu. Ranh giới từ vựng không được phân định bằng ký tự khoảng trắng như tiếng Anh mà hình thành thông qua các cụm từ ghép đa âm tiết (ví dụ: "máy tính xách tay", "hợp đồng bảo hiểm nhân thọ"). Sự nhập nhằng trong phân định ranh giới từ khiến các mô hình trích xuất đoạn văn bản (Span Prediction) dễ gặp lỗi cắt cụt hoặc thừa từ ở các biên của thực thể, ảnh hưởng tiêu cực đến tính toàn vẹn ngữ nghĩa của giá trị tham số.

Khía cạnh thứ hai liên quan đến hệ thống biểu thức quy ước đời thường vô cùng đa dạng trong văn hóa giao tiếp của người Việt, đặc biệt là các cách biểu đạt thời gian và giá trị số lượng tiền tệ. Trong ngôn ngữ sinh hoạt hàng ngày, người dùng thường sử dụng các từ lóng hoặc cách nói tắt phi quy chuẩn như "hai củ rưỡi" (2.500.000 đồng), "ba vé" (300.000 đồng), "năm xị" (500.000 đồng), hay các mốc thời gian phụ thuộc ngữ cảnh tương đối như "thứ Năm tuần tới", "ngày rằm tháng Bảy", "đầu giờ chiều mai". Một hệ thống trích xuất thuần túy nếu chỉ trích xuất bề mặt chuỗi văn bản (surface text) sẽ khiến API downstream thất bại do không thể chuyển đổi thành các kiểu dữ liệu nguyên thủy (`integer`, `number`, `ISO-8601 string`). Điều này đòi hỏi sự hiện diện bắt buộc của một module Chuẩn hóa Giá trị (Value Normalizer) đứng sau bộ trích xuất.

Khía cạnh cuối cùng là sự nhạy cảm tuyệt đối với hệ thống dấu thanh điệu và các biến thể phương ngữ vùng miền. Tiếng Việt sở hữu sáu thanh điệu; việc thiếu dấu thanh hoặc sai lệch dấu (do lỗi gõ bàn phím Telex/VNI) có thể làm biến đổi hoàn toàn ý nghĩa của câu truy vấn hoặc tên thực thể địa danh (chẳng hạn: "Hà Nam" so với "Hà Nội", "Tam Kỳ" so với "Tam Điệp"). Hơn nữa, việc sử dụng các từ viết tắt phổ biến ("TP.HCM", "HN", "ĐN", "SG") và các từ ngữ địa phương ("xe gắn máy", "xe honda", "xe máy") đòi hỏi mô hình truy hồi phải có năng lực hiểu ngữ nghĩa đa tầng để ánh xạ chính xác về cùng một định danh công cụ duy nhất."""

with open("/home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md", "r", encoding="utf-8") as f:
    content = f.read()

start_marker = "## Chương 2. CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN"
end_marker = "## Chương 3. XÂY DỰNG BỘ TIÊU CHUẨN ĐÁNH GIÁ (BENCHMARK) CHO TIẾNG VIỆT"

idx_start = content.find(start_marker)
idx_end = content.find(end_marker)

if idx_start == -1 or idx_end == -1:
    print(f"Error: Markers not found! idx_start={idx_start}, idx_end={idx_end}")
    sys.exit(1)

# Preserve the delimiter before Chapter 3
delimiter = "---\n\n<div style=\"page-break-after: always;\"></div>\n\n"
new_content = content[:idx_start] + NEW_CH2 + "\n\n" + delimiter + content[idx_end:]

with open("/home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md", "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully replaced Chapter 2!")
