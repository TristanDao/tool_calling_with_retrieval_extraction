# Phương pháp đánh giá hệ thống Tool Calling tiếng Việt

## 1. Mục tiêu của phần đánh giá

Phần đánh giá được xây dựng nhằm xác định mức độ hiệu quả của các hệ thống Tool Calling trên dữ liệu tiếng Việt ở cả hai khía cạnh: **chất lượng dự đoán** và **hiệu quả vận hành**. Thay vì chỉ sử dụng một chỉ số tổng hợp duy nhất, nghiên cứu đánh giá hệ thống theo từng thành phần và trên toàn bộ pipeline. Cách tổ chức này cho phép trả lời không chỉ câu hỏi hệ thống nào tốt hơn, mà còn xác định hệ thống sai ở bước nào, mức độ tổng quát hóa ra sao và chi phí để tạo ra một lời gọi công cụ chính xác là bao nhiêu.

Khung đánh giá được thiết kế để trả lời bốn câu hỏi nghiên cứu:

1. Hệ thống có xác định đúng việc cần gọi công cụ, chọn đúng công cụ và trích xuất đúng tham số hay không?
2. Nếu dự đoán cuối cùng sai, nguyên nhân chủ yếu đến từ tầng truy hồi công cụ, lựa chọn công cụ, trích xuất tham số hay định dạng đầu ra?
3. Hiệu năng của hệ thống có ổn định trên các nhóm dữ liệu khác nhau, đặc biệt là công cụ chưa thấy, miền mới, multi-call và schema phức tạp hay không?
4. Chất lượng dự đoán đạt được với độ trễ, tài nguyên và chi phí như thế nào?

Ý tưởng đánh giá chức năng được tham khảo từ Track A của AISA-ArabicFC và metric ArgA của Ersoy et al. (2025). Tinh thần phân tầng và đo khoảng cách hiệu năng được kế thừa từ Track C, nhưng trục phương ngữ được thay bằng các lát dữ liệu phù hợp với bài toán tiếng Việt như seen/unseen tool, domain, độ khó và độ phức tạp schema. Phần stress test về số lượng công cụ tham khảo ý tưởng từ RAG-MCP.

## 2. Đối tượng được đánh giá

Nghiên cứu có **hai phương pháp chính** và **hai baseline thương mại**, tương ứng với bốn hệ thống thực nghiệm. Vì Method 2 gồm hai mô hình thành phần, cách gọi “bốn model” không hoàn toàn chính xác; thuật ngữ phù hợp hơn là “bốn hệ thống được so sánh”.

| Hệ thống | Phương pháp/kỹ thuật | Vai trò trong thực nghiệm | Ý nghĩa |
|---|---|---|---|
| Method 1: SLM End-to-End | Fine-tune Qwen2.5 0.5B/1.5B bằng instruction tuning; sinh trực tiếp tên tool và arguments | Phương pháp local end-to-end và đối chứng với kiến trúc modular | Kiểm tra khả năng một SLM sinh toàn bộ structured call trong một lần suy luận |
| Method 2: Bi-Encoder + Cross-Encoder | BGE-M3 Bi-Encoder truy hồi tool; BGE-M3 với hierarchical heads trích xuất tham số theo schema | Phương pháp modular được nghiên cứu chính | Kiểm tra giả thuyết tách retrieval và extraction có thể duy trì độ chính xác trong khi giảm latency/cost |
| OpenAI Function Calling | API generative function calling, dự kiến dùng gpt-4o-mini | Baseline thương mại thứ nhất | Cung cấp mốc so sánh với hệ thống đóng có khả năng function calling mạnh |
| Google Gemini Function Calling | API generative function calling, dự kiến dùng gemini-1.5-flash | Baseline thương mại thứ hai | Tạo thêm mốc đối chiếu độc lập về chất lượng, latency và cost |

Tất cả hệ thống được chạy trên cùng tập kiểm thử, cùng danh sách candidate tools và cùng schema. Phiên bản model/API, ngày chạy, cấu hình decoding, phần cứng và chính sách retry phải được cố định hoặc ghi lại trong báo cáo thực nghiệm.

## 3. Nguyên tắc thiết kế đánh giá

### 3.1 So sánh công bằng trên cùng dữ liệu

Mỗi hệ thống nhận cùng user query và cùng tập candidate tool schemas. Prediction của các hệ thống được chuyển về một contract chung trước khi chấm điểm. Mỗi prediction phải giữ nguyên `id` của mẫu gold để bảo đảm so sánh theo cặp trên cùng một dữ liệu.

### 3.2 Tách metric chung và metric riêng theo kiến trúc

Các metric end-to-end như Tool Set Accuracy, Argument F1, N-FCEM, Schema Validity và latency được áp dụng cho cả bốn hệ thống. Các metric bên trong pipeline chỉ áp dụng khi hệ thống có thành phần tương ứng:

- Recall@K và MRR chỉ có ý nghĩa trực tiếp với Method 2 vì hệ thống này xuất danh sách công cụ được truy hồi theo thứ hạng.
- Oracle-tool ArgEM chủ yếu dùng cho parameter extractor của Method 2 vì có thể cấp gold tool schema trực tiếp cho module này.
- OpenAI, Gemini và SLM End-to-End không có tầng retrieval độc lập nên các ô Recall@K/MRR được ghi là `N/A`, không được gán bằng 0.

Nguyên tắc này tránh kết luận sai rằng một hệ thống kém hơn chỉ vì kiến trúc của nó không xuất intermediate result cần cho một metric thành phần.

### 3.3 Đánh giá đa tầng

Khung đánh giá gồm ba tầng:

1. **Component-level evaluation**: call detection, tool retrieval, tool selection, parameter extraction và schema validation.
2. **End-to-end evaluation**: kiểm tra toàn bộ structured function call từ query đến tool và arguments.
3. **Robustness and efficiency evaluation**: phân tích theo lát dữ liệu, khoảng cách hiệu năng, latency, throughput, token và cost.

### 3.4 Không sử dụng composite score tùy ý làm kết quả chính

Nghiên cứu không cộng các metric chất lượng và hiệu quả bằng các trọng số tự chọn. Ví dụ, một công thức như `0.4 × ToolAcc + 0.4 × ArgEM + 0.2 × LatencyScore` khó giải thích về mặt khoa học vì trọng số không có cơ sở khách quan. Thay vào đó, từng metric được báo cáo riêng và trade-off được phân tích trực tiếp, chẳng hạn N-FCEM tăng bao nhiêu điểm phần trăm, p95 latency giảm bao nhiêu phần trăm và cost/correct call thay đổi như thế nào.

## 4. Dữ liệu đầu vào và đầu ra chuẩn hóa

### 4.1 Gold benchmark

Mỗi mẫu gold tuân theo unified master structure:

```json
{
  "id": "glaive_00042",
  "source": "glaive",
  "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
  "function_calls": [
    {
      "name": "search_tutors",
      "arguments": {"subject": "Toán", "location": "Hà Nội"}
    }
  ],
  "tools": [
    {
      "name": "search_tutors",
      "description": "Tìm gia sư theo môn học và khu vực.",
      "feature_group": "Tìm kiếm & Kết nối",
      "parameters": {
        "type": "object",
        "properties": {
          "subject": {"type": "string"},
          "location": {"type": "string"}
        },
        "required": ["subject", "location"]
      }
    }
  ]
}
```

`function_calls` là danh sách nên hỗ trợ cả single-call và multi-call. Một negative/no-call sample được biểu diễn bằng `function_calls: []`.

### 4.2 Prediction contract

Prediction của mỗi hệ thống được đưa về cùng định dạng:

```json
{
  "id": "glaive_00042",
  "function_calls": [
    {
      "name": "search_tutors",
      "arguments": {"subject": "Toán", "location": "Hà Nội"}
    }
  ],
  "ranked_tools": [
    {"name": "search_tutors", "score": 0.94}
  ],
  "telemetry": {
    "latency_ms": 12.8,
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0.0,
    "gpu_peak_memory_mb": 1840.0
  }
}
```

`ranked_tools` là tùy chọn và chỉ cần thiết đối với hệ thống có retrieval. `telemetry` được ghi tại thời điểm inference. Prediction no-call dùng danh sách `function_calls` rỗng. Output không parse được vẫn phải giữ `id` và raw output để evaluator ghi nhận lỗi định dạng thay vì loại bỏ mẫu.

### 4.3 Chuẩn hóa đối xứng

Gold và prediction được chuẩn hóa bằng cùng một hàm dựa trên JSON Schema. Các kỹ thuật gồm:

- Unicode normalization, loại khoảng trắng thừa và case folding đối với natural-language string.
- Coercion integer, number và boolean chỉ khi schema khai báo kiểu tương ứng.
- Chuẩn hóa `date` và `date-time` khi schema có `format` tương ứng.
- Áp dụng alias cho currency, country hoặc closed-class value theo cấu hình định trước.
- Giữ strict đối với identifier như ID, UUID, IBAN và account number; không làm mất số 0 ở đầu.
- Chuẩn hóa đệ quy cho array và object.

Normalization phải được thiết kế trước khi xem prediction trên test set để tránh điều chỉnh luật theo lỗi của một hệ thống cụ thể. Mọi alias rule phải được lưu cùng cấu hình thí nghiệm.

### 4.4 So khớp multi-call

Thứ tự các function calls không làm thay đổi ý nghĩa của một multi-call sample nếu benchmark không quy định thứ tự thực thi. Vì vậy evaluator so sánh multiset calls thay vì so sánh theo vị trí. Khi nhiều calls có cùng tên tool, evaluator ghép cặp gold–prediction sao cho mức trùng khớp arguments là cao nhất rồi mới tính metric. Cách làm này tránh phạt sai chỉ vì hai lời gọi tương đương xuất hiện ở thứ tự khác nhau.

## 5. Đánh giá ở mức thành phần

### 5.1 Phát hiện yêu cầu gọi công cụ

Một mẫu được xem là positive nếu gold chứa ít nhất một function call. Nếu prediction chứa ít nhất một call, hệ thống dự đoán `CALL`; ngược lại là `NO_CALL`.

Với `TP`, `FP`, `FN` và `TN` lần lượt là số dự đoán đúng call, gọi thừa, bỏ sót call và đúng no-call:

\[
Precision_{call}=\frac{TP}{TP+FP}
\]

\[
Recall_{call}=\frac{TP}{TP+FN}
\]

\[
F1_{call}=\frac{2\times Precision_{call}\times Recall_{call}}
{Precision_{call}+Recall_{call}}
\]

Hai chỉ số lỗi được bổ sung:

\[
HallucinatedCallRate=\frac{FP}{N_{negative}}
\]

\[
MissedCallRate=\frac{FN}{N_{positive}}
\]

**Vai trò:** đánh giá call gate độc lập với khả năng chọn tool.  
**Ý nghĩa:** Hallucinated Call Rate phản ánh nguy cơ gọi API không cần thiết, làm tăng chi phí hoặc tạo side effect; Missed Call Rate phản ánh số yêu cầu cần công cụ nhưng hệ thống không xử lý.

Nếu test set không chứa no-call samples, Hallucinated Call Rate không thể tính và Call F1 không đủ để kết luận về khả năng chống tool hallucination. Do đó benchmark cần có một negative subset được xây dựng và kiểm soát riêng.

### 5.2 Truy hồi công cụ

Metric này áp dụng cho Bi-Encoder trong Method 2. Gọi \(G_i\) là tập tên gold tools duy nhất của mẫu \(i\), và \(R_i^K\) là K tools đứng đầu danh sách truy hồi.

Micro Recall@K được tính bằng:

\[
Recall@K=\frac{\sum_i |G_i\cap R_i^K|}{\sum_i |G_i|}
\]

Với multi-call, Full Recall@K yêu cầu tất cả gold tools đều có trong top-K:

\[
FullRecall@K=\frac{1}{N}\sum_i \mathbb{1}[G_i\subseteq R_i^K]
\]

MRR đo thứ hạng của gold tool xuất hiện sớm nhất:

\[
MRR=\frac{1}{N}\sum_i \frac{1}{rank_i}
\]

Nếu không có gold tool nào trong danh sách, reciprocal rank của mẫu bằng 0.

**Vai trò:** đo khả năng Bi-Encoder giữ gold schema trong candidate set cho các module sau.  
**Ý nghĩa:** Recall@K là upper bound thực tế của downstream pipeline. Nếu gold tool bị loại ở retrieval, selector hoặc extractor phía sau không thể sửa lỗi.

### 5.3 Selection Conversion@K

Selection Conversion@K được tính trên các mẫu mà toàn bộ gold tools đã xuất hiện trong top-K:

\[
SelectionConversion@K=
P(\widehat{G_i}=G_i\mid G_i\subseteq R_i^K)
\]

**Vai trò:** tách lỗi retrieval khỏi lỗi lựa chọn cuối cùng.  
**Ý nghĩa:** Recall@K cao nhưng Selection Conversion thấp cho thấy gold tool đã được truy hồi nhưng tầng chọn/rerank vẫn nhầm; ngược lại, Selection Conversion cao nhưng Recall@K thấp cho thấy bottleneck nằm ở Bi-Encoder.

### 5.4 Lựa chọn công cụ

Tool Set Accuracy trên positive samples được định nghĩa:

\[
ToolSetAcc=\frac{1}{N_{positive}}
\sum_i \mathbb{1}[\widehat{G_i}=G_i]
\]

Phép so sánh dùng multiset tên tool để hỗ trợ trường hợp một tool được gọi nhiều lần. Ngoài exact accuracy, evaluator tính precision, recall và F1 ở cấp tool. Macro-F1 là trung bình F1 của các tool, trong khi micro-F1 gộp toàn bộ quyết định trước khi tính.

**Vai trò:** đo đúng tầng function selection mà chưa yêu cầu arguments đúng.  
**Ý nghĩa:** Tool Set Accuracy cho biết tỷ lệ mẫu có toàn bộ tập tool đúng; Macro-F1 giúp phát hiện hệ thống chỉ tốt trên tool phổ biến nhưng yếu trên tool ít xuất hiện.

### 5.5 Trích xuất tham số

#### Normalized Argument Exact Match

Normalized ArgEM chỉ được tính khi predicted tool multiset đúng. Một mẫu đạt exact match khi toàn bộ argument keys và normalized values của tất cả calls đều trùng khớp:

\[
N\text{-}ArgEM=\frac{
\sum_{i\in C_{tool}}\mathbb{1}[N(\widehat{A_i})=N(A_i)]
}{|C_{tool}|}
\]

trong đó \(C_{tool}\) là tập positive samples có tool set đúng và \(N(\cdot)\) là hàm normalization theo schema.

**Vai trò:** đo chất lượng extraction khi upstream đã chọn đúng schema.  
**Ý nghĩa:** đây là chỉ số strict, phản ánh khả năng tạo một bộ arguments hoàn chỉnh, nhưng không cho biết hệ thống đúng một phần đến mức nào.

#### Argument Key F1

Mỗi parameter path được xem như một item. Ví dụ nested field được biểu diễn bằng đường dẫn như `passenger.contact.phone`.

- True positive: parameter key xuất hiện ở cả gold và prediction.
- False positive: prediction sinh thêm parameter không có trong gold.
- False negative: prediction bỏ sót parameter gold.

Key Precision, Recall và F1 được tính theo công thức phân loại chuẩn.

**Vai trò:** đánh giá `has_value`/khả năng quyết định parameter nào cần xuất hiện.  
**Ý nghĩa:** Key Precision thấp cho thấy model thường tự thêm optional parameter; Key Recall thấp cho thấy model bỏ sót thông tin người dùng đã cung cấp.

#### Argument Pair F1

Mỗi cặp `(parameter_path, normalized_value)` được xem là một item. Chỉ khi cả key và value đúng, item mới được tính true positive.

**Vai trò:** cung cấp metric mềm hơn ArgEM ở cấp parameter.  
**Ý nghĩa:** hai prediction cùng có ArgEM bằng 0 vẫn có thể được phân biệt: một prediction chỉ sai một value sẽ có Pair F1 cao hơn prediction sai toàn bộ arguments.

#### Value Accuracy và breakdown theo type

Value Accuracy đo tỷ lệ normalized values đúng trên các parameter keys có thể đối chiếu. Kết quả tiếp tục được phân tầng theo `string`, `integer`, `number`, `boolean`, `enum`, `array` và `object`.

**Vai trò:** xác định loại parameter gây lỗi.  
**Ý nghĩa:** breakdown theo type đặc biệt quan trọng với Method 2 vì kiến trúc sử dụng các sub-head khác nhau cho span, enum và boolean.

### 5.6 Oracle-tool extraction

Trong thí nghiệm oracle, parameter extractor nhận trực tiếp gold tool schema thay vì tool do tầng trước dự đoán. Sau đó evaluator tính Oracle-tool ArgEM và Argument Pair F1.

**Vai trò:** đo upper bound của module extraction khi loại bỏ lỗi upstream.  
**Ý nghĩa:** nếu Oracle ArgEM cao nhưng N-FCEM thực tế thấp, degradation chủ yếu đến từ retrieval/selection. Nếu Oracle ArgEM vẫn thấp, parameter extractor là bottleneck chính.

Oracle-tool metric là metric chẩn đoán của kiến trúc modular; không bắt buộc đối với SLM, OpenAI hay Gemini vì các hệ thống này không tách một extractor độc lập.

### 5.7 Tính hợp lệ theo schema

Schema Validity kiểm tra:

- Output có parse được thành structured function call hay không.
- Tool name có thuộc candidate tool set hay không.
- Arguments có phải JSON object hay không.
- Parameter có tồn tại trong schema hay bị sinh thêm.
- Required fields có đầy đủ hay không.
- Type, enum và các constraint JSON Schema có hợp lệ hay không.

Evaluator báo cáo Prediction Parse Validity, Call Schema Validity và Sample Schema Validity.

**Vai trò:** đánh giá khả năng sử dụng đầu ra trong hệ thống thật.  
**Ý nghĩa:** một prediction có semantic value gần đúng nhưng sai type hoặc enum vẫn có thể bị API từ chối. Schema Validity vì vậy bổ sung góc nhìn thực thi mà accuracy ngữ nghĩa không phản ánh đầy đủ.

## 6. Đánh giá end-to-end

### 6.1 Normalized Function Call Exact Match

Metric chính của nghiên cứu là **Normalized Function Call Exact Match trên positive samples**, viết tắt là N-FCEM-positive:

\[
N\text{-}FCEM_{pos}=\frac{1}{N_{positive}}
\sum_i \mathbb{1}[
N(\widehat{Y_i})=N(Y_i)
]
\]

Trong đó \(Y_i\) là multiset gold function calls, mỗi call gồm tool name và arguments. Một mẫu chỉ đúng khi:

1. Output parse hợp lệ.
2. Số lượng calls đúng.
3. Multiset tool names đúng.
4. Toàn bộ argument keys và normalized values đúng.

N-FCEM kế thừa tinh thần ArgA của Ersoy et al. (2025), nhưng được mở rộng bằng schema-aware normalization và so khớp multi-call không phụ thuộc thứ tự.

**Vai trò:** headline quality metric dùng chung để so sánh cả bốn hệ thống.  
**Ý nghĩa:** phản ánh xác suất một positive query được chuyển thành toàn bộ structured function call chính xác.

### 6.2 Overall Success

Overall Success tính trên cả positive và negative samples:

\[
OverallSuccess=\frac{
CorrectPositiveCalls+CorrectNoCalls
}{N_{total}}
\]

Positive sample chỉ đúng khi đạt N-FCEM; negative sample chỉ đúng khi hệ thống không gọi tool.

**Vai trò:** đo toàn bộ hành vi từ call decision đến structured output.  
**Ý nghĩa:** phù hợp với kịch bản triển khai có cả yêu cầu dùng tool và hội thoại không cần tool. Tuy nhiên chỉ số này phải luôn được báo cáo cùng tỷ lệ positive/negative vì phân bố no-call có thể ảnh hưởng mạnh đến kết quả.

## 7. Đánh giá robustness và khả năng tổng quát hóa

Track C của AISA phân tầng kết quả theo phương ngữ. Trong khóa luận này, cùng nguyên tắc được tổng quát hóa sang các thuộc tính phù hợp với contribution của đề tài:

| Lát đánh giá | Câu hỏi được trả lời |
|---|---|
| Source: Glaive/xLAM | Hệ thống có phụ thuộc vào nguồn dữ liệu không? |
| Seen/Unseen Tool | Hệ thống có tổng quát hóa sang schema chưa thấy khi train không? |
| In-domain/Cross-domain | Hệ thống có chịu được domain shift không? |
| Single-call/Multi-call | Chất lượng giảm thế nào khi cần nhiều calls? |
| Candidate tool count | Hệ thống có ổn định khi registry lớn hơn không? |
| Random/Hard distractors | Hệ thống có phân biệt được các tool gần nghĩa không? |
| Schema parameter count | Extraction suy giảm thế nào khi schema phức tạp? |
| Parameter type | Model yếu ở span, enum, boolean, array hay object? |

Với metric \(M\) trên các nhóm \(g\), performance gap được tính:

\[
Gap_M=\max_g M_g-\min_g M_g
\]

**Vai trò:** phát hiện degradation bị che khuất bởi score trung bình.  
**Ý nghĩa:** một hệ thống tốt cần đồng thời có score tổng thể cao và gap thấp. Ví dụ, N-FCEM cao trên seen tools nhưng giảm mạnh trên unseen tools cho thấy model ghi nhớ tool thay vì thực sự sử dụng semantic description/schema.

Mỗi group phải báo cáo số lượng mẫu. Gap trên nhóm quá nhỏ có phương sai lớn nên không nên diễn giải độc lập nếu không có confidence interval hoặc kiểm tra bổ sung.

## 8. Đánh giá hiệu quả hệ thống

### 8.1 Latency

Latency được đo từ khi hệ thống nhận input đã chuẩn bị đến khi tạo xong prediction, không bao gồm thời gian thực thi tool bên ngoài. Báo cáo gồm mean, p50, p95 và p99.

- p50 biểu diễn trải nghiệm điển hình.
- p95 và p99 biểu diễn tail latency.
- p95 được chọn làm chỉ số latency chính vì mean có thể che giấu một số request rất chậm.

Đối với API baseline, latency bao gồm thời gian truyền mạng và thời gian dịch vụ phản hồi. Đối với model local, phải ghi rõ phần cứng, batch size, precision, warm-up và việc có tính thời gian load model hay không.

### 8.2 Throughput

Throughput được báo cáo bằng queries per second. Khi chỉ có latency tuần tự, evaluator ước lượng:

\[
QPS\approx\frac{1000}{MeanLatency_{ms}}
\]

Nếu chạy batching hoặc concurrent requests, throughput cần được đo trực tiếp bằng tổng số query chia cho wall-clock time.

### 8.3 Token và chi phí

Đối với API, ghi nhận input tokens, output tokens và chi phí thực tế hoặc chi phí tính theo bảng giá được khóa tại ngày thí nghiệm. Đối với model local, token count vẫn hữu ích cho Method 1; Method 2 không sinh autoregressive output nên cần báo thêm GPU memory, model size và throughput.

Cost per Correct Call được tính:

\[
CostPerCorrect=\frac{TotalCost}{NumberOfOverallSuccessfulSamples}
\]

**Vai trò:** kết hợp chất lượng với chi phí mà không cần một composite score tùy ý.  
**Ý nghĩa:** một hệ thống rẻ hơn trên mỗi query nhưng accuracy quá thấp có thể không thật sự hiệu quả; cost/correct phản ánh lượng chi phí cần thiết để thu được một kết quả đúng.

Chi phí API và chi phí local không hoàn toàn tương đương. API cost là chi phí biến đổi trực tiếp; local cost cần giả định về giá GPU, điện năng và mức sử dụng. Vì vậy báo cáo phải trình bày rõ cách ước lượng, đồng thời giữ latency, throughput và VRAM như các chỉ số độc lập.

## 9. Độ tin cậy thống kê

### 9.1 Bootstrap confidence interval

Đối với N-FCEM, Overall Success và các metric dạng tỷ lệ, nghiên cứu lấy mẫu bootstrap trên test samples với replacement. Với mỗi lần lặp, metric được tính lại; khoảng percentile 2.5%–97.5% tạo thành 95% confidence interval. Seed và số lần bootstrap phải được cố định; framework mặc định dùng 1.000 lần và seed 42.

**Vai trò:** biểu diễn độ bất định của score thay vì chỉ báo cáo một point estimate.  
**Ý nghĩa:** chênh lệch nhỏ giữa hai hệ thống có thể chỉ do biến động của test set.

### 9.2 Exact McNemar test

Vì các hệ thống được chạy trên cùng test IDs, outcome đúng/sai là paired data. McNemar test sử dụng hai nhóm bất đồng:

- \(n_{10}\): hệ thống A đúng, B sai.
- \(n_{01}\): hệ thống A sai, B đúng.

Exact two-sided p-value kiểm tra giả thuyết hai hệ thống có xác suất lỗi bằng nhau.

**Vai trò:** kiểm tra sự khác biệt paired trên cùng test set.  
**Ý nghĩa:** phù hợp hơn một phép kiểm định coi kết quả của hai hệ thống là hai mẫu độc lập.

Ngoài p-value, báo cáo paired bootstrap confidence interval cho chênh lệch score `A − B`. Không nên chỉ dựa vào ngưỡng `p < 0.05`; cần trình bày cả effect size và confidence interval.

## 10. Ma trận áp dụng metric

| Metric | SLM End-to-End | Bi+Cross | OpenAI FC | Gemini FC |
|---|:---:|:---:|:---:|:---:|
| Call Precision/Recall/F1 | Có | Có | Có | Có |
| Tool Set Accuracy | Có | Có | Có | Có |
| Argument Key/Pair F1 | Có | Có | Có | Có |
| Normalized ArgEM | Có | Có | Có | Có |
| Schema Validity | Có | Có | Có | Có |
| N-FCEM-positive | Có | Có | Có | Có |
| Overall Success | Có | Có | Có | Có |
| Recall@K, Full Recall@K, MRR | N/A | Có | N/A | N/A |
| Selection Conversion@K | N/A | Có | N/A | N/A |
| Oracle-tool ArgEM | N/A | Có | N/A | N/A |
| Robustness slices/gaps | Có | Có | Có | Có |
| Latency | Có | Có | Có | Có |
| API token/cost | N/A | N/A | Có | Có |
| Local VRAM/throughput | Có | Có | N/A | N/A |

`N/A` có nghĩa metric không áp dụng do khác biệt kiến trúc, không có nghĩa hệ thống đạt điểm 0.

## 11. Quy trình thực nghiệm

Quy trình đánh giá được thực hiện theo các bước sau:

1. Cố định test set và kiểm tra không có sample/tool leakage từ test sang train ngoài thiết kế split.
2. Cố định candidate tool set cho từng sample.
3. Khóa phiên bản model/API, prompt, decoding parameters, phần cứng và cấu hình inference.
4. Chạy mỗi hệ thống trên cùng test IDs và xuất prediction contract chuẩn.
5. Ghi `ranked_tools` cho Method 2 và telemetry cho tất cả hệ thống.
6. Chạy oracle extraction của Method 2 bằng gold tool schema.
7. Áp dụng cùng normalization config cho gold và mọi prediction.
8. Kiểm tra prediction coverage, parse validity và schema validity trước khi đọc accuracy.
9. Tính component metrics, N-FCEM, Overall Success, robustness slices và efficiency metrics.
10. Tính bootstrap confidence intervals và paired comparisons.
11. Thực hiện error analysis trên các mẫu mà hai phương pháp chính có kết quả khác nhau.
12. Lưu report, resolved config, input hashes và per-sample correctness để tái lập.

Nếu fine-tuning có tính ngẫu nhiên đáng kể, mỗi model local nên được train với tối thiểu ba random seeds và báo cáo `mean ± standard deviation`. API baseline có thể được chạy lặp lại trên một subset để ước lượng độ không ổn định nếu dịch vụ không hoàn toàn deterministic.

## 12. Cách diễn giải kết quả

Một số mẫu diễn giải chẩn đoán:

| Quan sát | Diễn giải hợp lý |
|---|---|
| Recall@5 thấp | Bi-Encoder loại gold tool quá sớm; downstream không thể phục hồi |
| Recall@5 cao, Selection Conversion thấp | Tầng chọn/rerank nhầm giữa các tool đã được truy hồi |
| Tool Set Accuracy cao, ArgEM thấp | Bottleneck nằm ở parameter extraction |
| Oracle ArgEM cao, N-FCEM thấp | Extractor tốt khi có schema đúng; lỗi chủ yếu đến từ retrieval/selection |
| Oracle ArgEM thấp | Cần cải thiện `has_value`, span/enum/boolean heads hoặc normalization |
| N-FCEM cao, Schema Validity thấp | Normalization che được khác biệt semantic nhưng raw output chưa đủ an toàn để thực thi |
| Overall score cao, unseen-tool gap lớn | Hệ thống tốt in-distribution nhưng tổng quát hóa kém |
| Accuracy tương đương, p95/cost thấp hơn rõ rệt | Hệ thống hiệu quả hơn về vận hành |
| Chênh lệch nhỏ, CI chứa 0, McNemar không có ý nghĩa | Chưa đủ bằng chứng kết luận hai hệ thống khác nhau |

Các diễn giải trên là chỉ báo chẩn đoán, không tự động chứng minh quan hệ nhân quả. Kết luận cuối cùng cần kết hợp metric, ablation và phân tích lỗi định tính.

## 13. Bảng kết quả đề xuất cho khóa luận

### 13.1 Bảng so sánh chính

| Hệ thống | Call F1 ↑ | Tool Set Acc ↑ | Arg Pair F1 ↑ | N-FCEM ↑ | Schema Valid ↑ | p95 ms ↓ | Cost/correct ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SLM End-to-End | | | | | | | |
| Bi-Encoder + Cross-Encoder | | | | | | | |
| OpenAI Function Calling | | | | | | | |
| Gemini Function Calling | | | | | | | |

### 13.2 Bảng chẩn đoán Method 2

| R@1 | R@5 | Full R@5 | MRR | Selection Conversion@5 | Oracle ArgEM | Actual N-FCEM |
|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | |

### 13.3 Bảng robustness

| Evaluation slice | #Samples | Tool Set Acc | ArgEM | N-FCEM | Gap so với nhóm đối chứng |
|---|---:|---:|---:|---:|---:|
| Seen tools | | | | | — |
| Unseen tools | | | | | |
| Single-call | | | | | — |
| Multi-call | | | | | |
| Normal distractors | | | | | — |
| Hard distractors | | | | | |

### 13.4 Bảng parameter analysis

| Parameter type | Support | Key Precision | Key Recall | Key F1 | Value Accuracy |
|---|---:|---:|---:|---:|---:|
| String/span | | | | | |
| Integer/number | | | | | |
| Enum | | | | | |
| Boolean | | | | | |
| Array/object | | | | | |

## 14. Nguy cơ ảnh hưởng tính hợp lệ

### 14.1 Tính hợp lệ nội tại

- Luật normalization quá rộng có thể biến prediction sai thành đúng. Cần khóa rules trước test evaluation và công bố toàn bộ alias.
- Nếu bốn hệ thống không nhận cùng candidate schemas, kết quả không còn so sánh trực tiếp được.
- API retry hoặc timeout bị loại khỏi mẫu sẽ làm score thiên lệch. Timeout phải được giữ như prediction thất bại và telemetry phải được ghi lại.
- Đo latency không warm-up đồng nhất có thể bất lợi cho model local.

### 14.2 Tính hợp lệ ngoại tại

- Benchmark dịch từ Glaive và xLAM có thể chưa đại diện đầy đủ cho truy vấn tự nhiên của người dùng Việt Nam.
- Kết quả trên tool pool hiện tại chưa đảm bảo giữ nguyên khi registry lớn hơn hoặc tool descriptions thay đổi.
- API model có thể được nhà cung cấp cập nhật theo thời gian; cần ghi model snapshot/version và ngày thí nghiệm.
- Cost phụ thuộc bảng giá và hạ tầng tại thời điểm đo.

### 14.3 Tính hợp lệ kết luận

- Một test set nhỏ có thể tạo confidence interval rộng.
- Báo cáo nhiều slices làm tăng nguy cơ diễn giải ngẫu nhiên; cần ưu tiên các slices được định nghĩa trước.
- Statistical significance không đồng nghĩa với practical significance. Cần báo cả chênh lệch tuyệt đối, tương đối và chi phí vận hành.

## 15. Hiện thực và khả năng tái lập

Framework được hiện thực trong `src/evaluation/` với các nhóm module:

- `detection_metrics.py`, `retrieval_metrics.py`, `selection_metrics.py` và `extraction_metrics.py` cho component metrics.
- `normalization.py` và `schema_validation.py` cho so sánh schema-aware và structural validity.
- `end_to_end_metrics.py` cho N-FCEM và Overall Success.
- `robustness.py`, `efficiency_metrics.py` và `statistics.py` cho phân tầng, hiệu quả và kiểm định.
- `evaluator.py`, `compare.py`, `report.py` và `cli.py` cho orchestration và xuất báo cáo.

Mỗi lần đánh giá sinh ba file chính:

```text
report.json
per_sample.jsonl
summary.md
```

So sánh nhiều hệ thống sinh thêm `comparison.json`, `comparison.csv` và `comparison.md`. Report lưu resolved configuration, đường dẫn input và SHA-256 của gold/prediction files. Thiết kế này bảo đảm bảng kết quả có thể truy ngược về đúng dữ liệu và cấu hình đã sử dụng.

## 16. Kết luận về vai trò của khung đánh giá

Khung đánh giá không chỉ dùng để xếp hạng bốn hệ thống. Vai trò quan trọng hơn là liên kết kết quả thực nghiệm với câu hỏi nghiên cứu của khóa luận. N-FCEM xác định chất lượng end-to-end; component metrics chỉ ra bottleneck; oracle evaluation tách lỗi upstream và downstream; robustness gaps đo khả năng tổng quát hóa; latency và cost/correct phản ánh giá trị vận hành; confidence interval và paired test xác định mức độ đáng tin cậy của chênh lệch.

Nhờ đó, khóa luận có thể đưa ra kết luận cụ thể hơn việc “mô hình A có accuracy cao hơn mô hình B”. Ví dụ, nghiên cứu có thể kết luận rằng kiến trúc Bi-Encoder + Cross-Encoder duy trì N-FCEM tương đương SLM trong confidence interval, cải thiện unseen-tool generalization, đồng thời giảm p95 latency và cost per correct call. Đây mới là bằng chứng trực tiếp cho giả thuyết về lợi ích của phương pháp modular.

## 17. Tài liệu tham khảo trực tiếp cho phần đánh giá

1. AISA-ArabicFC Shared Task, ArabicNLP 2026 — Track A: Function Call Detection & Selection; Track C: Cross-Dialect Robustness.
2. Ersoy et al. (2025), *Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning* — ArgA và đánh giá end-to-end tool calling.
3. Gao et al. (2025), *RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation* — stress test khi số candidate tools tăng.
4. JSON Schema Draft 2020-12 — kiểm tra type, enum, required fields và structural constraints.
