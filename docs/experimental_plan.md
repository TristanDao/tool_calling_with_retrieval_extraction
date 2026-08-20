# Experimental Plan — Vietnamese Tool Calling

## 1. Mục tiêu

Tài liệu này mô tả kế hoạch thực nghiệm cho đề tài Tool Calling tiếng Việt. Thiết kế lấy phần SLM end-to-end của Ersoy et al. (2025) làm cơ sở, sau đó mở rộng bằng Bi-Encoder + Cross-Encoder, hai API baseline và stress test với số lượng tool tăng dần.

Mục tiêu chính:

1. Đo ảnh hưởng của dữ liệu tool-calling tiếng Việt so với dữ liệu tiếng Anh.
2. Đo lợi ích của huấn luyện song ngữ Anh-Việt.
3. Đo khả năng tổng quát hóa sang `CustomTools-VI`, đặc biệt là tool unseen.
4. So sánh accuracy, latency, cost và khả năng mở rộng của bốn phương pháp.

## 2. Phạm vi phương pháp

| Method | Thành phần | Mục tiêu |
|---|---|---|
| Method 1 | `Qwen/Qwen3.5-2B` hoặc `Qwen/Qwen3.5-4B` + SFT | Tool selection và argument extraction trong một model sinh |
| Method 2 | BGE-M3 Bi-Encoder + Cross-Encoder | Retrieval tool và extraction parameter chuyên biệt |
| Baseline 1 | OpenAI Function Calling | Mốc tham chiếu API thương mại |
| Baseline 2 | Gemini Function Calling | Mốc tham chiếu API thương mại |

Phạm vi đánh giá gồm tool selection, parameter extraction, JSON/schema validation, latency và cost. Không đánh giá tool execution hoặc multi-turn agent orchestration.

## 3. Dataset Và Split

### 3.1 Dataset sử dụng

| Dataset | Vai trò |
|---|---|
| Glaive English | Huấn luyện tool-calling tổng quát bằng tiếng Anh |
| xLAM English | Huấn luyện single-turn và multi-call bằng tiếng Anh |
| Glaive/xLAM Vietnamese | Huấn luyện và đánh giá tool-calling tiếng Việt |
| `data/custom_vi/train.jsonl` | Huấn luyện ngữ cảnh và tool đặc trưng Việt Nam |
| `data/custom_vi/val_seen.jsonl` | Validation cho tool đã xuất hiện |
| `data/custom_vi/val_unseen.jsonl` | Validation cho tool unseen |
| `data/custom_vi/test_seen.jsonl` | Đánh giá CustomTools với tool đã gặp |
| `data/custom_vi/test_unseen.jsonl` | Đánh giá zero-shot trên tool chưa gặp |

Các file CustomTools hiện có trong `data/custom_vi/` phải được giữ nguyên. Không đưa bất kỳ sample nào từ `test_seen` hoặc `test_unseen` vào training.

### 3.2 Benchmark canonical và snapshot theo experiment

`data/benchmark_vi/` là benchmark canonical dùng chung cho mọi experiment, không phải thư mục output riêng của từng model. Không xóa hoặc rebuild thư mục này cho mỗi experiment vì sẽ làm mất khả năng so sánh công bằng giữa các phương pháp.

Quy tắc quản lý:

- Freeze một revision benchmark trước khi chạy các kết quả chính.
- Ghi `metadata.json` và manifest SHA-256 cho các file input, split và tool schema.
- Mỗi experiment ghi composition dữ liệu, checkpoint, seed và manifest benchmark vào thư mục kết quả riêng.
- Nếu thay đổi dữ liệu hoặc split, tạo benchmark revision/snapshot mới; không ghi đè revision đã dùng cho kết quả trước đó.
- `CustomTools-VI` giữ split riêng: chỉ `train.jsonl` và validation được phép đưa vào training; `test_seen.jsonl` và `test_unseen.jsonl` chỉ dùng đánh giá.

Artifact hiện có trong workspace là một pilot/rebuild snapshot gồm 51,227 positive samples, split 40,981/5,122/5,124 theo 80/10/10, seed 42. Snapshot không trùng ID giữa các split, nhưng kiểm tra hiện tại phát hiện 1,099 nhóm query trùng giữa các split. Đây chưa phải full benchmark cuối cùng theo quy mô dự kiến ~105k samples và chưa được dùng cho kết luận cuối trước khi hoàn tất translation, deduplication và QA.

### 3.3 Việc phải hoàn thành trước khi train

- [ ] Freeze phiên bản dataset và ghi manifest SHA-256.
- [ ] Xác nhận train/validation/test không có duplicate query hoặc duplicate scenario.
- [ ] Xác nhận tool unseen không xuất hiện trong tool schema của training.
- [ ] Xác nhận mỗi sample dùng master schema thống nhất.
- [ ] Xác nhận cách biểu diễn negative là `function_calls=[]` hoặc `<no_tool_call>`.
- [ ] Bảo đảm mỗi test set dùng để báo cáo negative recall có negative samples.
- [ ] Xác nhận quy tắc đánh giá multi-call: không phạt thứ tự với các call độc lập.
- [ ] Tạo snapshot dataset cho từng experiment, không sửa trực tiếp test data.

### 3.4 Quy mô dữ liệu và nguyên tắc sử dụng

Thiết kế của Ersoy et al. dùng toàn bộ training split cho từng cấu hình, giữ test split cố định, sau đó tăng dữ liệu ở bilingual và tool-specific experiments. Dự án này giữ nguyên nguyên tắc đó nhưng bổ sung một main controlled track để tách ảnh hưởng của ngôn ngữ khỏi ảnh hưởng của số lượng mẫu.

Quy mô dữ liệu hiện có trước final deduplication:

| Pool | Số mẫu | Trạng thái | Vai trò |
|---|---:|---|---|
| Glaive positive | 45,593 | EN và bản dịch VI tương ứng | Core tool-calling |
| xLAM positive | 60,000 | EN và bản dịch VI tương ứng | Core single-turn/multi-call |
| Glaive negative | 15,141 | EN và bản dịch VI tương ứng | No-tool-call training/evaluation |
| CustomTools-VI train | 5,600 | VI, 3,600 positive + 2,000 negative | Tool-specific SFT của E5 |
| CustomTools-VI validation | 800 | VI, seen/unseen | Model selection và diagnostic |
| CustomTools-VI test | 1,600 | VI, seen/unseen | Chỉ đánh giá cuối |

Các con số trên là quy mô input hiện tại, không phải số lượng cuối sau deduplication. Số lượng chính thức phải lấy từ manifest của benchmark revision đã freeze.

#### Main controlled track

- Mỗi experiment E1–E4 dùng tối đa `60,000` core examples trong training: `54,000` positive và `6,000` negative.
- Với positive examples, giữ tỷ lệ nguồn gần dữ liệu gốc: `23,000` Glaive và `31,000` xLAM.
- E1 dùng 60,000 mẫu tiếng Anh; E2 dùng đúng các sample IDs tương ứng bằng tiếng Việt.
- E3 nếu chạy dùng cùng 60,000 mẫu tiếng Anh như E1, sau general SFT.
- E4 dùng `30,000 EN + 30,000 VI`, gồm cùng một tỷ lệ Glaive/xLAM và positive/negative ở mỗi ngôn ngữ.
- E5 dùng toàn bộ 60,000 mẫu của E4 cộng toàn bộ `5,600` mẫu `CustomTools-VI/train.jsonl`, tổng `65,600` training examples. Không đưa bất kỳ file CustomTools validation/test nào vào training.
- Lấy mẫu một lần bằng seed cố định và lưu manifest; không lấy lại subset khác cho Qwen3.5-2B.

Cap 60,000 không phải giới hạn lý thuyết của model 2B/4B. Đây là ngân sách chính để các experiment ngôn ngữ có cùng quy mô. Khoảng 60k tool-calling examples là đủ lớn cho SFT pilot, còn dữ liệu đa dạng hơn sẽ được kiểm tra ở full-data track.

#### Full-data track

Sau khi hoàn tất deduplication và QA, chạy thêm một full-data run cho mỗi kích thước model:

- E1-full: toàn bộ core train tiếng Anh.
- E2-full: toàn bộ core train tiếng Việt.
- E4-full: toàn bộ core train song ngữ, giữ tỷ lệ EN/VI `1:1`.
- E5-full: E4-full cộng `CustomTools-VI/train.jsonl`.

Full-data track không thay thế main controlled track và phải báo cáo riêng số training examples, số token, số step và compute. Nếu tài nguyên hạn chế, ưu tiên main controlled track; không tự ý cắt test hoặc validation để giảm chi phí.

#### Validation và test cố định

- Core dataset được split theo scenario/source ID trước khi tạo bản EN và VI; bản dịch EN/VI của cùng một scenario phải cùng split.
- Tỷ lệ core split là `80/10/10` theo train/validation/test sau deduplication.
- Không cap validation hoặc test. Các test rows không được dùng để chọn checkpoint, điều chỉnh prompt hoặc chọn hyperparameter.
- E1–E4 chọn checkpoint bằng core validation tương ứng; CustomTools validation chỉ là zero-shot diagnostic, không dùng để chọn checkpoint.
- E5 có thể dùng thêm `val_seen` và `val_unseen` để chọn checkpoint, nhưng phải báo cáo riêng từng kết quả và không dùng bất kỳ test split nào.
- Test bắt buộc gồm core test EN/VI và `CustomTools-VI/test_seen.jsonl` + `test_unseen.jsonl`, trong đó CustomTools phải tách positive/negative khi báo cáo.

## 4. Chuẩn Output Chung

Mọi model phải được chuyển về cùng một output contract trước khi tính metric.

Tool call hợp lệ:

```text
<tool_call>
{"name":"tool_name","arguments":{"arg":"value"}}
</tool_call>
```

Negative hợp lệ:

```text
<no_tool_call>
```

Parser chung phải xử lý và ghi nhận:

- Một hoặc nhiều tool call.
- JSON thuần và JSON nằm trong code fence.
- Output thiếu tag.
- JSON không parse được.
- Tool name không tồn tại.
- Argument thiếu, thừa hoặc sai type.
- Output chứa text thừa ngoài tool call.

Không dùng parser hoặc quy tắc sửa lỗi riêng cho từng model.

## 5. Method 1 — SLM End-to-End

### 5.1 Format huấn luyện

- System: danh sách tool gồm name, description và JSON Schema.
- User: query tiếng Việt hoặc tiếng Anh tùy experiment.
- Assistant: `<tool_call>...</tool_call>` hoặc `<no_tool_call>`.
- Multi-call: assistant sinh danh sách hoặc chuỗi call theo một format đã cố định.

### 5.2 Thí nghiệm bắt buộc

Main result dùng controlled track 60k; full-data track được báo cáo riêng như robustness/scale-up. Mỗi dòng phải chạy độc lập cho cả `Qwen/Qwen3.5-4B` và `Qwen/Qwen3.5-2B` với cùng manifest sample IDs.

| ID | Model state | Core train | Custom train | Validation để chọn checkpoint | Test cố định | Mục đích |
|---|---|---:|---:|---|---|---|
| E0 | Qwen3.5 post-trained, chưa fine-tune | 0 | 0 | Không fine-tune | Core EN/VI + CustomTools test | Zero-shot baseline |
| E1 | Qwen3.5 post-trained | 60k EN | 0 | Core EN val | Core EN/VI test + CustomTools test | Đo transfer EN → VI |
| E2 | Qwen3.5 post-trained | 60k VI | 0 | Core VI val | Core EN/VI test + CustomTools test | Đo lợi ích của dữ liệu tiếng Việt |
| E3 | Qwen3.5 post-trained + general SFT | 60k EN | 0 | Core EN val | Core EN/VI test + CustomTools test | Đo ảnh hưởng của general instruction tuning |
| E4 | Qwen3.5 post-trained + general SFT | 30k EN + 30k VI | 0 | Core EN/VI val | Core EN/VI test + CustomTools test | Đo bilingual training ở cùng ngân sách 60k |
| E5 | Qwen3.5 post-trained + general SFT | 30k EN + 30k VI | 5,600 VI | Core EN/VI val + CustomTools val | Core EN/VI test + CustomTools test | Đo tool-specific fine-tuning |

E3 chỉ bắt buộc nếu có general instruction dataset độc lập. Nếu chưa có, đánh dấu E3 là optional và không dùng dữ liệu test để thay thế. Nếu chạy E3, general SFT phải được thực hiện trước tool-calling SFT và phải ghi riêng checkpoint trung gian.

Mỗi experiment nên chạy cho `Qwen/Qwen3.5-4B` trước. `Qwen/Qwen3.5-2B` chạy sau để đo ảnh hưởng kích thước model. Dự án chỉ dùng các checkpoint Qwen3.5 post-trained không có hậu tố `-Base`; không dùng `Qwen/Qwen3.5-2B-Base` hoặc `Qwen/Qwen3.5-4B-Base`.

### 5.3 Quy tắc huấn luyện

- Dùng cùng prompt template giữa các experiment.
- Dùng cùng tokenizer, max sequence length, seed và effective batch size.
- Dùng cùng sample IDs, số epoch mục tiêu và training budget cho 2B/4B trong cùng một experiment.
- Main controlled track bắt đầu với 1 epoch; chỉ tăng epoch nếu core validation cải thiện và phải ghi rõ early stopping.
- Chỉ dùng validation để chọn checkpoint và hyperparameter.
- Inference dùng deterministic decoding: `temperature=0`, không sampling.
- Lưu config, checkpoint path, commit hash, seed và log cho từng run.
- Không thay đổi test set dựa trên kết quả trung gian.

## 6. Method 2 — Bi-Encoder + Cross-Encoder

### 6.1 Bi-Encoder

Input training pair:

```text
(query, tool_description)
```

- Base model: `BAAI/bge-m3`.
- Loss: MultipleNegativesRankingLoss.
- Positive: tool xuất hiện trong `function_calls`.
- Negative: tool khác trong cùng candidate pool.
- Hard negative: tool cùng feature group hoặc bị retrieve sai ở vòng trước.

Metric:

- Recall@1, Recall@3, Recall@5, Recall@10.
- MRR.
- NDCG nếu cần đánh giá ranking sâu hơn.

### 6.2 Cross-Encoder

Input mỗi parameter:

```text
[CLS] query [SEP] Param=<name>. Desc=<description>. Type=<type>[. Enum=...] [SEP]
```

Heads:

- `has_value`: parameter có được đề cập hay không.
- Span head: string, integer, number.
- Enum head: enum value.
- Boolean head: true/false.

Metric:

- `has_value` accuracy và F1.
- Span Exact Match và Span F1.
- Enum accuracy.
- Boolean accuracy.
- Required argument accuracy.
- End-to-end argument F1 và Exact Match.

### 6.3 Hai chế độ đánh giá bắt buộc

1. Oracle retrieval: đưa đúng tool vào Cross-Encoder để đo riêng extraction.
2. Full pipeline: Bi-Encoder retrieve top-k rồi Cross-Encoder extract.

Chênh lệch giữa hai chế độ cho biết lỗi chính nằm ở retrieval hay extraction.

## 7. Baseline API

OpenAI và Gemini phải nhận cùng query, candidate tools và JSON Schema với các method local.

Mỗi request cần lưu:

- Raw response.
- Parsed function calls.
- HTTP status và error.
- Input/output tokens nếu API cung cấp.
- Latency.
- Estimated cost.

Phải ghi rõ model version, ngày chạy, region và tình trạng warm/cold request.

## 8. Metrics

### 8.1 Tool selection

Xem `<no_tool_call>` là một class riêng và báo cáo:

- Tool accuracy.
- Weighted precision.
- Weighted recall.
- Macro F1.
- Positive recall: nhận ra query cần gọi tool.
- Negative recall: không gọi tool khi không cần.
- Hallucinated tool rate.

### 8.2 Argument Population Accuracy

ArgA là metric chính của Method 1 và phải được giữ để so sánh với bài báo:

```text
ArgA = số positive sample có tool name và toàn bộ arguments đúng
       / tổng số positive sample
```

Một sample chỉ đúng khi tool name, argument keys, argument values và số lượng call đều chính xác.

Với multi-call:

- So sánh tập các function call sau canonicalization.
- Không phạt thứ tự giữa các call độc lập.
- Phạt call thiếu hoặc call thừa.

### 8.3 Argument và output validity

- Argument Exact Match.
- Argument F1.
- Required argument accuracy.
- Optional `has_value` accuracy.
- Enum accuracy.
- Boolean accuracy.
- Numeric accuracy sau normalization.
- JSON valid rate.
- Schema valid rate.
- Required field complete rate.
- Invalid output rate.

### 8.4 Normalization

Tính cả strict metric và normalized metric.

Normalization được phép:

- Chuẩn hóa Unicode tiếng Việt.
- Xóa whitespace dư.
- Chuẩn hóa chữ hoa/chữ thường khi không làm thay đổi identifier.
- Chuẩn hóa số nguyên, số thực, ngày tháng và boolean.

Không tự động dùng synonym hoặc LLM judge cho metric chính vì có thể làm sai ý nghĩa Exact Match.

## 9. Error Analysis

Lấy mẫu lỗi từ các nhóm sau:

- Đúng tool nhưng sai argument.
- Sai tool.
- Đáng lẽ gọi tool nhưng trả `<no_tool_call>`.
- Đáng lẽ không gọi tool nhưng lại gọi.
- JSON không hợp lệ.
- Thiếu required argument.
- Argument sai ngôn ngữ.
- Sai enum, boolean hoặc numeric format.
- Sai số lượng hoặc cấu trúc multi-call.

Phân loại theo bốn nhóm của bài báo:

- `W`: Wrong argument value.
- `T`: Translation/language discrepancy.
- `P`: Paraphrasing variance.
- `I`: Incomplete context.

Báo cáo số lượng và tỷ lệ từng nhóm theo model và experiment.

## 10. Latency, Cost Và Stress Test

### 10.1 Latency/cost

Đo:

- Mean latency.
- P50 latency.
- P95 latency.
- Throughput.
- Peak memory/GPU memory.
- Cost per 1,000 queries.

Method 2 phải tách thời gian query embedding, retrieval, Cross-Encoder và validation. Tool embedding pre-compute chỉ đo một lần riêng.

### 10.2 Stress test

Candidate tool pool:

```text
N = 3, 10, 50, 100, 500, 1000
```

Distractor strategy:

- Random.
- Same-domain.

Thiết kế đầy đủ:

```text
200 queries × 6 tool counts × 2 distractor strategies = 2,400 instances
```

Đánh giá cả bốn methods theo:

- Tool accuracy.
- ArgA.
- JSON validity.
- P50/P95 latency.
- Cost.
- Memory/context usage.

## 11. Bảng Kết Quả Bắt Buộc

### Bảng A — Dataset statistics

| Dataset | Language | Train | Val | Test | Tools | Positive | Negative |
|---|---|---:|---:|---:|---:|---:|---:|
| Glaive | EN/VI | | | | | | |
| xLAM | EN/VI | | | | | | |
| CustomTools-VI | VI | 5,600 | 800 | 1,600 | 40 | 4,800 | 3,200 |

### Bảng B — SLM experiments

| Model | Experiment | Test set | Tool P | Tool R | Negative R | ArgA | JSON valid |
|---|---|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.5-4B` | E0–E5 | | | | | | |
| `Qwen/Qwen3.5-2B` | E0–E5 | | | | | | |

### Bảng C — Method comparison

| Method | Tool Acc | Arg F1 | ArgA | JSON valid | P50 ms | P95 ms | Cost/1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| SLM | | | | | | | |
| Bi+Cross | | | | | | | |
| OpenAI FC | | | | | | | |
| Gemini FC | | | | | | | |

### Bảng D — CustomTools generalization

| Method | Seen Tool R | Unseen Tool R | Seen ArgA | Unseen ArgA |
|---|---:|---:|---:|---:|
| SLM | | | | |
| Bi+Cross | | | | |
| OpenAI FC | | | | |
| Gemini FC | | | | |

### Bảng E — Stress test

| Method | N tools | Tool Acc | ArgA | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| SLM | 3/10/50/100/500/1000 | | | | |
| Bi+Cross | 3/10/50/100/500/1000 | | | | |
| OpenAI FC | 3/10/50/100/500/1000 | | | | |
| Gemini FC | 3/10/50/100/500/1000 | | | | |

## 12. Thứ Tự Triển Khai

- [ ] Freeze dataset, split và manifest.
- [ ] Hoàn thiện parser output chung.
- [ ] Hoàn thiện JSON Schema validator.
- [ ] Hoàn thiện weighted P/R, F1, ArgA và multi-call metrics.
- [ ] Chạy E0 zero-shot.
- [ ] Chạy Method 1 với `Qwen/Qwen3.5-4B`.
- [ ] Chạy E1 English-only.
- [ ] Chạy E2 Vietnamese-only.
- [ ] Chạy E4 bilingual.
- [ ] Chạy E5 bilingual + CustomTools.
- [ ] Lặp lại các experiment chính với `Qwen/Qwen3.5-2B`.
- [ ] Train và đánh giá Bi-Encoder.
- [ ] Train và đánh giá Cross-Encoder với oracle retrieval.
- [ ] Đánh giá full Bi+Cross pipeline.
- [ ] Chạy OpenAI và Gemini baseline.
- [ ] Đo latency, throughput và cost.
- [ ] Chạy stress test.
- [ ] Phân tích lỗi ArgA.
- [ ] Tạo bảng, biểu đồ và kết luận.

## 13. Acceptance Criteria

Một experiment chỉ được xem là hoàn tất khi:

- [ ] Có config và commit hash.
- [ ] Có seed và thông tin model/checkpoint.
- [ ] Có raw predictions.
- [ ] Có parsed predictions.
- [ ] Có metrics JSON/CSV.
- [ ] Có log lỗi và số sample bị bỏ qua.
- [ ] Có latency và resource log nếu là experiment comparison.
- [ ] Có thể tái chạy trên cùng snapshot dataset.

Bản kết quả cuối phải trả lời được bốn câu hỏi:

1. Dữ liệu tiếng Việt có cải thiện tool calling so với English-only không?
2. Bilingual training có cải thiện kết quả trên test tiếng Việt không?
3. CustomTools-VI có cải thiện seen và unseen tool generalization không?
4. Bi+Cross có đạt trade-off accuracy/latency tốt hơn SLM và API baseline không?

## 14. Deliverables

- `data/` snapshot và dataset manifest.
- Instruction data cho Method 1.
- Bi-Encoder training pairs và checkpoint.
- Cross-Encoder labels, checkpoint và inference output.
- Raw predictions của bốn methods.
- Metrics theo sample, dataset, experiment và model.
- Latency/cost logs.
- Stress test results.
- Bảng và biểu đồ trong `results/tables_figures/`.
- Error analysis report.
- Final comparison report cho luận văn.

## 15. Quyết Định Cần Chốt Trước Khi Chạy Chính Thức

- [ ] Ghi exact checkpoint ID của Qwen3.5 post-trained cho mỗi experiment; không dùng Base checkpoint.
- [ ] Có hoặc không có general instruction dataset cho E3.
- [ ] Chọn LoRA hay full fine-tuning.
- [ ] Chốt format multi-call duy nhất.
- [ ] Chốt strict và normalized comparison rules.
- [ ] Chốt số seed cho mỗi experiment; khuyến nghị ít nhất 3 seed cho các kết quả chính nếu tài nguyên cho phép.
- [ ] Chốt model/API version của OpenAI và Gemini.
- [ ] Chốt hardware và quy tắc đo latency local/API.
