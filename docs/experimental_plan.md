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
| Method 1 | Qwen2.5-0.5B/1.5B + SFT | Tool selection và argument extraction trong một model sinh |
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
| `data/custom_vi/v1/train.jsonl` | Huấn luyện ngữ cảnh và tool đặc trưng Việt Nam |
| `data/custom_vi/v1/val_seen.jsonl` | Validation cho tool đã xuất hiện |
| `data/custom_vi/v1/val_unseen.jsonl` | Validation cho tool unseen |
| `data/custom_vi/v1/test_seen.jsonl` | Đánh giá CustomTools với tool đã gặp |
| `data/custom_vi/v1/test_unseen.jsonl` | Đánh giá zero-shot trên tool chưa gặp |

Các file CustomTools hiện có trong `data/custom_vi/v1/` phải được giữ nguyên. Không đưa bất kỳ sample nào từ `test_seen` hoặc `test_unseen` vào training.

### 3.2 Việc phải hoàn thành trước khi train

- [ ] Freeze phiên bản dataset và ghi manifest SHA-256.
- [ ] Xác nhận train/validation/test không có duplicate query hoặc duplicate scenario.
- [ ] Xác nhận tool unseen không xuất hiện trong tool schema của training.
- [ ] Xác nhận mỗi sample dùng master schema thống nhất.
- [ ] Xác nhận cách biểu diễn negative là `function_calls=[]` hoặc `<no_tool_call>`.
- [ ] Xác nhận quy tắc đánh giá multi-call: không phạt thứ tự với các call độc lập.
- [ ] Tạo snapshot dataset cho từng experiment, không sửa trực tiếp test data.

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

| ID | Model state | Dữ liệu tool-calling | Mục đích |
|---|---|---|---|
| E0 | Base/Instruct, chưa fine-tune | Không có | Zero-shot baseline |
| E1 | Base | Glaive EN + xLAM EN | Đo transfer EN → VI |
| E2 | Base | Glaive VI + xLAM VI | Đo lợi ích của dữ liệu tiếng Việt |
| E3 | Instruct hoặc model đã general SFT | Glaive EN + xLAM EN | Đo ảnh hưởng của general instruction tuning |
| E4 | Instruct hoặc model đã general SFT | Glaive EN/VI + xLAM EN/VI | Đo bilingual training |
| E5 | Instruct hoặc model đã general SFT | E4 + CustomTools train | Đo tool-specific fine-tuning |

E3 chỉ bắt buộc nếu có general instruction dataset độc lập. Nếu chưa có, đánh dấu E3 là optional và không dùng dữ liệu test để thay thế.

Mỗi experiment nên chạy cho Qwen2.5-1.5B trước. Qwen2.5-0.5B chạy sau để đo ảnh hưởng kích thước model.

### 5.3 Quy tắc huấn luyện

- Dùng cùng prompt template giữa các experiment.
- Dùng cùng tokenizer, max sequence length, seed và effective batch size.
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
| Qwen2.5-1.5B | E0–E5 | | | | | | |
| Qwen2.5-0.5B | E0–E5 | | | | | | |

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
- [ ] Chạy Method 1 với Qwen2.5-1.5B.
- [ ] Chạy E1 English-only.
- [ ] Chạy E2 Vietnamese-only.
- [ ] Chạy E4 bilingual.
- [ ] Chạy E5 bilingual + CustomTools.
- [ ] Lặp lại các experiment chính với Qwen2.5-0.5B.
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

- [ ] Chọn Qwen Base hay Qwen Instruct cho E0–E2.
- [ ] Có hoặc không có general instruction dataset cho E3.
- [ ] Chọn LoRA hay full fine-tuning.
- [ ] Chốt format multi-call duy nhất.
- [ ] Chốt strict và normalized comparison rules.
- [ ] Chốt số seed cho mỗi experiment; khuyến nghị ít nhất 3 seed cho các kết quả chính nếu tài nguyên cho phép.
- [ ] Chốt model/API version của OpenAI và Gemini.
- [ ] Chốt hardware và quy tắc đo latency local/API.
