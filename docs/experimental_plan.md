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
| Method 1 | `unsloth/Qwen3.5-2B` hoặc `unsloth/Qwen3.5-4B` + Unsloth QLoRA/SFT | Tool selection và argument extraction trong một model sinh |
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

`data/benchmark_core/<revision>/` là benchmark canonical dùng chung cho mọi
experiment; `data/benchmark_vi/` chỉ là active Vietnamese export. Không xóa
hoặc rebuild revision theo từng model vì sẽ làm mất khả năng so sánh công bằng.

Quy tắc quản lý:

- Freeze một revision benchmark trước khi chạy các kết quả chính.
- Ghi `metadata.json` và manifest SHA-256 cho các file input, split và tool schema.
- Mỗi experiment ghi composition dữ liệu, checkpoint, seed và manifest benchmark vào thư mục kết quả riêng.
- Nếu thay đổi dữ liệu hoặc split, tạo benchmark revision/snapshot mới; không ghi đè revision đã dùng cho kết quả trước đó.
- `CustomTools-VI` giữ split riêng: chỉ `train.jsonl` đưa vào training; validation dùng
  cho model selection/diagnostic; `test_seen.jsonl` và `test_unseen.jsonl` chỉ dùng đánh giá.

Revision đã freeze `2026-09-02-full-dedup-seed42` gồm `77,028` paired records
(`18,210` Glaive, `58,818` xLAM), `4,817` negative và `4,421` unique tools.
Split cố định là `61,615/7,701/7,712`, seed `42`; EN/VI counterpart luôn cùng
split. Các hash và rejected records nằm trong revision manifest.

### 3.3 Việc phải hoàn thành trước khi train

- [x] Freeze phiên bản dataset và ghi manifest SHA-256.
- [x] Xác nhận train/validation/test không có duplicate query hoặc duplicate scenario.
- [ ] Xác nhận tool unseen không xuất hiện trong tool schema của training.
- [x] Xác nhận mỗi sample dùng master schema thống nhất.
- [x] Xác nhận negative master là `function_calls=[]`; native row dùng normal assistant content.
- [ ] Bảo đảm mỗi test set dùng để báo cáo negative recall có negative samples.
- [ ] Xác nhận quy tắc đánh giá multi-call: không phạt thứ tự với các call độc lập.
- [x] Tạo manifest training cho từng experiment, không sửa trực tiếp test data.

### 3.4 Quy mô dữ liệu và nguyên tắc sử dụng

Thiết kế của Ersoy et al. dùng toàn bộ training split cho từng cấu hình, giữ test split cố định, sau đó tăng dữ liệu ở bilingual và tool-specific experiments. Dự án này giữ nguyên nguyên tắc đó nhưng bổ sung một main controlled track để tách ảnh hưởng của ngôn ngữ khỏi ảnh hưởng của số lượng mẫu.

Quy mô input và evaluation hiện hành:

| Pool | Số mẫu | Trạng thái | Vai trò |
|---|---:|---|---|
| Frozen core revision | 77,028 | EN/VI paired, 4,817 negative | Core train/val/test |
| Frozen core unique tools | 4,421 | Shared EN/VI tool pool | Retrieval và native prompt |
| CustomTools-VI train | 5,600 | VI, 3,600 positive + 2,000 negative | Tool-specific SFT của E4 |
| CustomTools-VI validation | 800 | VI, seen/unseen | Model selection và diagnostic |
| CustomTools-VI test | 1,600 | VI, seen/unseen | Chỉ đánh giá cuối |

Các con số trên là quy mô input hiện tại, không phải số lượng cuối sau deduplication. Số lượng chính thức phải lấy từ manifest của benchmark revision đã freeze.

#### Main controlled track

- Revision frozen có `61,615` core train rows. Sau dedup, quota đăng ký lại là
  `60,000` rows: `56,151` positive (`10,712` Glaive + `45,439` xLAM) và
  `3,849` negative. Không lấy validation/test để bù quota.
- E1 dùng 60,000 mẫu tiếng Anh; E2 dùng đúng các sample IDs tương ứng bằng tiếng Việt.
- E3 dùng `30,000 EN + 30,000 VI`; mỗi ngôn ngữ gồm `27,000` positive
  (`10,712` Glaive + `16,288` xLAM) và `3,000` negative.
- E4 dùng toàn bộ 60,000 mẫu của E3 cộng toàn bộ `5,600` mẫu `CustomTools-VI/train.jsonl`, tổng `65,600` training examples. Không đưa bất kỳ file CustomTools validation/test nào vào training.
- Lấy mẫu một lần bằng seed cố định và lưu manifest; không lấy lại subset khác cho Qwen3.5-2B.

Cap 60,000 không phải giới hạn lý thuyết của model 2B/4B. Đây là ngân sách
chính để các experiment ngôn ngữ có cùng quy mô. Vì scenario dedup làm pool
Glaive nhỏ hơn quota ban đầu, manifest ghi rõ composition thực tế thay vì
đưa sample từ validation/test vào train.

#### Full-data track

Sau khi hoàn tất deduplication và QA, chạy thêm một full-data run cho mỗi kích thước model:

- E1-full: toàn bộ core train tiếng Anh.
- E2-full: toàn bộ core train tiếng Việt.
- E3-full: toàn bộ core train song ngữ, giữ tỷ lệ EN/VI `1:1`.
- E4-full: E3-full cộng `CustomTools-VI/train.jsonl`.

Full-data track không thay thế main controlled track và phải báo cáo riêng số training examples, số token, số step và compute. Nếu tài nguyên hạn chế, ưu tiên main controlled track; không tự ý cắt test hoặc validation để giảm chi phí.

#### Validation và test cố định

- Core dataset được split theo scenario/source ID trước khi tạo bản EN và VI; bản dịch EN/VI của cùng một scenario phải cùng split.
- Tỷ lệ core split là `80/10/10` theo train/validation/test sau deduplication.
- Không cap validation hoặc test. Các test rows không được dùng để chọn checkpoint, điều chỉnh prompt hoặc chọn hyperparameter.
- E1–E4 chọn checkpoint bằng core validation tương ứng; CustomTools validation chỉ là zero-shot diagnostic, không dùng để chọn checkpoint.
- E4 có thể dùng thêm `val_seen` và `val_unseen` để chọn checkpoint, nhưng phải báo cáo riêng từng kết quả và không dùng bất kỳ test split nào.
- Test bắt buộc gồm core test EN/VI và `CustomTools-VI/test_seen.jsonl` + `test_unseen.jsonl`, trong đó CustomTools phải tách positive/negative khi báo cáo.

#### Cấu trúc thư mục để chạy

Materialize toàn bộ controlled track bằng:

```bash
bash scripts/data/prepare_experiments.sh
```

Script tạo `data/experiments/{e0,e1,e2,e3,e4}/`. Mỗi thư mục chỉ có
`manifest.json`, `train.jsonl` và native `instruction/train_chat.jsonl` khi có
training data. Validation/test đọc trực tiếp từ frozen revision và
`data/custom_vi/`; Method 1 dùng `instruction/train_chat.jsonl`.

E0 không có file train vì đây là zero-shot. E1 train EN, E2 train VI,
E3 train song ngữ 30k EN + 30k VI, E4 dùng đúng E3 cộng
`data/custom_vi/train.jsonl`. Các file test/validation không được đưa vào training.

## 4. Chuẩn Output Chung

Mọi model phải được chuyển về cùng một output contract trước khi tính metric.

Master ground truth dùng `function_calls=[]` cho negative. Tool call native hợp lệ:

```text
<tool_call>
<function=tool_name>
<parameter=arg>
value
</parameter>
</function>
</tool_call>
```

Negative native hợp lệ là assistant content bình thường, không có tool call:

```text
Tôi chưa thể thực hiện yêu cầu này.
```

Parser chung phải xử lý và ghi nhận:

- Một hoặc nhiều tool call.
- JSON thuần và JSON nằm trong code fence.
- Output thiếu tag hoặc dùng protocol cũ.
- JSON không parse được.
- Tool name không tồn tại.
- Argument thiếu, thừa hoặc sai type.
- Output chứa text thừa ngoài tool call.

Parser chung là `src/evaluation/native_output.py`; output malformed không được
coi là negative. Không dùng parser hoặc quy tắc sửa lỗi riêng cho từng model.

## 5. Method 1 — SLM End-to-End

### 5.1 Format huấn luyện

- System: prompt nền; danh sách tool truyền qua trường native `tools`.
- User: query tiếng Việt hoặc tiếng Anh tùy experiment.
- Assistant positive: structured `tool_calls` được render bằng template của checkpoint.
- Assistant negative: normal assistant content, không dùng marker tự chế.
- Multi-call: nhiều phần tử trong cùng `tool_calls` list; thứ tự không dùng trong metric chính.

### 5.2 Thí nghiệm bắt buộc

Main result dùng controlled track 60k; full-data track được báo cáo riêng như robustness/scale-up. Mỗi dòng phải chạy độc lập cho cả `unsloth/Qwen3.5-4B` và `unsloth/Qwen3.5-2B` với cùng manifest sample IDs.

| ID | Model state | Core train | Custom train | Validation để chọn checkpoint | Test cố định | Mục đích |
|---|---|---:|---:|---|---|---|
| E0 | Qwen3.5 post-trained, chưa fine-tune | 0 | 0 | Không fine-tune | Core EN/VI + CustomTools test | Zero-shot baseline |
| E1 | Qwen3.5 post-trained | 60k EN | 0 | Core EN val | Core EN/VI test + CustomTools test | Đo transfer EN → VI |
| E2 | Qwen3.5 post-trained | 60k VI | 0 | Core VI val | Core EN/VI test + CustomTools test | Đo lợi ích của dữ liệu tiếng Việt |
| E3 | Qwen3.5 post-trained | 30k EN + 30k VI | 0 | Core EN/VI val | Core EN/VI test + CustomTools test | Đo bilingual training ở cùng ngân sách 60k |
| E4 | Qwen3.5 post-trained | 30k EN + 30k VI | 5,600 VI | Core EN/VI val + CustomTools val | Core EN/VI test + CustomTools test | Đo tool-specific fine-tuning |

E3 trong project là bilingual tool-calling SFT, không phải E3 của bài Arabic.
Qwen3.5 post-trained/Instruct được dùng làm checkpoint khởi đầu cho toàn bộ main
track; không có general-SFT stage riêng.

Mỗi experiment nên chạy cho `unsloth/Qwen3.5-4B` trước. `unsloth/Qwen3.5-2B` chạy sau để đo ảnh hưởng kích thước model. Dự án chỉ dùng các checkpoint Qwen3.5 post-trained/Instruct không có hậu tố `-Base` trong main track.

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

Xem output không có tool call hợp lệ là một class riêng và báo cáo:

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
- Đáng lẽ gọi tool nhưng trả normal answer hoặc output rỗng.
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
| `unsloth/Qwen3.5-4B` | E0, E1, E2, E3, E4 | | | | | | |
| `unsloth/Qwen3.5-2B` | E0, E1, E2, E3, E4 | | | | | | |

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

- [x] Freeze dataset, split và manifest.
- [x] Hoàn thiện parser output chung.
- [x] Hoàn thiện JSON Schema validator.
- [ ] Hoàn thiện weighted P/R, F1, ArgA và multi-call metrics.
- [ ] Chạy E0 zero-shot.
- [ ] Chạy Method 1 với `unsloth/Qwen3.5-4B`.
- [ ] Chạy E1 English-only.
- [ ] Chạy E2 Vietnamese-only.
- [ ] Chạy E3 bilingual.
- [ ] Chạy E4 bilingual + CustomTools.
- [ ] Lặp lại các experiment chính với `unsloth/Qwen3.5-2B`.
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

- [x] Ghi exact checkpoint ID của Qwen3.5 post-trained cho mỗi experiment; không dùng Base checkpoint.
- [x] Chọn LoRA QLoRA thay vì full fine-tuning.
- [x] Chốt format multi-call native `tool_calls` list.
- [x] Chốt strict và normalized comparison rules.
- [ ] Chốt số seed cho mỗi experiment; khuyến nghị ít nhất 3 seed cho các kết quả chính nếu tài nguyên cho phép.
- [ ] Chốt model/API version của OpenAI và Gemini.
- [ ] Chốt hardware và quy tắc đo latency local/API.
