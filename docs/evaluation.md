# Evaluation Protocol

Tài liệu này mô tả contract và cách vận hành evaluator. Phần trình bày học thuật về phương pháp, kỹ thuật, vai trò và ý nghĩa để đưa vào báo cáo khóa luận nằm tại `docs/evaluation_methodology_thesis.md`.

## 1. Mục tiêu

Evaluation framework nhận gold benchmark theo unified master structure và prediction của một hoặc nhiều phương pháp. Framework đánh giá độc lập từng tầng, toàn pipeline, độ bền theo các lát dữ liệu và hiệu quả suy luận. Mọi method dùng cùng normalization và cùng rule so khớp.

## 2. Gold input

Mỗi dòng của `gold.jsonl` là một JSON object:

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
  ],
  "metadata": {
    "tool_split": "seen",
    "domain": "education",
    "difficulty": "normal"
  }
}
```

Quy ước:

- `id` là duy nhất.
- `function_calls: []` là gold no-call.
- Có thể có nhiều calls và nhiều calls cùng tên; thứ tự không được dùng để quyết định đúng/sai.
- `tools[]` là candidate schemas dùng cho schema-aware normalization và validation.
- `metadata` là tùy chọn, dùng cho robustness slices.
- Loader cũng đọc được master cũ dạng `conversation[]` và lấy user query cùng assistant `function_calls[]` đầu tiên.

## 3. Prediction input

Mỗi dòng của `predictions.jsonl` có cùng `id` với gold:

```json
{
  "id": "glaive_00042",
  "function_calls": [
    {
      "name": "search_tutors",
      "arguments": {"subject": "toán", "location": "Hà Nội"}
    }
  ],
  "ranked_tools": [
    {"name": "search_tutors", "score": 0.94},
    {"name": "search_courses", "score": 0.62}
  ],
  "telemetry": {
    "latency_ms": 12.8,
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0.0,
    "gpu_peak_memory_mb": 1840.0
  },
  "metadata": {"model": "method_2"}
}
```

Các field tối thiểu là `id` và `function_calls`. No-call được ghi `function_calls: []`. `ranked_tools` chỉ cần cho method có retrieval. `telemetry` là tùy chọn; metric thiếu dữ liệu trả về `null`, không tự điền 0.

Loader hỗ trợ thêm:

- `function_call` là một call đơn.
- OpenAI-style `tool_calls[].function.name/arguments`.
- `raw_output` chứa một hoặc nhiều `<tool_call>...</tool_call>` hoặc `<no_tool_call>`.

Output không parse được phải giữ `id` và `raw_output`; evaluator đánh dấu `parse_valid=false` và xem như không có call.

## 4. Oracle-tool prediction

Để tách lỗi extraction khỏi lỗi chọn tool, có thể truyền thêm `oracle_predictions.jsonl`. File này có cùng contract prediction nhưng parameter extractor đã nhận gold tool schema. Báo cáo oracle dùng riêng file này và không thay prediction end-to-end.

## 5. Normalization

Normalization luôn áp dụng đối xứng cho gold và prediction:

1. Unicode NFC, trim và collapse whitespace.
2. Case-fold string nếu bật trong structured config.
3. Coerce integer/number/boolean chỉ khi JSON Schema khai báo type tương ứng.
4. Chuẩn hóa `date`/`date-time` khi JSON Schema khai báo `format` tương ứng.
5. Chuẩn hóa đệ quy array/object theo schema.
6. Áp dụng alias global hoặc alias theo `tool.parameter_path` từ JSON config tùy chọn.
7. Array giữ nguyên thứ tự mặc định; chỉ sort tại path được khai báo unordered.

Ví dụ alias config:

```json
{
  "global": {
    "đô la mỹ": "USD",
    "việt nam": "VN"
  },
  "by_parameter": {
    "convert_currency.source_currency": {
      "đồng việt nam": "VND"
    }
  },
  "unordered_array_paths": [
    "search_products.categories"
  ]
}
```

Một identifier như account number có schema `string` nên `"001234"` vẫn là `"001234"`.
Các field có tên kết thúc bằng `_id`, `_uuid`, `_iban`, `*_number` thuộc nhóm định danh, hoặc schema có `format: uuid|iban`/`x-identifier: true`, mặc định không case-fold.

## 6. Metric definitions

### Call detection

- Call Precision, Recall, F1.
- Hallucinated Call Rate: `FP / N_negative`.
- Missed Call Rate: `FN / N_positive`.

Call detection chỉ có ý nghĩa đầy đủ khi test set chứa cả positive và negative samples. Nếu benchmark hiện tại chỉ có function-call positives, Hallucinated Call Rate trả về `null`; cần bổ sung no-call subset trước khi dùng Call F1 làm kết luận về call gate.

### Retrieval

- Micro Recall@K trên toàn bộ unique gold tool names của từng sample.
- Full Recall@K: tỉ lệ positive samples có tất cả gold tools trong top-K.
- MRR: reciprocal rank của gold tool xuất hiện sớm nhất.
- Retrieval Coverage: tỉ lệ positive samples có `ranked_tools`.

### Tool selection

- Tool Set Accuracy: exact match multiset tên tools trên positive samples.
- Tool micro/macro F1.
- Selection Conversion@K: Tool Set Accuracy trong các mẫu có full retrieval coverage tại K.

### Parameter extraction

- Normalized ArgEM given correct tool set.
- Key Precision/Recall/F1.
- Argument Pair Precision/Recall/F1.
- Value Accuracy.
- Breakdown theo JSON Schema parameter type.
- Oracle-tool versions nếu có oracle prediction file.

### Schema validity

- Prediction parse validity.
- Call schema validity.
- Sample schema validity trên samples có predicted call.

### End-to-end

- N-FCEM-positive: exact tool-call multiset sau normalization trên positive samples.
- Overall Success: N-FCEM trên positive và đúng no-call trên negative.

### Robustness

Mỗi slice báo sample count, N-FCEM-positive, Overall Success, Tool Set Accuracy và ArgEM. Gap là score lớn nhất trừ nhỏ nhất. Built-in slices gồm `source`, `call_count`, `schema_parameter_count`, `candidate_tool_count` và `feature_group`; field tùy chỉnh đọc từ `metadata.<field>`.

### Efficiency

- Latency mean/p50/p95/p99.
- Throughput ước lượng theo latency tuần tự.
- Input/output/total tokens mỗi query.
- Cost/query, cost/1k và cost/correct call.
- Peak GPU memory.

## 7. Output

Một lần evaluate tạo:

```text
results/evaluation/<method>/
├── report.json
├── per_sample.jsonl
└── summary.md
```

`report.json` là báo cáo machine-readable đầy đủ. `per_sample.jsonl` chứa correctness flags để error analysis và paired tests. `summary.md` là bảng tóm tắt cho luận văn.
Report lưu resolved structured config, input paths và SHA-256 của gold/prediction files để tái lập đúng lần chạy.

Lệnh compare nhiều method tạo thêm:

```text
results/evaluation/comparison/
├── <method>/report.json
├── <method>/per_sample.jsonl
├── <method>/summary.md
├── comparison.json
├── comparison.csv
└── comparison.md
```

`comparison.json` chứa exact McNemar test và paired bootstrap CI trên các ID chung.

## 8. Các bước chạy

### Bước 1 — xuất predictions

Mỗi model chạy trên cùng `test.jsonl`, giữ nguyên `id`, rồi ghi contract ở mục 3. Method 2 nên xuất cả `ranked_tools`; tất cả method nên đo telemetry ngay tại inference.

### Bước 2 — đánh giá một method

```bash
python -m src.evaluation.cli evaluate \
  --gold data/benchmark_vi/test.jsonl \
  --predictions results/slm/predictions.jsonl \
  --output-dir results/evaluation/slm
```

PowerShell tương đương:

```powershell
python -m src.evaluation.cli evaluate `
  --gold data/benchmark_vi/test.jsonl `
  --predictions results/slm/predictions.jsonl `
  --output-dir results/evaluation/slm
```

Có oracle extractor và alias config:

```bash
python -m src.evaluation.cli evaluate \
  --gold data/benchmark_vi/test.jsonl \
  --predictions results/retrieval/predictions.jsonl \
  --oracle-predictions results/extraction/oracle_predictions.jsonl \
  --normalization-config configs/eval/normalization_aliases.json \
  --slice metadata.tool_split \
  --slice metadata.domain \
  --output-dir results/evaluation/method_2
```

### Bước 3 — so sánh bốn methods

```bash
python -m src.evaluation.cli compare \
  --gold data/benchmark_vi/test.jsonl \
  --prediction slm=results/slm/predictions.jsonl \
  --prediction method_2=results/retrieval/predictions.jsonl \
  --prediction openai=results/baselines/openai_predictions.jsonl \
  --prediction gemini=results/baselines/gemini_predictions.jsonl \
  --output-dir results/evaluation/comparison
```

Có thể thêm oracle extraction cho method modular:

```bash
  --oracle-prediction method_2=results/extraction/oracle_predictions.jsonl
```

### Bước 4 — đọc kết quả

1. Kiểm tra coverage và parse/schema validity trước.
2. Dùng Call F1 để đánh giá no-call gate.
3. Với Method 2, kiểm tra Recall@K rồi Selection Conversion@K.
4. So Oracle ArgEM với actual N-FCEM để định vị lỗi upstream/downstream.
5. Dùng N-FCEM-positive làm headline quality metric.
6. Dùng robustness gaps để phân tích generalization.
7. Dùng p95 latency và cost/correct để kết luận quality-efficiency trade-off.

## 9. Reproducibility

Structured config dùng dataclass trong `src/evaluation/config.py`. Default retrieval K là `[1, 3, 5, 10]`, seed là `42`, bootstrap là `1000` lần và confidence level là `95%`. Report lưu toàn bộ resolved config.
