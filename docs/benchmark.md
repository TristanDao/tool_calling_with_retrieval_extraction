# Benchmark tiếng Việt — Tool Calling

## 1. Mục tiêu

Xây dựng bộ benchmark tiếng Việt phục vụ:
- **Huấn luyện** Method 1 (SLM instruction-tuning) và Method 2 (Bi-Encoder + Cross-Encoder).
- **Đánh giá** pipeline 2 thành phần.
- **So sánh** 4 methods (SLM, Bi+Cross, OpenAI FC, Gemini FC).

## 2. Nguồn dữ liệu

| Dataset | Số samples | Đặc điểm |
|---|---|---:|---|
| Glaive Function Calling v2 | 112,960 | Multi-turn chat (74%), 1 tool trong `system` |
| xLAM (Salesforce) | 60,000 | Flat query, multi-call (53%), full tool pool per sample |

## 3. Schema master (canonical — single-turn + multi-call)

Mỗi sample trong `data/benchmark_core/<revision>/{en,vi}/{train,val,test}.jsonl`.
`data/benchmark_vi/{train,val,test}.jsonl` chỉ là active Vietnamese export của
revision đã freeze:

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
          "subject":  {"type": "string",  "description": "Môn học cần tìm"},
          "location": {"type": "string",  "description": "Thành phố hoặc khu vực"}
        },
        "required": ["subject", "location"]
      }
    }
  ]
}
```

### 3.1 Quy ước bắt buộc

| Trường | Ngôn ngữ | Quy tắc |
|---|---|---|
| `id` | — | Unique string, format `<source>_<index>` (VD: `glaive_00042`, `xlam_00123`) |
| `source` | — | `glaive` hoặc `xlam` |
| `query` | **VI** | User query tiếng Việt |
| `function_calls[].name` | **EN** | snake_case identifier, không dấu |
| `function_calls[].arguments` keys | **EN** | snake_case |
| `function_calls[].arguments` values | VI/EN | tùy natural language hay identifier |
| `tools[].name` | **EN** | snake_case identifier |
| `tools[].description` | **VI** | Mô tả tự nhiên tiếng Việt |
| `tools[].feature_group` | **VI** | Tên nhóm chức năng (do LLM classify, cache theo tool name) |
| `tools[].parameters` | — | JSON Schema chuẩn (xem §3.2) |

### 3.2 JSON Schema `type` chuẩn

| Schema chuẩn | xLAM raw mapping | Ghi chú |
|---|---|---|
| `"string"` | `"str"`, `"str, optional"` | |
| `"integer"` | `"int"`, `"int, optional"` | |
| `"number"` | `"float"`, `"float, optional"` | |
| `"boolean"` | `"bool"`, `"bool, optional"` | |
| `"array"` | `"list"`, `"List[int]"`, `"List[str]"`, ... | Phải có `items: {type: T}` |
| `"object"` | — | Nested object hiếm gặp |

**Optional flag**: Tách `", optional"` suffix thành top-level `required: []` array (không có trong `required` = optional). Enum dùng JSON Schema hợp lệ: `type: "string"` kèm field `enum`, không dùng `type: "enum"`.

### 3.3 Tool Schema (file riêng trong `tool_schema/`)

Mỗi tool 1 file `<tool_name>.json`:

```json
{
  "name": "search_tutors",
  "description": "Tìm gia sư theo môn học và khu vực.",
  "feature_group": "Tìm kiếm & Kết nối",
  "parameters": {
    "type": "object",
    "properties": {
      "subject":  {"type": "string", "description": "Môn học cần tìm"},
      "location": {"type": "string", "description": "Thành phố hoặc khu vực"}
    },
    "required": ["subject", "location"]
  }
}
```

### 3.4 Tool pool gộp — `tool_pool.json`

File tổng hợp unique tools từ cả 2 dataset (sau khi dịch + normalize):

```json
[
  {"name": "search_tutors", "description": "Tìm gia sư...", "feature_group": "...", "parameters": {...}},
  ...
]
```

Dùng cho: Method 1 system prompt, Method 2 Bi-Encoder index, stress test haystack.

## 4. Sample count sau pairing, validation và dedup

| Dataset | Raw/normalized input | Frozen revision | Ghi chú |
|---|---|---|---|
| Glaive | 45,593 positive + 15,141 negative | **18,210** | Sau scenario dedup và schema validation |
| xLAM | 60,000 positive | **58,818** | Sau scenario dedup và schema validation |
| **Total** | 120,734 paired inputs | **77,028 paired** | 4,817 negative |

Revision hiện hành là `2026-09-02-full-dedup-seed42`, với split
`61,615/7,701/7,712` theo train/val/test và `4,421` unique tools. Metadata,
input hashes và rejected records nằm trong revision manifest; không suy ra số
liệu chính thức từ các pilot cũ.

## 5. Pipeline xây dựng

```
data/raw/{glaive,xlam}/
   │
   │  src/data/collect.py
   ▼
data/raw/ (downloaded)
   │
   │  ┌─────────────────────────────────────────────────┐
   │  │  BỘ 1: src/data/translate.py                   │
   │  │  - Async batch K=10, concurrency=8              │
   │  │  - Qwen-MT qua ALIBABA_URL                     │
   │  │  - Append JSONL, flush per sample, atomic cp    │
   │  └─────────────────────────────────────────────────┘
   ▼
data/translations/
   ├── {glaive,xlam}_vi.jsonl         (output incremental)
   ├── failed/                        (sample fail sau 3 retry)
   ├── .checkpoint/                   (resume state)
   ├── logs/                          (progress log)
   │
   │  src/data/qa_translation.py
   │  - Rule check (identifier, JSON parse, required keys)
   │  - LLM judge (qwen3.7-max) cho 5% sample
   ▼
data/translations/qa_samples/         (QC pass/fail)
   │
    │  ┌─────────────────────────────────────────────────┐
    │  │  FROZEN CORE: src/data/rebuild_benchmark.py    │
    │  │  - Pair EN/VI, giữ positive và negative        │
    │  │  - Validate schema, arguments và scenario      │
    │  │  - Split paired 80/10/10 (seed=42)             │
   │  └─────────────────────────────────────────────────┘
   ▼
data/benchmark_core/<revision>/
    ├── en/{train,val,test}.jsonl + vi/{train,val,test}.jsonl
    ├── split_manifest.json + manifest.json
    ├── tool_schema/ + tool_pool.json
    └── metadata.json
data/benchmark_vi/  (active export của vi split)
    └── train.jsonl / val.jsonl / test.jsonl
   │
   │  ┌─────────────────────────────────────────────────┐
   │  │  Method 1: src/data/convert_to_instruction.py  │
    │  │  - Convert master → native messages/tool_calls  │
   │  └─────────────────────────────────────────────────┘
   ▼
    └── data/experiments/{e1,e2,e4,e5}/instruction/train_chat.jsonl
```

## 6. Splits

| Split | Tỷ lệ (đề xuất) | Mục đích |
|---|---|---|
| train | 80% | Huấn luyện Method 1 + Method 2 |
| val | 10% | Hyperparameter tuning, model selection |
| test | 10% | Đánh giá cuối, so sánh 4 methods |

> Seed đăng ký: `42`. Split được ghi bất biến trong `split_manifest.json`.

### 6.1 Trạng thái snapshot hiện tại

`data/benchmark_core/<revision>/` là canonical benchmark dùng chung, không phải
output riêng cho từng experiment. `data/benchmark_vi/` chỉ export split VI của
revision active. Không xóa hoặc rebuild benchmark theo từng model; mỗi
experiment chỉ ghi composition và reference tới revision.

Revision hiện hành `2026-09-02-full-dedup-seed42` được build bằng
`src/data/rebuild_benchmark.py`, gồm `77,028` paired records, `4,817`
negative và `4,421` unique tools. Split là `61,615/7,701/7,712`; EN và VI
counterpart luôn cùng split. Negative core records có `function_calls=[]`.

Trước khi chạy kết quả chính thức cần:

- Kiểm tra `manifest.json`, input hashes và rejected records của revision.
- Kiểm tra integrity của tool unseen và candidate tools.
- Không dùng `test_seen.jsonl` hoặc `test_unseen.jsonl` trong training.

## 7. Thống kê cần sinh (`src/data/stats.py`)

Output lưu `data/statistics/`:

- Tổng số samples sau QC (mỗi split).
- Phân bố số function_calls/query (1, 2, 3+).
- Phân bố số parameters/tool.
- Phân bố `feature_group` (top 20).
- Độ dài trung bình user query (token).
- Số unique tools trong `tool_pool.json`.

## 8. QC tiêu chí

Mỗi sample qua benchmark phải thỏa:

1. **JSON hợp lệ**: parse được, đủ 5 key `id`, `source`, `query`, `function_calls`, `tools`.
2. **Identifier integrity**: function name, argument keys đều snake_case EN (regex `^[a-z][a-z0-9_]*$`).
3. **Schema hợp lệ**: JSON Schema parse được, `type` ∈ chuẩn JSON Schema; enum là `type: "string"` + `enum`.
4. **Tool pool coverage**: Tất cả `function_calls[].name` có trong `tools[]` của sample.
5. **Feature_group**: 100% tool có `feature_group` non-empty.
6. **Vietnamese quality** (LLM judge 5%): query/description tự nhiên, không lỗi font.

## 9. Stress Test Set (Phase 7)

### 9.1 Tool pool — `data/benchmark_core/<revision>/tool_pool.json`
- Gộp unique tools từ Glaive + xLAM (sau dịch VI, dedupe theo `name`).
- Mỗi tool: `{name, description_VI, feature_group, parameters}`.

### 9.2 Anchors — `data/processed/stress_test/anchors.jsonl`
- 200 samples từ `benchmark_core/<revision>/vi/test.jsonl`.
- Mỗi sample: `{query, ground_truth_tool, gold_arguments}`.
- Tiêu chí: đa dạng `feature_group`, query rõ ràng.

### 9.3 Augmented instances — `data/processed/stress_test/augmented/`
- 200 × 6 × 2 = 2,400 instances.
- N values: [3, 10, 50, 100, 500, 1000].
- Strategies: `random` + `same_domain`.

## 10. Files output

| File | Format | Mô tả |
|---|---|---|
| `tool_pool.json` | JSON | All unique tools |
| `tool_schema/<name>.json` | JSON | 1 file/tool |
| `train.jsonl` | JSONL | 80% samples (schema master) |
| `val.jsonl` | JSONL | 10% samples (schema master) |
| `test.jsonl` | JSONL | 10% samples (schema master) |
| `split_manifest.json` | JSON | ID → split mapping bất biến |
| `manifest.json` | JSON | Input/output SHA-256 và build config |
| `metadata.json` | JSON | Counts, dedup và validation report |
| `data/experiments/e*/instruction/train_chat.jsonl` | JSONL | Native Method 1 training view |
