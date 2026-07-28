# Benchmark tiếng Việt — Tool Calling

## 1. Mục tiêu

Xây dựng bộ benchmark tiếng Việt phục vụ:
- **Huấn luyện** Bi-Encoder (retrieval) và Cross-Encoder (extraction).
- **Đánh giá** pipeline 2 thành phần.
- **So sánh** với Generative LLM baseline (OpenAI FC, Gemini FC).

## 2. Nguồn dữ liệu — chỉ dùng 2 nguồn chính

Dữ liệu EN được thu thập từ 2 bộ Function Calling công khai:

| Dataset | Số samples | Đặc điểm |
|---|---:|---|
| Glaive Function Calling v2 | 112,960 | Multi-turn chat (74%), 1 tool trong `system` |
| xLAM (Salesforce) | 60,000 | Flat query, multi-call (53%), full tool pool per sample |

**Lý do chỉ dùng 2 nguồn**:
- Dataset đủ lớn (~170k samples tổng) để train/val/test.
- Schema rõ ràng, dễ chuẩn hóa.
- Đa dạng domain (Glaive) + function call chuẩn (xLAM).
- Tiết kiệm thời gian thu thập + xử lý (3 tháng khóa luận).

## 3. Kiến trúc 2 bộ (cập nhật 2026-07-28)

> **Quyết định**: Tách thành **2 bộ** riêng biệt — 1 bộ để dịch, 1 bộ task format chuẩn.

| Bộ | Path | Format | Mục đích |
|---|---|---|---|
| **Bộ 1 (dịch)** | `data/translations/` | Gần raw (giữ `chat` text cho Glaive, `answers` JSON list cho xLAM); chỉ dịch natural language | LLM dịch dễ, ít pre-processing |
| **Bộ 2 (task)** | `data/benchmark_vi/` | Multi-turn + multi-call chuẩn (xem §4) | Train + eval Bi-Encoder / Cross-Encoder |

**Lý do tách**:
- Bộ 1 giữ format gần raw → LLM dịch ít rủi ro corrupt JSON / mất ngữ nghĩa.
- Bộ 2 restructure từ Bộ 1 (parse + normalize) → schema task riêng, độc lập với raw.
- Khi Bộ 1 hỏng → re-translate; Bộ 2 không bị ảnh hưởng.

## 4. Schema Bộ 2 (chuẩn cho task — multi-turn + multi-call)

Mỗi sample trong `data/benchmark_vi/{train,val,test}.jsonl`:

```json
{
  "id": "glaive_00042",
  "source": "glaive",
  "conversation": [
    {
      "role": "user",
      "content": "Tôi muốn đặt vé máy bay từ Hà Nội đi Tokyo."
    },
    {
      "role": "assistant",
      "content": null,
      "function_calls": [
        {
          "name": "search_flights",
          "arguments": {"origin": "Hà Nội", "destination": "Tokyo"}
        },
        {
          "name": "search_hotels",
          "arguments": {"city": "Tokyo"}
        }
      ]
    },
    {
      "role": "function",
      "name": "search_flights",
      "content": "[{...flight 1...}, {...flight 2...}]"
    },
    {
      "role": "function",
      "name": "search_hotels",
      "content": "[{...hotel 1...}]"
    },
    {
      "role": "assistant",
      "content": "Tôi tìm được 2 chuyến bay và 1 khách sạn phù hợp..."
    }
  ],
  "tools": [
    {
      "name": "search_flights",
      "description": "Tìm chuyến bay theo điểm đi/đến/ngày.",
      "feature_group": "Đặt vé & Du lịch",
      "parameters": {
        "type": "object",
        "properties": {
          "origin": {"type": "string", "description": "Thành phố khởi hành"},
          "destination": {"type": "string", "description": "Thành phố đến"}
        },
        "required": ["origin", "destination"]
      }
    },
    {
      "name": "search_hotels",
      "description": "Tìm khách sạn theo thành phố.",
      "feature_group": "Đặt vé & Du lịch",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "Thành phố"}
        },
        "required": ["city"]
      }
    }
  ]
}
```

### 4.1 Quy ước bắt buộc

| Trường | Ngôn ngữ | Quy tắc |
|---|---|---|
| `id` | — | Unique string, format `<source>_<index>` (VD: `glaive_00042`, `xlam_00123`) |
| `source` | — | `glaive` hoặc `xlam` |
| `conversation[].role` | — | ∈ `{"user", "assistant", "function"}` |
| `conversation[].content` | **VI** | Natural language (user / function response / final assistant text) |
| `conversation[].function_calls[].name` | **EN** | snake_case identifier, không dấu |
| `conversation[].function_calls[].arguments` keys | **EN** | snake_case |
| `conversation[].function_calls[].arguments` values | VI/EN | tùy natural language hay identifier |
| `tools[].name` | **EN** | snake_case identifier |
| `tools[].description` | **VI** | Mô tả tự nhiên tiếng Việt |
| `tools[].feature_group` | **VI** | Tên nhóm chức năng (do LLM classify, cache theo tool name) |
| `tools[].parameters` | — | JSON Schema chuẩn (xem §4.2) |

### 4.2 JSON Schema `type` chuẩn

| Schema chuẩn | xLAM raw mapping | Ghi chú |
|---|---|---|
| `"string"` | `"str"`, `"str, optional"` | |
| `"integer"` | `"int"`, `"int, optional"` | |
| `"number"` | `"float"`, `"float, optional"` | |
| `"boolean"` | `"bool"`, `"bool, optional"` | |
| `"array"` | `"list"`, `"List[int]"`, `"List[str]"`, `"List[Union[int, float]]"`, … | Phải có `items: {type: T}` |
| `"object"` | — | Nested object hiếm gặp |

**Optional flag**: Tách `", optional"` suffix thành top-level `required: []` array (không có trong `required` = optional).

**VÍ dụ xLAM normalization**:
```json
// xLAM raw
{"type": "str, optional", "description": "..."}

// → Schema chuẩn
{"type": "string", "description": "..."}
// và parameter này KHÔNG có trong `required: [...]`
```

### 4.3 Tool Schema (file riêng trong `tool_schema/`)

Mỗi tool 1 file `<tool_name>.json`:

```json
{
  "name": "search_tutors",
  "description": "Tìm gia sư theo môn học và khu vực.",
  "feature_group": "Tìm kiếm & Kết nối",
  "parameters": {
    "type": "object",
    "properties": {
      "subject": {"type": "string", "description": "Môn học cần tìm"},
      "location": {"type": "string", "description": "Thành phố hoặc khu vực"}
    },
    "required": ["subject", "location"]
  }
}
```

### 4.4 Tool pool gộp — `tool_pool.json`

File tổng hợp unique tools từ cả 2 dataset (sau khi dịch + normalize):

```json
[
  {"name": "search_tutors", "description": "Tìm gia sư...", "feature_group": "...", "parameters": {...}},
  ...
]
```

Dùng cho: Bi-Encoder index, stress test haystack, OpenAI/Gemini baseline (cần full tool list).

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
   │  │  - Async batch K=50, concurrency=20             │
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
    │  │  BỘ 2: src/data/build_benchmark.py             │
    │  │  - Parse Bộ 1 → multi-turn schema              │
    │  │  - Pre-label feature_group trong translate     │
    │  │  - src/data/normalize_schema.py (type mapping) │
   │  │  - src/data/feature_group_classify.py          │
   │  │  - src/data/build_tool_pool.py                 │
   │  │  - Split train/val/test (80/10/10, seed=42)    │
   │  └─────────────────────────────────────────────────┘
   ▼
data/benchmark_vi/
   ├── tool_schema/   (mỗi tool 1 file JSON)
   ├── tool_pool.json (gộp unique tools từ 2 dataset)
   ├── train.jsonl / val.jsonl / test.jsonl
   └── metadata.json  (số sample, split ratio, statistics)
```

## 6. Splits

| Split | Tỷ lệ (đề xuất) | Mục đích |
|---|---|---|
| train | 80% | Huấn luyện Bi-Encoder, Cross-Encoder |
| val | 10% | Hyperparameter tuning, model selection |
| test | 10% | Đánh giá cuối, so sánh baseline |

> Seed: `42`. Có thể thay đổi khi build benchmark thực tế.

## 7. Thống kê cần sinh (`src/data/stats.py`)

Output lưu `data/statistics/`:

- Tổng số samples sau QC (mỗi split).
- Phân bố số turn/conversation (1, 2, 3+).
- Phân bố số function_calls/turn (1, 2, 3+).
- Phân bố số parameters/tool.
- Phân bố `feature_group` (top 20).
- Độ dài trung bình user query (token).
- Số unique tools trong `tool_pool.json`.

## 8. QC tiêu chí (Bộ 2)

Mỗi sample qua benchmark phải thỏa:

1. **JSON hợp lệ**: parse được, đủ 3 key `id`, `conversation`, `tools`.
2. **Identifier integrity**: function name, argument keys đều snake_case EN (regex `^[a-z][a-z0-9_]*$`).
3. **Schema hợp lệ**: JSON Schema parse được, `type` ∈ chuẩn JSON Schema.
4. **Multi-turn consistency**:
   - Nếu `role=assistant` có `function_calls[]` → `content` = null.
   - Nếu `role=function` → có `name` (tool name) + `content`.
5. **Tool pool coverage**: Tất cả `function_calls[].name` có trong `tools[]` của sample.
6. **Feature_group**: 100% tool có `feature_group` non-empty.
7. **Vietnamese quality** (LLM judge 5%): query/description tự nhiên, không lỗi font.

## 9. Stress Test Set (RAG-MCP inspired) — Phase 7

> Giữ nguyên từ version trước. Xem chi tiết ở version cũ nếu cần.

### 9.1 Tool pool — `data/benchmark_vi/tool_pool.json`
- Gộp unique tools từ Glaive + xLAM (sau dịch VI, dedupe theo `name`).
- Mỗi tool: `{name, description_VI, feature_group, parameters}`.
- Ước tính: 500-2000 tools (sẽ xác nhận khi build).

### 9.2 Anchors — `data/processed/stress_test/anchors.jsonl`
- 200 samples từ `benchmark_vi/test.jsonl`.
- Mỗi sample: `{query, ground_truth_tool, gold_arguments}`.
- Tiêu chí: đa dạng `feature_group`, query rõ ràng.

### 9.3 Augmented instances — `data/processed/stress_test/augmented/`
- 200 × 6 × 2 = 2,400 instances.
- Strategies: `random` + `same_domain`.

## 10. Files output

| File | Format | Mô tả |
|---|---|---|
| `tool_pool.json` | JSON | All unique tools |
| `tool_schema/<name>.json` | JSON | 1 file/tool |
| `train.jsonl` | JSONL | 80% samples |
| `val.jsonl` | JSONL | 10% samples |
| `test.jsonl` | JSONL | 10% samples |
| `metadata.json` | JSON | Stats, split info, dataset card |
