# Benchmark tiếng Việt — Tool Calling

## 1. Mục tiêu

Xây dựng bộ benchmark tiếng Việt phục vụ:
- **Huấn luyện** Bi-Encoder (retrieval) và Cross-Encoder (extraction).
- **Đánh giá** pipeline 2 thành phần.
- **So sánh** với Generative LLM baseline (OpenAI FC, Gemini FC).

## 2. Nguồn dữ liệu — chỉ dùng 2 nguồn chính

Dữ liệu EN được thu thập từ 2 bộ Function Calling công khai:

| Dataset | Số samples (ước tính) | Đặc điểm |
|---|---|---|
| Glaive Function Calling v2 | ~110k | Đa dạng tool, có multi-turn |
| xLAM (Salesforce) | ~60k | Function call chuẩn, schema rõ ràng |

**Lý do chỉ dùng 2 nguồn**:
- Dataset đủ lớn (~170k samples tổng) để train/val/test.
- Schema rõ ràng, dễ chuẩn hóa.
- Đa dạng domain (Glaive) + function call chuẩn (xLAM).
- Tiết kiệm thời gian thu thập + xử lý (3 tháng khóa luận).
- ToolBench, ToolACE không dùng trong khóa luận này.

**Lưu ý**: Số liệu trên là ước tính, sẽ được xác nhận khi thu thập thực tế trong Phase 1.

## 3. Schema chuẩn

Mỗi sample có cấu trúc:

```json
{
  "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
  "label": {
    "function_call": {
      "name": "search_tutors",
      "arguments": {
        "subject": "Toán",
        "location": "Hà Nội"
      }
    }
  },
  "tools_summary": [
    {
      "feature_group": "Tìm kiếm & Kết nối",
      "tools": [
        {"name": "search_tutors", "description": "Tìm gia sư theo môn và khu vực."},
        {"name": "view_tutor_details", "description": "Xem hồ sơ chi tiết của gia sư."}
      ]
    }
  ]
}
```

### 3.1 Quy ước bắt buộc

| Trường | Ngôn ngữ | Quy tắc |
|---|---|---|
| `query` | **VI** | Câu hỏi tự nhiên tiếng Việt |
| `label.function_call.name` | **EN** | snake_case identifier, không dấu |
| `label.function_call.arguments` keys | **EN** | snake_case, không dấu |
| `label.function_call.arguments` values | VI/EN | tùy natural language hay identifier |
| `tools_summary[].feature_group` | **VI** | Tên nhóm chức năng tiếng Việt |
| `tools_summary[].tools[].name` | **EN** | snake_case identifier |
| `tools_summary[].tools[].description` | **VI** | Mô tả tiếng Việt |

### 3.2 Tool Schema

Cấu trúc JSON Schema riêng cho mỗi tool, lưu trong `data/benchmark_vi/tool_schema/`:

```json
{
  "name": "search_tutors",
  "description": "Tìm gia sư theo môn học và khu vực.",
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

## 4. Pipeline xây dựng

```
data/raw/{glaive,xlam}/
        │
        │  src/data/collect.py
        ▼
data/raw/ (downloaded)
        │
        │  src/data/normalize_schema.py
        ▼
data/processed/{tools,queries,parameters}/ (đã chuẩn hóa EN)
        │
        │  src/data/translate.py  (Qwen-MT)
        │  src/data/translate_guidelines.py  (bảo vệ identifier)
        ▼
data/translations/
        │
        │  src/data/qa_translation.py  (LLM judge + rule)
        ▼
data/translations/qa_samples/ (QC pass)
        │
        │  src/data/build_benchmark.py
        ▼
data/benchmark_vi/
    ├── tool_schema/  (các tool schemas, JSON)
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

## 5. Splits

| Split | Tỷ lệ (dự kiến) | Mục đích |
|---|---|---|
| train | 80% | Huấn luyện Bi-Encoder, Cross-Encoder, Unsloth |
| val | 10% | Hyperparameter tuning, model selection |
| test | 10% | Đánh giá cuối, so sánh baseline |

> **Open question** (xem `AGENTS.md` section 10): Tỷ lệ train/val/test sẽ được quyết định khi thu thập xong dataset thực tế.

## 6. Thống kê dự kiến

Sẽ được sinh tự động bởi `src/data/stats.py`, output lưu trong `data/statistics/`:

- Tổng số samples sau khi QC.
- Phân bố số tools/sample (1, 2, 3+).
- Phân bố số arguments/tool.
- Phân bố domain (education, finance, e-commerce, …).
- Độ dài trung bình query (token).
- Số unique tools trong benchmark.

## 7. Định dạng file

- **JSONL** (1 sample/dòng) cho train/val/test.
- **JSON** cho tool_schema (mỗi tool 1 file, hoặc 1 file gộp).

Lý do dùng JSONL: dễ stream, dễ shuffle, dễ load với `datasets` library.

## 8. QC tiêu chí

Mỗi sample qua benchmark phải thỏa:

1. **JSON hợp lệ**: parse được, không thiếu key bắt buộc.
2. **Function name hợp lệ**: snake_case, không dấu, không space.
3. **Schema hợp lệ**: JSON Schema parse được, type đúng.
4. **Arguments khớp schema**:
   - Span values: nằm trong query (start, end hợp lệ).
   - Enum values: thuộc danh sách enum.
   - Required fields: đủ, không null.
5. **Query dịch tự nhiên**: LLM judge đánh giá.
6. **Description dịch đúng nghĩa**: LLM judge đánh giá.

## 9. Quyết định đang chờ

- **[ ]** Số lượng tool trong benchmark (10? 50? 100?).
- **[ ]** Tỷ lệ train/val/test chính xác.
- **[ ]** Có dùng negative sampling trong train (cho Bi-Encoder) không, tỷ lệ bao nhiêu.
- **[ ]** Có augment data không (paraphrase query, swap synonym …).

---

## 10. Stress Test Set (RAG-MCP inspired) — Phase 7

### Mục đích
Test pipeline với tool pool lớn (lên đến **1000 tools**) để đo **degradation curve**.
Bổ sung cho test set chính, không thay thế.

### Thành phần

#### 10.1 Tool pool — `data/benchmark_vi/tool_pool.json`
- Gộp unique tools từ **Glaive + xLAM** (sau khi dịch VI, dedupe theo `name`).
- Mỗi tool có: `{name, description_VI, feature_group, parameters}`.
- Kích thước ước tính: **500-2000 tools** (sẽ xác nhận khi build).
- Đây là **"haystack"** của stress test.

#### 10.2 Anchors — `data/processed/stress_test/anchors.jsonl`
- 200 samples lấy từ `benchmark_vi/test.jsonl`.
- Mỗi sample = `{query, ground_truth_tool, gold_arguments}`.
- Tiêu chí chọn:
  - Đa dạng `feature_group` (tránh tập trung 1 domain).
  - Query rõ ràng, ground truth có arguments không rỗng.
- Đây là **"needle"** của stress test.

#### 10.3 Augmented instances — `data/processed/stress_test/augmented/`
- Với mỗi anchor × N × strategy → 1 instance.
- Format mỗi instance:
  ```json
  {
    "query": "...",
    "ground_truth_tool": "search_tutors",
    "gold_arguments": {"subject": "Toán", "location": "Hà Nội"},
    "candidate_tools": [
      {"name": "search_tutors", "description": "..."},
      {"name": "distractor_1", "description": "..."},
      ...
      {"name": "distractor_N-1", "description": "..."}
    ],
    "strategy": "random" | "same_domain",
    "N": 100
  }
  ```
- Tổng: 200 × 6 × 2 = **2,400 instances**.

### Distractor strategies

| Strategy | Mô tả | Mức độ khó |
|---|---|---|
| `random` | Random từ tool pool, loại trừ ground truth | Dễ |
| `same_domain` | Random từ cùng `feature_group` với ground truth | Khó hơn (semantic confusion) |

### Build pipeline

```
src/data/build_tool_pool.py              # Gộp tools từ raw Glaive + xLAM
src/data/extract_anchors.py              # Lấy 200 samples từ test.jsonl
src/data/augment_with_distractors.py     # Generate instances
src/evaluation/stress_test.py            # Chạy pipeline + baseline
src/evaluation/plot_stress_test.py       # Matplotlib plot
```

### Khác biệt với test set chính

| | Test set (`test.jsonl`) | Stress test set |
|---|---|---|
| N tools | ~1-10 (gốc từ data) | 3 → 1000 (augmented) |
| Mục đích | Accuracy pipeline bình thường | Degradation curve |
| Sample size | ~10% of total | 200 anchors |
| Distractors | Không | `random` + `same_domain` |
| Pipeline output | 1 metric (accuracy) | 4 metrics (accuracy, retrieval, latency, tokens) |

### Xem thêm
- Protocol chi tiết: `docs/methodology.md` Section 6.5.
- Lý do tham khảo paper: `docs/references.md` Section 12.
