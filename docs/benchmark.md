# Benchmark tiếng Việt — Tool Calling

## 1. Mục tiêu

Xây dựng bộ benchmark tiếng Việt phục vụ:
- **Huấn luyện** Bi-Encoder (retrieval) và Cross-Encoder (extraction).
- **Đánh giá** pipeline 2 thành phần.
- **So sánh** với Generative LLM baseline (OpenAI FC, Gemini FC, Qwen2.5/Llama-3.1 local).

## 2. Nguồn dữ liệu

Dữ liệu EN được thu thập từ các bộ Function Calling công khai:

| Dataset | Số samples (ước tính) | Đặc điểm |
|---|---|---|
| Glaive Function Calling v2 | ~110k | Đa dạng tool, có multi-turn |
| ToolBench (API-Bank + RestBench) | ~100k | API thực tế, phức tạp |
| xLAM (Salesforce) | ~60k | Đơn tool, schema rõ ràng |
| ToolACE (Huawei) | ~11k | Tool call chuẩn, ít noise |

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
data/raw/{glaive,toolbench,xlam,toolace}/
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
3. **Arguments khớp schema**: đúng type, đủ required fields.
4. **Query dịch tự nhiên**: LLM judge đánh giá.
5. **Description dịch đúng nghĩa**: LLM judge đánh giá.

## 9. Quyết định đang chờ

- **[ ]** Số lượng tool trong benchmark (10? 50? 100?).
- **[ ]** Tỷ lệ train/val/test chính xác.
- **[ ]** Có dùng negative sampling trong train (cho Bi-Encoder) không, tỷ lệ bao nhiêu.
- **[ ]** Có augment data không (paraphrase query, swap synonym …).
