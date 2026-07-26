# Methodology — Phương pháp nghiên cứu

## 1. Tổng quan phương pháp

Đề tài thực hiện theo hướng **nghiên cứu thực nghiệm + xây dựng hệ thống**, bao gồm:

1. **Khảo sát** các phương pháp Tool Calling hiện có (Toolformer, Gorilla, ToolBench, ToolLLM, ToolACE, xLAM, AutoTool, OpenAI FC, Gemini FC).
2. **Xây dựng dữ liệu** tiếng Việt (thu thập → chuẩn hóa → dịch → QC → benchmark).
3. **Nghiên cứu & huấn luyện** 2 mô hình chuyên biệt (Bi-Encoder + Cross-Encoder).
4. **Tích hợp pipeline** end-to-end.
5. **Đánh giá** trên benchmark tiếng Việt và so sánh với Generative LLM baseline.

---

## 2. Cơ sở lý thuyết

### 2.1 Tool Calling (Function Calling)

Tool Calling là khả năng của LLM/AI Agent **chọn** và **gọi** external function/API dựa trên user intent. Một quy trình Tool Calling hoàn chỉnh gồm:

1. **Tool selection**: chọn tool phù hợp từ tập tool.
2. **Parameter extraction**: trích xuất tham số từ user query theo Tool Schema.
3. **Validation**: kiểm tra JSON hợp lệ + khớp schema.
4. **Execution**: gọi API thực tế.

Phạm vi đề tài: bước 1, 2, 3. (Bước 4 nằm ngoài scope.)

### 2.2 Semantic Retrieval với Bi-Encoder

**Bi-Encoder** mã hóa query và tool **độc lập** thành vector, sau đó tính cosine similarity:

```
sim(q, t) = cos(E_q(q), E_t(t))
```

- **Ưu điểm**: pre-compute embedding của tools → truy vấn cực nhanh (nearest neighbor search).
- **Nhược điểm**: query và tool được encode riêng, không "thấy" nhau → có thể miss semantic match tinh tế.
- **Phù hợp**: giai đoạn retrieval top-k khi tool pool lớn.

### 2.3 Cross-Encoder cho Parameter Extraction

**Cross-Encoder** cho (query, tool_schema) đi qua chung 1 encoder, output tương tác trực tiếp giữa 2 phía.

- **Ưu điểm**: hiểu sâu quan hệ query ↔ schema → extraction chính xác hơn.
- **Nhược điểm**: không pre-compute được → chậm hơn Bi-Encoder nếu apply cho toàn bộ tools.
- **Phù hợp**: chạy sau Bi-Encoder trên top-k candidates.

### 2.4 Tại sao tách 2 thành phần?

| Tiêu chí | End-to-end Generative LLM | Pipeline 2 thành phần |
|---|---|---|
| Latency | Cao (autoregressive) | Thấp (Bi-Encoder retrieval + Cross-Encoder extraction) |
| Cost | Cao (token-based) | Thấp (forward pass cố định) |
| Scalability (nhiều tools) | Kém (context dài) | Tốt (retrieval lọc trước) |
| Hallucination | Cao | Thấp (validator lọc output) |
| Accuracy | Tốt (nếu model mạnh) | Cạnh tranh (chuyên biệt) |

---

## 3. Pipeline đề xuất

```
query (VI)
   │
   ▼
┌──────────────────┐
│  Bi-Encoder      │  embedding top-k
│  (Retrieval)     │ ─────────────────────┐
└──────────────────┘                      │
                                          ▼
                              ┌──────────────────────┐
                              │  Cross-Encoder       │
                              │  (Schema-aware       │
                              │   Parameter Extract) │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  JSON Schema         │
                              │  Validator           │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              function_call (hợp lệ)
```

---

## 4. Dataset & Benchmark

### 4.1 Nguồn dữ liệu (EN)

- **Glaive Function Calling v2** (~110k samples)
- **ToolBench** (API-Bank, RestBench)
- **xLAM** (Salesforce)
- **ToolACE** (Huawei)

### 4.2 Pipeline xây dựng benchmark tiếng Việt

1. **Collect**: tải dataset từ HuggingFace / GitHub repo gốc.
2. **Normalize**: chuẩn hóa schema về format chung (xem `docs/benchmark.md`).
3. **Translate**: dùng **Qwen-MT** (Alibaba, 1M token context) qua DashScope API.
   - Áp dụng **Translation Guidelines** (`docs/translation_guidelines.md`):
     - KHÔNG dịch: function name (snake_case), argument keys, identifier rõ ràng.
     - CHỈ dịch: user query, tool description, argument values (nếu là natural language).
4. **QA**: LLM judge tự động kiểm tra:
   - Function name + argument keys còn nguyên vẹn (string match).
   - JSON structure không bị phá.
   - Translation ngữ nghĩa hợp lý.
5. **Build benchmark**: chia train/val/test, sinh file jsonl.

### 4.3 Schema chuẩn của benchmark

Mỗi sample:

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

---

## 5. Phương pháp huấn luyện

### 5.1 Bi-Encoder (Retrieval)

- **Base model**: `BAAI/bge-m3` (multilingual, hỗ trợ tiếng Việt tốt).
- **Loss**: InfoNCE (contrastive) với in-batch negatives.
- **Hard negative mining**: dùng chính Bi-Encoder retrieve top-k sai → dùng làm hard negative.
- **Hyperparameter**: lr=2e-5, batch=32, epochs=3.
- **Metric đánh giá**: Recall@1, Recall@5, MRR, NDCG@10.

### 5.2 Cross-Encoder (Parameter Extraction)

- **Base model**: `BAAI/bge-m3` (encoder) + generation/classification head.
- **Input**: `[CLS] query [SEP] tool_schema_formatted [SEP]`.
- **Output**: chuỗi JSON `{"tool": name, "arguments": {...}}`.
- **Loss**: cross-entropy trên token output.
- **Hyperparameter**: lr=2e-5, batch=16, max_len=1024 (chứa schema), epochs=3.
- **Metric đánh giá**: JSON validity, schema-validity, argument-level F1/EM.

### 5.3 Local LLM Baseline (Unsloth)

- **Base model**: `Qwen/Qwen2.5-7B-Instruct` hoặc `meta-llama/Llama-3.1-8B-Instruct`.
- **Method**: LoRA fine-tune trên benchmark_vi/train.
- **Serving**: vLLM.
- **Mục đích**: so sánh công bằng với pipeline 2 thành phần dưới cùng điều kiện local.

---

## 6. Phương pháp đánh giá

### 6.1 Metric cho Retrieval

- **Recall@k** (k=1, 3, 5, 10): có tool đúng trong top-k không.
- **MRR** (Mean Reciprocal Rank).
- **NDCG@k**.

### 6.2 Metric cho Parameter Extraction

- **JSON validity rate**: % output parse được thành JSON.
- **Schema validity rate**: % output khớp schema.
- **Argument-level Exact Match (EM)**.
- **Argument-level F1**.

### 6.3 Metric cho End-to-End Pipeline

- **End-to-end accuracy**: tool name đúng + tất cả arguments khớp label.
- **Latency** (ms/query).
- **Throughput** (queries/sec).
- **Cost** ($/1k queries) — ước lượng cho OpenAI/Gemini API; tính theo GPU-hour cho local.

### 6.4 So sánh

Bảng so sánh chính sẽ gồm các hàng:

| Method | Tool Acc | Arg F1 | JSON Valid | Latency | Cost/1k |
|---|---|---|---|---|---|
| Pipeline (ours) | … | … | … | … | … |
| OpenAI FC | … | … | … | … | … |
| Gemini FC | … | … | … | … | … |
| Qwen2.5 local | … | … | … | … | … |
| Llama-3.1 local | … | … | … | … | … |

---

## 7. Reproducibility

- `src/utils/seed.py` set seed toàn cục (Python, NumPy, PyTorch, CUDA).
- Hydra config log lại toàn bộ tham số.
- `wandb` (optional) log metrics real-time.
- `requirements.txt` / `pyproject.toml` pin version.

---

## 8. Hạn chế & hướng phát triển

### Hạn chế trong phạm vi đề tài
- Chưa xử lý multi-tool planning (mỗi query chỉ 1 tool).
- Chưa benchmark trên số tool cực lớn (>1000).
- Translation chưa đạt chất lượng "human-level" vì dùng MT tự động.

### Hướng phát triển (ngoài scope)
- Rerank layer sau retrieval.
- Multi-tool planning với constraint solver.
- Human-in-the-loop translation QC.
- Continual learning khi có tool mới.
