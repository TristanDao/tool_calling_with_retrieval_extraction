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

### 4.1 Nguồn dữ liệu (EN) — chỉ dùng 2 nguồn chính

- **Glaive Function Calling v2** (~110k samples, đa dạng tool, có multi-turn).
- **xLAM** (Salesforce, ~60k samples, function call chuẩn, schema rõ ràng).

> Quyết định chỉ dùng 2 nguồn: Glaive + xLAM. Các dataset khác (ToolBench, ToolACE) không dùng trong khóa luận này. Lý do: dataset đủ lớn, schema rõ ràng, đa dạng domain.

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
- **Framework**: **FlagEmbedding** (BAAI official).
- **Loss**: **MultipleNegativesRankingLoss** (contrastive) với in-batch negatives.
- **Hard negative mining**: dùng chính Bi-Encoder retrieve top-k sai → dùng làm hard negative.
- **Hyperparameter**: lr=2e-5, batch=32, epochs=3.
- **Metric đánh giá**: Recall@1, Recall@5, MRR, NDCG@10.

### 5.2 Cross-Encoder (Parameter Extraction) — Hierarchical Span Prediction

- **Base model**: `BAAI/bge-m3` (encoder) + custom Hierarchical heads.
- **Framework**: HuggingFace Transformers (cần custom head, FlagEmbedding không hỗ trợ).
- **Input (BERT-QA style)**: `[CLS] <query> [SEP] Param=<name>. Desc=<desc>. Type=<type>[. Enum=<v1>|<v2>|...] [SEP]`
  - Query làm context, param schema làm question.
  - Per-parameter forward pass (tool có N params = N passes).
  - max_length=1024, truncation="only_first" (cắt query nếu cần).
- **Architecture**: shared BGE-M3 encoder + **Hierarchical heads** (schema-driven routing, không cần type-prediction head).
  - **Head 1: `has_value` (BCE)** — phân biệt null vs có giá trị.
  - **Sub-head 2a: Span** (string/number) — 2 đầu `(start, end)` chỉ trên query tokens.
  - **Sub-head 2b: Enum** — N-way classification.
  - **Sub-head 2c: Boolean** — 2-way (true/false).
- **Loss (gated multi-task)**:
  ```
  L = BCE(has_value)
    + 𝟙[has_value=1, type∈{string,number}] · (CE_start + CE_end)
    + 𝟙[has_value=1, type=enum]           · CE(enum)
    + 𝟙[has_value=1, type=boolean]        · CE(boolean)
  ```
  - Default weight = 1.0 cho mỗi sub-loss; có thể tune nếu imbalance.
- **Hyperparameter**: lr=2e-5, batch=16, max_len=1024, epochs=3.
- **Metric đánh giá**:
  - `has_value_acc`: % params predict đúng có/không có giá trị.
  - `span_F1` (SQuAD-style) cho string/number, restricted trên query tokens.
  - `span_EM` (Exact Match).
  - `enum_acc`, `boolean_acc`.
  - `end_to_end_F1`: has_value đúng + sub-head đúng.
  - `argument_level_F1/EM`: aggregate trên toàn bộ arguments của 1 query.
- **Label generation rule-based** (`src/models/crossencoder/label_generator.py`):
  - String/number: substring match gold value trong query → (start, end) token positions.
  - Enum: match gold value với enum options → enum index.
  - Boolean: rule "có"/"không"/"muốn"/... → true/false; phủ định ngầm skip sample.
  - Null: param optional + không có trong gold arguments → has_value=0.
- **Tại sao Hierarchical + BERT-QA**:
  - **Null tách bạch khỏi type**: has_value head học "absence of value" (state) thay vì "null là 1 loại value" → training ổn định.
  - **Schema-driven routing**: type đã có trong input, model tập trung sub-head tương ứng → ít confusion, dễ debug.
  - **Khớp pre-train BGE-M3**: BERT-QA format quen thuộc, span head áp dụng tự nhiên.
  - **1 forward pass / param** → low latency.
  - **Span bắt buộc có trong query** → ít hallucination.

### 5.3 Baselines (API-based)

- **OpenAI Function Calling** (`gpt-4o-mini`): API generative LLM.
- **Google Gemini Function Calling** (`gemini-1.5-flash`): API generative LLM.
- Cả 2 dùng **JSON Schema của tool** làm input (giống Cross-Encoder) nhưng sinh JSON tự do qua autoregressive generation.

> **Ngoài scope**: Local LLM baseline (Qwen2.5/Llama Unsloth + vLLM) đã bỏ do timeline 3 tháng.

## 6. Phương pháp đánh giá

### 6.1 Metric cho Retrieval

- **Recall@k** (k=1, 3, 5, 10): có tool đúng trong top-k không.
- **MRR** (Mean Reciprocal Rank).
- **NDCG@k**.

### 6.2 Metric cho Parameter Extraction (Span Prediction)

- **Type accuracy**: % params predict đúng type (span/enum/null).
- **Span F1 (SQuAD-style)**: chuẩn metric cho span prediction, computed trên (start, end) coordinates.
- **Span EM (Exact Match)**: % spans khớp 100% với gold span.
- **Enum accuracy**: % enum values khớp gold.
- **End-to-end F1**: type đúng + (span đúng HOẶC enum đúng).
- **Argument-level F1/EM**: aggregate trên toàn bộ arguments của 1 query.

### 6.3 Metric cho End-to-End Pipeline

- **End-to-end accuracy**: tool name đúng + tất cả arguments khớp label.
- **End-to-end F1**: argument-level F1 aggregate trên toàn sample.
- **Latency** (ms/query).
- **Throughput** (queries/sec).
- **Cost** ($/1k queries) — ước lượng cho OpenAI/Gemini API.

### 6.4 So sánh

Bảng so sánh chính sẽ gồm các hàng:

| Method | Tool Acc | Arg F1 | Span F1 | Latency | Cost/1k |
|---|---|---|---|---|---|
| Pipeline (ours) | … | … | … | … | … |
| OpenAI FC | … | … | … | … | … |
| Gemini FC | … | … | … | … | … |

---

## 6.5 Stress Test (RAG-MCP inspired) — Phase 7

### Mục đích
Đánh giá khả năng **scale** của pipeline khi tool pool tăng lên.
Lấy cảm hứng từ **RAG-MCP** (arXiv:2505.03275) — biến thể của Needle-in-a-Haystack cho tool selection.

### Concept (mượn từ paper)
- Mỗi trial: 1 **ground-truth tool** + **(N-1) distractor tools**.
- Vary `N` (số candidate tools) và đo **degradation curve**.
- 4 metric chính: selection accuracy, task success, prompt token usage, latency.

### Điểm khác biệt với paper
- Paper dùng `random` distractor trên generic MCP (web search).
- Đề tài dùng `random` + `same_domain` distractor trên **domain-specific VI function calling**.
- Paper test 1 model end-to-end. Đề tài tách **retrieval vs extraction** → đo riêng từng thành phần.

### Setup
- **Tool pool**: gộp unique tools từ Glaive + xLAM (sau dịch VI).
  Lưu trong `data/benchmark_vi/tool_pool.json` (~500-2000 tools).
- **Anchors**: 200 samples từ `benchmark_vi/test.jsonl`.
  Lưu trong `data/processed/stress_test/anchors.jsonl`.
  Đa dạng `feature_group` để cover nhiều domain.
- **N values**: `[3, 10, 50, 100, 500, 1000]` (số candidate tools).
- **Distractor strategies**:
  - `random`: random từ tool pool, loại trừ ground truth.
  - `same_domain`: random từ cùng `feature_group` với ground truth.
- **Augmented test set**: mỗi `(anchor, N, strategy)` → 1 test instance.
  Lưu trong `data/processed/stress_test/augmented/`.
  Tổng: 200 × 6 × 2 = **2,400 test instances**.

### Protocol

```
for N in [3, 10, 50, 100, 500, 1000]:
    for strategy in [random, same_domain]:
        for anchor in anchors:
            distractors = sample(tool_pool, N-1, strategy, exclude=ground_truth)
            candidate_tools = [ground_truth] + distractors
            instance = {query, ground_truth, candidate_tools, gold_arguments}
            result = run_pipeline(instance)  # gồm Bi-Encoder + Cross-Encoder
            run_baselines(instance)  # OpenAI, Gemini, Qwen/Llama
            collect_metrics(result, baseline_results)
```

### Metric
- `retrieval_recall@1`: Bi-Encoder lấy đúng ground truth top-1 không.
- `end_to_end_accuracy`: cả retrieval + extraction đều đúng.
- `latency_p50`, `latency_p95`: tail latency.
- `tokens_consumed`: số token đưa vào LLM (cho OpenAI/Gemini, đây là chi phí chính).
- **Đặc biệt**: đo riêng `extraction_acc_given_correct_tool` (khi biết trước tool đúng, Cross-Encoder accuracy bao nhiêu).

### Output
- **Figure**: Accuracy vs N cho 2 strategies (2 line) + 2 baselines (OpenAI FC, Gemini FC) → tổng 4-5 line.
- **Table**: Latency tại N=100 và N=1000.
- File lưu trong `results/tables_figures/stress_test/`.
- Notebook phân tích: `notebooks/07_stress_test_results.ipynb`.

### So sánh kỳ vọng
- Pipeline (Bi-Encoder + Cross-Encoder): **giảm chậm** nhờ retrieval lọc trước.
- OpenAI FC / Gemini FC: **giảm nhanh** khi N tăng vì context dài.
- Local LLM (Qwen2.5, Llama-3.1): tương tự OpenAI/Gemini.
- Khoảng cách giữa pipeline và baselines = **bằng chứng thuyết phục** cho đề tài.

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
