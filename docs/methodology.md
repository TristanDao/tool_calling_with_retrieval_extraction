# Methodology — Phương pháp nghiên cứu

## 1. Tổng quan phương pháp

Đề tài thực hiện theo hướng **nghiên cứu thực nghiệm + so sánh 2 phương pháp**:

1. **Khảo sát** các phương pháp Tool Calling hiện có (Toolformer, Gorilla, ToolBench, ToolLLM, ToolACE, xLAM, AutoTool, OpenAI FC, Gemini FC).
2. **Xây dựng dữ liệu** tiếng Việt (thu thập → chuẩn hóa → dịch → QC → benchmark).
3. **Triển khai Method 1**: Fine-tune Qwen2.5 nhỏ (0.5B/1.5B) end-to-end cho tool calling (theo Ersoy et al. 2025).
4. **Triển khai Method 2**: Bi-Encoder (retrieval) + Cross-Encoder (extraction).
5. **Đánh giá & so sánh** 4 methods trên benchmark VI: Method 1 (SLM), Method 2 (Bi+Cross), OpenAI FC, Gemini FC.

---

## 2. Cơ sở lý thuyết

### 2.1 Tool Calling (Function Calling)

Tool Calling là khả năng của LLM/AI Agent **chọn** và **gọi** external function/API dựa trên user intent. Một quy trình Tool Calling hoàn chỉnh gồm:

1. **Tool selection**: chọn tool phù hợp từ tập tool.
2. **Parameter extraction**: trích xuất tham số từ user query theo Tool Schema.
3. **Validation**: kiểm tra JSON hợp lệ + khớp schema.
4. **Execution**: gọi API thực tế.

Phạm vi đề tài: bước 1, 2, 3. (Bước 4 nằm ngoài scope.)

### 2.2 Method 1 — SLM End-to-End Instruction Tuning

**Theo Ersoy et al. (2025)**: Fine-tune một LLM nhỏ (Small Language Model) làm toàn bộ pipeline tool calling.

- **Format**: instruction-tuning (system prompt + user query + assistant response).
- **System prompt**: liệt kê các tool có sẵn (name, description, parameters).
- **User**: query tiếng Việt.
- **Assistant output**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` hoặc `<no_tool_call>`.
- **Model**: Qwen2.5 0.5B / 1.5B.
- **Framework**: LLaMA-Factory (SFT).
- **Metric chính**: **ArgA (Argument Population Accuracy)** — tỉ lệ function call có cả tên tool và toàn bộ tham số chính xác.

**Tại sao chọn SLM**:
- So sánh công bằng với Method 2 (Bi+Cross) — cả 2 đều nhẹ, không dùng LLM thương mại.
- Kiểm chứng hypothesis: "retrieval-based approach (Method 2) có cạnh tranh với fine-tune SLM không?"
- Đúng tinh thần Ersoy et al. — fine-tune open-weight model cho low-resource language.

### 2.3 Method 2 — Bi-Encoder + Cross-Encoder

**Bi-Encoder** mã hóa query và tool **độc lập** thành vector, sau đó tính cosine similarity:

```
sim(q, t) = cos(E_q(q), E_t(t))
```

- **Ưu điểm**: pre-compute embedding của tools → truy vấn cực nhanh.
- **Nhược điểm**: query và tool encode riêng, không "thấy" nhau.

**Cross-Encoder** cho (query, tool_schema) đi qua chung 1 encoder → hiểu sâu quan hệ query ↔ schema.

### 2.4 Tại sao so sánh 2 phương pháp?

| Tiêu chí | Method 1: SLM End-to-End | Method 2: Bi+Cross Encoder |
|---|---|---|
| Cơ chế | Generative (autoregressive) | Retrieval + Span Prediction |
| Latency | Cao hơn (generation) | Thấp (forward pass cố định) |
| Cost | Thấp (local, không API) | Thấp (local) |
| Scalability (nhiều tools) | Kém (context dài) | Tốt (retrieval lọc trước) |
| Hallucination | Có thể sinh JSON sai | Thấp (span prediction trong query) |
| Accuracy | Kỳ vọng tốt (fine-tune) | Cạnh tranh (chuyên biệt) |

---

## 3. Dataset & Benchmark

### 3.1 Nguồn dữ liệu (EN)

- **Glaive Function Calling v2** (~113k samples): multi-turn chat, lọc first turn → 45,593.
- **xLAM** (~60k samples): flat query, multi-call (53%), giữ toàn bộ.

### 3.2 Schema master (single-turn + multi-call)

```json
{
  "id": "glaive_00042",
  "source": "glaive",
  "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
  "function_calls": [
    {"name": "search_tutors", "arguments": {"subject": "Toán", "location": "Hà Nội"}}
  ],
  "tools": [
    {"name": "search_tutors", "description": "Tìm gia sư theo môn và khu vực.", "feature_group": "...", "parameters": {...}}
  ]
}
```

Dùng chung cho cả 2 method:
- **Method 1**: convert sang instruction format (system + user + assistant).
- **Method 2**: dùng trực tiếp (query → Bi-Encoder, query + param_schema → Cross-Encoder).

### 3.3 Pipeline xây dựng

1. **Collect**: tải dataset từ HuggingFace.
2. **Translate**: Qwen-MT qua Alibaba OpenAI-compatible API.
3. **Normalize**: schema về JSON Schema chuẩn, map xLAM types.
4. **QA**: rule check + LLM judge 5%.
5. **Build benchmark**: Bộ 1 → schema master, split 80/10/10.
6. **Convert**: schema master → instruction format cho Method 1.

---

## 4. Phương pháp huấn luyện

### 4.1 Method 1 — SLM Instruction Tuning

- **Model**: Qwen2.5 0.5B / 1.5B.
- **Framework**: LLaMA-Factory (SFT).
- **Learning rate**: ~5e-7, cosine schedule.
- **Metric**: **ArgA** — end-to-end accuracy (tool name + all arguments đúng).
- **Evaluation**: weighted precision/recall per tool + ArgA.

### 4.2 Method 2 — Bi-Encoder (Retrieval)

- **Base model**: `BAAI/bge-m3`.
- **Framework**: FlagEmbedding.
- **Loss**: MultipleNegativesRankingLoss với in-batch negatives.
- **Hard negative mining**: dùng Bi-Encoder retrieve top-k sai.
- **Metric**: Recall@1, Recall@5, MRR.

### 4.3 Method 2 — Cross-Encoder (Parameter Extraction)

- **Base model**: `BAAI/bge-m3` + custom Hierarchical heads.
- **Input format** (BERT-QA): `[CLS] query [SEP] Param=<name>. Desc=<desc>. Type=<type>[. Enum=...] [SEP]`
- **Heads**: `has_value` (BCE) + span/enum/boolean sub-heads (schema-driven routing).
- **Loss (gated multi-task)**:
  ```
  L = BCE(has_value)
    + 𝟙[has_value=1, type∈{string,number}] · (CE_start + CE_end)
    + 𝟙[has_value=1, type=enum]           · CE(enum)
    + 𝟙[has_value=1, type=boolean]        · CE(boolean)
  ```
- **Per-parameter forward pass** (tool có N params = N passes).
- **Metric**: Span F1, Enum accuracy, End-to-end F1.

### 4.4 Baselines (API-based)

- **OpenAI Function Calling** (gpt-4o-mini).
- **Google Gemini Function Calling** (gemini-1.5-flash).

---

## 5. Phương pháp đánh giá

Framework đánh giá dùng cùng một protocol cho Method 1, Method 2 và hai API baseline. Không dùng một composite score có trọng số tùy ý; kết quả được báo cáo theo từng tầng và toàn pipeline.

### 5.1 Call detection

Một mẫu là positive khi gold có ít nhất một phần tử trong `function_calls[]`. Prediction rỗng là no-call. Báo cáo Call Precision, Recall, F1, Hallucinated Call Rate `FP/N_negative` và Missed Call Rate `FN/N_positive`.

### 5.2 Tool retrieval và selection

Retriever được đo bằng micro Recall@K, Full Recall@K cho multi-call và MRR của gold tool đầu tiên. Selection được đo bằng exact match của multiset tên tool trên positive samples, tool micro/macro F1 và Selection Conversion@K khi toàn bộ gold tools đã có trong top-K.

### 5.3 Parameter extraction

Arguments được chuẩn hóa đối xứng cho gold/prediction theo schema: Unicode/whitespace/case, numeric/boolean coercion theo type và alias khai báo. Identifier thuộc schema `string` không bị ép kiểu số. Multi-call cùng tên được ghép cặp tối ưu trước khi tính:

- Normalized Argument Exact Match khi tool set đúng.
- Argument Key Precision/Recall/F1.
- Argument Pair Precision/Recall/F1 trên cặp `(parameter_path, normalized_value)`.
- Value Accuracy trên các parameter cùng key.
- Breakdown theo parameter type.
- Oracle-tool ArgEM từ một file prediction dùng gold tool schema riêng.

### 5.4 Structural validity

Schema Validity kiểm tra tên tool thuộc candidate set, arguments là object, required fields, JSON Schema type/enum/constraint và output parse được. Báo cáo validity theo call, theo sample và prediction-format validity.

### 5.5 End-to-end

Metric chính là **Normalized Function Call Exact Match trên positive samples (N-FCEM-positive)**. Một mẫu chỉ đúng khi multiset function calls, tên tool và toàn bộ normalized arguments đều đúng; thứ tự các call không ảnh hưởng kết quả. `Overall Success` mở rộng N-FCEM bằng cách tính đúng cho negative sample khi prediction là no-call.

### 5.6 Robustness theo tinh thần Track C

Không phân tầng theo phương ngữ. Evaluator breakdown N-FCEM, Tool Set Accuracy, ArgEM và Overall Success theo `source`, seen/unseen tool, domain, difficulty, single/multi-call, candidate-set size, feature group và schema complexity. Với mỗi slice, gap là `max(group score) − min(group score)`.

### 5.7 Efficiency

Từ telemetry từng prediction, báo cáo mean/p50/p95/p99 latency, throughput, input/output tokens mỗi query, cost/query, cost/1k queries, cost/correct call và peak GPU memory. p95 latency và cost/correct là hai chỉ số hiệu quả chính.

### 5.8 Statistical reliability

N-FCEM, Overall Success và các binary headline metrics có bootstrap confidence interval với seed cố định. So sánh hai method trên cùng test IDs dùng exact McNemar test và paired bootstrap confidence interval của chênh lệch.

### 5.9 Bảng so sánh chính

| Method | Call F1 | R@5 | Tool Set Acc | Oracle ArgEM | N-FCEM | Schema Valid | p95 ms | Cost/correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Method 1: SLM (ours) | … | — | … | … | … | … | … | … |
| Method 2: Bi+Cross (ours) | … | … | … | … | … | … | … | … |
| OpenAI FC (gpt-4o-mini) | … | — | … | … | … | … | … | … |
| Gemini FC (gemini-1.5-flash) | … | — | … | … | … | … | … | … |

Định nghĩa field, input/output và hướng dẫn chạy đầy đủ nằm trong `docs/evaluation.md`.
Phần trình bày học thuật có thể sử dụng trực tiếp trong báo cáo nằm tại `docs/evaluation_methodology_thesis.md`.

---

## 6. Stress Test (Phase 7)

### Mục đích
Đánh giá khả năng **scale** của cả 2 method + 2 baselines khi tool pool tăng.

### Setup
- **Tool pool**: `data/benchmark_vi/tool_pool.json`.
- **Anchors**: 200 samples từ `test.jsonl`.
- **N values**: [3, 10, 50, 100, 500, 1000].
- **Distractor strategies**: `random` + `same_domain`.
- **Tổng**: 200 × 6 × 2 = **2,400 instances**.

### Output
- **Figure**: Accuracy vs N cho 4 methods × 2 strategies.
- **Table**: Latency tại N=100 và N=1000.

---

## 7. Reproducibility

- `src/utils/seed.py` set seed toàn cục.
- Hydra config log toàn bộ tham số.
- `wandb` (optional) log metrics.
- `requirements.txt` / `pyproject.toml` pin version.

---

## 8. Hạn chế & hướng phát triển

### Hạn chế
- Chưa xử lý multi-turn conversation.
- Chưa benchmark trên số tool cực lớn (>1000).
- Translation chưa đạt chất lượng "human-level".

### Hướng phát triển (ngoài scope)
- Multi-turn, multi-agent orchestration.
- Rerank layer sau retrieval.
- Continual learning khi có tool mới.
