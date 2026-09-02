# Methodology — Phương pháp nghiên cứu

## 1. Tổng quan phương pháp

Đề tài thực hiện theo hướng **nghiên cứu thực nghiệm + so sánh 2 phương pháp**:

1. **Khảo sát** các phương pháp Tool Calling hiện có (Toolformer, Gorilla, ToolBench, ToolLLM, ToolACE, xLAM, AutoTool, OpenAI FC, Gemini FC).
2. **Xây dựng dữ liệu** tiếng Việt (thu thập → chuẩn hóa → dịch → QC → benchmark).
3. **Triển khai Method 1**: Fine-tune Qwen3.5 nhỏ (2B/4B) end-to-end cho tool calling (theo Ersoy et al. 2025).
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

- **Format**: native `messages` + `tools` + structured `tool_calls`.
- **System prompt**: prompt nền; danh sách tool truyền riêng qua `tools=`.
- **User**: query tiếng Việt.
- **Assistant output**: checkpoint-native `<tool_call><function=...>...</function></tool_call>` hoặc normal answer.
- **Model**: checkpoint Qwen3.5 post-trained `unsloth/Qwen3.5-2B` / `unsloth/Qwen3.5-4B`; không dùng checkpoint `-Base`.
- **Framework**: Unsloth QLoRA/SFT, assistant-only loss.
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

Sau pairing, schema validation và scenario dedup, revision frozen
`2026-09-02-full-dedup-seed42` có `77,028` paired records: `18,210` Glaive,
`58,818` xLAM; trong đó `4,817` là negative. Split cố định là
`61,615/7,701/7,712` theo train/val/test. Các con số chính thức lấy từ
`data/benchmark_core/<revision>/metadata.json`.

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
- **Method 1**: convert sang native `messages`/`tools`/`tool_calls`, sau đó dùng chat template của checkpoint.
- **Method 2**: dùng trực tiếp (query → Bi-Encoder, query + param_schema → Cross-Encoder).

### 3.3 Pipeline xây dựng

1. **Collect**: tải dataset từ HuggingFace.
2. **Translate**: Qwen-MT qua Alibaba OpenAI-compatible API.
3. **Normalize**: schema về JSON Schema chuẩn, map xLAM types.
4. **QA**: rule check + LLM judge 5%.
5. **Build benchmark**: pairing + validation + dedup → frozen revision, split 80/10/10.
6. **Convert**: training rows → native `messages`/`tools`/`tool_calls` cho Method 1.

---

## 4. Phương pháp huấn luyện

### 4.1 Method 1 — Native Tool-Call SFT

- **Model**: checkpoint Qwen3.5 post-trained `unsloth/Qwen3.5-2B` / `unsloth/Qwen3.5-4B`; không dùng checkpoint `-Base`.
- **Framework**: Unsloth QLoRA/SFT.
- **Serialization**: `apply_chat_template(..., tools=..., enable_thinking=False)` từ exact checkpoint.
- **Loss**: chỉ assistant response tokens; system/user/tool definitions được mask.
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

### 5.1 Metric chung

- **Tool Accuracy**: tỉ lệ chọn đúng tool name.
- **Arg F1 / EM**: argument-level F1 và Exact Match.
- **End-to-end accuracy**: tool name + all arguments đúng.
- **Latency** (ms/query).
- **Cost** ($/1k queries) — ước lượng cho API baselines.

### 5.2 Metric riêng Method 1

- **ArgA** (Argument Population Accuracy, Ersoy et al.): tỉ lệ function call có cả tên tool và tất cả arguments chính xác, tính trên positive cases (có tool call).

Negative core records giữ `function_calls=[]`. Native training row không có
`tool_calls` và dùng assistant content bình thường; không dùng marker
`<no_tool_call>`. Output parser xem output không có call hợp lệ là negative,
nhưng output malformed vẫn là invalid.

### 5.3 Metric riêng Method 2

- **Recall@k** (k=1,3,5,10): retrieval.
- **Span F1 (SQuAD-style)**: extraction.
- **has_value_acc**: % params predict đúng có/không có giá trị.

### 5.4 Bảng so sánh chính

| Method | Tool Acc | Arg F1 | Latency (ms) | Cost/1k |
|---|---|---|---|---|
| Method 1: SLM (ours) | … | … | … | … |
| Method 2: Bi+Cross (ours) | … | … | … | … |
| OpenAI FC (gpt-4o-mini) | … | … | … | … |
| Gemini FC (gemini-1.5-flash) | … | … | … | … |

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
