# Architecture — Tool Calling VI: So sánh 2 phương pháp

## 1. Tổng quan hệ thống

Hệ thống so sánh **2 phương pháp** Tool Calling tiếng Việt + 2 API baselines. Cả 4 được đánh giá trên cùng benchmark VI.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     BENCHMARK VI (single-turn + multi-call)                       │
│                          data/benchmark_vi/{train,val,test}.jsonl                 │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   METHOD 1: SLM     │ │  METHOD 2: Bi-Enc   │ │    BASELINES        │
│   End-to-End        │ │  + Cross-Enc        │ │                     │
│                     │ │                     │ │  OpenAI FC           │
│  Qwen2.5 0.5B/1.5B  │ │  Bi-Encoder         │ │  (gpt-4o-mini)       │
│  + LLaMA-Factory    │ │  (BGE-M3 + MNRL)    │ │                     │
│                     │ │    │                │ │  Gemini FC           │
│  ┌───────────────┐  │ │    ▼                │ │  (gemini-1.5-flash)  │
│  │ Instruction   │  │ │  Cross-Encoder      │ │                     │
│  │ Tuning Format │  │ │  (BGE-M3 + Heads)   │ └─────────────────────┘
│  │ (system+user  │  │ │    │                │
│  │  +assistant)  │  │ │    ▼                │
│  └───────────────┘  │ │  Validator          │
│    │                 │ │                     │
│    ▼                 │ │                     │
│  <tool_call>...</>   │ │  function_call JSON │
└─────────────────────┘ └─────────────────────┘
           │                       │
           └───────────────────────┴───────────────────────┐
                                                           ▼
                                               ┌─────────────────────┐
                                               │   EVALUATION        │
                                               │  (compare.py)       │
                                               │                     │
                                               │  Tool Acc, Arg F1,  │
                                               │  Latency, Cost/1k   │
                                               └──────────┬──────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────────┐
                                               │  STRESS TEST        │
                                               │  (Phase 7)          │
                                               │  N=[3,10,50,100,    │
                                               │     500,1000]       │
                                               │  random + same_dom  │
                                               └─────────────────────┘
```

## 2. Method 1 — SLM End-to-End

Fine-tune Qwen2.5 nhỏ (0.5B/1.5B) làm tool selection + parameter extraction trong 1 model.
Theo hướng tiếp cận của Ersoy et al. (2025).

```
┌──────────────────────────────────────────────────────────────────┐
│                    METHOD 1: SLM END-TO-END                       │
│                                                                   │
│  Training data: instruction-tuning format                         │
│  ┌─────────────────────────────────────────────┐                  │
│  │ System: Available tools:                    │                  │
│  │  [{"name":"search_tutors","description":...}]│                 │
│  │                                             │                  │
│  │ User: Tôi muốn tìm gia sư Toán ở Hà Nội.    │                  │
│  │                                             │                  │
│  │ Assistant: <tool_call>                      │                  │
│  │  {"name":"search_tutors",                   │                  │
│  │   "arguments":{"subject":"Toán",            │                  │
│  │               "location":"Hà Nội"}}          │                  │
│  │  </tool_call>                               │                  │
│  └─────────────────────────────────────────────┘                  │
│                                                                   │
│  Model: Qwen2.5 0.5B / 1.5B                                      │
│  Framework: LLaMA-Factory (SFT)                                   │
│  Output: <tool_call>JSON</tool_call> hoặc <no_tool_call>          │
│  Metric chính: ArgA (Ersoy et al.) — end-to-end accuracy          │
└──────────────────────────────────────────────────────────────────┘
```

- **Input**: system prompt (tool list) + user query (VI).
- **Output**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` hoặc `<no_tool_call>`.
- **Train**: LLaMA-Factory instruction tuning, learning rate ~5e-7, cosine schedule.
- **Data**: convert từ schema master qua `src/data/convert_to_instruction.py`.
- **Reference**: Ersoy et al. (2025) — Tool Calling for Arabic LLMs.

## 3. Method 2 — Bi-Encoder + Cross-Encoder

Pipeline 2 thành phần chuyên biệt, không autoregressive.

```
┌──────────────────────────────────────────────────────────────────┐
│                    METHOD 2: BI + CROSS ENCODER                   │
│                                                                   │
│  user query (VI)                                                  │
│       │                                                           │
│       ▼                                                           │
│  ┌────────────┐    top-k tools     ┌─────────────────┐            │
│  │  BI-       │ ─────────────────► │   CROSS-        │            │
│  │  ENCODER   │                    │   ENCODER       │            │
│  │ (BGE-M3)   │                    │ (BGE-M3 + Heads)│            │
│  └────────────┘                    └────────┬────────┘            │
│       ▲                                     │                     │
│       │ tool descriptions (VI)              ▼                     │
│       │                              ┌─────────────┐              │
│       │                              │  VALIDATOR  │              │
│       │                              └──────┬──────┘              │
│       │                                     │                     │
│       │                                     ▼                     │
│       │                              function_call JSON           │
│       │                                                           │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 Bi-Encoder — Semantic Tool Retrieval

- **Base model**: `BAAI/bge-m3`.
- **Framework**: `FlagEmbedding`.
- **Loss**: `MultipleNegativesRankingLoss`.
- **Input**: query VI + tool description VI.
- **Metric**: Recall@1, Recall@5, MRR.

### 3.2 Cross-Encoder — Schema-aware Parameter Extraction

- **Base model**: `BAAI/bge-m3` + hierarchical heads.
- **Input format** (BERT-QA): `[CLS] query [SEP] Param=<name>. Desc=<desc>. Type=<type>[. Enum=...] [SEP]`
- **Heads**: `has_value` (binary) + span/enum/boolean sub-heads (schema-driven routing).
- **Per-parameter forward pass**.
- **Metric**: Span F1, Enum accuracy, End-to-end F1.

### 3.3 Validator

- Kiểm tra JSON hợp lệ, enum hợp lệ, required fields đủ, type đúng.

## 4. Data flow

```
data/raw/glaive_raw.jsonl + data/raw/xlam_raw.jsonl
   │
   ├─ collect.py (đã có)
   │
   ▼
data/processed/glaive_single_turn_raw.jsonl (lọc Glaive, giữ system/chat)
    │
    ▼
data/translations/ (Bộ 1 — raw VI)
   │
   ├─ translate.py (async, K=10, concurrency=8)
   ├─ translate_guidelines.py
   ├─ qa_translation.py
   │
   ▼
src/data/normalize_schema.py  (xLAM type → JSON Schema chuẩn)
src/data/build_benchmark.py  (Bộ 1 → schema master + split)
src/data/feature_group_classify.py  (LLM classify unique tools, cache)
   │
   ▼
data/benchmark_vi/
   ├── tool_pool.json
   ├── tool_schema/<name>.json
   ├── train.jsonl / val.jsonl / test.jsonl  (schema master)
   │
   ├──► Method 1: convert_to_instruction.py
   │    → data/benchmark_vi/instruction/{train,val,test}_chat.jsonl
   │
   └──► Method 2: train trực tiếp trên schema master
        → Bi-Encoder: (query, tool_desc) pairs
        → Cross-Encoder: (query, param_schema, label) per-param

Đường chạy thay thế khi cần dịch trực tiếp schema master:

data/normalized_en/glaive_normalized.jsonl
    │
    ▼
scripts/data/run_translate_glaive_normalized.sh
    │
    ▼
data/translations/glaive_normalized_vi.jsonl
```

## 5. Comparison Table

So sánh 4 methods trên benchmark VI:

| Method | Tool Acc | Arg F1 | Latency (ms) | Cost/1k |
|---|---|---|---|---|
| Method 1: SLM (ours) | … | … | … | … |
| Method 2: Bi+Cross (ours) | … | … | … | … |
| OpenAI FC (gpt-4o-mini) | … | … | … | … |
| Gemini FC (gemini-1.5-flash) | … | … | … | … |

## 6. Stress Test (Phase 7)

So sánh cả 4 methods khi N tools tăng [3, 10, 50, 100, 500, 1000], 2 distractor strategies (random + same_domain). 200 anchors × 6 × 2 = 2,400 instances. Plot accuracy vs N.
