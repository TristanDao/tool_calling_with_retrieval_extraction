# Architecture — Tool Calling VI: So sánh 2 phương pháp

## 1. Tổng quan hệ thống

Hệ thống so sánh **2 phương pháp** Tool Calling tiếng Việt + 2 API baselines. Cả 4 được đánh giá trên cùng benchmark VI.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│             FROZEN CORE BENCHMARK (paired EN/VI, single-turn + multi-call)        │
│       data/benchmark_core/<revision>/{en,vi}/{train,val,test}.jsonl               │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   METHOD 1: SLM     │ │  METHOD 2: Bi-Enc   │ │    BASELINES        │
│   End-to-End        │ │  + Cross-Enc        │ │                     │
│                     │ │                     │ │  OpenAI FC           │
│  Qwen3.5 2B/4B      │ │  Bi-Encoder         │ │  (gpt-4o-mini)       │
│  + Unsloth QLoRA    │ │  (BGE-M3 + MNRL)    │ │                     │
│                     │ │    │                │ │  Gemini FC           │
│  ┌───────────────┐  │ │    ▼                │ │  (gemini-1.5-flash)  │
│  │ Instruction   │  │ │  Cross-Encoder      │ │                     │
│  │ Tuning Format │  │ │  (BGE-M3 + Heads)   │ └─────────────────────┘
│  │ (system+user  │  │ │    │                │
│  │  +assistant)  │  │ │    ▼                │
│  └───────────────┘  │ │  Validator          │
│    │                 │ │                     │
│    ▼                 │ │                     │
│  native tool calls   │ │  function_call JSON │
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

Fine-tune checkpoint Qwen3.5 post-trained nhỏ (`unsloth/Qwen3.5-2B` hoặc `unsloth/Qwen3.5-4B`) làm tool selection + parameter extraction trong 1 model. Dự án không dùng các checkpoint có hậu tố `-Base`.
Theo hướng tiếp cận của Ersoy et al. (2025).

```
┌──────────────────────────────────────────────────────────────────┐
│                    METHOD 1: SLM END-TO-END                       │
│                                                                   │
│  Training data: native messages/tool_calls                         │
│  ┌─────────────────────────────────────────────┐                  │
│  │ System: prompt nền; tools truyền qua tools= │                  │
│  │  [{"type":"function","function":{...}}]  │                  │
│  │                                             │                  │
│  │ User: Tôi muốn tìm gia sư Toán ở Hà Nội.    │                  │
│  │                                             │                  │
│  │ Assistant: tool_calls=[{name, arguments}]   │                  │
│  └─────────────────────────────────────────────┘                  │
│                                                                   │
│  Model: Qwen3.5 2B / 4B                                          │
│  Framework: Unsloth QLoRA/SFT                                      │
│  Output: native XML tool call hoặc normal answer                   │
│  Metric chính: ArgA (Ersoy et al.) — end-to-end accuracy          │
└──────────────────────────────────────────────────────────────────┘
```

- **Input**: system prompt (tool list) + user query (VI).
- **Output**: Qwen3.5 native `<tool_call><function=...>...</function></tool_call>` hoặc normal answer.
- **Train**: Unsloth QLoRA/SFT, assistant-only loss, learning rate ~5e-7, cosine schedule.
- **Data**: convert master rows thành native `messages`/`tool_calls` qua `src/data/convert_to_instruction.py`; template được load từ checkpoint.
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
data/normalized_en/{glaive_normalized,xlam_normalized}.jsonl
    │
    ▼
data/translations/*_normalized_vi.jsonl (schema master VI)
   │
    ├─ translate.py (async, K=10, concurrency=12)
   ├─ translate_guidelines.py
   ├─ qa_translation.py
   │
   ▼
src/data/rebuild_benchmark.py  (paired revision + split + validation)
src/data/feature_group_classify.py  (cached labels, fallback Khác)
   │
   ▼
data/benchmark_core/<revision>/
    ├── en/{train,val,test}.jsonl + vi/{train,val,test}.jsonl
    ├── split_manifest.json + manifest.json
    ├── tool_pool.json + tool_schema/<name>.json
    └── .cache/feature_group.json

data/benchmark_vi/  (active export của vi split)
    ├── train.jsonl / val.jsonl / test.jsonl
   │
   ├──► Method 1: convert_to_instruction.py
    │    → data/experiments/{e1,e2,e4,e5}/instruction/train_chat.jsonl
   │
   └──► Method 2: train trực tiếp trên schema master
        → Bi-Encoder: (query, tool_desc) pairs
        → Cross-Encoder: (query, param_schema, label) per-param

Raw translation legacy vẫn được giữ để audit và khôi phục:

data/translations/glaive_vi.jsonl
data/translations/xlam_vi.jsonl
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
