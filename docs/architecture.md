# Architecture — Tool Calling VI Pipeline

## 1. Tổng quan hệ thống

Hệ thống Tool Calling tiếng Việt gồm **2 thành phần chuyên biệt** + **1 validator**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE END-TO-END                            │
│                                                                              │
│  ┌────────────┐    top-k tools     ┌─────────────────┐    JSON args         │
│  │  BI-       │ ─────────────────► │   CROSS-        │ ───────────────┐    │
│  │  ENCODER   │                    │   ENCODER       │                │    │
│  │  (Retrieval)│                   │  (Extraction)   │                ▼    │
│  └────────────┘                    └─────────────────┘      ┌─────────────┐│
│        ▲                                   ▲                │  VALIDATOR  ││
│        │                                   │                │  (JSON +    ││
│        │                                   │                │   Schema)   ││
│        │ query (VI)                        │ tool schema    └──────┬──────┘│
│        │ + tool pool                       │                      │       │
└────────┼───────────────────────────────────┼──────────────────────┼───────┘
         │                                   │                      │
         ▼                                   ▼                      ▼
   user input                          tool_schema.json        {name, args}
                                                                       │
                                                                       ▼
                                                               (ready to call)
```

---

## 2. Các thành phần

### 2.1 Bi-Encoder — Semantic Tool Retrieval

- **Mục tiêu**: Cho 1 query VI, trả về top-k tool phù hợp nhất.
- **Base model**: `BAAI/bge-m3`.
- **Framework**: **FlagEmbedding** (BAAI official).
- **Kiến trúc**: 2 tower (query encoder + tool encoder), chia sẻ trọng số, similarity bằng cosine.
- **Input**:
  - Query: `query` (VI)
  - Tool: `name (EN) + description (VI)`
- **Output**: top-k `(tool_name, score)`
- **Loss**: **MultipleNegativesRankingLoss** (contrastive) với in-batch negatives + hard negative mining.
- **Module**: `src/models/biencoder/`

### 2.2 Cross-Encoder — Schema-aware Parameter Extraction (Hierarchical Span Prediction)

- **Mục tiêu**: Cho (query VI, tool schema), trích xuất arguments đúng schema.
- **Base model**: `BAAI/bge-m3` (encoder).
- **Framework**: HuggingFace Transformers (vì cần custom head, FlagEmbedding không hỗ trợ).
- **Input format (BERT-QA style)**: `[CLS] <query> [SEP] Param=<name>. Desc=<desc>. Type=<type>[. Enum=<v1>|<v2>|...] [SEP]`
  - **Query** đóng vai trò "context" (chứa answer span).
  - **Parameter schema** đóng vai trò "question" (mô tả cần trích gì).
  - **Per-parameter forward pass**: 1 parameter = 1 forward pass; tool có N params = N passes.
  - **max_length = 1024**, `truncation="only_first"` (cắt query nếu quá dài, giữ nguyên schema question).
  - Ví dụ: `[CLS] Tôi muốn tìm gia sư Toán ở Hà Nội. [SEP] Param=subject. Desc=Môn học. Type=string [SEP]`
- **Architecture**: Shared BGE-M3 encoder + **Hierarchical heads** (schema-driven routing).
  - **Head 1: `has_value` (binary, BCE)** — phân biệt null (param optional + user không cung cấp) vs có giá trị.
  - **Sub-head 2a: Span (string/number)** — 2 đầu tuyến tính độc lập `(start, end)` chỉ tính trên query tokens.
  - **Sub-head 2b: Enum** — N-way classification chọn giá trị từ danh sách enum trong schema.
  - **Sub-head 2c: Boolean** — 2-way classification (true/false).
  - **Type head: KHÔNG CÓ** — type lấy từ schema question, không cần model học/predict.
- **Loss (gated multi-task)**:
  ```
  L = BCE(has_value, has_value_label)   # always
    + 𝟙[has_value=1 AND type=string|number] · (CE(span_start) + CE(span_end))
    + 𝟙[has_value=1 AND type=enum]      · CE(enum_logits)
    + 𝟙[has_value=1 AND type=boolean]   · CE(boolean_logits)
  ```
- **Module**: `src/models/crossencoder/`
  - `heads.py` — `CrossEncoderHeads` (has_value + span + enum + boolean)
  - `losses.py` — `HierarchicalLoss` (gated theo schema type)
  - `data_collator.py` — tokenize + align span labels theo token positions
  - `inference.py` — convert logits → JSON arguments
  - `model.py` — `CrossEncoderForExtraction` (BGE-M3 + heads wrapper)
  - `label_generator.py` — rule-based label generation cho string/enum/boolean/null
- **Tại sao Hierarchical heads + BERT-QA format**:
  - **Schema-driven routing**: type đã có sẵn trong input, model tập trung vào span/enum/boolean → ít confusion hơn.
  - **Null ≠ type**: has_value head tách "absence" (state) khỏi "value type" (semantic) → training ổn định hơn.
  - **BERT-QA alignment**: query làm context, schema làm question — khớp pre-train objective của BGE-M3, span head áp dụng tự nhiên.
  - **1 forward pass / param** (nhanh, low latency).
  - **Span phải có trong query** → ít hallucination.
  - **F1/EM evaluation chuẩn** (SQuAD-style cho span, accuracy cho enum/boolean).

### 2.3 Validator

- **Mục tiêu**: Đảm bảo arguments hợp lệ + khớp schema.
- **Thư viện**: `jsonschema` (Python) hoặc `pydantic`.
- **Kiểm tra**:
  - Span extracted nằm trong query (start, end hợp lệ)
  - Enum value hợp lệ (có trong danh sách enum)
  - Required fields đủ (không null nếu required)
  - Type đúng (string, number, boolean, array, object)
- **Module**: `src/pipeline/validator.py`

> Cross-Encoder (Span Prediction) đã đảm bảo schema-aware by design → validator chỉ kiểm tra post-processing (consistency, type safety).

---

## 3. Data flow chi tiết

```
                    ┌──────────────────────────────────────────────┐
                    │   data/raw/  (Glaive, xLAM)                  │
                    └──────────────────┬───────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │   src/data/collect.py                        │
                    │   src/data/normalize_schema.py               │
                    │   → data/processed/{tools,queries,params}    │
                    └──────────────────┬───────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │   src/data/translate.py  (Qwen-MT)            │
                    │   + translate_guidelines.py (bảo vệ ID)     │
                    │   + qa_translation.py (LLM judge)            │
                    │   → data/translations/                       │
                    └──────────────────┬───────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │   src/data/build_benchmark.py                │
                    │   → data/benchmark_vi/{train,val,test}.jsonl │
                    └──────────────────┬───────────────────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
        │ Biencoder/train  │  │ CrossEnc/ train  │  │ Baselines (eval) │
        └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                 │                     │                     │
                 ▼                     ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
        │ Biencoder/infer  │  │ CrossEnc/ infer  │  │ OpenAI/Gemini/   │
        └────────┬─────────┘  └────────┬─────────┘  │ Qwen results     │
                 │                     │            └────────┬─────────┘
                 └──────────┬──────────┘                     │
                            ▼                                │
                   ┌──────────────────┐                      │
                   │ pipeline/        │◄─────────────────────┘
                   │   tool_caller.py │
                   │   validator.py   │
                   └────────┬─────────┘
                            ▼
                   ┌──────────────────┐
                   │ evaluation/      │
                   │   compare.py     │
                   └────────┬─────────┘
                            ▼
                   results/tables_figures/
```

---

## 4. Pipeline Tool Caller (end-to-end)

`src/pipeline/tool_caller.py` là orchestrator duy nhất:

```python
def call(query: str) -> FunctionCall:
    # 1. Retrieve top-k tools
    candidates = biencoder.retrieve(query, k=5)
    # 2. Rerank (optional) — sẽ cân nhắc sau
    # 3. Extract args với Cross-Encoder
    for tool in candidates:
        args = crossencoder.extract(query, tool)
        # 4. Validate
        if validator.is_valid(tool, args):
            return FunctionCall(name=tool.name, arguments=args)
    raise NoValidToolCall
```

**Mục tiêu hiệu năng**: latency < generative LLM baseline (vì cả Bi-Encoder + Cross-Encoder đều không autoregressive).

---

## 5. Baselines (so sánh)

| Baseline | Kiểu | Phụ thuộc |
|---|---|---|
| OpenAI FC | API generative LLM | `OPENAI_API_KEY` |
| Gemini FC | API generative LLM | `GEMINI_API_KEY` |

Cả 2 baseline đều dùng **JSON Schema của tool** làm input (giống Cross-Encoder) nhưng sinh JSON tự do qua autoregressive generation.

> **Ngoài scope khóa luận**: Local LLM baseline (Qwen2.5/Llama Unsloth + vLLM) đã bỏ do timeline 3 tháng. Nếu cần mở rộng sau, xem `AGENTS.md` section 10.

---

## 6. Cấu hình (Hydra Structured Config)

`configs/config.yaml` là entry point, các sub-config là Python `@dataclass`:

```python
# configs/config.py
@dataclass
class DataConfig:
    benchmark_dir: str = "data/benchmark_vi"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "test.jsonl"

@dataclass
class BiEncoderConfig:
    model_name: str = "BAAI/bge-m3"
    max_length: int = 512
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3

@dataclass
class CrossEncoderConfig:
    model_name: str = "BAAI/bge-m3"
    max_length: int = 1024  # vì input chứa schema
    schema_format: str = "json_schema"  # json_schema | flattened | hybrid
    batch_size: int = 16
    learning_rate: float = 2e-5

@dataclass
class PipelineConfig:
    retrieval_k: int = 5
    schema_format: str = "json_schema"

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    biencoder: BiEncoderConfig = field(default_factory=BiEncoderConfig)
    crossencoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
```

---

## 7. Mở rộng trong tương lai (ngoài scope đề tài)

- Rerank layer (Cross-Encoder rerank top-k từ Bi-Encoder).
- Multi-tool planning.
- Tool execution engine.
- Dynamic tool creation.
- Real-time API integration.

Các mục này **không nằm trong phạm vi đề tài**, chỉ ghi nhận để mở rộng sau.
