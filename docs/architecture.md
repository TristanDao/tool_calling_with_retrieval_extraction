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
- **Base model**: `BAAI/bge-m3` hoặc `intfloat/multilingual-e5-large` (sentence-transformers).
- **Kiến trúc**: 2 tower (query encoder + tool encoder), chia sẻ trọng số, similarity bằng cosine.
- **Input**:
  - Query: `query` (VI)
  - Tool: `name (EN) + description (VI)`
- **Output**: top-k `(tool_name, score)`
- **Loss**: InfoNCE (contrastive) với in-batch negatives + hard negative mining.
- **Module**: `src/models/biencoder/`

### 2.2 Cross-Encoder — Schema-aware Parameter Extraction

- **Mục tiêu**: Cho (query VI, tool schema), sinh ra arguments đúng schema dưới dạng JSON.
- **Base model**: BGE-M3 (encoder-only) + classification/seq2seq head, hoặc multilingual generative model.
- **Input format** (nhiều lựa chọn — sẽ quyết sau khi thử):
  - **Option A — JSON Schema dump**: `[CLS] query [SEP] {json_schema} [SEP]`
  - **Option B — Flattened**: `[CLS] query [SEP] name: search_tutors | desc: ... | params: subject (string, ...), location (string, ...) [SEP]`
  - **Option C — Hybrid**: JSON Schema + thêm natural-language description cho mỗi param
- **Output**: JSON `{"tool": "...", "arguments": {...}}` (chuỗi rồi parse, hoặc structured head).
- **Loss**: Token-level cross-entropy trên chuỗi JSON output.
- **Module**: `src/models/crossencoder/`
- **Helper**: `src/models/crossencoder/schema_formatter.py` (chuyển schema thành format A/B/C).

### 2.3 Validator

- **Mục tiêu**: Đảm bảo JSON hợp lệ + khớp schema.
- **Thư viện**: `jsonschema` (Python) hoặc `pydantic`.
- **Kiểm tra**:
  - JSON parse được không
  - Đúng type (string, number, boolean, array, object)
  - Required fields đủ không
  - Enum values hợp lệ không
- **Module**: `src/pipeline/validator.py`

---

## 3. Data flow chi tiết

```
                    ┌──────────────────────────────────────────────┐
                    │   data/raw/  (Glaive, ToolBench, xLAM, …)   │
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
| Qwen2.5 / Llama-3.1 local | Local generative LLM (Unsloth LoRA + vLLM) | GPU local |

Cả 3 baseline đều dùng **JSON Schema của tool** làm input (giống Cross-Encoder) nhưng sinh JSON tự do qua autoregressive generation.

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
