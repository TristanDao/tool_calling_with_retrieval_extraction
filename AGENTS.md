# AGENTS.md — Cross-session memory for AI coding assistants

> **Mục đích**: File này là "bộ nhớ dài hạn" giữa các phiên làm việc với AI.
> Mọi agent (opencode, Claude, GPT, Cursor, v.v.) **phải đọc file này đầu tiên**
> trước khi bắt đầu bất kỳ thay đổi nào trong repo.
> Cập nhật file này bất cứ khi nào có quyết định kiến trúc, công cụ, hoặc hướng đi mới.

---

## 1. Project Identity

- **Tên dự án**: Tool Calling tiếng Việt — So sánh 2 phương pháp: SLM End-to-End vs Bi-Encoder + Cross-Encoder
- **Repo path**: `/home/thinh/project/UIT/tool_calling_with_retrieval_extraction`
- **Loại**: Đồ án / luận văn UIT (nghiên cứu thực nghiệm + xây dựng hệ thống)
- **Tác giả**: Thinh
- **Trạng thái**: Phase 1 — Data pipeline (pilot translation + master benchmark rebuild; full translation pending)

---

## 2. Problem & Goal

### Vấn đề
Các hệ thống Tool Calling hiện tại (OpenAI FC, Gemini FC, Toolformer, Gorilla, ToolBench,
ToolLLM, ToolACE, xLAM, AutoTool) chủ yếu dựa trên **generative LLM**:
- Chi phí suy luận cao
- Độ trễ lớn (autoregressive generation)
- Hallucination + JSON không hợp lệ
- Hiệu năng suy giảm khi số tool tăng (context quá dài)
- **Ít nghiên cứu / benchmark cho tiếng Việt**

### Mục tiêu
**So sánh 2 phương pháp Tool Calling cho tiếng Việt**:

| Phương pháp | Mô tả | Tham khảo |
|---|---|---|
| **Method 1: SLM End-to-End** | Fine-tune Qwen3.5 (2B/4B) làm tool selection + parameter extraction trong 1 model | Ersoy et al. (2025) |
| **Method 2: Bi-Encoder + Cross-Encoder** | Tách thành 2 thành phần chuyên biệt: Bi-Encoder chọn tool, Cross-Encoder trích xuất tham số | Thiết kế ban đầu |

Cả 2 được so sánh với:
- **OpenAI Function Calling** (gpt-4o-mini)
- **Google Gemini Function Calling** (gemini-1.5-flash)

→ Mục tiêu: **Method 2 cạnh tranh về accuracy, thắng về latency/cost. Method 1 là baseline fine-tune local.** So sánh cả 4 trên benchmark VI.

### Phạm vi
- ✅ Tool Retrieval, Parameter Extraction, JSON validation
- ✅ Benchmark tiếng Việt
- ✅ Single-turn + multi-call (1 query có thể gọi 1+ tool)
- ❌ Multi-turn conversation, tool execution, multi-agent orchestration, dynamic tool creation, real-time integration

---

## 3. Tech Stack (đã chốt)

| Thành phần | Công nghệ |
|---|---|
| Framework chính | **PyTorch + Transformers** |
| Config | **Hydra** với **structured config** (Python `@dataclass`) |
| **Method 1: SLM End-to-End** | **Qwen3.5 2B/4B** + **Unsloth** (QLoRA/SFT) |
| Method 2: Bi-Encoder (Retrieval) | **BGE-M3** + **FlagEmbedding** + **MultipleNegativesRankingLoss** |
| Method 2: Cross-Encoder (Extraction) | **BGE-M3** + **Hierarchical heads** (1 binary `has_value` + schema-driven sub-head: span / enum / boolean), format `[CLS] query [SEP] Param=<name>. Desc=... Type=<type>[. Enum=...] [SEP]` (BERT-QA style) |
| Dịch dataset | **Alibaba OpenAI-compatible API** (qwen3.7-flash / qwen3.7-max) |
| Baseline 1 | **OpenAI Function Calling** (gpt-4o-mini) |
| Baseline 2 | **Google Gemini Function Calling** (gemini-1.5-flash) |

**Lưu ý kiến trúc**:
- Method 1 SLM: dùng Unsloth để QLoRA/SFT checkpoint `unsloth/Qwen3.5-2B` hoặc `unsloth/Qwen3.5-4B`. Training view dùng native `messages` + `tools` + structured `tool_calls`; render bằng chat template của exact checkpoint với `enable_thinking=False`, loss chỉ tính trên assistant response. Negative dùng assistant content bình thường, không dùng `<no_tool_call>`. Giống Ersoy et al. (2025) ở thiết kế SFT, không ở serialization.
- Method 2 Bi-Encoder: dùng `FlagEmbedding` (BAAI official) + `MultipleNegativesRankingLoss`.
- Method 2 Cross-Encoder: **Hierarchical Span Prediction** (BGE-M3 base + custom heads). Schema-driven routing: type lấy từ schema question nên model không cần học/predict type — chỉ activate 1 sub-head phù hợp (span/enum/boolean). 1 binary `has_value` head riêng để phân biệt null (absent) vs có giá trị. Input format BERT-QA: query làm context, param schema làm question.

---

## 4. Data Schema (master canonical — single-turn + multi-call)

Schema master dùng chung cho toàn bộ hệ thống. Cả Method 1 và Method 2 đều dùng cùng 1 nguồn data, khác cách convert khi train.

```json
{
  "id": "glaive_00042",
  "source": "glaive",
  "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
  "function_calls": [
    {
      "name": "search_tutors",
      "arguments": {"subject": "Toán", "location": "Hà Nội"}
    }
  ],
  "tools": [
    {
      "name": "search_tutors",
      "description": "Tìm gia sư theo môn học và khu vực.",
      "feature_group": "Tìm kiếm & Kết nối",
      "parameters": {
        "type": "object",
        "properties": {
          "subject": {"type": "string", "description": "Môn học cần tìm"},
          "location": {"type": "string", "description": "Thành phố hoặc khu vực"}
        },
        "required": ["subject", "location"]
      }
    }
  ]
}
```

### Quy ước bắt buộc

| Trường | Ngôn ngữ | Quy tắc |
|---|---|---|
| `id` | — | Unique string, format `<source>_<index>` (VD: `glaive_00042`, `xlam_00123`) |
| `source` | — | `glaive` hoặc `xlam` |
| `query` | **EN/VI** | User query theo language split |
| `function_calls[].name` | **EN** | snake_case identifier, không dấu |
| `function_calls[].arguments` keys | **EN** | snake_case |
| `function_calls[].arguments` values | VI/EN | tùy natural language hay identifier |
| `tools[].name` | **EN** | snake_case identifier |
| `tools[].description` | **EN/VI** | Mô tả tự nhiên theo language split |
| `tools[].feature_group` | **VI** | Tên nhóm chức năng (do LLM classify, cache theo tool name) |
| `tools[].parameters` | — | JSON Schema chuẩn (`string`/`integer`/`number`/`boolean`/`array`/`object`); enum dùng `type: "string"` + `enum` |

**Lưu ý**: `function_calls[]` là list (multi-call) — 1 query có thể gọi 1+ tool.

### JSON Schema `type` chuẩn

| Schema chuẩn | xLAM raw mapping | Ghi chú |
|---|---|---|
| `"string"` | `"str"`, `"str, optional"` | |
| `"integer"` | `"int"`, `"int, optional"` | |
| `"number"` | `"float"`, `"float, optional"` | |
| `"boolean"` | `"bool"`, `"bool, optional"` | |
| `"array"` | `"list"`, `"List[int]"`, ... | Phải có `items: {type: T}` |
| `"object"` | — | Nested object hiếm gặp |

### Sample count sau frozen pairing và dedup

| Dataset | Input paired | Frozen revision | Ghi chú |
|---|---|---|---|
| Glaive | 60,734 | **18,210** | 45,593 positive + 15,141 negative trước dedup/validation |
| xLAM | 60,000 | **58,818** | Positive single-turn/multi-call |
| **Total** | 120,734 | **77,028 paired** | 944 rejected, 42,762 scenario duplicates |

### Data flow: từ frozen master schema → train views

```
Schema master (data/benchmark_core/<revision>/{en,vi}/*.jsonl)
   │
   ├──► Method 1 (SLM):
   │    src/data/convert_to_instruction.py
   │    → data/experiments/{e1,e2,e4,e5}/instruction/train_chat.jsonl
   │      native Qwen3.5 messages/tools/tool_calls
   │
   └──► Method 2 (Bi+Cross):
        Dùng trực tiếp schema master
        → Bi-Encoder: (query, tool_description) pairs
        → Cross-Encoder: (query, param_schema, label) per-param
```

---

## 5. Translation Guidelines (tóm tắt)

File đầy đủ: `docs/translation_guidelines.md`.

| KHÔNG dịch | CHỈ dịch |
|---|---|
| Function name (snake_case) | User query |
| Argument keys | Tool description |
| Identifier rõ ràng (UUID, brand, proper noun chuẩn) | Argument values (nếu là natural language) |
| Mã tiền tệ, mã thành phố chuẩn hóa | |

**Tuyệt đối giữ cấu trúc JSON** — không thêm/bớt key, giữ format.

---

## 6. Folder Structure

```
tool_calling_with_retrieval_extraction/
├── README.md
├── AGENTS.md                  ← file này
├── pyproject.toml
├── requirements.txt           (sẽ thêm sau)
├── .gitignore
├── .env.example
│
├── configs/                   # Hydra structured config (Python dataclass)
│   ├── data/ model/ pipeline/ baseline/ eval/
│
├── data/
│   ├── raw/                   # EN datasets (Glaive, xLAM)
│   ├── processed/
│   │   └── stress_test/       # ← Phase 7
│   │       ├── anchors.jsonl
│   │       └── augmented/
│   ├── benchmark_core/        # Frozen paired EN/VI revisions (gitignored)
│   │   └── <revision>/{en,vi}/{train,val,test}.jsonl + manifests
│   ├── benchmark_vi/          # Active VI export of the selected revision
│   ├── experiments/            # Method 1 train-only artifacts (gitignored)
│   │   └── e0, e1, e2, e4, e5/
│   ├── legacy/                # Read-only archive of generated pilot outputs
│   ├── translations/          # Qwen-MT logs + QA samples
│   └── statistics/
│
├── src/
│   ├── data/
│   │   ├── collect.py
│   │   ├── normalize_schema.py
│   │   ├── translate.py
│   │   ├── translate_guidelines.py
│   │   ├── qa_translation.py
│   │   ├── build_benchmark.py          # Legacy raw parser compatibility
│   │   ├── rebuild_benchmark.py        # Frozen paired revision builder
│   │   ├── convert_to_instruction.py  # ← Method 1: native Qwen view
│   │   ├── build_tool_pool.py
│   │   ├── extract_anchors.py
│   │   ├── augment_with_distractors.py
│   │   └── stats.py
│   ├── models/
│   │   ├── slm/               # ← Method 1: Qwen3.5 fine-tune (Unsloth)
│   │   ├── biencoder/         # Method 2: Semantic Tool Retrieval
│   │   ├── crossencoder/      # Method 2: Schema-aware Parameter Extraction
│   │   └── baselines/         # OpenAI FC, Gemini FC
│   ├── pipeline/
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── extraction_metrics.py
│   │   ├── latency.py
│   │   ├── cost.py
│   │   ├── throughput.py
│   │   ├── compare.py         # So sánh 4 methods
│   │   ├── stress_test.py
│   │   └── plot_stress_test.py
│   └── utils/
│
├── scripts/                   # data/, train/, serve/, eval/
├── notebooks/                 # 01 EDA + analysis
├── tests/                     # unit tests
├── checkpoints/               # slm/, biencoder/, crossencoder/
├── results/
│   ├── slm/ retrieval/ extraction/ baselines/
│   └── tables_figures/
│       └── stress_test/
├── docs/                      # architecture, methodology, benchmark, translation_guidelines, references
└── logs/                      # train/, eval/
```

---

## 7. Implementation Phases (roadmap)

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Skeleton (folders + .md files) | ✅ Done |
| 1 | Data pipeline (collect, normalize, translate, build benchmark) | ⏳ In progress |
| 2 | Method 2: Bi-Encoder | ⏳ |
| 3 | Method 2: Cross-Encoder | ⏳ (skeleton có sẵn) |
| 4 | Method 1: SLM fine-tune + instruction data | ⏳ |
| 5 | Baselines (OpenAI FC, Gemini FC) | ⏳ |
| 6 | Evaluation & comparison (4 methods) | ⏳ |
| 7 | Stress test (RAG-MCP inspired, so sánh cả 4) | ⏳ |

### Phase detail

**Phase 1 — Data pipeline**:
- Collect → normalize → translate (Bộ 1) → QA → frozen paired revision → native training views
- Output: frozen `data/benchmark_core/<revision>/`, active `data/benchmark_vi/` và train artifacts trong `data/experiments/`

**Phase 2 — Bi-Encoder** (Method 2):
- Train BGE-M3 + MNRL cho tool retrieval
- Metric: Recall@1, Recall@5, MRR

**Phase 3 — Cross-Encoder** (Method 2):
- Train BGE-M3 + hierarchical heads cho parameter extraction
- Metric: Span F1, Enum accuracy, End-to-end F1

**Phase 4 — SLM** (Method 1):
- Fine-tune `unsloth/Qwen3.5-2B/4B` với Unsloth QLoRA/SFT trên native rows
- Format: `messages` + `tools` → assistant structured `tool_calls` hoặc normal answer
- Template load trực tiếp từ checkpoint, loss chỉ tính trên assistant response
- Metric: End-to-end accuracy (giống Ersoy et al. ArgA)

**Phase 5 — Baselines**:
- OpenAI FC (gpt-4o-mini), Gemini FC (gemini-1.5-flash)
- Đo latency, cost, accuracy

**Phase 6 — Evaluation**:
- Bảng so sánh 4 methods: Tool Acc, Arg F1, Latency, Cost/1k
- Cả Method 1 + Method 2 + 2 baselines

**Phase 7 — Stress test**:
- So sánh cả 4 methods khi N tools tăng [3, 10, 50, 100, 500, 1000]
- 2 distractor strategies: random + same_domain
- Tổng: 200 × 6 × 2 = 2,400 instances
- Output: accuracy vs N plot, latency table

### Reference liên quan
- **Ersoy et al. (2025)** — Tool Calling for Arabic LLMs. Phương pháp chính cho Method 1 (SLM fine-tune). Cùng dùng Glaive + xLAM, dịch sang ngôn ngữ đích, fine-tune LLM end-to-end. Đề tài mở rộng thêm Method 2 để so sánh.
- **RAG-MCP (2025)** — Stress test concept với varying N.
- **BFCL** — Tham khảo categories khi xây benchmark.

---

## 8. Conventions (cho AI agents)

### Khi viết code
1. **Không thêm comment trừ khi user yêu cầu**.
2. **Luôn đọc** `docs/translation_guidelines.md` trước khi viết code dịch.
3. **Luôn đọc** `docs/architecture.md` + `docs/methodology.md` trước khi implement model/pipeline.
4. **Hydra config dùng structured config** (Python `@dataclass`), không dùng YAML composing.
5. **Type hints đầy đủ** cho mọi public function.
6. **Mỗi thay đổi kiến trúc phải cập nhật file .md tương ứng**.
7. **Tên file .py đặt theo snake_case**, tên class PascalCase.
8. **Mỗi module Python có docstring mô tả ngắn** ở đầu file.
9. **Reproducibility**: set seed qua `utils/seed.py`.
10. **Không commit data, checkpoint, log, .env** vào git (đã có .gitignore).

### Khi user yêu cầu thêm tính năng
1. Cập nhật section tương ứng trong `AGENTS.md` + `docs/`
2. Cập nhật phase trong roadmap
3. Sau đó mới viết code

### Khi gặp xung đột quyết định cũ
- Mặc định ưu tiên quyết định **mới nhất trong AGENTS.md**
- Nếu nghi ngờ, hỏi user trước khi sửa

---

## 9. Key Files (quick reference)

| File | Mục đích |
|---|---|
| `AGENTS.md` | File này — bộ nhớ cross-session |
| `README.md` | Project overview, quick start |
| `docs/architecture.md` | Sơ đồ 2 pipeline (Method 1 + Method 2) |
| `docs/methodology.md` | Phương pháp nghiên cứu chi tiết |
| `docs/benchmark.md` | Cấu trúc benchmark tiếng Việt |
| `docs/translation_guidelines.md` | Quy tắc dịch |
| `docs/references.md` | Papers & resources |
| `.env.example` | Template biến môi trường |
| `pyproject.toml` | Project metadata + tool config |

---

## 10. Open Questions / Decisions Pending

- **[x] Single-turn + multi-call**: Đã chốt — chỉ lấy first turn từ Glaive, giữ multi-call từ xLAM. Frozen revision hiện hành có `77,028` paired records sau validation/dedup.
- **[x] Schema master**: Đã chốt — `{id, source, query, function_calls[], tools[]}`. Dùng chung cho cả 2 method; EN/VI counterpart giữ cùng split.
- **[x] Method 1 model**: Đã chốt — `unsloth/Qwen3.5-2B` và `unsloth/Qwen3.5-4B` (Small LM), fine-tune với Unsloth.
- **[x] Method 1 data format**: Đã chốt — native `messages`/`tools`/`tool_calls`, convert từ frozen revision qua `convert_to_instruction.py`, render bằng template exact checkpoint.
- **[x] Comparison table**: Đã chốt — 4 methods (Method 1 SLM + Method 2 Bi+Cross + OpenAI FC + Gemini FC).
- **[x] Stress test**: Đã chốt — giữ, so sánh cả 4 methods.
- **[x] Số lượng tool trong benchmark**: `4,421` unique tools trong revision `2026-09-02-full-dedup-seed42`.
- **[x] Splits train/val/test ratio**: `80/10/10`, seed=`42`; counts mỗi language `61,615/7,701/7,712`.
- **[ ] Metric chính Method 1**: ArgA (Ersoy et al.) hay dùng metric chung với Method 2?
- **[x] Fine-tune Method 1 framework**: Dùng Unsloth cho QLoRA/SFT Qwen3.5. Không dùng LLaMA-Factory hoặc ShareGPT trong training path; training dùng native Qwen3.5 chat template và response-only loss.
- **[x] Phase 1 in progress** (data pipeline):
  - [x] Download Glaive + xLAM
  - [x] EDA (notebook 01)
  - [x] Decision: single-turn + multi-call
  - [x] `src/data/translate.py`
  - [x] `src/data/translate_guidelines.py`
  - [x] `src/data/translation_checkpoint.py`
  - [x] `src/data/qa_translation.py`
  - [x] `src/data/normalize_schema.py`
  - [x] `src/data/feature_group_classify.py`
  - [x] `src/data/rebuild_benchmark.py` (production builder)
  - [x] `src/data/convert_to_instruction.py`
  - [ ] `src/data/stats.py`
  - [ ] `src/data/push_hf.py`
- **[x] Translation pipeline design** (chốt 2026-07-28):
  - K=10 samples/batch, concurrency=8.
  - 3 retry/sample với exp backoff. Fail → `failed/`.
  - Validate per-sample. Output: append JSONL + flush + fsync.
  - Resume: atomic checkpoint JSON.
  - Pilot: 100+100 → 1k+1k → frozen revision; full source translation remains pending.
- **[x] Schema clarification + pilot rebuild** (2026-08-04):
  - `data/translations/` là Bộ 1, giữ format gần raw (`system/chat` cho Glaive; `id/query/answers/tools` cho xLAM) theo thiết kế.
  - `data/benchmark_vi/` là Bộ 2, output schema master `{id, source, query, function_calls[], tools[]}`.
  - Rebuild pilot hiện có: 209 samples, 395 unique tools, split 167/20/22; instruction output đã tạo đủ 3 split.
  - Pilot artifacts đã archive; frozen revision mới được ghi riêng dưới `data/benchmark_core/`.
- **[x] Feature-group API smoke test** (2026-08-04):
  - Chạy một request không ghi cache bằng `bash scripts/data/run_feature_group.sh data/benchmark_vi/tool_pool.json --smoke-test`.
  - `feature_group` dùng `${ALIBABA_MODEL}`; smoke test hiện thành công với `qwen3.7-flash` và trả category hợp lệ.
  - `ALIBABA_BACKUP_MODEL*` chỉ dùng cho translation pipeline, chưa dùng cho feature-group classifier.
- **[x] Qwen3.5 checkpoint policy** (2026-08-20):
  - Method 1 chỉ dùng checkpoint `unsloth/Qwen3.5-2B` và `unsloth/Qwen3.5-4B`.
  - Không dùng checkpoint `-Base`; exact checkpoint tự cung cấp chat template.
- **[x] Benchmark revision policy** (2026-09-02):
  - `data/benchmark_core/2026-09-02-full-dedup-seed42/` là frozen canonical revision; `data/benchmark_vi/` là active VI export.
  - Mỗi experiment ghi revision, composition, seed, ID hashes và checkpoint policy vào manifest riêng.
  - Revision có `77,028` paired records, `4,817` negative, `4,421` tools; split `61,615/7,701/7,712`.
- **[x] Method 1 data budget policy** (2026-08-20):
  - Main controlled track dùng 60,000 core examples cho E1–E4 và 65,600 cho E5 (thêm toàn bộ `data/custom_vi/train.jsonl`).
  - Qwen3.5-2B và Qwen3.5-4B phải dùng cùng sample IDs, seed, epoch target và training budget.
  - Full-data runs là robustness/scale-up track riêng, không thay thế main controlled track.
- **[x] Experiment data materialization** (2026-08-27):
  - Thêm `src/data/prepare_experiments.py` và `scripts/data/prepare_experiments.sh`.
  - Tạo `data/experiments/e0`, `e1`, `e2`, `e4`, `e5` với manifest và train-only native data; validation/test dùng shared revision.
  - E1/E2 có 60,000 mẫu, E4 có 60,000 mẫu song ngữ, E5 có 65,600 mẫu gồm CustomTools train; E3 chờ general-SFT prerequisite.
- **[x] Kaggle upload helper** (2026-08-27):
  - Thêm `scripts/data/upload_experiments_to_kaggle.py` dùng `kagglehub` để upload train artifacts cùng frozen revision/CustomTools tùy chọn.
  - Kaggle token lấy từ `~/.kaggle/access_token`; không lưu credential trong repo.
- **[x] Experiment instruction formatting** (2026-08-27):
  - EN và VI dùng system prompt đúng ngôn ngữ; negative samples dùng assistant content bình thường, không có `<no_tool_call>`.
  - E4/E5 ghép native rows trong `train_chat.jsonl`; source/language vẫn được giữ trong `train.jsonl`.
- **[x] Kaggle notebook guide** (2026-09-02):
  - `docs/kaggle_notebook_guide.md` dùng Unsloth, native template, shared validation/test và không copy evaluation vào experiment.
- **[x] Method 1 trainer migration** (2026-08-31):
  - Chốt dùng Unsloth cho QLoRA/SFT checkpoint `unsloth/Qwen3.5-2B` và `unsloth/Qwen3.5-4B`.
  - Không dùng LLaMA-Factory/ShareGPT trong training path; training dùng native Qwen3.5 chat template và chỉ tính loss trên assistant response.
- **[x] Translation backup chain + API smoke test** (2026-08-04):
  - Translation retry chain: `ALIBABA_MODEL` → `ALIBABA_BACKUP_MODELS` (74 models, comma-separated, ordered by quality tier).
  - Smoke test 1 sample không ghi output/checkpoint bằng `bash scripts/data/run_translate_glaive.sh 175 176 --smoke-test`.
  - Smoke test đã thử đủ 4 model và translation trả JSON hợp lệ sau khi config dùng `${ALIBABA_URL}`.
- **[x] Translation input filtering + clean pilot** (2026-08-04):
  - Glaive filter giữ format raw: 112,960 → 45,593 positive first-turn records.
  - Xóa translation/benchmark pilot cũ; tắt feature_group trong translation.
  - Pilot mới: Glaive 10/10 và xLAM 10/10; QA rule pass toàn bộ.
- **[x] Translation API pilot resume** (2026-08-04):
  - Chạy thêm 25 mẫu/dataset với `qwen3.7-flash` và backup `deepseek-v4-flash-0731`.
  - Glaive: 24/25 thành công; xLAM: 25/25 thành công.
  - Checkpoint hiện tại: Glaive index 175, xLAM index 230.
- **[x] CustomTools-VI dataset strategy** (chốt 2026-08-11):
  - Tạo 8,000 samples đặc trưng VN (4,800 positive + 3,200 negative) thuộc 40 tools / 10 nhóm chức năng.
  - Split 70/10/20: 5,600 train + 800 val + 1,600 test.
  - Giữ CustomTools độc lập; chỉ `train.jsonl` đưa vào E5, validation/test dùng evaluation view riêng.
  - Plan chi tiết tại `docs/custom_vi_dataset_plan.md`.
- **[x] Phase 2/3 deferred** (quay lại sau khi data xong):
  - [ ] 4 configs/crossencoder/ (model, heads, losses, training)
  - [ ] 4 tests/crossencoder/ (heads, losses, label_generator, inference)
- **[x] Frozen benchmark rebuild + native preparation** (2026-09-02):
  - Revision `data/benchmark_core/2026-09-02-full-dedup-seed42/` có `77,028` paired records, `4,817` negative và `4,421` unique tools.
  - Archive pilot tại `data/legacy/pilot_20260902T000000Z/`; source raw/normalized/translation/CustomTools được giữ nguyên.
  - Native converter, assistant-only collator, Unsloth trainer và output parser đã có; GPU/template smoke test còn pending vì môi trường hiện thiếu PyTorch/Transformers.

---

## 11. Workflow khi bắt đầu session mới

1. **Đọc `AGENTS.md`** (file này) đầu tiên.
2. **Đọc `docs/architecture.md`** + `docs/methodology.md`.
3. **Kiểm tra phase hiện tại** trong section 7.
4. **Kiểm tra open questions** trong section 10.
5. **Hỏi user** nếu có nghi nghi gì về phase hiện tại.
6. **Bắt đầu code**, sau đó cập nhật phase + open questions.

---

## 12. Change Log

| Ngày | Thay đổi |
|---|---|
| 2026-07-25 | Khởi tạo repo, chốt tech stack, tạo skeleton + .md files |
| 2026-07-25 | Thêm Phase 7 (Stress Test) + folder structure |
| 2026-07-26 | Đóng decision: BGE-M3 base cho cả 2 model. Cross-Encoder: Hierarchical heads, BERT-QA input format |
| 2026-07-26 | Cross-Encoder skeleton code 6/6 files done. Tạm dừng model, làm data trước |
| 2026-07-26 | Download data thành công. EDA: Glaive 45,593 single-turn, xLAM 28,461 single-call |
| 2026-07-28 | Pivot sang multi-turn + multi-call (đã revert sau) |
| 2026-08-03 | **Pivot lớn**: Chuyển sang so sánh 2 phương pháp. Method 1: SLM Qwen2.5 end-to-end (theo Ersoy et al.). Method 2: Bi-Encoder + Cross-Encoder. Schema master single-turn + multi-call. 4-method comparison. Update toàn bộ docs. |
| 2026-08-04 | Xác nhận Bộ 1 dịch gần raw là chủ đích; xóa benchmark pilot schema cũ và rebuild theo schema master: 209 samples, 395 tools; tạo instruction format cho cả 3 split. |
| 2026-08-04 | Thêm smoke test feature_group không ghi cache; xác nhận endpoint đang hết free quota (`403 insufficient_quota`). |
| 2026-08-04 | Sửa feature_group config dùng `ALIBABA_MODEL` thay vì hardcode; smoke test thành công với `qwen3.7-flash`. |
| 2026-08-04 | Chạy pilot translation thêm 25 mẫu/dataset: Glaive 24/25, xLAM 25/25; QA rule pass toàn bộ output hiện có. |
| 2026-08-04 | Đổi translation batch từ 25 xuống 10; commit output + checkpoint trước feature_group để lỗi classifier không làm mất batch đã dịch. |
| 2026-08-04 | Thêm backup chain 3 tầng và smoke test dịch 1 sample không ghi output/checkpoint. |
| 2026-08-04 | Smoke test translation thử đủ 4 model sau khi đổi API key; endpoint hiện trả `403 access_denied`, không ghi output/checkpoint. |
| 2026-08-04 | Sửa các data config dùng `${ALIBABA_URL}` thay vì hardcode endpoint cũ; translation và feature_group smoke test đều thành công. |
| 2026-08-04 | Tạo filtered raw Glaive 45,593 mẫu, reset output dịch/benchmark cũ, tắt feature_group trong translation và chạy pilot sạch 10+10 pass. |
| 2026-08-04 | **Multi-model fallback chain**: Curate 74 models (loại 19: thinking/OCR/video/persona) xếp theo tier chất lượng. Primary: `qwen-mt-plus`. Backup: comma-separated `ALIBABA_BACKUP_MODELS`. concurrency 8→12. Thêm MODEL override trong shell scripts. Daily budget: ~74M tokens → ~4.5 ngày cho 105k samples. |
| 2026-08-10 | Thêm đường chạy dịch trực tiếp `data/normalized_en/glaive_normalized.jsonl` với dataset `glaive_normalized`, output/checkpoint/QA riêng; không ghi đè Bộ 1 raw hiện có. |
| 2026-08-10 | Chuyển entrypoint dịch Glaive và xLAM sang normalized schema; raw outputs giữ lại để audit, thêm chuyển đổi/resume không gọi API lại cho phần đã dịch. |
| 2026-08-11 | **CustomTools-VI Strategy**: Thống nhất kế hoạch xây dựng 8,000 samples (40 tools / 10 nhóm) cho ngữ cảnh Việt Nam, split 70/10/20. Tạo `docs/custom_vi_dataset_plan.md` và `implementation_plan.md`. |
| 2026-08-20 | **Cập nhật backbone model Method 1**: Chuyển mô hình SLM từ Qwen2.5 (0.5B/1.5B) sang **Qwen3.5 (2B/4B)** theo định hướng thực nghiệm mới. Đồng bộ toàn bộ tài liệu và kế hoạch thực nghiệm. |
| 2026-08-31 | **Cập nhật trainer Method 1**: Chuyển QLoRA/SFT từ LLaMA-Factory sang **Unsloth**. Training path dùng native Qwen3.5 chat template + response-only loss, không dùng ShareGPT. |
| 2026-09-02 | **Frozen benchmark + experiment preparation**: Rebuild paired EN/VI revision có negative và group-level dedup, archive pilot, materialize E0/E1/E2/E4/E5 train-only artifacts với native tool calls và manifest reproducibility. |
