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
- **Trạng thái**: Phase 0 — Skeleton + chờ Phase 1 (data pipeline)

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
| **Method 1: SLM End-to-End** | Fine-tune Qwen2.5 (0.5B/1.5B) làm tool selection + parameter extraction trong 1 model | Ersoy et al. (2025) |
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
| **Method 1: SLM End-to-End** | **Qwen2.5 0.5B/1.5B** + **LLaMA-Factory** (instruction tuning) |
| Method 2: Bi-Encoder (Retrieval) | **BGE-M3** + **FlagEmbedding** + **MultipleNegativesRankingLoss** |
| Method 2: Cross-Encoder (Extraction) | **BGE-M3** + **Hierarchical heads** (1 binary `has_value` + schema-driven sub-head: span / enum / boolean), format `[CLS] query [SEP] Param=<name>. Desc=... Type=<type>[. Enum=...] [SEP]` (BERT-QA style) |
| Dịch dataset | **Alibaba OpenAI-compatible API** (qwen3.7-flash / qwen3.7-max) |
| Baseline 1 | **OpenAI Function Calling** (gpt-4o-mini) |
| Baseline 2 | **Google Gemini Function Calling** (gemini-1.5-flash) |

**Lưu ý kiến trúc**:
- Method 1 SLM: dùng instruction-tuning format (system prompt chứa tool list, user query, assistant sinh `<tool_call>...</tool_call>`). Giống Ersoy et al. (2025).
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
| `query` | **VI** | User query tiếng Việt |
| `function_calls[].name` | **EN** | snake_case identifier, không dấu |
| `function_calls[].arguments` keys | **EN** | snake_case |
| `function_calls[].arguments` values | VI/EN | tùy natural language hay identifier |
| `tools[].name` | **EN** | snake_case identifier |
| `tools[].description` | **VI** | Mô tả tự nhiên tiếng Việt |
| `tools[].feature_group` | **VI** | Tên nhóm chức năng (do LLM classify, cache theo tool name) |
| `tools[].parameters` | — | JSON Schema chuẩn (`string`/`integer`/`number`/`boolean`/`array`/`object`) |

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

### Sample count sau filter single-turn

| Dataset | Raw | Sau filter | Mất |
|---|---|---|---|
| Glaive | 112,960 | **45,593** (40%) | 67,367 multi-turn bị bỏ |
| xLAM | 60,000 | **60,000** (100%) | 0 (đã flat) |
| **Total** | 172,960 | **~105,593** | ~39% |

### Data flow: từ master schema → 2 format train

```
Schema master (data/benchmark_vi/*.jsonl)
   │
   ├──► Method 1 (SLM):
   │    src/data/convert_to_instruction.py
   │    → data/benchmark_vi/instruction/
   │      train_chat.jsonl (LLaMA-Factory format)
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
│   ├── benchmark_vi/          # Final Vietnamese benchmark
│   │   ├── tool_pool.json     # Gộp unique tools từ Glaive + xLAM
│   │   ├── tool_schema/
│   │   ├── train.jsonl / val.jsonl / test.jsonl
│   │   └── instruction/       # ← Method 1: convert sang chat format
│   │       ├── train_chat.jsonl / val_chat.jsonl / test_chat.jsonl
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
│   │   ├── build_benchmark.py
│   │   ├── convert_to_instruction.py  # ← Method 1: master → chat format
│   │   ├── build_tool_pool.py
│   │   ├── extract_anchors.py
│   │   ├── augment_with_distractors.py
│   │   └── stats.py
│   ├── models/
│   │   ├── slm/               # ← Method 1: Qwen2.5 fine-tune (LLaMA-Factory)
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
| 6 | Evaluation & comparison (4 methods) | ⏳ Framework done; chờ predictions/experiments |
| 7 | Stress test (RAG-MCP inspired, so sánh cả 4) | ⏳ |

### Phase detail

**Phase 1 — Data pipeline**:
- Collect → normalize → translate (Bộ 1) → QA → build benchmark (Bộ 2/master) → convert instruction format
- Output: `data/benchmark_vi/` + `data/benchmark_vi/instruction/`

**Phase 2 — Bi-Encoder** (Method 2):
- Train BGE-M3 + MNRL cho tool retrieval
- Metric: Recall@1, Recall@5, MRR

**Phase 3 — Cross-Encoder** (Method 2):
- Train BGE-M3 + hierarchical heads cho parameter extraction
- Metric: Span F1, Enum accuracy, End-to-end F1

**Phase 4 — SLM** (Method 1):
- Fine-tune Qwen2.5 0.5B/1.5B với LLaMA-Factory trên instruction data
- Format: system (tool list) + user (query) → assistant (`<tool_call>...</tool_call>`)
- Metric: End-to-end accuracy (giống Ersoy et al. ArgA)

**Phase 5 — Baselines**:
- OpenAI FC (gpt-4o-mini), Gemini FC (gemini-1.5-flash)
- Đo latency, cost, accuracy

**Phase 6 — Evaluation**:
- Component-level: Call F1, Recall@K/MRR, Tool Set Accuracy, Key/Argument Pair F1, Normalized ArgEM, Schema Validity.
- End-to-end: **N-FCEM-positive** (tool multiset + toàn bộ arguments đúng sau normalization) và Overall Success có no-call.
- Robustness: breakdown + gap theo source, seen/unseen tool, domain, độ khó, số call, số candidate và độ phức tạp schema; không dùng dialect.
- Efficiency: p50/p95/p99 latency, throughput, token/cost/query, cost/correct call, GPU memory nếu có.
- Statistical reliability: bootstrap confidence interval; paired comparison dùng McNemar và paired bootstrap.
- Input chuẩn: gold master JSONL + prediction JSONL cùng `id`; hỗ trợ native multi-call và adapter cho master dạng `conversation` cũ.
- Output chuẩn: `report.json`, `per_sample.jsonl`, `summary.md`; compare 4 methods sinh thêm CSV/Markdown và paired tests.
- Framework code + CLI + tests đã hoàn tất; phần còn lại của Phase 6 là sinh predictions của 4 methods và chạy thí nghiệm.

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
| `docs/evaluation_methodology_thesis.md` | Trình bày học thuật về phương pháp, kỹ thuật, vai trò và ý nghĩa của framework đánh giá |
| `docs/translation_guidelines.md` | Quy tắc dịch |
| `docs/references.md` | Papers & resources |
| `.env.example` | Template biến môi trường |
| `pyproject.toml` | Project metadata + tool config |

---

## 10. Open Questions / Decisions Pending

- **[x] Single-turn + multi-call**: Đã chốt — chỉ lấy first turn từ Glaive, giữ multi-call từ xLAM. ~105k samples.
- **[x] Schema master**: Đã chốt — `{id, source, query, function_calls[], tools[]}`. Dùng chung cho cả 2 method.
- **[x] Method 1 model**: Đã chốt — Qwen2.5 0.5B/1.5B (Small LM, đúng tinh thần "SLM"), fine-tune với LLaMA-Factory.
- **[x] Method 1 data format**: Đã chốt — instruction-tuning (system prompt + user + assistant), convert từ schema master qua `convert_to_instruction.py`.
- **[x] Comparison table**: Đã chốt — 4 methods (Method 1 SLM + Method 2 Bi+Cross + OpenAI FC + Gemini FC).
- **[x] Stress test**: Đã chốt — giữ, so sánh cả 4 methods.
- **[ ] Số lượng tool trong benchmark**: Chưa quyết (sau khi build tool_pool.json).
- **[ ] Splits train/val/test ratio**: Đề xuất 80/10/10, seed=42.
- **[x] Metric chính Method 1**: dùng metric chung **N-FCEM-positive** (tương đương tinh thần ArgA nhưng schema-aware normalization và hỗ trợ multi-call); ArgEM/Arg-F1 là metric chẩn đoán.
- **[ ] Fine-tune Method 1 pipeline**: Dùng LLaMA-Factory CLI hay tích hợp training script trong repo?
- **[x] Phase 1 in progress** (data pipeline):
  - [x] Download Glaive + xLAM
  - [x] EDA (notebook 01)
  - [x] Decision: single-turn + multi-call
  - [ ] `src/data/translate.py`
  - [ ] `src/data/translate_guidelines.py`
  - [ ] `src/data/translation_checkpoint.py`
  - [ ] `src/data/qa_translation.py`
  - [ ] `src/data/normalize_schema.py`
  - [ ] `src/data/feature_group_classify.py`
  - [ ] `src/data/build_benchmark.py`
  - [ ] `src/data/convert_to_instruction.py`
  - [ ] `src/data/stats.py`
  - [ ] `src/data/push_hf.py`
- **[x] Translation pipeline design** (chốt 2026-07-28):
  - K=25 samples/batch, concurrency=8.
  - 3 retry/sample với exp backoff. Fail → `failed/`.
  - Validate per-sample. Output: append JSONL + flush + fsync.
  - Resume: atomic checkpoint JSON.
  - Pilot: 100+100 → 1k+1k → full ~105k.
- **[x] Phase 2/3 deferred** (quay lại sau khi data xong):
  - [ ] 4 configs/crossencoder/ (model, heads, losses, training)
  - [ ] 4 tests/crossencoder/ (heads, losses, label_generator, inference)
- **[x] Phase 6 evaluation design** (chốt 2026-08-07):
  - Không dùng composite score tùy ý làm headline.
  - Component metrics + N-FCEM + robustness slices + efficiency.
  - Prediction contract dùng `function_calls[]`, `ranked_tools[]`, `telemetry`.
  - No-call là `function_calls: []`; multi-call so khớp không phụ thuộc thứ tự.
  - Oracle-tool extraction được đánh giá bằng file prediction riêng tùy chọn.

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
| 2026-08-07 | Chốt và triển khai Phase 6 evaluator: component metrics, N-FCEM, robustness không theo dialect, efficiency, bootstrap/McNemar; native multi-call. |
| 2026-08-07 | Thêm tài liệu phương pháp đánh giá theo văn phong báo cáo khóa luận, phân biệt 2 phương pháp chính và 2 baseline. |
