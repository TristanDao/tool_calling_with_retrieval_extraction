# AGENTS.md — Cross-session memory for AI coding assistants

> **Mục đích**: File này là "bộ nhớ dài hạn" giữa các phiên làm việc với AI.
> Mọi agent (opencode, Claude, GPT, Cursor, v.v.) **phải đọc file này đầu tiên**
> trước khi bắt đầu bất kỳ thay đổi nào trong repo.
> Cập nhật file này bất cứ khi nào có quyết định kiến trúc, công cụ, hoặc hướng đi mới.

---

## 1. Project Identity

- **Tên dự án**: Tool Calling tiếng Việt theo hướng **Semantic Retrieval + Schema-aware Parameter Extraction**
- **Repo path**: `/home/thinh/project/UIT/tool_calling_with_retrieval_extraction`
- **Loại**: Đồ án / luận văn UIT (nghiên cứu thực nghiệm + xây dựng hệ thống)
- **Tác giả**: Thinh
- **Trạng thái**: Phase 0.5 — Skeleton + Cross-Encoder code (6 files done) + chờ Phase 1 (data pipeline)

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
Xây dựng hệ thống Tool Calling tiếng Việt **tách thành 2 thành phần chuyên biệt**:
1. **Semantic Tool Retrieval** (Bi-Encoder) — chọn tool phù hợp
2. **Schema-aware Parameter Extraction** (Cross-Encoder) — sinh arguments theo schema

→ Mục tiêu: **giảm latency + cost** so với generative LLM, **giữ độ chính xác cạnh tranh**.

### Phạm vi
- ✅ Tool Retrieval, Parameter Extraction, JSON validation
- ✅ Benchmark tiếng Việt
- ❌ Tool execution, multi-tool planning, multi-agent orchestration, dynamic tool creation, real-time integration

---

## 3. Tech Stack (đã chốt)

| Thành phần | Công nghệ |
|---|---|
| Framework chính | **PyTorch + Transformers** |
| Config | **Hydra** với **structured config** (Python `@dataclass`) |
| Bi-Encoder (Retrieval) | **BGE-M3** + **FlagEmbedding** + **MultipleNegativesRankingLoss** |
| Cross-Encoder (Extraction) | **BGE-M3** + **Hierarchical heads** (1 binary `has_value` + schema-driven sub-head: span / enum / boolean), format `[CLS] query [SEP] Param=<name>. Desc=... Type=<type>[. Enum=...] [SEP]` (BERT-QA style) |
| Dịch dataset | **Qwen-MT (Alibaba, 1M token context)** qua DashScope API |
| LLM baseline 1 | **OpenAI Function Calling** (gpt-4o-mini) |
| LLM baseline 2 | **Google Gemini Function Calling** (gemini-1.5-flash) |
| ~~LLM baseline 3 (local)~~ | **ĐÃ BỎ** — ngoài scope khóa luận 3 tháng (xem section 10) |

**Lưu ý kiến trúc**:
- Không dùng Unsloth, vLLM, hay fine-tune LLM local.
- Bi-Encoder dùng `FlagEmbedding` (BAAI official) + `MultipleNegativesRankingLoss`.
- Cross-Encoder là **Hierarchical Span Prediction** (BGE-M3 base + custom heads). Schema-driven routing: type lấy từ schema question nên model không cần học/predict type — chỉ activate 1 sub-head phù hợp (span/enum/boolean). 1 binary `has_value` head riêng để phân biệt null (absent) vs có giá trị. Input format BERT-QA: query làm context, param schema làm question.

---

## 4. Data Schema (chuẩn — theo mẫu user cung cấp)

```json
{
  "id": "glaive_00042",
  "source": "glaive",
  "conversation": [
    {
      "role": "user",
      "content": "Tôi muốn tìm gia sư Toán ở Hà Nội."
    },
    {
      "role": "assistant",
      "content": null,
      "function_calls": [
        {
          "name": "search_tutors",
          "arguments": {"subject": "Toán", "location": "Hà Nội"}
        }
      ]
    },
    {
      "role": "function",
      "name": "search_tutors",
      "content": "[{...tutor 1...}, {...tutor 2...}]"
    },
    {
      "role": "assistant",
      "content": "Tôi tìm được 2 gia sư phù hợp..."
    }
  ],
  "tools": [
    {
      "name": "search_tutors",
      "description": "Tìm gia sư theo môn và khu vực.",
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

**Quy ước bắt buộc** (cập nhật 2026-07-28 — multi-turn + multi-call):
- `id` unique string (format: `<source>_<index>`)
- `source` ∈ `{"glaive", "xlam"}`
- `conversation[]` là list các turn theo thứ tự. Mỗi turn có:
  - `role` ∈ `{"user", "assistant", "function"}`
  - `user` / `function` / `assistant.final`: có `content` (string)
  - `assistant` có tool call: `content: null` + `function_calls[]` (list of `{name, arguments}`)
  - `function`: có `name` (tool name) + `content` (response text)
- `tools[]` flat list (KHÔNG group), mỗi tool có `feature_group` (string VI, do LLM classify)
- `function_calls[].name` luôn English identifier (snake_case)
- `function_calls[].arguments` keys luôn English, values có thể VI/EN
- `tools[].description` tiếng Việt
- `tools[].name` English identifier
- `tools[].parameters` chuẩn JSON Schema (`string`/`integer`/`number`/`boolean`/`array`/`object`)

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

## 6. Folder Structure (đã tạo)

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
│   ├── raw/                   # EN datasets (Glaive, ToolBench, xLAM, ToolACE)
│   ├── processed/
│   │   ├── tools/ queries/ parameters/  # đã chuẩn hóa
│   │   └── stress_test/                # ← MỚI (Phase 7)
│   │       ├── anchors.jsonl           # 200 anchors từ test.jsonl
│   │       └── augmented/              # random_N10.jsonl, same_domain_N100.jsonl, ...
│   ├── benchmark_vi/          # Final Vietnamese benchmark
│   │   ├── tool_pool.json    # ← MỚI: tool pool lớn gộp từ Glaive + xLAM
│   │   ├── tool_schema/
│   │   ├── train/ val/ test/  (jsonl)
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
│   │   ├── build_tool_pool.py           # ← MỚI (Phase 7)
│   │   ├── extract_anchors.py           # ← MỚI (Phase 7)
│   │   ├── augment_with_distractors.py  # ← MỚI (Phase 7)
│   │   └── stats.py
│   ├── models/
│   │   ├── biencoder/         # Semantic Tool Retrieval (BGE-M3 + FlagEmbedding + MNRL)
│   │   ├── crossencoder/      # Schema-aware Parameter Extraction (BGE-M3 + Span Prediction head)
│   │   └── baselines/         # OpenAI FC, Gemini FC
│   ├── pipeline/              # tool_caller end-to-end
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── extraction_metrics.py
│   │   ├── latency.py
│   │   ├── cost.py
│   │   ├── throughput.py
│   │   ├── compare.py
│   │   ├── stress_test.py              # ← MỚI (Phase 7)
│   │   └── plot_stress_test.py         # ← MỚI (Phase 7)
│   └── utils/
│
├── scripts/                   # data/, train/, serve/, eval/
├── notebooks/                 # 01-06 EDA + analysis + 07 stress test
├── tests/                     # unit tests
├── checkpoints/               # biencoder/, crossencoder/
├── results/
│   ├── retrieval/ extraction/ pipeline/ baselines/
│   └── tables_figures/
│       └── stress_test/                # ← MỚI (Phase 7)
├── docs/                      # architecture, methodology, benchmark, translation_guidelines, references
└── logs/                      # train/, eval/
```

**Nguyên tắc**:
- `data/` chỉ lưu data, `src/data/` chỉ chứa code xử lý data
- 2 model tách biệt hoàn toàn
- Baselines tách riêng để dễ so sánh
- `pipeline/tool_caller.py` là orchestrator duy nhất
- Tất cả config qua Hydra, không hardcode
- Stress test data tách riêng trong `data/processed/stress_test/`

---

## 7. Implementation Phases (roadmap)

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Skeleton (folders + .md files) | ✅ **Đang ở đây** |
| 1 | Data pipeline (collect, normalize, translate, build benchmark) | ⏳ |
| 2 | Bi-Encoder (Semantic Tool Retrieval) | ⏳ |
| 3 | Cross-Encoder (Schema-aware Parameter Extraction) | ⏳ |
| 4 | Pipeline + JSON validator | ⏳ |
| 5 | Baselines (OpenAI, Gemini, local Qwen/Llama) | ⏳ |
| 6 | Evaluation & comparison (test chính) | ⏳ |
| 7 | **Stress test (RAG-MCP inspired)** | ⏳ |

### Stress Test Plan (Phase 7 — tóm tắt)
- **Mục đích**: Đo khả năng scale của pipeline khi tool pool tăng (lấy cảm hứng từ RAG-MCP arXiv:2505.03275).
- **Tool pool**: gộp unique tools từ Glaive + xLAM (sau dịch VI) → `data/benchmark_vi/tool_pool.json`.
- **Anchors**: 200 samples từ `benchmark_vi/test.jsonl` → `data/processed/stress_test/anchors.jsonl`.
- **N values**: [3, 10, 50, 100, 500, 1000] (số candidate tools).
- **Distractor strategies**:
  - `random`: random từ tool pool (loại trừ ground truth).
  - `same_domain`: random từ cùng `feature_group` với ground truth.
- **Tổng instances**: 200 × 6 × 2 = 2,400 augmented test instances.
- **Metrics**: retrieval_recall@1, end_to_end_accuracy, latency_p50/p95, tokens_consumed.
- **Output**: plot accuracy vs N cho mỗi strategy, so sánh với OpenAI/Gemini baseline.
- **Điểm khác biệt với paper 2505.03275**: paper chỉ dùng random + generic MCP, đề tài dùng `random + same_domain` + domain-specific VI tools + tách riêng retrieval vs extraction metric.

### Reference liên quan
- **Ersoy et al. (2025)** — Arabic tool-calling, dịch 2 dataset open-source sang Arabic. Tham khảo chiến lược dịch + adapt dataset.
- **RAG-MCP (2025)** — stress test concept với varying N, paper dùng MCP web search, đề tài mượn ý tưởng.
- **BFCL** — tham khảo categories (Live Simple/Multiple/Parallel, Multi-turn) khi xây benchmark_vi.

---

## 8. Conventions (cho AI agents)

### Khi viết code
1. **Không thêm comment trừ khi user yêu cầu** (xem rule project).
2. **Luôn đọc** `docs/translation_guidelines.md` trước khi viết code dịch.
3. **Luôn đọc** `docs/architecture.md` + `docs/methodology.md` trước khi implement model/pipeline.
4. **Hydra config dùng structured config** (Python `@dataclass`), không dùng YAML composing.
5. **Type hints đầy đủ** cho mọi public function.
6. **Mỗi thay đổi kiến trúc phải cập nhật file .md tương ứng** (cùng commit).
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
| `docs/architecture.md` | Sơ đồ pipeline end-to-end |
| `docs/methodology.md` | Phương pháp nghiên cứu chi tiết |
| `docs/benchmark.md` | Cấu trúc benchmark tiếng Việt |
| `docs/translation_guidelines.md` | Quy tắc dịch Qwen-MT |
| `docs/references.md` | Papers & resources |
| `.env.example` | Template biến môi trường |
| `pyproject.toml` | Project metadata + tool config |

---

## 10. Open Questions / Decisions Pending

Cập nhật mục này khi có câu hỏi chưa giải quyết:

- **[x] Cross-Encoder architecture**: Đã quyết — **Hierarchical heads** (1 binary `has_value` + schema-driven sub-head). Sub-head active dựa trên `Type` trong schema question. Sub-head gồm: span (start/end) / enum (N-way) / boolean (2-way). Không dùng generation, không dùng type-prediction head (type đã có sẵn trong schema).
- **[x] Cross-Encoder input format**: Đã quyết — **BERT-QA style**: `[CLS] query [SEP] Param=<name>. Desc=<desc>. Type=<type>[. Enum=...] [SEP]` (query làm context, param schema làm question).
- **[x] Có nên dùng paper 2505.03275 (RAG-MCP) trực tiếp?**
  → **KHÔNG.** Mượn concept (vary N, plot degradation curve), tự build distractor generator
  + VI tool pool + tách riêng retrieval vs extraction metric. Xem Phase 7 ở section 7.
- **[x] Local LLM baseline (Qwen2.5/Llama Unsloth)**: **ĐÃ BỎ** — ngoài scope 3 tháng.
  Chỉ so sánh với OpenAI FC + Gemini FC.
- **[x] Datasets**: Chỉ dùng **2 nguồn chính** — Glaive Function Calling v2 + xLAM.
- **[x] Phase 0 done**: Cross-Encoder code skeleton 6/6 files (heads, losses, data_collator, label_generator, inference, model).
- **[ ] Phase 1 in progress** (data pipeline):
  - [x] Decision: Glaive CHỈ single-turn
  - [x] Decision: Translation model = `ALIBABA_MODEL` env (qwen3.7-flash)
  - [x] Decision: QA judge model = `qwen3.7-max`
  - [x] Decision: Alibaba OpenAI-compatible API (KHÔNG dùng dashscope)
  - [x] Download Glaive Function Calling v2 từ HF
  - [x] Download xLAM function-calling-60k từ HF
  - [x] EDA format (notebook 01)
  - [ ] `src/data/translate.py` (async batch K=50, concurrency=20)
  - [ ] `src/data/translate_guidelines.py` (protect identifiers)
  - [ ] `src/data/translation_checkpoint.py` (atomic save/load)
  - [ ] `src/data/qa_translation.py` (rule + qwen3.7-max judge)
  - [ ] Mini pilot 100+100 → 1k+1k pilot → full 170k
  - [ ] `src/data/normalize_schema.py` (type mapping xLAM → JSON Schema chuẩn)
  - [ ] `src/data/feature_group_classify.py` (LLM classify, cache; có thể pre-label trong translate)
  - [ ] `src/data/build_benchmark.py` (Bộ 1 → Bộ 2, multi-turn schema)
  - [ ] `src/data/push_hf.py` (upload dataset)
  - [ ] Fix Cross-Encoder `normalize_schema_type()` silent bug
- **[ ] Phase 2/3 deferred** (quay lại sau khi data xong):
  - [ ] 4 configs/crossencoder/ (model, heads, losses, training)
  - [ ] 4 tests/crossencoder/ (heads, losses, label_generator, inference)
- **[x] Normalization decisions** (đã chốt 2026-07-28):
  - [x] **xLAM multi-call (53% có 2+ tools)**: **GIỮ multi-call** + multi-turn Glaive (giữ nguyên số call/turn).
  - [x] **feature_group (cả 2 dataset không có)**: **LLM classify 1 lần, cache theo tool name** (cần cho stress test `same_domain`).
  - [x] **Pilot translate size**: 100+100 mini pilot → 1k+1k pilot → full 170k.
  - [x] **JSON Schema `type` standard**: chuẩn `string`/`integer`/`number`/`boolean`/`array`/`object` (map xLAM `str`→`string`, `int`→`integer`, `float`→`number`, `bool`→`boolean`, `list/List[T]`→`array`+`items`).
  - [x] **Architecture**: **2 bộ riêng** — (1) `data/translations/` raw VI để dịch, (2) `data/benchmark_vi/` task format chuẩn cho model.
- **[x] Translation pipeline design** (chốt 2026-07-28):
  - K=50 samples/batch, concurrency=20 (asyncio.Semaphore).
  - 3 retry/sample với exp backoff (1s, 2s, 4s). Fail → `data/translations/failed/<ds>_failed.jsonl`.
  - Validate per-sample ngay (rule check: identifier snake_case, JSON parse, required keys).
  - Output: append JSONL + flush per sample + `os.fsync()`.
  - Resume: atomic checkpoint JSON (`data/translations/.checkpoint/<ds>.json`, ghi tmp + rename).
  - Memory: O(K) RAM (~250KB peak). Stream I/O.
  - Pilot: 100 Glaive + 100 xLAM (mini pilot) → 1k+1k → full.
- **[ ] Số lượng tool trong benchmark**: Chưa quyết (sau khi build tool_pool.json).
- **[ ] Splits train/val/test ratio**: Đề xuất 80/10/10, seed=42 — confirm khi build benchmark.
- **[ ] Metric chính để so sánh**: End-to-end accuracy (retrieval@1 + extraction F1) — confirm khi viết evaluation.

---

## 11. Workflow khi bắt đầu session mới

1. **Đọc `AGENTS.md`** (file này) đầu tiên.
2. **Đọc `docs/architecture.md`** + `docs/methodology.md`.
3. **Kiểm tra phase hiện tại** trong section 7.
4. **Kiểm tra open questions** trong section 10.
5. **Hỏi user** nếu có nghi nghi gì về phase hiện tại.
6. **Bắt đầu code**, sau đó cập nhật phase + open questions.

---

## 12. Change Log (cập nhật khi có thay đổi lớn)

| Ngày | Thay đổi |
|---|---|
| 2026-07-25 | Khởi tạo repo, chốt tech stack, tạo skeleton + .md files |
| 2026-07-25 | Thêm Phase 7 (Stress Test RAG-MCP inspired) + folder structure + cập nhật docs. Đóng decision về việc KHÔNG dùng trực tiếp paper 2505.03275. Tham khảo Ersoy et al. (2025) cho chiến lược dịch dataset tool-calling sang ngôn ngữ ít tài nguyên. |
| 2026-07-26 | Đóng decision: BGE-M3 base cho cả 2 model (FlagEmbedding + MNRL cho Bi-Encoder, custom head cho Cross-Encoder); bỏ Unsloth + local LLM baseline; chỉ dùng 2 nguồn dataset (Glaive + xLAM); Cross-Encoder format = schema first `[CLS] schema [SEP] query [SEP]`. Update AGENTS.md + toàn bộ docs. |
| 2026-07-26 | Đổi Cross-Encoder architecture: từ generation sang **Span Prediction** (3 output: span/enum/null). Lý do: nhanh hơn, ít hallucination, khớp "Schema-aware", F1/EM evaluation chuẩn. Update docs. |
| 2026-07-26 | Refactor Cross-Encoder sang **Hierarchical heads** (1 binary `has_value` + schema-driven sub-head: span/enum/boolean). Bỏ `value_type` head vì type đã có sẵn trong schema question. Null coi là "absence of value" (gate qua `has_value`) thay vì "một loại giá trị". Đổi input format sang **BERT-QA style**: `[CLS] query [SEP] Param=... Type=...[. Enum=...] [SEP]`. Per-parameter forward pass (N passes / query), max_length=1024, truncation="only_first" (cắt query nếu quá dài). Tạo skeleton `src/models/crossencoder/` (6 files) + `configs/crossencoder/` (4 files) + `tests/crossencoder/` (4 files). Update architecture.md, methodology.md, references.md, AGENTS.md. |
| 2026-07-26 | Cross-Encoder skeleton code 6/6 files done (heads.py, losses.py, data_collator.py, label_generator.py, inference.py, model.py). CHƯA tạo configs/crossencoder/ (0/4) + tests/crossencoder/ (0/4). Tạm dừng model work để làm data trước (Phase 1). |
| 2026-07-26 | Decision data: Glaive CHỈ single-turn; Qwen-MT token không giới hạn (nhiều model free Alibaba); API key chưa có (cần đăng ký); khi quay lại model: Configs → Tests. |
| 2026-07-26 | Update .env.example theo .env của user: dùng Alibaba OpenAI-compatible API (`ALIBABA_URL` + `ALIBABA_MODEL` + `ALIBABA_QA_MODEL=qwen3.7-max`) thay cho DashScope SDK. Bỏ section vLLM local. Update pyproject.toml: thêm `openai>=1.0`, `datasets>=2.18`, bỏ `dashscope`. Start Phase 1: data pipeline (collect → normalize → translate → QA → benchmark). |
| 2026-07-26 | Download data thành công: Glaive 112,960 + xLAM 60,000 → `data/raw/`. EDA findings: Glaive 45,593 single-turn usable (40%), xLAM 28,461 single-call (47%). Tool pool: 1,040 + 3,605 unique. 3 decision pending (multi-call, feature_group, pilot size) chờ user hội ý nhóm. |
| 2026-07-28 | **Pivot lớn**: Đổi từ single-turn/single-call → **full multi-turn + multi-call** (giữ 74% Glaive multi-turn + 53% xLAM multi-call). Schema mới: `conversation[]` (user/assistant/function turns) + `tools[]` flat. Quyết `feature_group` cần dùng LLM classify. Quyết **2 bộ riêng**: (1) translations/ giữ raw, (2) benchmark_vi/ task format. Quyết translation pipeline: async batch K=50, concurrency=20, 3 retry/sample, validate per-sample, append JSONL + atomic checkpoint, resume tự động. Pilot 100+100 → 1k+1k → full 170k. Bắt đầu code Phase A (skeleton) → Phase B (translate). |
| 2026-07-28 | Translation pipeline cập nhật: pre-label `feature_group` ngay trong batch dịch và ghi cache `data/benchmark_vi/.cache/feature_group.json` để build benchmark/stress test dùng lại. QA được sửa để judge cặp `(original EN, translated VI)` thay vì tự so với chính sample đã dịch. |
