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
| 2 | Method 2: Bi-Encoder | Strict round1 + round2, mining/index/calibration/val đã chạy và kiểm tra |
| 3 | Method 2: Cross-Encoder | Đã train/eval; đang kiểm định validation gate |
| 4 | Method 1: SLM fine-tune + instruction data | ⏳ |
| 5 | Baselines (OpenAI FC, Gemini FC) | ⏳ |
| 6 | Evaluation & comparison (4 methods) | ⏳ |
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

- **[x] Method 2 completion protocol (2026-09-06)**: User chọn strict unseen. Loại test-unseen tools khỏi mọi positive/negative và mining trong training Bi-Encoder; train lại cả hai round từ base, không dùng run cũ làm mining teacher. Test gold giữ nguyên. Snapshot/output cũ chỉ là legacy negative-exposure run.
- **[x] Method 2 implementation thực tế**: BGE-M3 + LoRA + sentence-transformers CachedMNRL; Cross-Encoder XLM-R base + hierarchical heads theo `docs/method2_plan.md`, BGE-M3 CE chỉ là ablation tùy quota. Các mô tả BGE-M3 CE/FlagEmbedding ở trên là thiết kế ban đầu.
- **[ ] Method 2 nghiệm thu**: strict/normalized re-evaluation, multi-call error review, validation argument gate, strict unseen rerun, stress và báo cáo; xem `docs/method2_completion.md`. Ablation Phase 6 là tùy quota, không phải điều kiện cứng. Không dùng test để tune threshold/hyperparameter. Metric normalized-vs-strict không thay thế ablation normalizer bật/tắt.
- **[x] Method 2 local completion tooling (2026-09-06)**: evaluator strict type/whitespace, oracle gap cùng mẫu số, error alignment multi-call, gate incomplete và raw logits đã bổ sung; report legacy 12,155 queries đã tái chấm. Snapshot strict loại 20 tool val/test-unseen, sửa 905 negative exposures, giữ 94,634 pairs. Sáu notebook completion có hash manifest, validation checkpoint provenance và xuất report/figure tự động. GPU retraining/validation/stress chưa chạy; pure same-domain N=1000 chưa khả thi với nhãn pool hiện tại. Hướng dẫn tại `docs/method2_completion.md`.
- **[x] Kaggle LoRA environment fix**: notebook 02 dừng trước optimizer vì PEFT 0.20.0 phát hiện torchao 0.10.0 không tương thích. Setup completion notebook gỡ torchao (run không dùng quantization), smoke LoRA trên encoder nhỏ không tải mạng trước training. Dùng notebook local mới với bundle cũ; không đổi hash data/config/checkpoint và không cần chạy lại validation 01 vì sửa môi trường.
- **[x] Strict round1 verified (2026-09-06)**: output `output_train/output_method2-completion-02-round1` hoàn thành 3 epoch/921 step, 78,435 training examples, 9.033 h, peak 10,888.9 MB. Đã kiểm tra 114 bundle hashes + 10 final checkpoint hashes. Full-pool pair-level val R@1=55.35%, R@5=85.13%, MRR@10=0.6804. Tiếp notebook 03 dùng output 02 làm cả bundle lẫn teacher input (không gắn thêm bundle trùng manifest); có gói gọn `artifacts/method2_completion/round2_input_from_round1.zip`. CE validation 01 đã đo đủ nhưng enum/argument gate chưa đạt; chưa freeze cho evaluation cuối.

- **[x] Single-turn + multi-call**: Đã chốt — chỉ lấy first turn từ Glaive, giữ multi-call từ xLAM. ~105k samples.
- **[x] Schema master**: Đã chốt — `{id, source, query, function_calls[], tools[]}`. Dùng chung cho cả 2 method.
- **[x] Method 1 model**: Đã chốt — Qwen2.5 0.5B/1.5B (Small LM, đúng tinh thần "SLM"), fine-tune với LLaMA-Factory.
- **[x] Method 1 data format**: Đã chốt — instruction-tuning (system prompt + user + assistant), convert từ schema master qua `convert_to_instruction.py`.
- **[x] Comparison table**: Đã chốt — 4 methods (Method 1 SLM + Method 2 Bi+Cross + OpenAI FC + Gemini FC).
- **[x] Stress test**: Đã chốt — giữ, so sánh cả 4 methods.
- **[ ] Số lượng tool trong benchmark**: Chưa quyết (sau khi build tool_pool.json).
- **[ ] Splits train/val/test ratio**: Đề xuất 80/10/10, seed=42.
- **[ ] Metric chính Method 1**: ArgA (Ersoy et al.) hay dùng metric chung với Method 2?
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
  - K=10 samples/batch, concurrency=8.
  - 3 retry/sample với exp backoff. Fail → `failed/`.
  - Validate per-sample. Output: append JSONL + flush + fsync.
  - Resume: atomic checkpoint JSON.
  - Pilot: 100+100 → 1k+1k → full ~105k.
- **[x] Schema clarification + pilot rebuild** (2026-08-04):
  - `data/translations/` là Bộ 1, giữ format gần raw (`system/chat` cho Glaive; `id/query/answers/tools` cho xLAM) theo thiết kế.
  - `data/benchmark_vi/` là Bộ 2, output schema master `{id, source, query, function_calls[], tools[]}`.
  - Rebuild pilot hiện có: 209 samples, 395 unique tools, split 167/20/22; instruction output đã tạo đủ 3 split.
  - Full translation vẫn pending: pilot sạch hiện có 10 Glaive + 10 xLAM samples thành công.
- **[x] Feature-group API smoke test** (2026-08-04):
  - Chạy một request không ghi cache bằng `bash scripts/data/run_feature_group.sh data/benchmark_vi/tool_pool.json --smoke-test`.
  - `feature_group` dùng `${ALIBABA_MODEL}`; smoke test hiện thành công với `qwen3.7-flash` và trả category hợp lệ.
  - `ALIBABA_BACKUP_MODEL*` chỉ dùng cho translation pipeline, chưa dùng cho feature-group classifier.
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
  - Train/Val merge vào master benchmark; Test giữ riêng `test_custom_vi.jsonl` làm bộ đánh giá VN-specific.
  - Plan chi tiết tại `docs/custom_vi_dataset_plan.md`.
- **[x] Phase 2/3 deferred** (quay lại sau khi data xong):
  - [ ] 4 configs/crossencoder/ (model, heads, losses, training)
  - [ ] 4 tests/crossencoder/ (heads, losses, label_generator, inference)

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

Notebook06 verified and consolidated report (2026-09-08): 114 bundle + 29 artifact hashes verified; identical weights/index/thresholds to04, no notebook errors, full pipeline/oracle/raw ID coverage12,155 queries. OFF normalized ArgA benchmark20.99%, seen33.00%, unseen7.00%; ON advantage19.22/34.75/15.75 percentage points. Rankings and selected tool sequences unchanged for all queries; calls arguments changed4510/271/216. PDF/Markdown/evidence report at reports/method2_20260908. 189 event review queue remains unreviewed; CE gate false, pure same-domain pending. Review found CE planned4546/actual4542 step mismatch: budget ceil vs loop stepping only complete gradient accumulation groups, no final partial flush; preserve frozen runs, document as future training implementation fix, impact not measured. No rerun01–06 required for reporting these results.

Notebook05 stress verified (2026-09-07): 114 bundle + 29 artifact hashes match, checkpoint/threshold identical to notebook04, no notebook cell errors. Same 200 unique test_seen anchors (100 positive/100 negative) across all six N=[3,10,50,100,500,1000]; all 1,200 predictions/raw records present. Random stress ArgA 81/79/78/77/62/54%; total P50 58.16/57.15/55.82/58.55/87.44/91.88 ms. At N1000 all100 positive anchors retain every gold tool in top3, but32 have extra selected tool(s);43/100 negative anchors generate calls. No gold truncation. Latency is not flat across full N range. Random stress complete; pure same-domain still pending, CE gate still false. Next notebook06 uses full output04, not required to rerun05; human error review/final report remain. Audit: artifacts/method2_completion/stress05_verification.json.

Notebook04 evaluation verified (2026-09-07): output_method2-completion-04-evaluation completed without cell errors; 114 bundle hashes + 29 artifact hashes match, including prior strict round2 and selected CE repair01b. Pipeline/oracle/raw prediction IDs cover all 10,555 benchmark + 800 custom_seen + 800 custom_unseen queries without duplicates. Positive-only normalized ArgA: 40.21% / 67.75% / 22.75%; oracle: 42.97% / 82.00% / 36.50%. Quality gate remains false; no test-based tuning. Next: local notebook05_stress with ONLY full output04, random 200x6=1,200 instances; optional notebook06_normalizer_ablation also uses full output04 in a separate session. No rerun01–04 required. Audit: artifacts/method2_completion/evaluation04_verification.json. Pure same-domain, human error review and final reporting remain pending.

CE repair 01b verified (2026-09-07): actual Kaggle output completed without notebook errors; 114 base bundle + 7 addon + 5 selected CE file hashes verified. Candidate repair_v1 selected by the predeclared validation rule and copied identically to run01/final. Oracle Argument EM 210/460→285/460 (45.65%→61.96%); repaired Custom span EM 75.64%→93.64%; enum 75.71%→74.49%. Gate remains false (enum<90%, argument<70%); Glaive/xLAM span validation regressed. Recommendation: freeze this validation-selected candidate for the current evaluation, continue local notebook04 with full output03 + full output01b, preserve quality_gate_passed=false and report limitations; no rerun01–03 and no test-based tuning. Loader verified against both actual inputs. Audit: artifacts/method2_completion/ce_repair_01b_verification.json; steps: docs/method2_ce_repair.md. Evaluation/stress and final acceptance still pending.

CE repair protocol (2026-09-07): rebuild CustomTools CE train/val supervision from validated argument_mentions (call_index + parameter_path + exact character offsets + canonical value); preserve benchmark rows and existing retained training sample IDs. Supervise the annotated surface, including numeric units, even when a canonical string needs unsupported alias mapping; audit canonical mismatch separately. Train a fresh XLM-R base with the same 2+2 curriculum to isolate the data repair. Compare old/new on the same repaired component validation and unchanged oracle val gold; choose by argument EM then enum accuracy, retain old on ties. No test tuning, no forced lowering has_value threshold, no changes to strict Bi-Encoder artifacts. Ship an additive CE repair package with its own hashes and parent completion manifest so notebook 04 can still verify the existing strict BI lineage.

CE repair local preparation complete: custom train 12,253→15,318 pairs (+3,065, corrected693 spans); val1,434→1,684 (+250, corrected109 spans). Real XLM-R tokenizer checked all17,002 custom pairs: no out-of-window drops or labels outside query. Notebook `method2_ce_repair_01b.ipynb` + additive `ce_repair_v1.zip` ready; actual GPU checkpoint training remains pending user Kaggle run. Instructions: `docs/method2_ce_repair.md`. Input01b = complete output01 + addon; later04 uses output03 + output01b selected checkpoint.

Strict round2 verified (2026-09-07): `output_train/output_method2-completion-03-round2` hoàn thành 3 epoch/921 step, 11.31 giờ, 114 bundle hashes + 24 artifact hashes khớp. Mined 94,634 rows không có heldout exposure. Candidate val full R@5=99.99%, pool full R@5=83.96%; threshold gap tau=0.35/delta=0.21/k_max=3 trên val. CE vẫn chưa freeze: val gate enum=75.71%, argument EM=45.65%. Audit coverage: 250 gold argument occurrences của CustomTools val không có pair component tương ứng (integer72, number47, string131), nên component span metric chỉ chấm phần alignable, không đại diện mọi gold argument. Cần xử lý CE label/span/normalization và kiểm định trên validation trước evaluation cuối; không tự tăng epoch để thay bước điều tra.

Notebook completion input fix (2026-09-07): Kaggle có thể thấy marker ở cả output đầy đủ và archive reports đã giải nén. Chọn stage input dựa trên checkpoint hashes và bundle identity; bỏ bản thiếu weights, chấp nhận bản trùng cùng checkpoint, dừng nếu có nhiều checkpoint khác nhau. Sửa notebook loader ngoài bundle, giữ nguyên bundle/training/output đã hoàn thành.

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
| 2026-08-23 | **Method 2 Phase 0+1**: khôi phục `src/evaluation` từ nhánh chưa merge; sửa 4 bug chặn train của Cross-Encoder (index schema_type trong loss, padding 1024→dynamic 256, rule đoán boolean, decode span); build `data/benchmark_vi` (105,539 sample) và tool pool canonical 4,464 tool (gộp từ 23,374 biến thể); sinh pairs Bi-Encoder + Cross-Encoder; viết đủ code Phase 2-5 (`biencoder/`, `crossencoder/{dataset,train,normalize,evaluate}`, `pipeline/{method2,validator}`), `configs/method2/*.yaml`, 3 notebook Kaggle. Xem `docs/method2.md`. |
