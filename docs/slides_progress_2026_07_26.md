# Báo cáo tiến độ — Tool Calling tiếng Việt

**Ngày báo cáo**: 2026-07-26
**Người báo cáo**: Thinh
**Đồ án / Luận văn UIT**

---

## Slide 1 — Trang bìa

**Tool Calling tiếng Việt: Semantic Retrieval + Schema-aware Parameter Extraction**

- Hướng tiếp cận: Tách làm 2 thành phần chuyên biệt (thay vì generative LLM end-to-end)
- Mục tiêu: Giảm latency + cost, giữ accuracy cạnh tranh
- Ngôn ngữ: Tiếng Việt (ít nghiên cứu benchmark)
- Ngày: 2026-07-26

---

## Slide 2 — Vấn đề & Mục tiêu

### Vấn đề của hệ thống Tool Calling hiện tại
- Generative LLM (OpenAI FC, Gemini FC, Toolformer, Gorilla, ...)
- Latency cao (autoregressive generation)
- Cost cao (token-based)
- Hallucination + JSON không hợp lệ
- Hiệu năng suy giảm khi số tool tăng
- **Ít nghiên cứu cho tiếng Việt**

### Mục tiêu đề tài
1. **Semantic Tool Retrieval** (Bi-Encoder) — chọn tool phù hợp
2. **Schema-aware Parameter Extraction** (Cross-Encoder) — sinh arguments
3. **Benchmark tiếng Việt** — đánh giá công bằng
4. So sánh với OpenAI FC, Gemini FC baseline

---

## Slide 3 — Kiến trúc tổng quan

```
User Query (VI)
      │
      ▼
┌──────────────┐    top-k tools
│  BI-ENCODER  │ ──────────────────►
│  (Retrieval) │
│  BGE-M3 + MNRL│
└──────────────┘
      ▲
      │ query + tool pool
      │
      ▼
                                ┌──────────────┐    JSON args
                          ─────► │ CROSS-ENCODER│ ──────────►
                                │ (Extraction) │
                                │ BGE-M3 + heads│
                                └──────────────┘
                                       ▲
                                       │ tool schema
                                       │
                                ┌──────────────┐
                                │  VALIDATOR   │
                                │ (JSON schema)│
                                └──────────────┘
                                       │
                                       ▼
                                function_call
```

**Đặc điểm**:
- 2 model chuyên biệt, không generation
- 1 forward pass / parameter (low latency)
- Validator lọc output cuối cùng

---

## Slide 4 — Tech stack đã chốt

| Thành phần | Công nghệ |
|---|---|
| Framework | PyTorch + Transformers |
| Config | Hydra (structured config) |
| Bi-Encoder | BGE-M3 + FlagEmbedding + MultipleNegativesRankingLoss |
| Cross-Encoder | BGE-M3 + Hierarchical heads (1 binary `has_value` + schema-driven sub-head) |
| Translation | Alibaba OpenAI-compatible API (qwen3.7-flash / qwen3.7-max) |
| Baseline 1 | OpenAI Function Calling (gpt-4o-mini) |
| Baseline 2 | Google Gemini Function Calling (gemini-1.5-flash) |
| ~~Local LLM~~ | ~~Đã bỏ~~ (ngoài scope 3 tháng) |

---

## Slide 5 — Cross-Encoder (điểm nhấn kỹ thuật)

### Architecture: Hierarchical Span Prediction

- **Base model**: BGE-M3 (multilingual, mạnh về tiếng Việt)
- **Input format** (BERT-QA style):
  ```
  [CLS] <query> [SEP] Param=<name>. Desc=<desc>. Type=<type>[. Enum=...] [SEP]
  ```
  → Query làm context, param schema làm question
- **Heads** (không dùng generation, không dùng type-prediction):
  - `has_value` (binary BCE) — có/không có giá trị
  - `span` (start/end) — string/number params
  - `enum` (N-way) — enum params
  - `boolean` (2-way) — boolean params
- **Routing**: schema-driven, type lấy từ input → chỉ 1 sub-head active / param
- **Loss**: Gated multi-task (BCE has_value + CE sub-head có điều kiện)

### Lợi ích
- Schema-aware by design
- Ít hallucination (span phải có trong query)
- F1/EM evaluation chuẩn (SQuAD-style)
- Per-parameter forward (1 pass / param) → nhanh

---

## Slide 6 — Tiến độ Phase 0 (Skeleton + Cross-Encoder)

✅ **Đã hoàn thành**:
- Folder structure đầy đủ
- Cross-Encoder skeleton code (6 files):
  - `heads.py` — hierarchical heads
  - `losses.py` — gated multi-task loss
  - `data_collator.py` — BERT-QA tokenization
  - `label_generator.py` — rule-based labels
  - `inference.py` — extract arguments
  - `model.py` — BGE-M3 + heads wrapper
- Documentation: `architecture.md`, `methodology.md`, `references.md`

⏳ **Deferred** (chờ data xong):
- 4 Hydra configs (model, heads, losses, training)
- 4 unit tests (heads, losses, label_generator, inference)

---

## Slide 7 — Tiến độ Phase 1 (Data pipeline) — bắt đầu

### 7.1. Download datasets ✅
| Dataset | Samples | Size | Status |
|---|---:|---:|---|
| Glaive Function Calling v2 | 112,960 | 256 MB | ✅ |
| xLAM function-calling-60k | 60,000 | 91 MB | ✅ |
| **Tổng** | **172,960** | **347 MB** | ✅ |

### 7.2. EDA findings ✅

**Glaive**:
- Format: `system` (1 tool schema) + `chat` (multi-turn)
- Function call pattern: `<functioncall> {"name": "...", "arguments": '{...}'} <|endoftext|>`
- 40% single-turn có FC (45,593 usable)
- 73% multi-turn (bỏ)
- 1,040 unique tools (chỉ 109 thực sự được gọi)

**xLAM**:
- Format: `id`, `query`, `answers` (JSON str list), `tools` (JSON str list)
- 47% single-call (28,461), 42% có 2 calls, 11% có 3+ calls
- 3,605 unique tools (rất diverse — real-world APIs)

### 7.3. Usable samples sau filter

| | Glaive | xLAM | Tổng |
|---|---:|---:|---:|
| Raw | 112,960 | 60,000 | 172,960 |
| **Single-turn + single-call** | 45,593 | 28,461 | **~74,000** |
| Loss | 60% | 53% | 57% |

---

## Slide 8 — Quyết định kỹ thuật đã chốt

| Decision | Choice | Lý do |
|---|---|---|
| Glaive format | Single-turn only | Khớp benchmark phổ biến (BFCL Live Simple) |
| xLAM multi-call | First call only (MVP) | Đơn giản, khớp schema 1 query = 1 tool |
| Translation model | `ALIBABA_MODEL` env (qwen3.7-flash) | Free, đa dụng, có thể switch |
| QA judge model | `qwen3.7-max` | Mạnh hơn, chính xác cho rule check |
| API | Alibaba OpenAI-compatible (không dùng DashScope) | Đơn giản hơn, cùng endpoint cho mọi model |
| HF token | Optional | 2 dataset public, không bắt buộc |

---

## Slide 9 — Pending decisions (3 — chờ thầy hội ý)

1. **xLAM multi-call**: first only / expand N samples / giữ multi-call
2. **`feature_group`**: default "Tools" / LLM classify / cluster rule-based
3. **Pilot translate size**: 2k / 20k / full ~74k

→ Sẽ chốt sau khi thầy phản hồi, tiếp tục `normalize_schema.py`

---

## Slide 10 — Kế hoạch tiếp theo

| Bước | Công việc | Output |
|---|---|---|
| 1 | Viết `normalize_schema.py` (sau khi chốt 3 decision trên) | `data/processed/{glaive,xlam}_normalized.jsonl` |
| 2 | Translate pilot 2k (1k + 1k) | `data/translations/pilot_2k.jsonl` |
| 3 | QA judge với qwen3.7-max | `data/translations/qa_report.json` |
| 4 | Validate QA pass rate (target > 90%) | OK để scale |
| 5 | Translate full ~74k | `data/translations/full.jsonl` |
| 6 | `build_benchmark.py` (split 80/10/10) | `data/benchmark_vi/{train,val,test}.jsonl` |
| 7 | Quay lại Phase 2/3: 4 configs + 4 tests | Cross-Encoder hoàn chỉnh |
| 8 | Train Bi-Encoder + Cross-Encoder | `checkpoints/` |
| 9 | Pipeline end-to-end + baselines | `results/` |
| 10 | Stress test (RAG-MCP inspired) | `results/tables_figures/stress_test/` |

**Timeline ước tính**:
- Data pipeline: 1-2 tuần
- Training: 1-2 tuần
- Evaluation: 1 tuần
- **Tổng: ~5 tuần** (còn ~2 tháng)

---

## Slide 11 — Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Translation quality thấp | Pilot 2k trước, QA judge strict, fallback model |
| Qwen-MT còn free? | Nhiều model Alibaba dự phòng, có thể switch |
| Cross-Encoder F1 thấp | Hierarchical heads giảm confusion, BERT-QA format chuẩn |
| Tool pool quá lớn → retrieval kém | Bi-Encoder pre-compute + stress test vary N |
| Multi-call bị mất data | MVP mất 50% xLAM, accept; mở rộng Phase 2 |
| API rate limit | Pilot nhỏ, scale dần, có thể chia batch |

---

## Slide 12 — Demo / Q&A

**Đã có sẵn**:
- Code skeleton đầy đủ (6 files Cross-Encoder)
- Data downloaded (347 MB)
- EDA notebook với real findings
- Documentation đầy đủ (architecture, methodology, references)

**Câu hỏi dự kiến**:
1. Tại sao không dùng generative LLM end-to-end?
   → Latency + cost + hallucination; trade-off so sánh trong thesis
2. Tại sao BGE-M3?
   → Multilingual mạnh, có FlagEmbedding chính thức, MNRL chuẩn
3. Tại sao tách 2 model thay vì 1?
   → Bi-Encoder retrieval nhanh (pre-compute), Cross-Encoder extraction chính xác
4. Tiếng Việt khác gì EN trong bài toán này?
   → Tokenize, schema question có thể VI, translation pipeline riêng

---

## Appendix — Files đã tạo

### Docs (4 updated)
- `AGENTS.md` — cross-session memory
- `docs/architecture.md` — pipeline diagram
- `docs/methodology.md` — research method
- `docs/references.md` — papers
- `.env.example` — env template
- `pyproject.toml` — deps

### Code (5 created)
- `src/data/__init__.py`
- `src/data/collect.py` — download từ HF
- `configs/data/collect.yaml` — Hydra config
- `notebooks/01_eda_raw_data.ipynb` — EDA với real findings
- `scripts/data/run_collect.sh` — run script

### Cross-Encoder (6 skeleton — done từ Phase 0)
- `src/models/crossencoder/{heads,losses,data_collator,label_generator,inference,model}.py`

### Data (3 files)
- `data/raw/glaive_raw.jsonl` (256 MB)
- `data/raw/xlam_raw.jsonl` (91 MB)
- `data/raw/EDA_SUMMARY.md` (4.5 KB)
