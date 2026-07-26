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
- **Trạng thái**: Phase 0 — Skeleton (chỉ có .md + folder, chưa code Python)

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
| Fine-tune LLM | **Unsloth** (LoRA) |
| Base encoder (Bi-Encoder & Cross-Encoder) | **BGE-M3** hoặc **multilingual-e5** (sentence-transformers) |
| Dịch dataset | **Qwen-MT (Alibaba, 1M token context)** qua DashScope API |
| LLM baseline 1 | **OpenAI Function Calling** (gpt-4o-mini) |
| LLM baseline 2 | **Google Gemini Function Calling** (gemini-1.5-flash) |
| LLM baseline 3 | **Qwen2.5 / Llama-3.1 local** (Unsloth LoRA + vLLM serve) |

---

## 4. Data Schema (chuẩn — theo mẫu user cung cấp)

```json
{
  "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
  "label": {
    "function_call": {
      "name": "search_tutors",
      "arguments": {
        "subject": "Toán",
        "location": "Hà Nội"
      }
    }
  },
  "tools_summary": [
    {
      "feature_group": "Tìm kiếm & Kết nối",
      "tools": [
        {"name": "search_tutors", "description": "Tìm gia sư theo môn và khu vực."}
      ]
    }
  ]
}
```

**Quy ước bắt buộc**:
- `query` luôn tiếng Việt
- `function_call.name` luôn English identifier (snake_case)
- `function_call.arguments` keys luôn English, values có thể VI/EN
- `tools_summary[].tools[].description` tiếng Việt
- `tools_summary[].tools[].name` English identifier

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
│   ├── processed/             # tools/, queries/, parameters/ đã chuẩn hóa
│   ├── benchmark_vi/          # Final Vietnamese benchmark
│   │   ├── tool_schema/
│   │   ├── train/ val/ test/  (jsonl)
│   ├── translations/          # Qwen-MT logs + QA samples
│   └── statistics/
│
├── src/
│   ├── data/                  # collect, normalize, translate, build_benchmark
│   ├── models/
│   │   ├── biencoder/         # Semantic Tool Retrieval
│   │   ├── crossencoder/      # Schema-aware Parameter Extraction
│   │   ├── local_baseline/    # Unsloth LoRA + vLLM
│   │   └── baselines/         # OpenAI FC, Gemini FC
│   ├── pipeline/              # tool_caller end-to-end
│   ├── evaluation/            # retrieval, extraction, latency, cost, throughput, compare
│   └── utils/
│
├── scripts/                   # data/, train/, serve/, eval/
├── notebooks/                 # 01-06 EDA + analysis
├── tests/                     # unit tests
├── checkpoints/               # biencoder/, crossencoder/, unsloth_lora/
├── results/                   # retrieval/, extraction/, pipeline/, baselines/, tables_figures/
├── docs/                      # architecture, methodology, benchmark, translation_guidelines, references
└── logs/                      # train/, eval/
```

**Nguyên tắc**:
- `data/` chỉ lưu data, `src/data/` chỉ chứa code xử lý data
- 2 model tách biệt hoàn toàn
- Baselines tách riêng để dễ so sánh
- `pipeline/tool_caller.py` là orchestrator duy nhất
- Tất cả config qua Hydra, không hardcode

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
| 6 | Evaluation & comparison | ⏳ |

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

- **[ ] Cross-Encoder input format**: User sẽ quyết sau khi thử nghiệm.
  Cần `schema_formatter.py` hỗ trợ nhiều format (JSON Schema gốc / flattened / hybrid).
- **[ ] Số lượng tool trong benchmark**: Chưa quyết (10? 50? 100?).
- **[ ] Splits train/val/test ratio**: Chưa quyết.
- **[ ] Metric chính để so sánh**: Chưa quyết (End-to-end accuracy? F1 từng thành phần?).
- **[ ] Local LLM baseline chính**: Qwen2.5 hay Llama-3.1 (hay cả hai).
- **[ ] Qwen-MT context size**: Tận dụng 1M token để dịch whole-file hay per-sample.

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
