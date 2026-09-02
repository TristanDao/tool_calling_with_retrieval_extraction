# Tool Calling tiếng Việt — So sánh 2 Phương pháp

> Đồ án / luận văn UIT: so sánh SLM End-to-End vs Bi-Encoder + Cross-Encoder cho Tool Calling tiếng Việt.

## 1. Tổng quan

Hệ thống so sánh **2 phương pháp** Tool Calling cho tiếng Việt:

| Phương pháp | Cách làm | Model |
|---|---|---|
| **Method 1: SLM End-to-End** | Fine-tune SLM chọn tool + điền tham số bằng native tool calls | Unsloth QLoRA/SFT + `unsloth/Qwen3.5-4B` |
| **Method 2: Bi-Encoder + Cross-Encoder** | Tách retrieval (Bi-Encoder) + extraction (Cross-Encoder) | BGE-M3 + FlagEmbedding + custom heads |

Baselines so sánh:
- OpenAI Function Calling (gpt-4o-mini)
- Google Gemini Function Calling (gemini-1.5-flash)

## 2. Cấu trúc thư mục

```
tool_calling_with_retrieval_extraction/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/              # Hydra structured config
├── data/                 # sources / frozen benchmark / experiment train artifacts
├── src/                  # data/ + models/ + pipeline/ + evaluation/
├── scripts/              # CLI wrappers
├── notebooks/            # EDA + analysis
├── tests/                # unit tests
├── checkpoints/          # model weights (gitignored)
├── results/              # metrics + tables/figures
├── docs/                 # architecture, methodology, benchmark, translation_guidelines, references
└── logs/                 # training/eval logs
```

Chi tiết xem `AGENTS.md` section 6 và `docs/architecture.md`.

## 3. Quick start

> Core benchmark đã có revision frozen. Training vẫn cần môi trường GPU có Unsloth.

```bash
# 1. Cài dependencies
pip install -e ".[dev,translate,train]"

# 2. Copy & chỉnh env
cp .env.example .env
# điền OPENAI_API_KEY, GEMINI_API_KEY, ALIBABA_API_KEY, ...

# 3. Collect dữ liệu
bash scripts/data/run_collect.sh

# 4. Translate Glaive + xLAM
bash scripts/data/run_translate_glaive.sh
bash scripts/data/run_translate_xlam.sh

# 5. QA translation
bash scripts/data/run_qa.sh

# 6. Archive generated pilot artifacts (dry-run first, then apply)
bash scripts/data/run_cleanup.sh
bash scripts/data/run_cleanup.sh --apply --timestamp 20260902T000000Z

# 7. Build and promote one frozen paired EN/VI revision
bash scripts/data/run_benchmark.sh \
  --revision 2026-09-02-full-dedup-seed42 \
  --feature-group-cache data/legacy/pilot_20260902T000000Z/benchmark_vi/.cache/feature_group.json

# 8. Materialize train-only artifacts (E0, E1, E2, E4, E5)
bash scripts/data/prepare_experiments.sh --overwrite

# 9. Smoke-test the exact Qwen3.5 templates in a GPU/Internet environment
bash scripts/train/smoke_native_qwen.sh
```

## 4. Data Schema

Schema master single-turn + multi-call (dùng chung cho cả 2 method):

```json
{
  "id": "glaive_00042",
  "source": "glaive",
  "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
  "function_calls": [
    {"name": "search_tutors", "arguments": {"subject": "Toán", "location": "Hà Nội"}}
  ],
  "tools": [
    {"name": "search_tutors", "description": "Tìm gia sư theo môn và khu vực.", "feature_group": "Tìm kiếm & Kết nối", "parameters": {"type": "object", "properties": {"subject": {"type": "string", "description": "Môn học cần tìm"}, "location": {"type": "string", "description": "Thành phố hoặc khu vực"}}, "required": ["subject", "location"]}}
  ]
}
```

Quy ước: `query` và `tools[].description` theo language split EN/VI; `function_calls[].name` và `arguments` keys là EN, values có thể VI/EN, `tools[].feature_group` là VI.

- **Method 1**: convert sang native `messages`/`tool_calls` qua `src/data/convert_to_instruction.py`; training render dùng `apply_chat_template()` của checkpoint.
- **Method 2**: dùng trực tiếp schema master để train Bi-Encoder + Cross-Encoder.

Sau bước materialize, `data/experiments/{e0,e1,e2,e4,e5}/` chỉ có train input native và `manifest.json`. Test/validation dùng trực tiếp từ frozen revision và `data/custom_vi/`; E3 chỉ tạo khi có general-SFT checkpoint độc lập.

Upload training data cùng frozen benchmark và CustomTools test lên Kaggle Dataset:

```bash
pip install kagglehub
python scripts/data/upload_experiments_to_kaggle.py <kaggle-username>/tool-calling-vi-experiments --dry-run
python scripts/data/upload_experiments_to_kaggle.py \
  <kaggle-username>/tool-calling-vi-experiments \
  --benchmark-revision data/benchmark_core/2026-09-02-full-dedup-seed42 \
  --custom-data data/custom_vi
```

`kagglehub` tự đọc token từ `~/.kaggle/access_token`; không ghi token vào source code hoặc commit vào Git.

## 5. Tech stack

| Thành phần | Công nghệ |
|---|---|
| Framework | PyTorch + Transformers |
| Config | Hydra (structured config, Python dataclass) |
| Method 1: SLM | `unsloth/Qwen3.5-4B`/`2B` + Unsloth QLoRA/SFT |
| Method 2: Bi-Encoder | BGE-M3 + FlagEmbedding + MultipleNegativesRankingLoss |
| Method 2: Cross-Encoder | BGE-M3 + Hierarchical heads, BERT-QA format |
| Dịch dataset | Alibaba OpenAI-compatible API (qwen3.7-flash / qwen3.7-max) |
| Baselines | OpenAI FC (gpt-4o-mini), Gemini FC (gemini-1.5-flash) |

## 6. Tài liệu chi tiết

- `AGENTS.md` — bộ nhớ cross-session.
- `docs/architecture.md` — sơ đồ 2 pipeline.
- `docs/methodology.md` — phương pháp nghiên cứu.
- `docs/benchmark.md` — cấu trúc benchmark tiếng Việt.
- `docs/translation_guidelines.md` — quy tắc dịch.
- `docs/references.md` — papers & resources.
- `docs/kaggle_notebook_guide.md` — chạy E0-E5 trên Kaggle.

## 7. Trạng thái

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Skeleton + docs | ✅ Done |
| 1 | Data pipeline | ⏳ In progress |
| 2 | Method 2: Bi-Encoder | ⏳ |
| 3 | Method 2: Cross-Encoder | ⏳ (skeleton có sẵn) |
| 4 | Method 1: SLM fine-tune | ⏳ |
| 5 | Baselines (OpenAI FC, Gemini FC) | ⏳ |
| 6 | Evaluation & comparison (4 methods) | ⏳ |
| 7 | Stress test (RAG-MCP inspired) | ⏳ |

## 8. References chính

- Ersoy et al. (2025) — Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning.
- RAG-MCP (2025) — arXiv:2505.03275.
- BGE-M3 (BAAI, 2024).

## 9. License

MIT
