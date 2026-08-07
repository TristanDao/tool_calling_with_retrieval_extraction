# Tool Calling tiếng Việt — So sánh 2 Phương pháp

> Đồ án / luận văn UIT: so sánh SLM End-to-End vs Bi-Encoder + Cross-Encoder cho Tool Calling tiếng Việt.

## 1. Tổng quan

Hệ thống so sánh **2 phương pháp** Tool Calling cho tiếng Việt:

| Phương pháp | Cách làm | Model |
|---|---|---|
| **Method 1: SLM End-to-End** | Fine-tune LLM chọn tool + điền tham số (instruction-tuning) | Qwen2.5 0.5B/1.5B + LLaMA-Factory |
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
├── data/                 # raw / translations / benchmark_vi
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

Phase 1 data pipeline và Phase 6 evaluator framework đang được triển khai song song.

```bash
# 1. Cài dependencies
pip install -e ".[dev,translate]"

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

# 6. Build benchmark
bash scripts/data/run_benchmark.sh

# 7. Evaluate một prediction file
python -m src.evaluation.cli evaluate \
  --gold data/benchmark_vi/test.jsonl \
  --predictions results/slm/predictions.jsonl \
  --output-dir results/evaluation/slm
```

Evaluator xuất `report.json`, `per_sample.jsonl`, `summary.md` và hỗ trợ compare nhiều method. Contract prediction, metric definitions và lệnh đầy đủ: `docs/evaluation.md`.

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

Quy ước: `query` VI, `function_calls[].name` + `arguments` keys EN, values có thể VI/EN, `tools[].description` VI, `tools[].feature_group` VI.

- **Method 1**: convert sang instruction format qua `src/data/convert_to_instruction.py`.
- **Method 2**: dùng trực tiếp schema master để train Bi-Encoder + Cross-Encoder.

## 5. Tech stack

| Thành phần | Công nghệ |
|---|---|
| Framework | PyTorch + Transformers |
| Config | Hydra (structured config, Python dataclass) |
| Method 1: SLM | Qwen2.5 0.5B/1.5B + LLaMA-Factory |
| Method 2: Bi-Encoder | BGE-M3 + FlagEmbedding + MultipleNegativesRankingLoss |
| Method 2: Cross-Encoder | BGE-M3 + Hierarchical heads, BERT-QA format |
| Dịch dataset | Alibaba OpenAI-compatible API (qwen3.7-flash / qwen3.7-max) |
| Baselines | OpenAI FC (gpt-4o-mini), Gemini FC (gemini-1.5-flash) |

## 6. Tài liệu chi tiết

- `AGENTS.md` — bộ nhớ cross-session.
- `docs/architecture.md` — sơ đồ 2 pipeline.
- `docs/methodology.md` — phương pháp nghiên cứu.
- `docs/benchmark.md` — cấu trúc benchmark tiếng Việt.
- `docs/evaluation.md` — contract input/output, metric và hướng dẫn chạy evaluator.
- `docs/evaluation_methodology_thesis.md` — phương pháp, kỹ thuật, vai trò và ý nghĩa của phần đánh giá theo văn phong khóa luận.
- `docs/translation_guidelines.md` — quy tắc dịch.
- `docs/references.md` — papers & resources.

## 7. Trạng thái

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Skeleton + docs | ✅ Done |
| 1 | Data pipeline | ⏳ In progress |
| 2 | Method 2: Bi-Encoder | ⏳ |
| 3 | Method 2: Cross-Encoder | ⏳ (skeleton có sẵn) |
| 4 | Method 1: SLM fine-tune | ⏳ |
| 5 | Baselines (OpenAI FC, Gemini FC) | ⏳ |
| 6 | Evaluation & comparison (4 methods) | ⏳ Framework done; chờ predictions/experiments |
| 7 | Stress test (RAG-MCP inspired) | ⏳ |

## 8. References chính

- Ersoy et al. (2025) — Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning.
- RAG-MCP (2025) — arXiv:2505.03275.
- BGE-M3 (BAAI, 2024).

## 9. License

MIT
