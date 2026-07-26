# Tool Calling tiếng Việt — Semantic Retrieval + Schema-aware Parameter Extraction

> Đồ án / luận văn UIT: Nghiên cứu bài toán **Tool Calling (Function Calling)** cho tiếng Việt
> theo hướng **tách thành 2 thành phần chuyên biệt** (Bi-Encoder retrieval + Cross-Encoder extraction)
> nhằm **giảm latency + cost** so với generative LLM, **giữ độ chính xác cạnh tranh**.

---

## 1. Tổng quan

Hệ thống pipeline gồm 3 bước chính:

```
User query (VI) ──► [Bi-Encoder: chọn tool] ──► [Cross-Encoder: trích xuất args] ──► JSON hợp lệ
                       │                              │
                       ▼                              ▼
                  tool_schema (EN)               tool_schema (EN)
```

- **Bi-Encoder** (Semantic Tool Retrieval): dùng **BGE-M3** + **FlagEmbedding** + **MultipleNegativesRankingLoss** để retrieve top-k tool phù hợp.
- **Cross-Encoder** (Schema-aware Parameter Extraction): dùng **BGE-M3** + **Span Prediction head** (3 loại output: span/enum/null) để trích xuất arguments.
- **Validator**: kiểm tra arguments hợp lệ + khớp schema.

Baselines so sánh:
- OpenAI Function Calling (`gpt-4o-mini`)
- Google Gemini Function Calling (`gemini-1.5-flash`)

> Local LLM baseline (Qwen2.5/Llama-3.1) ngoài scope khóa luận 3 tháng.

---

## 2. Cấu trúc thư mục

```
tool_calling_with_retrieval_extraction/
├── AGENTS.md             # Cross-session memory (đọc đầu tiên)
├── README.md             # File này
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── configs/              # Hydra structured config
├── data/                 # raw / processed / benchmark_vi / translations
├── src/                  # data/ models/ pipeline/ evaluation/ utils/
├── scripts/              # CLI scripts (data, train, serve, eval)
├── notebooks/            # EDA + analysis
├── tests/                # unit tests
├── checkpoints/          # model weights (gitignored)
├── results/              # metrics + tables/figures
├── docs/                 # architecture, methodology, benchmark, translation_guidelines, references
└── logs/                 # training/eval logs
```

Chi tiết xem `AGENTS.md` section 6 và `docs/architecture.md`.

---

## 3. Quick start (sau khi code sẵn sàng)

> Hiện tại repo đang ở **Phase 0 — Skeleton**. Các command dưới đây sẽ hoạt động khi code được implement.

```bash
# 1. Cài dependencies
pip install -e ".[dev,translate]"

# 2. Copy & chỉnh env
cp .env.example .env
# điền OPENAI_API_KEY, GEMINI_API_KEY, DASHSCOPE_API_KEY, ...

# 3. Build benchmark
bash scripts/data/05_build_benchmark.sh

# 4. Train Bi-Encoder
bash scripts/train/train_biencoder.sh

# 5. Train Cross-Encoder
bash scripts/train/train_crossencoder.sh

# 6. Run pipeline + baselines + comparison
bash scripts/compare_all.sh
```

---

## 4. Data Schema

Mỗi sample trong benchmark tiếng Việt:

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

Quy ước: `query` VI, `function_call.name` + `arguments.keys` EN, values có thể VI/EN, `description` VI.

---

## 5. Tech stack

| Thành phần | Công nghệ |
|---|---|
| Framework | PyTorch + Transformers |
| Config | Hydra (structured config, Python dataclass) |
| Bi-Encoder | BGE-M3 + FlagEmbedding + MultipleNegativesRankingLoss |
| Cross-Encoder | BGE-M3 + Span Prediction head (3 output: span/enum/null), format `[CLS] schema [SEP] query [SEP]` |
| Dịch dataset | Qwen-MT (Alibaba, DashScope API) |
| LLM baseline | OpenAI FC, Gemini FC |

---

## 6. Tài liệu chi tiết

- [AGENTS.md](./AGENTS.md) — bộ nhớ cross-session (đọc đầu tiên mỗi phiên).
- [docs/architecture.md](./docs/architecture.md) — sơ đồ pipeline.
- [docs/methodology.md](./docs/methodology.md) — phương pháp nghiên cứu.
- [docs/benchmark.md](./docs/benchmark.md) — cấu trúc benchmark tiếng Việt.
- [docs/translation_guidelines.md](./docs/translation_guidelines.md) — quy tắc dịch Qwen-MT.
- [docs/references.md](./docs/references.md) — papers & resources.

---

## 7. Trạng thái

- [x] **Phase 0**: Skeleton (folders + .md files)
- [ ] Phase 1: Data pipeline
- [ ] Phase 2: Bi-Encoder
- [ ] Phase 3: Cross-Encoder
- [ ] Phase 4: Pipeline + validator
- [ ] Phase 5: Baselines
- [ ] Phase 6: Evaluation & comparison (test chính)
- [ ] **Phase 7**: Stress test (RAG-MCP inspired) — vary N candidates, plot degradation curve

---

## 8. References chính

- **Ersoy et al. (2025)** — *Tool Calling for Arabic LLMs* (ArabicNLP 2025). Tham khảo chiến lược dịch dataset tool-calling sang ngôn ngữ ít tài nguyên.
- **RAG-MCP (2025)** — arXiv:2505.03275. Mượn concept stress test (vary N, plot curve).
- **BGE-M3** (BAAI, 2024) — base encoder cho retrieval.
- Xem đầy đủ tại `docs/references.md`.

---

## 9. License

MIT
