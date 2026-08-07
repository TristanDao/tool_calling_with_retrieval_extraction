# References — Papers & Resources

## 1. Tool Calling / Function Calling — nền tảng

### Papers

- **Toolformer** (Schick et al., 2023) — *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761.
- **Gorilla** (Patil et al., 2023) — *Gorilla: Large Language Model Connected with Massive APIs*. arXiv:2305.15334.
- **ToolLLM** (Qin et al., 2023) — *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*. arXiv:2307.16789.
- **xLAM** (Salesforce, 2024) — *xLAM: A Family of Large Action Models*. arXiv:2409.03215.
- **ToolACE** (Liu et al., 2024) — *ToolACE: Winning the Points of LLM Function Calling*. arXiv:2409.00920.
- **AutoTool** (2024) — *AutoTool: A Tool Selection System for Large Language Models*.

### Datasets công khai (dùng trong đề tài — chỉ 2 nguồn chính)

- **Glaive Function Calling v2** — `glaiveai/glaive-function-calling-v2` trên HuggingFace (~110k samples).
- **xLAM** — `Salesforce/xlam-function-calling-60k` trên HuggingFace (~60k samples).

> Chỉ dùng 2 nguồn này. ToolBench, ToolACE không dùng trong khóa luận này (xem `AGENTS.md` section 10).

## 2. Tool Calling cho ngôn ngữ ít tài nguyên (multilingual / non-English)

- **Ersoy et al. (2025)** — *Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning*. ArabicNLP 2025 (co-located EMNLP), Suzhou, China. arXiv:2509.20957. ACL Anthology: 2025.arabicnlp-main.28.
  - **Primary reference cho Method 1 (SLM end-to-end)**:
    - Họ **dịch 2 open-source tool-calling dataset (Glaive + xLAM) sang tiếng Arabic**.
    - Fine-tune open-weight Arabic LLM (Fanar 9B) với instruction-tuning format.
    - LLM sinh `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` hoặc `<no_tool_call>`.
    - Metric chính: **ArgA (Argument Population Accuracy)** — end-to-end accuracy.
    - Dùng LLaMA-Factory, lr=5e-7 cosine schedule.
  - **Đề tài làm tương tự cho tiếng Việt** với Qwen2.5 0.5B/1.5B (SLM).
  - **3 câu hỏi họ nghiên cứu**:
    1. Cần dữ liệu tool-calling bằng chính ngôn ngữ đó không?
    2. Hiệu quả của general-purpose instruction tuning?
    3. Giá trị của fine-tune trên specific high-priority tools?
  - **Kết luận chính**: in-language data quan trọng cho ArgA; paraphrasing variance (50.2%) là nguyên nhân lỗi hàng đầu; tool-specific fine-tune cho kết quả gần perfect.

- **Berkeley Function Calling Leaderboard (BFCL)** — chuẩn benchmark phổ biến nhất cho tool calling (EN).
  - Đề tài tham khảo cấu trúc metric + categories (Live Simple, Live Multiple, Live Parallel, Multi-turn) khi xây benchmark_vi.

## 3. Semantic Retrieval

- **BGE-M3** (BAAI, 2024) — *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216.
- **Multilingual E5** (Wang et al., 2024) — *Multilingual E5 Text Embeddings: A Technical Report*. arXiv:2402.05672.
- **Sentence-BERT** (Reimers & Gurevych, 2019) — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.

## 4. Cross-Encoder & Schema-aware

- **BERT-QA (SQuAD)** (Devlin et al., 2019) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
  - Input format: `[CLS] question [SEP] context [SEP]`, span head `(start, end)` trên context tokens.
- **UIE — Universal Information Extraction** (Lu et al., 2022) — *Unified Structure Generation for Universal Information Extraction*. arXiv:2203.12277.
- **GLiNER** (Zaratiana et al., 2023) — *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer*. arXiv:2311.08526.

## 5. Generative LLM baselines

- **OpenAI Function Calling** — https://platform.openai.com/docs/guides/function-calling.
- **Google Gemini Function Calling** — https://ai.google.dev/gemini-api/docs/function-calling.
- **Qwen2.5** (Alibaba, 2024) — *Qwen2.5: A Party of Foundation Models*. Dùng làm base model cho Method 1 (SLM fine-tune 0.5B/1.5B).
- **LLaMA-Factory** (Zheng et al., 2024) — *LLaMAFactory: Unified Efficient Fine-Tuning of 100+ Language Models*. ACL 2024. Dùng cho Method 1 instruction tuning.

## 6. Translation

- **Qwen-MT** (Alibaba) — *Qwen-MT: Multilingual Translation Model* với 1M token context window.
- **DashScope API** — Alibaba Cloud API cho Qwen models.
- **WMT** — workshop về machine translation (general references).

## 7. Frameworks & Tools

- **PyTorch** — https://pytorch.org/.
- **HuggingFace Transformers** — https://huggingface.co/docs/transformers.
- **FlagEmbedding** (BAAI) — https://github.com/FlagOpen/FlagEmbedding (dùng cho Bi-Encoder).
- **Hydra** — https://hydra.cc/ (config composition).
- **Weights & Biases** — https://wandb.ai/ (experiment tracking, optional).

## 8. Đánh giá

- **AISA-ArabicFC Shared Task (ArabicNLP 2026)** — Track A đánh giá call detection, function selection và argument extraction; Track C phân tầng robustness và đo performance gap. Đề tài kế thừa cách phân rã chức năng và tổng quát hóa ý tưởng Track C sang seen/unseen tool, domain, multi-call và schema complexity. https://huggingface.co/spaces/TuwaiqAcademy/AISA-ArabicFC-Shared-Task
- **Recall@k, MRR, NDCG** — chuẩn metric cho retrieval.
- **BLEU, ROUGE** — chuẩn metric cho text generation (ít dùng cho tool calling).
- **JSON Schema validation** — RFC 8259 + JSON Schema draft-07/2020-12.
- **Argument-level F1/EM** — tự định nghĩa theo schema.

## 9. Tiếng Việt NLP

- **PhoBERT** (Nguyen & Nguyen, 2020) — *PhoBERT: Pre-trained language models for Vietnamese*. arXiv:2003.00744.
- **ViT5** (Phan et al., 2022) — *ViT5: Pretrained Text-to-Text Transformer for Vietnamese Language Generation*. arXiv:2205.06457.
- **BabelNet, VLSP** — nguồn tham khảo tiếng Việt.

## 10. Stress Test (RAG-MCP inspired)

- **RAG-MCP** (Gao et al., 2025) — *RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation*. arXiv:2505.03275.
  - Concept: vary số candidate tools N, đo degradation curve.
  - Setup: 1 ground-truth + (N-1) random distractors, vary N từ 1 → 11100.
  - **Đề tài mượn concept**:
    - So sánh cả 4 methods (Method 1 SLM + Method 2 Bi+Cross + OpenAI FC + Gemini FC).
    - Distractor strategies: `random` + `same_domain` (paper chỉ dùng random).
    - N range: 3 → 1000.
    - Tool pool: gộp unique tools từ Glaive + xLAM (sau dịch VI).
- **Needle-in-a-Haystack (NIAH)** — gốc concept stress test với N varying.

---

> **Lưu ý**: Danh sách sẽ được cập nhật trong quá trình thực hiện. Mỗi paper/component implement xong sẽ cite đầy đủ trong báo cáo.
