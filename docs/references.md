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
  - **Rất liên quan đến đề tài**: họ **dịch 2 open-source tool-calling dataset sang tiếng Arabic**, nghiên cứu 3 câu hỏi:
    1. Có cần dữ liệu tool-calling bằng chính ngôn ngữ đó (Arabic) không, hay chỉ cần cross-lingual transfer?
    2. Hiệu quả của general-purpose instruction tuning lên tool-calling performance?
    3. Giá trị của fine-tune trên specific high-priority tools?
  - Dùng open-weight Arabic LLM (base + post-trained).
  - **Bài học rút ra cho đề tài**: chiến lược dịch + adapt dataset, đánh giá tác động của in-language data.

- **Berkeley Function Calling Leaderboard (BFCL)** — chuẩn benchmark phổ biến nhất cho tool calling (EN).
  - Đề tài tham khảo cấu trúc metric + categories (Live Simple, Live Multiple, Live Parallel, Multi-turn) khi xây benchmark_vi.

## 3. Semantic Retrieval

- **BGE-M3** (BAAI, 2024) — *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216.
- **Multilingual E5** (Wang et al., 2024) — *Multilingual E5 Text Embeddings: A Technical Report*. arXiv:2402.05672.
- **Sentence-BERT** (Reimers & Gurevych, 2019) — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.

## 4. Cross-Encoder & Schema-aware

- **BERT-QA (SQuAD)** (Devlin et al., 2019) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
  - Input format: `[CLS] question [SEP] context [SEP]`, span head `(start, end)` trên context tokens.
  - Đề tài **mượn ý tưởng**: query làm context, param schema làm question. Span head chỉ tính trên query tokens.
- **UIE — Universal Information Extraction** (Lu et al., 2022) — *Unified Structure Generation for Universal Information Extraction*. arXiv:2203.12277.
  - Span-based extraction cho structured information.
  - Đề tài tham khảo: dùng span prediction thay vì generation.
- **GLiNER** (Zaratiana et al., 2023) — *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer*. arXiv:2311.08526.
  - Span prediction NER, generalize cho nhiều loại entity.
  - Đề tài tham khảo: kiến trúc span + null classification.
- **DeepStruct** — span-based structured prediction.
- **Cross-Encoders for Re-ranking** — blog sentence-transformers (cho Bi-Encoder).
- **JSONFormer** (2023) — structured JSON generation (đã bỏ, không dùng trong đề tài).
- **Schema-Guided Reasoning** — tổng quan về schema-conditioned generation.

## 5. Generative LLM baselines

- **OpenAI Function Calling** — docs tại https://platform.openai.com/docs/guides/function-calling.
- **Google Gemini Function Calling** — docs tại https://ai.google.dev/gemini-api/docs/function-calling.

> Local LLM baseline (Qwen2.5/Llama-3.1 Unsloth + vLLM) **ngoài scope** khóa luận 3 tháng. Nếu mở rộng sau, tham khảo:
> - **Qwen2.5** (Alibaba, 2024) — *Qwen2.5 Technical Report*.
> - **Llama-3.1** (Meta, 2024) — *The Llama 3 Herd of Models*.
> - **Unsloth** — https://github.com/unslothai/unsloth.
> - **vLLM** (Kwon et al., 2023) — *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP'23.

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
  - **Đề tài mượn concept, tự build**:
    - Distractor strategies: `random` + `same_domain` (paper chỉ dùng random).
    - N range: 3 → 1000 (paper: 1 → 11100).
    - Tool pool: gộp unique tools từ Glaive + xLAM (sau dịch VI), domain-specific thay vì generic MCP.
    - Đo riêng retrieval vs extraction (paper chỉ test 1 model end-to-end).
- **Needle-in-a-Haystack (NIAH)** — gốc concept stress test với N varying.

---

> **Lưu ý**: Danh sách sẽ được cập nhật trong quá trình thực hiện. Mỗi paper/component implement xong sẽ cite đầy đủ trong báo cáo.
