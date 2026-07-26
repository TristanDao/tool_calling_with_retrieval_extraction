# References — Papers & Resources

## 1. Tool Calling / Function Calling — nền tảng

### Papers

- **Toolformer** (Schick et al., 2023) — *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761.
- **Gorilla** (Patil et al., 2023) — *Gorilla: Large Language Model Connected with Massive APIs*. arXiv:2305.15334.
- **ToolLLM** (Qin et al., 2023) — *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*. arXiv:2307.16789.
- **xLAM** (Salesforce, 2024) — *xLAM: A Family of Large Action Models*. arXiv:2409.03215.
- **ToolACE** (Liu et al., 2024) — *ToolACE: Winning the Points of LLM Function Calling*. arXiv:2409.00920.
- **AutoTool** (2024) — *AutoTool: A Tool Selection System for Large Language Models*.

### Datasets công khai (dùng trong đề tài)

- **Glaive Function Calling v2** — `glaiveai/glaive-function-calling-v2` trên HuggingFace.
- **ToolBench** — `OpenBMB/ToolBench` (API-Bank, RestBench subsets).
- **xLAM** — `Salesforce/xlam-function-calling-60k`.
- **ToolACE** — `Team-ACE/ToolACE` trên HuggingFace.

## 2. Semantic Retrieval

- **BGE-M3** (BAAI, 2024) — *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216.
- **Multilingual E5** (Wang et al., 2024) — *Multilingual E5 Text Embeddings: A Technical Report*. arXiv:2402.05672.
- **Sentence-BERT** (Reimers & Gurevych, 2019) — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.

## 3. Cross-Encoder & Schema-aware

- **Cross-Encoders for Re-ranking** — blog sentence-transformers.
- **JSONFormer** (2023) — structured JSON generation.
- **Schema-Guided Reasoning** — tổng quan về schema-conditioned generation.

## 4. Generative LLM baselines

- **OpenAI Function Calling** — docs tại https://platform.openai.com/docs/guides/function-calling.
- **Google Gemini Function Calling** — docs tại https://ai.google.dev/gemini-api/docs/function-calling.
- **Qwen2.5** (Alibaba, 2024) — *Qwen2.5 Technical Report*.
- **Llama-3.1** (Meta, 2024) — *The Llama 3 Herd of Models*.

## 5. Fine-tuning (Unsloth + LoRA)

- **Unsloth** — https://github.com/unslothai/unsloth.
- **LoRA** (Hu et al., 2021) — *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- **PEFT** (HuggingFace) — https://huggingface.co/docs/peft.

## 6. Serving

- **vLLM** (Kwon et al., 2023) — *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP'23.
- **Ollama** — alternative cho local LLM serve.

## 7. Translation

- **Qwen-MT** (Alibaba) — *Qwen-MT: Multilingual Translation Model* với 1M token context window.
- **DashScope API** — Alibaba Cloud API cho Qwen models.
- **WMT** — workshop về machine translation (general references).

## 8. Frameworks & Tools

- **PyTorch** — https://pytorch.org/.
- **HuggingFace Transformers** — https://huggingface.co/docs/transformers.
- **Hydra** — https://hydra.cc/ (config composition).
- **Weights & Biases** — https://wandb.ai/ (experiment tracking, optional).

## 9. Đánh giá

- **Recall@k, MRR, NDCG** — chuẩn metric cho retrieval.
- **BLEU, ROUGE** — chuẩn metric cho text generation (ít dùng cho tool calling).
- **JSON Schema validation** — RFC 8259 + JSON Schema draft-07/2020-12.
- **Argument-level F1/EM** — tự định nghĩa theo schema.

## 10. Tiếng Việt NLP

- **PhoBERT** (Nguyen & Nguyen, 2020) — *PhoBERT: Pre-trained language models for Vietnamese*. arXiv:2003.00744.
- **ViT5** (Phan et al., 2022) — *ViT5: Pretrained Text-to-Text Transformer for Vietnamese Language Generation*. arXiv:2205.06457.
- **BabelNet, VLSP** — nguồn tham khảo tiếng Việt.

---

> **Lưu ý**: Danh sách sẽ được cập nhật trong quá trình thực hiện. Mỗi paper/component implement xong sẽ cite đầy đủ trong báo cáo.
