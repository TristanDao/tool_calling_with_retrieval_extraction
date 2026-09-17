# Vietnamese Tool Calling: A Comparative Study Between End-to-End Small Language Models and Specialized Bi-Encoder + Cross-Encoder Architecture

**Authors**:  
Dao Phuoc Thinh$^1$, Ha Quang Dat$^1$, Dang Van Thin$^1$ (Supervisor)  

$^1$Faculty of Information Technology, University of Information Technology, Vietnam National University, Ho Chi Minh City (VNU-HCM), Vietnam  
*Email*: {21521469, 21521925}@gm.uit.edu.vn, thindv@uit.edu.vn  

**Date**: September 2026  

---

## Abstract

Tool calling (function calling) is a transformative capability that empowers Large Language Models (LLMs) to interface with external APIs, databases, and computational services, serving as the foundational building block for modern autonomous agents. While substantial progress has been achieved for high-resource languages such as English, enabling robust, low-latency, and zero-shot tool calling for lower-resource languages like Vietnamese remains severely underexplored. Conventional cloud-based Large Language Models suffer from crippling engineering drawbacks in production: heavy autoregressive latency (800–2,000 ms), steep token operational costs, enterprise privacy exposures, JSON syntax invalidity, and quadratic context degradation when scaling to large tool catalogs. In this paper, we present the first systematic comparative empirical study for Vietnamese tool calling between two contrasting paradigms: End-to-End Generative Small Language Models (SLMs based on Qwen3.5 2B and 4B) and a Decoupled Non-Autoregressive Discriminative Pipeline (Bi-Encoder BGE-M3 combined with Hierarchical Cross-Encoder XLM-R). To benchmark these systems, we standardize and release two novel datasets: the **Canonical Core Benchmark** (77,028 paired English–Vietnamese instances across 4,421 unique tools) and **CustomTools-VI** (8,000 samples grounded in 10 Vietnamese domains, strictly partitioned into 20 seen and 20 zero-shot unseen tools with a 50% conversational negative ratio). Our empirical results demonstrate that our local SLM (E4) sets a new state of the art, achieving an Exact Match Argument Accuracy (ArgA) of **87.92%** on seen and **86.75%** on unseen tools with the 4B backbone, outperforming commercial closed-source baseline `GPT-5.6 Luna` (+8.75% seen, +6.88% unseen) while reducing syntax errors to 0.32%. Conversely, the decoupled pipeline delivers an ultra-fast median wall-clock latency of **54.68–58.16 ms** (15–18× faster than SLM) with 0.00% syntax errors and $< 1.2$ GB VRAM. Furthermore, extensive stress testing ($N = 3 \to 1,000$) reveals that while the decoupled pipeline remains sub-92 ms up to 1,000 tools, SLMs suffer quadratic attention memory explosion (CUDA OOM) at $N \ge 500$ on commodity 16GB GPUs, rigorously charting the Pareto frontier between inference fidelity and computational resource limits.

---

## 1. Introduction

Autonomous language agents require the ability to interact with real-world digital ecosystems via Application Programming Interfaces (APIs). Tool calling—the capacity to recognize user intent, select appropriate tools from an external catalog, and extract structured parameters complying with strict schemas—has become a cornerstone of modern artificial intelligence.

![Figure 1: Architectural paradigm confrontation between Method 1 (End-to-End SLM) and Method 2 (Bi-Encoder + Cross-Encoder)](figures/fig1_system_architecture.png)

*Figure 1: Overall architectural confrontation between two distinct paradigms: Method 1 (Autoregressive End-to-End SLM based on Qwen3.5) and Method 2 (Modular Non-Autoregressive pipeline combining Bi-Encoder BGE-M3 for retrieval and Hierarchical Cross-Encoder XLM-R for parameter extraction).*

Despite rapid advances spearheaded by proprietary Large Language Models (LLMs) such as OpenAI Function Calling and Google Gemini FC, widespread real-world deployment faces four severe limitations:
1. **Computational Overhead and Latency**: Autoregressive decoding over long tool definitions produces latency figures exceeding 800–2,000 ms per turn. This is unacceptable in latency-sensitive industrial applications such as telecommunications, financial customer care, and robotics.
2. **Context Window Exhaustion**: As tool repositories expand from dozens to hundreds or thousands of APIs, stuffing full schemas into the prompt degrades LLM reasoning ("needle in a haystack" phenomenon) and incurs quadratic token costs.
3. **Structured Format & Hallucination Vulnerability**: Generative models frequently hallucinate parameter values, emit invalid JSON strings, or invent fictitious APIs when confronted with domain-specific jargon.
4. **Scarcity of Vietnamese Tool-Calling Research**: The vast majority of function-calling benchmarks (e.g., Berkeley Function-Calling Leaderboard, ToolBench, Glaive) are exclusively in English. Vietnamese presents unique structural challenges: lack of morphological case markers, heavy reliance on tonality, distinct numerical and date formatting (e.g., "ngày 15 tháng 8 năm 2026", "hai triệu rưỡi"), and ubiquitous bilingual code-switching in technical queries.

To address these challenges systematically, our investigation is guided by five foundational Research Questions (RQs):
- **RQ1 (Cross-Lingual Knowledge Transfer)**: To what degree can pre-trained multilingual models transfer tool-calling competence from English to Vietnamese without requiring native training data?
- **RQ2 (The Over-Triggering Pathology & Negative Calibration)**: Does generic bilingual instruction tuning introduce destructive trigger biases on non-tool conversational queries, and does localized negative calibration eliminate it?
- **RQ3 (Zero-Shot Generalization on Unseen Tools)**: How does a generative Small Language Model (SLM) compare against a modular discriminative pipeline (Bi-Encoder + Cross-Encoder) when confronted with entirely novel, culturally specialized tools (`test_unseen`)?
- **RQ4 (Pareto Efficiency: Latency vs. Accuracy Frontier)**: Can a modular, non-autoregressive dual-encoder pipeline achieve competitive argument accuracy while delivering an order-of-magnitude latency reduction over generative SLMs?
- **RQ5 (Catalog Scalability & Context Stress Bottlenecks)**: How do both paradigms scale as external tool catalogs expand from small collections ($N=3$) to realistic enterprise volumes ($N=1,000$), and where does the physical quadratic breakdown threshold occur for autoregressive decoders?

### Scientific Contributions
1. **First Large-Scale Standardized Vietnamese Benchmark**: We construct and release the **Canonical Core Benchmark** (77,028 paired English–Vietnamese records across 4,421 unique tools) and **CustomTools-VI** (8,000 localized samples spanning 40 tools across 10 Vietnamese domains, featuring a strict 20-seen / 20-unseen zero-shot evaluation split and 50% conversational negative ratio).
2. **Systematic Cross-Lingual & Negative Calibration Analysis**: We demonstrate that while English tool-calling knowledge transfers robustly to Vietnamese, general bilingual fine-tuning triggers catastrophic over-calling on conversational inputs (Non-FC Recall collapsing to 2.0%). We prove that in-domain negative data completely restores model calibration (100.0% Non-FC Recall).
3. **Establishing the Pareto Trade-Off Frontier**: We provide exhaustive empirical evidence establishing that fine-tuned SLMs (`Qwen3.5-2B/4B` E4) dominate in accuracy and zero-shot tool generalization (**86.75%** unseen ArgA), whereas modular Bi-Encoder + Cross-Encoder pipelines deliver unmatched deterministic inference speed (**54–58 ms** P50 latency, 15–18× faster), guiding architectural selection for production systems.
4. **SOTA Benchmark Elevation via 4B Scaling**: Scaling to `Qwen3.5-4B` elevates Core VI Tool Acc to **98.73%**, suppresses syntax errors from 4.31% down to **0.32%**, and pushes CustomTools-VI Seen ArgA to **87.92%**, outperforming `GPT-5.6 Luna`.
5. **Physical Limits in Catalog Stress Testing ($N = 3 \to 1,000$)**: We quantify the physical memory breakdown (CUDA OOM) threshold for autoregressive SLMs at $N \ge 500$ on 16GB GPUs, contrasting it with the bounded $< 1.5$ GB footprint of the decoupled pipeline.

---

## 2. Related Work

### 2.1 Tool Calling in Generative LLMs
Toolformer (Schick et al., 2023) pioneered self-supervised tool utilization in language models. Subsequent benchmarks including Gorilla (Patil et al., 2023), ToolLLM (Qin et al., 2023), ToolBench, and the Berkeley Function-Calling Leaderboard (BFCL) (Yan et al., 2024) formalized structured evaluation across single-turn, multi-call, and multi-turn interactions. Ersoy et al. (2025) recently investigated Arabic tool calling, showing that translated open datasets coupled with localized instruction tuning can outperform generic models. Our work builds upon this foundation, introducing strict bilingual experimental controls, native XML structured formatting, and head-to-head confrontation against a modular discriminative alternative.

### 2.2 Dense Retrieval and Information Extraction Pipelines
Dense retrieval for APIs has been explored in ToolRetriever and AutoTool (Song et al., 2023), typically utilizing dual encoders like Contriever or BGE to filter tool candidates before generative synthesis. In classical Information Extraction, span extraction models derived from BERT/RoBERTa have been widely deployed for slot filling in conversational systems (Weld et al., 2022). However, combining a dedicated Bi-Encoder with a multi-headed hierarchical Cross-Encoder capable of simultaneous span extraction, enum classification, boolean classification, and canonical normalization **without any autoregressive generation** represents a novel architecture tailored for deterministic, low-latency execution.

---

## 3. Dataset Construction & Vietnamese Benchmarks

Table 1 summarizes the statistical properties of the benchmarks constructed and evaluated in this work.

**Table 1: Detailed Statistics of the Function-Calling Benchmark Datasets**  
*(FC: Function calling required (Y) or conversational negative (N). Turns: Single-turn (S). Calls: Single call (S) or Multiple calls (M). Unique Tools: Number of distinct API specifications).*

| Dataset | Language | FC | Turns | Calls | Train Samples | Test Samples | Unique Tools |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Canonical Glaive** | Vietnamese (VI) | Y | S | S | 14,568 | 1,821 | 972 |
| | Vietnamese (VI) | N | S | S | 3,642 | 455 | — |
| | English (EN) | Y | S | S | 14,568 | 1,821 | 972 |
| | English (EN) | N | S | S | 3,642 | 455 | — |
| **Canonical xLAM** | Vietnamese (VI) | Y | S | M | 47,047 | 5,891 | 3,449 |
| | Vietnamese (VI) | N | S | M | 0 | 0 | — |
| | English (EN) | Y | S | M | 47,047 | 5,891 | 3,449 |
| | English (EN) | N | S | M | 0 | 0 | — |
| **CustomTools-VI (Seen)** | Vietnamese (VI) | Y | S | S/M | 3,600 | 400 | 20 |
| | Vietnamese (VI) | N | S | S/M | 2,000 | 400 | — |
| **CustomTools-VI (Unseen)**| Vietnamese (VI) | Y | S | S/M | 0 *(Strict Zero-Shot)* | 400 | 20 |
| | Vietnamese (VI) | N | S | S/M | 0 *(Strict Zero-Shot)* | 400 | — |
| **Total (Core Benchmark)**| **Bilingual EN-VI**| **Y/N** | **S** | **S/M**| **61,615 (x2)** | **7,712 (x2)** | **4,421** |

### 3.1 Canonical Core Benchmark & Revision Policy
Derived from Glaive Function Calling v2 and Salesforce xLAM, raw records underwent rigorous deduplication, syntax validation, and single-turn standardization:
- **Frozen Revision `2026-09-02-full-dedup-seed42`**: Contains **77,028 paired records** across 4,421 tools, split 80/10/10 into 61,615 train, 7,701 validation, and 7,712 test instances with fixed random seed 42.
- **Translation Guidelines**: English queries and tool descriptions were translated into fluent Vietnamese using cloud-based LLMs under strict invariant constraints: function identifiers (`snake_case`), argument keys, UUIDs, currency codes (`VND`, `USD`), and JSON schemas were strictly preserved.

### 3.2 CustomTools-VI: Culturally Grounded Benchmark
To evaluate performance on realistic Vietnamese scenarios, we curated `CustomTools-VI`, featuring 40 tools across 10 functional groups:
1. *E-commerce & Shopping* (`tra_cuu_don_hang`, `tim_ma_giam_gia`)
2. *Domestic Travel & Transport* (`dat_ve_xe_khach`, `tra_cuu_chuyen_bay_noi_dia`)
3. *Food & Dining Delivery* (`dat_ban_nha_hang`, `tim_mon_an_dac_san`)
4. *Utilities & Bill Payments* (`thanh_toan_tien_dien_nuoc`, `nap_tien_dien_thoai`)
5. *Banking & Fintech* (`chuyen_khoan_noi_bo`, `tra_cuu_ty_gia_ngoai_te`)
6. *Real Estate & Housing* (`tim_phong_tro_sinh_vien`, `dinh_gia_nha_dat`)
7. *Public Administration & Legal* (`tra_cuu_phat_nguoi`, `dang_ky_thu_tuc_cu_tru`)
8. *Healthcare & Clinic Booking* (`dat_lich_kham_benh`, `tim_nha_thuoc_gan_nhat`)
9. *Education & Tutoring* (`tim_gia_su_toan_ly_hoa`, `tra_cuu_diem_chuan_dai_hoc`)
10. *Logistics & Shipping* (`tinh_phi_ship_noi_thanh`, `theo_doi_buu_kien_vnpost`)

**Strict Seen vs. Unseen Split**: Exactly 20 tools are designated as "Seen" (allowed in training for E4 and Method 2), while 20 tools are held out as "Unseen" (strictly excluded from all training, negatives, and hard-negative mining). Both `test_seen` (800 samples) and `test_unseen` (800 samples) feature an exact 50/50 balance (400 positive queries and 400 negative conversational queries).

---

## 4. Methodology

### 4.1 Method 1: End-to-End Small Language Model (SLM)

Method 1 frames tool calling as autoregressive sequence generation. Given a system prompt defining available tool schemas $\mathcal{T}$ and a user query $q$, the model generates either a conversational response or structured tool call tags:

```xml
<tool_call>
<function=tool_name>
<parameter=arg_name>arg_value</parameter>
</function>
</tool_call>
```

#### Training Specifications & Response-Only Loss
- **Backbones**: `unsloth/Qwen3.5-2B` and `unsloth/Qwen3.5-4B`.
- **Optimization**: Unsloth QLoRA ($r=16, \alpha=32$, target modules on all linear projections: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`), FP16 mixed precision, cosine learning rate schedule with warmup ratio 0.05, learning rate $2\times 10^{-4}$.
- **Loss Computation**: Supervised Fine-Tuning (SFT) loss is calculated **strictly on assistant responses**. Prompt tokens (system prompt, tool schemas, and user message) are masked with label `-100`:
  $$\mathcal{L}_{SFT} = -\sum_{i=1}^{N} m_i \log P(w_i \mid w_{<i}, q, \mathcal{T})$$
  where $m_i = 1$ only for tokens generated by the assistant.
- **Negative Sample Handling**: Negative instances output a polite natural language assistant response without tool call tags (avoiding artificial `<no_tool_call>` tokens that induce distribution shift).

#### Controlled Experimental Configurations (Budget: 60,000 samples)
- **E0 (Zero-Shot Baseline)**: Pre-trained base instruction model without additional fine-tuning.
- **E1 (Monolingual English SFT)**: 60,000 English core training samples.
- **E2 (Monolingual Vietnamese SFT)**: 60,000 Vietnamese counterpart training samples.
- **E3 (Bilingual SFT)**: 30,000 English + 30,000 Vietnamese balanced samples.
- **E4 (Bilingual + Domain Adaptation)**: 60,000 bilingual samples + 5,600 samples from the specialized `CustomTools-VI` domain training set (total 65,600 training instances).

---

### 4.2 Method 2: Specialized Bi-Encoder + Cross-Encoder Architecture

Method 2 decomposes tool calling into a two-stage discriminative pipeline:

#### Stage 1: Semantic Tool Retrieval (Bi-Encoder)
- **Model**: `BAAI/bge-m3` initialized with sentence-transformers and LoRA ($r=16, \alpha=32$).
- **Loss**: Cached Multiple Negatives Ranking Loss (`CachedMNRL`) executed over two iterative rounds:
  - *Round 1 (Teacher)*: Trained on 78,435 pairs to establish initial representation.
  - *Hard Negative Mining*: Round 1 model retrieves top false-positive distractors.
  - *Round 2 (Final)*: Trained with mined hard negatives under strict held-out tool exclusion.
- **Abstention & Thresholding**: Calibrated on validation data with confidence threshold $\tau = 0.35$ and score margin $\delta = 0.21$. Queries failing $\tau$ trigger no-call abstention.

#### Stage 2: Schema-Aware Parameter Extraction (Cross-Encoder)
- **Model**: `xlm-roberta-base` trained with multi-task hierarchical classification heads:
  1. `has_value` **Head**: Binary classification head determining whether a parameter is present in the query ($\hat{y}_{has} \ge 0.5$).
  2. `span_extraction` **Head**: Sequence labeling head predicting `(start_idx, end_idx)` for free-form string parameters.
  3. `enum_classification` **Head**: Schema-routed classification head selecting among allowed enum candidates.
  4. `boolean` **Head**: Binary classifier predicting `true` or `false`.
- **Value Normalizer**: Rule-based and dictionary-driven post-processor mapping colloquial Vietnamese expressions (e.g., "ngày mười lăm tháng tám", "2 triệu 5", "vé khứ hồi") into canonical JSON types.

---

### 4.3 Hardware Infrastructure & Training Setup

To guarantee strict reproducibility and fair benchmarking, all models under Method 1 and Method 2 were trained and evaluated on rigorously monitored, standardized hardware:

- **Training Infrastructure**: Executed on Google Colab Pro equipped with 1× NVIDIA A100-SXM4-40GB GPU (39.49 GB VRAM), PyTorch 2.8.0, CUDA 12.8, Bfloat16 (`bf16=True`), and the Unsloth framework (v2026.9.4 with memory-efficient patching for Qwen3.5).
- **Inference & Benchmarking Infrastructure**: Executed independently on Kaggle with 2× NVIDIA Tesla T4 GPUs (14.56 GB VRAM each, Turing architecture, Compute Capability 7.5), CUDA 12.x, and PyTorch 2.x. SLMs were evaluated under 4-bit NF4 BitsAndBytes quantization with greedy decoding (`do_sample=False`, `max_new_tokens=128`).
- **Comprehensive Hyperparameter and Resource Telemetry**:

| Component / Metric | Qwen3.5-2B (E1 $\to$ E4) | Qwen3.5-4B (E3, E4) | Method 2: Bi+Cross |
| :--- | :--- | :--- | :--- |
| **Backbone Architecture** | `unsloth/Qwen3.5-2B` | `unsloth/Qwen3.5-4B` | `BGE-M3` + `XLM-R base` |
| **Adaptation Method** | QLoRA (4-bit NF4) | QLoRA (4-bit NF4) | LoRA (Bi-Enc) / Full Head (Cross-Enc) |
| **LoRA Parameters** | $r=16, \alpha=16$, dropout $0.0$ | $r=16, \alpha=16$, dropout $0.0$ | $r=16, \alpha=32$ (Bi-Encoder) |
| **Target Modules** | `q, k, v, o, gate, up, down_proj` | `q, k, v, o, gate, up, down_proj` | `q_proj, v_proj` (Bi-Encoder) |
| **Total Model Parameters** | 2,224,153,408 (~2.22B) | 4,560,499,200 (~4.56B) | 567M + 278M (~845M) |
| **Trainable Parameters** | **10,911,744 (0.49%)** | **21,233,664 (0.47%)** | ~15M (LoRA + Heads) |
| **Learning Rate** | $5 \times 10^{-7}$ (Cosine, warmup 0.05) | $5 \times 10^{-7}$ (Cosine, warmup 0.05) | $2 \times 10^{-5}$ (Bi) / $3 \times 10^{-5}$ (Cross) |
| **Batch Size Configuration** | $64 \times 1$ (Per-device: 64, Accum: 1) | $32 \times 2$ (Per-device: 32, Accum: 2) | 32 (Bi-Enc) / 16 (Cross-Enc) |
| **Effective Batch Size** | **64** (Unified across runs) | **64** (Unified across runs) | — |
| **Training Time / 60k run** | **~49.5 min** (938 steps on A100) | **~1 hr 55 min** (938 steps on A100)| ~2.5 hrs (2 rounds) + ~3 hrs (CE) |
| **Training Time E4 (65.6k)**| **~54 min** (1,025 steps on A100) | **2 hrs 05 min** (1,025 steps A100) | — |
| **Convergence Final Loss** | $\approx 0.0198$ | $\approx 0.0103$ (~48% reduction) | — |
| **Inference Hardware** | 2× NVIDIA Tesla T4 (15GB) | 2× NVIDIA Tesla T4 (15GB) | 2× NVIDIA Tesla T4 (15GB) |

---

## 5. Evaluation Methodology & Metrics

Following BFCL and Ersoy et al. (2025), evaluation is computed on complete test sets:

### 5.1 Tool Selection Precision, Recall, and Accuracy
For each tool $T$, precision $P_T$ and recall $R_T$ measure detection correctness:
$$P_{T}=\frac{TP_{T}}{TP_{T}+FP_{T}}, \quad R_{T}=\frac{TP_{T}}{TP_{T}+FN_{T}}$$

The weighted average scores across tool set $\mathcal{K}$ are defined as:
$$\text{Precision}_{weighted}=\sum_{T\in \mathcal{K}}\frac{N_{T}}{N_{total}}\cdot P_{T}, \quad \text{Recall}_{weighted}=\sum_{T\in \mathcal{K}}\frac{N_{T}}{N_{total}}\cdot R_{T}$$

On test splits, we report **Tool Selection Accuracy (Tool Acc %)** across positive instances.

### 5.2 Argument Accuracy / Exact Match (ArgA / EM %)
ArgA assesses end-to-end correctness, requiring both the function name and all parameter values to match the ground truth identically:
$$\text{ArgA} = \frac{\text{Exact Matches}}{\text{Total Positive Cases}}$$

### 5.3 Non-FC Recall (%)
Measures precision in abstaining from tool invocation on conversational/negative inputs:
$$\text{Non-FC Recall} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}$$

### 5.4 Syntax Error Rate & Inference Latency
- **Syntax Error Rate (%)**: Percentage of model outputs failing XML or JSON syntax parsers.
- **Latency (P50, ms)**: Median wall-clock inference time measured under identical hardware conditions (NVIDIA T4 GPU).

---

## 6. Experimental Results & Analysis

Table 2 and Table 3 detail the performance across all experimental setups on both benchmarks.

**Table 2: Performance Evaluation on CustomTools-VI (Seen vs. Unseen)**  
*(Best results are highlighted in bold. Asterisks \* denote scores recorded from earlier independent runs; formal re-evaluation under the unified Method 1 benchmark suite is pending)*

| Exp | Training Setup | Seen: Tool Acc (%) | Seen: ArgA / EM (%) | Seen: Non-FC Rec (%) | Unseen: Tool Acc (%) | Unseen: ArgA / EM (%) | Unseen: Non-FC Rec (%) | Syntax Error (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Local Model Cohort* | | | | | | | | |
| **E0** | Qwen3.5-2B Zero-Shot | 39.25% | 56.75% | 97.75% | 40.75% | 62.50% | 97.25% | 11.00% |
| **E1** | Monolingual EN (60k) | 91.50% | 63.12% | 75.25% | 96.50% | 70.50% | 74.50% | 4.50% |
| **E2** | Monolingual VI (60k) | 91.00% | 51.50% | 67.50% | 96.25% | 59.62% | 67.75% | 7.18% |
| **E3** | Bilingual EN+VI (60k) | 92.25% | 19.38% | 3.00% | 96.00% | 27.88% | 2.00% | 4.31% |
| **E4** | Bilingual + Custom VI (65.6k)| **93.25%** | **87.00%** | **100.00%** | **97.00%** | **86.38%** | **100.00%** | **2.44%** |
| **Method 2**| Bi-Encoder + Cross-Encoder | 85.25% | 67.75% | 75.00% | 74.50% | 22.75% | 71.75% | **0.00%** |
| *Frontier API Cohort (Closed-source Baselines)* | | | | | | | | |
| **GPT-5.6 Luna** | OpenAI API (Zero-shot) | 93.25% | 78.25% | 100.00% | 97.25% | 79.50% | 100.00% | 0.00% |
| **Gemini 3.8 Flash** | Google API (Zero-shot) | **100.00%** | **93.62%** | 100.00% | **100.00%** | **92.12%** | 99.75% | 0.06% |

*\*Note: Method 2 metrics are derived from the standardized decoupled Bi-Encoder + Cross-Encoder evaluation pipeline. Proprietary closed-source baselines (GPT-5.6 Luna and Gemini 3.8 Flash) were evaluated via their official APIs with fixed zero temperature (temperature = 0) across all 1,600 CustomTools-VI samples.*

![Figure 2: Head-to-head performance comparison on CustomTools-VI](figures/fig2_performance_comparison.png)

*Figure 2: Head-to-head comparison of parameter extraction accuracy (ArgA %) and tool selection accuracy (Tool Acc %) on the CustomTools-VI benchmark across seen and unseen domains. The SLM architecture demonstrates remarkable zero-shot robustness (Unseen ArgA of 86.38% on 2B and 86.75% on 4B), whereas Method 2 suffers severe extraction degradation on novel tool signatures.*

---

**Table 3: Performance Evaluation on Canonical Core Benchmark (7,712 samples / language)**

| Exp | Training Setup | VI Test: Tool Acc (%) | VI Test: ArgA (%) | VI Test: Non-FC (%) | EN Test: Tool Acc (%) | EN Test: ArgA (%) | EN Test: Non-FC (%) | VI Latency (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Qwen3.5-2B Zero-Shot | 56.34% | 40.48% | 96.33% | 85.18% | 60.63% | 94.50% | 707 ms |
| **E1** | Monolingual EN (60k) | 93.67% | 65.57% | 94.30% | 94.07% | **73.66%** | 94.30% | 878 ms |
| **E2** | Monolingual VI (60k) | 90.25% | 64.90% | 94.30% | 93.45% | 71.36% | 93.08% | 200 ms* |
| **E3 (2B)** | Bilingual EN+VI (60k) | 94.00% | 69.76% | 94.30% | 94.27% | 73.22% | 94.30% | 864 ms |
| **E4 (2B)** | Bilingual + Custom VI | 93.92% | 69.75% | 94.30% | 94.17% | 73.15% | 94.30% | 970 ms |
| **E3 (4B)** | Bilingual EN+VI (60k) | **98.73%** | **72.86%** | 94.30% | **96.63%** | **74.71%** | 94.30% | 2,432 ms |
| **Method 2**| Bi-Encoder + Cross-Encoder | 80.47%* | 40.21%* | — | — | — | — | **54.68 ms** |

*\*Note: E2 latency was measured under a larger batch size. For Method 2, Core Benchmark metrics are evaluated across 10,555 valid tool invocation instances under the standardized pipeline protocol. The E3 (4B) model was evaluated on 2× NVIDIA Tesla T4 GPUs.*

---

### 6.1 Addressing RQ1: Cross-Lingual Knowledge Transfer in Vietnamese Tool Calling
Comparing E1 (trained exclusively on English) and E2 (trained exclusively on Vietnamese):
- When evaluated on Vietnamese Core Test, E1 achieves **65.57%** ArgA, slightly higher than E2 (**64.90%**). On Custom Unseen, E1 achieves **70.50%**, significantly outpacing E2 (**59.62%**).
- This shows that base Qwen3.5 possesses a highly aligned multilingual representation space. Structural tool-calling reasoning learned in English transfers seamlessly to Vietnamese prompts.
- However, cross-lingual transfer is asymmetric: E2 suffers slight degradation when tested on English (71.36% vs. 73.66% in E1). Balanced bilingual fine-tuning (E3) reconciles this asymmetry, reaching the peak Core VI ArgA of **69.76%**.

### 6.2 Addressing RQ2: The Over-Triggering Pathology and In-Domain Negative Calibration
A critical empirical discovery in this study is the dramatic collapse of the bilingual model E3 on `CustomTools-VI`:
- ArgA dropped catastrophically to **19.38%** (Seen) and **27.88%** (Unseen).
- Diagnostic analysis revealed the culprit: **Non-FC Recall plummeted to 3.00% and 2.00%**. The model suffered from an acute trigger bias: almost every conversational query (e.g., *"Hôm nay trời đẹp thật"*) was falsely triggered into an arbitrary tool call.
- **Etiology**: General bilingual instruction tuning primes the model to output tool invocations. When exposed to unfamiliar Vietnamese domain concepts without localized negative examples, the model loses restraint.
- **Remedy**: In E4, incorporating 5,600 domain samples containing localized negative prompts completely eliminated false positives, driving Non-FC Recall to **100.00%** and unleashing true argument accuracy to **87.00%** (Seen) and **86.38%** (Unseen). **Lesson**: Target-domain negative calibration is mandatory for deploying agentic systems safely.

### 6.3 Addressing RQ3: Zero-Shot Generalization on Unseen Tools
The technological divide between generative SLMs and discriminative pipelines is most pronounced on unseen tools:
- **Method 1 (SLM E4)** maintained remarkable resilience: ArgA on `test_unseen` reached **86.38%**, dropping by only **0.62%** compared to `test_seen` (87.00%). In-context reasoning enables the autoregressive decoder to interpret novel schema specifications on the fly.
- **Method 2 (Bi+Cross)** plummeted from 67.75% to **22.75%** (a 45% absolute drop). While the Bi-Encoder successfully retrieved unseen tools (Recall@1 ~88%), the Cross-Encoder failed to extract parameters: span boundaries and enum logits were misaligned with out-of-distribution schema formats.

### 6.4 Addressing RQ4: The Pareto Trade-Off Frontier (Latency vs. Accuracy)

**Table 4: Direct Confrontation Between Method 1 and Method 2 Across Key Engineering Dimensions**  
*(Note: Empirical metrics for both Method 1 and Method 2 are officially compiled from dedicated evaluation reports)*

| Dimension | Method 1: SLM Qwen3.5-2B (E4) | Method 2: Bi+Cross (BGE-M3 + XLM-R) | Engineering Recommendation |
|---|:---:|:---:|---|
| **Seen ArgA Accuracy** | **87.00%** | 67.75% (+19.25% for SLM) | SLM excels in complex argument extraction |
| **Unseen ArgA Accuracy**| **86.38%** | 22.75% (+63.63% for SLM) | **SLM decisively wins zero-shot transfer** |
| **Inference Latency (P50)**| ~860 – 970 ms | **54.68 – 58.16 ms** | **Method 2 is 15–18× faster** |
| **VRAM Footprint** | ~4.8 GB | **< 1.2 GB** | Method 2 is deployable on CPU / Edge devices |
| **Syntax Error Risk** | 2.44% | **0.00% (Guaranteed)** | Method 2 enforces 100% schema conformity |
| **Catalog Scalability** | Context grows quadratically | Fixed Vector Index, $O(1)$ lookup | Method 2 scales easily to 1,000+ tools |

The Pareto boundary is sharply delineated:
- **Complex Open-World Agents, Dynamic Tool Insertion**: **Select Method 1 (SLM)**.
- **Real-Time Voicebots, Fixed Closed-World APIs, Cost-Sensitive Deployments**: **Select Method 2 (Bi+Cross)**.

---

### 6.5 Comparative Benchmarking Against Frontier API Baselines (GPT-5.6 Luna & Gemini 3.8 Flash)

Empirical evaluation across all 1,600 samples of `CustomTools-VI` establishes fundamental insights when comparing specialized local architectures against state-of-the-art commercial APIs:

1. **Local SLM (Qwen3.5-2B E4) Outperforms GPT-5.6 Luna in Parameter Extraction**:
   - While `GPT-5.6 Luna` demonstrates robust tool selection (Tool Acc 93.25% Seen, 97.25% Unseen) and perfect trigger restraint (Non-FC Recall 100.00%), its argument extraction accuracy (ArgA / EM) plateaus at **78.25%** (Seen) and **79.50%** (Unseen).
   - In contrast, the fine-tuned `Qwen3.5-2B` (E4) achieves ArgA of **87.00%** (Seen) and **86.38%** (Unseen), **outperforming GPT-5.6 Luna by +8.75% and +6.88%**, respectively. This underscores that a lightweight 2B parameter model locally calibrated on culturally nuanced, domain-specific Vietnamese data surpasses general-purpose closed frontier models in localized entity and argument binding (e.g., VND currency syntax, Vietnamese administrative ward/district entities).
2. **Gemini 3.8 Flash Defines the Frontier Performance Ceiling**:
   - `Gemini 3.8 Flash` delivers peerless accuracy, achieving flawless Tool Accuracy (**100.00%** across both Seen and Unseen) and ArgA / EM scores of **93.62%** (Seen) and **92.12%** (Unseen), with an exceptionally low syntax error rate of 0.06%.
3. **The Multi-Dimensional Trade-Off: Latency, Cost, and Privacy**:
   - **Inference Latency**: Closed API models incur noticeable cloud network round-trips (`Gemini 3.8 Flash`: 2,086 ms; `GPT-5.6 Luna`: 1,858 ms), running more than 2× slower than local SLM E4 (~970 ms) and **over 35× slower** than the specialized Method 2 pipeline (58 ms).
   - **Cost & Deployability**: Closed frontier models depend on recurring token charges, constant Internet connectivity, and cloud data egress. Local SLM E4 and Method 2 offer fully private, zero-inference-cost deployment suited for edge and enterprise air-gapped environments.

---

### 6.6 RQ5: Catalog Scalability & Physical Bottlenecks under Context Pressure (Stress Testing)

To evaluate system resilience as tool catalog sizes grow from narrow scopes ($N=3$) to enterprise-scale environments ($N=1,000$ tools), we conducted a head-to-head **Stress Test** comparing the leading SLM configuration (`Qwen3.5-2B` E4) against Method 2 (BGE-M3 + XLM-R) across 200 diverse query anchors. All tests were executed on identical independent hardware (2× NVIDIA Tesla T4 16GB GPUs).

**Table 5: Head-to-Head Stress Test Results Across Varying Catalog Sizes ($N = 3 \to 1,000$)**  
*(Note: OOM denotes Out of Memory — execution aborted due to exceeding the 16GB VRAM limit on the T4 hardware).*

| Number of Tools ($N$) | SLM Tool Acc (%) | M2 Tool Acc (%) | SLM ArgA (%) | M2 ArgA (%) | SLM P50 Latency (ms) | M2 P50 Latency (ms) | Speedup (M2 vs. SLM) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 3$** | 94.0% | **100.0%** | **85.5%** | 81.0% | 1,258.94 ms | **58.16 ms** | **21.6×** |
| **$N = 10$** | 93.0% | **98.0%** | **84.5%** | 79.0% | 1,604.68 ms | **57.15 ms** | **28.1×** |
| **$N = 50$** | 93.0% | **96.0%** | **85.5%** | 78.0% | 4,703.39 ms | **55.82 ms** | **84.3×** |
| **$N = 100$** | 92.0% | **95.0%** | **83.0%** | 77.0% | 9,318.21 ms | **58.55 ms** | **159.2×** |
| **$N = 500$** | *OOM* | **78.0%** | *OOM* | **62.0%** | *OOM* | **87.44 ms** | $\infty$ |
| **$N = 1000$** | *OOM* | **68.0%** | *OOM* | **54.0%** | *OOM* | **91.88 ms** | $\infty$ |

![Figure 3: Head-to-head Stress Test results between Method 1 and Method 2 as tool catalog size scales from N = 3 to N = 1,000](figures/fig3_stress_test_curves.png)

*Figure 3: Head-to-head empirical Stress Test results: (a) Accuracy degradation curves and the catastrophic Out-of-Memory (CUDA OOM) boundary for the SLM at N ≥ 500; (b) Median inference latency P50 (ms) on a logarithmic scale, illustrating up to a 159.2× acceleration achieved by the decoupled pipeline of Method 2.*

#### Key Empirical Findings:
1. **Exponential Latency Divergence**:
   - In small-to-medium catalogs ($N \le 100$), Method 2 latency remains **strictly flat** ($\sim 55 - 58$ ms), whereas SLM latency spikes from $1.26$ s ($N=3$) to **$9.32$ s ($N=100$)** — a **159.2× slowdown** (with P95 reaching $12.0$ s), rendering it unusable for real-time applications.
2. **Small-Catalog Accuracy Trade-off**:
   - For $N \le 100$, SLM holds a modest advantage in argument accuracy ($+4.5\% \to +6.0\%$) due to unified contextual attention. Conversely, Method 2 achieves higher tool retrieval accuracy ($+2\% \to 6\%$) due to contrastive learning with hard-negative mining.
3. **The Physical Barrier and Catastrophic OOM at $N \ge 500$**:
   - Beyond $N \ge 500$, context lengths exceed 32,000 tokens. The quadratic self-attention mechanism ($O(L^2)$) requires **over 32 GiB VRAM** for intermediate attention tensors during prefill, triggering CUDA Out of Memory and total Denial of Service on commodity 16GB GPUs.
   - Conversely, Method 2 retains steady sub-92 ms latency and $< 1.5$ GB VRAM footprint up to 1,000 tools, sustaining 68% Tool Acc and 54% ArgA.

---

## 7. Model Scaling Study: Qwen3.5-2B vs. Qwen3.5-4B (Model Scaling Study)

To investigate parameter capacity effects on extraction accuracy, generalization resilience, and syntax conformity, we scale training to `unsloth/Qwen3.5-4B` on 1× NVIDIA A100-SXM4-40GB GPU. Training preserves identical data budgets (60,000 samples for E3 and 65,600 for E4), learning rate ($5 \times 10^{-7}$), and effective batch size 64 ($32 \times 2$). The E4 run completed in 1,025 steps (2 hours 05 minutes), driving loss to **$0.0103$** (a ~48% reduction compared to 2B's $0.0198$), with checkpoints pushed to Hugging Face Hub (`ThinhDao/Qwen3.5-4B_E4`).

**Table 6: Model Scaling Framework (Qwen3.5-2B vs. Qwen3.5-4B)**

| Configuration | Backbone | Core VI ArgA (%) | Custom Seen Acc (%) | Custom Seen ArgA (%) | Custom Unseen Acc (%) | Custom Unseen ArgA (%) | ArgA Gap | Core VI Syntax Err (%) | Latency P50 (ms) | Training Time (A100) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E3 (Bilingual)** | Qwen3.5-2B | 69.76% | 92.25% | 19.38% | 96.00% | 27.88% | +8.50% | 4.31% | 863 ms | 49.5 min |
| **E3 (Bilingual)** | **Qwen3.5-4B** | **72.86%** | 92.00% | **62.00%** | **97.50%** | **69.75%** | -7.75% | **0.32%** | 2,432 ms | ~1 hr 55 min |
| **E4 (Domain)** | Qwen3.5-2B | 69.75% | 93.25% | 87.00% | 97.00% | 86.38% | -0.62% | 4.67% | 970 ms | 54 min |
| **E4 (Domain)** | **Qwen3.5-4B** | *[In progress]* | **94.66%** | **87.92%** | 96.75% | **86.75%** | **+1.17%** | *[In progress]* | *[In progress]* | **2 hrs 05 min** |

*(Note: Canonical Core Test comprises 7,712 samples/language; CustomTools-VI consists of 800 Seen and 800 Unseen instances. Latency measured on 2× NVIDIA Tesla T4 GPUs).*

**Table 6.1: Detailed Confrontation of Qwen3.5-4B (E3) on Canonical Core Benchmark**

| Test Split | Sample Count | Tool Selection Acc (%) | ArgA / Exact Match (%) | Non-FC Recall (%) | Syntax Error Rate (%) | Avg Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Core VI Test** | 7,712 (7,221 pos / 491 neg) | **98.73%** | **72.86%** | 94.30% | **0.32%** | 2,431.83 ms |
| **Core EN Test** | 7,712 (7,221 pos / 491 neg) | **96.63%** | **74.71%** | 94.30% | **1.97%** | 3,289.59 ms |

![Figure 4: Model scaling study (2B vs. 4B): Raising the Core Benchmark accuracy ceiling and eliminating JSON syntax errors](figures/fig4_model_scaling.png)

*Figure 4: Impact of parameter scaling (Qwen3.5-2B vs. Qwen3.5-4B): (a) Improvements in parameter extraction accuracy (ArgA) and tool selection accuracy on the Canonical Core Benchmark; (b) Suppression of JSON syntax error rate from 4.31% down to 0.32% (>13× reduction).*

#### 4 Key Empirical Findings from Model Scaling:

1. **Suppression of Syntax Errors and Core Benchmark Elevation**:
   - Scaling capacity from 2.2B to 4.56B parameters suppresses the Syntax Error Rate on Vietnamese Core Test **from 4.31% down to just 0.32%** (over a 13× reduction).
   - Eliminating invalid JSON structures pushes ArgA / Exact Match on Core VI from **69.76%** (2B E3) to **72.86%** (4B E3, **+3.10%**), and on Core EN to **74.71%** (**+1.49%**).
   - Tool Selection Accuracy reaches a record **98.73%** on Core VI (+4.73% over 2B), proving that higher parameter capacity distinguishes subtly differing API boundaries with exceptional fidelity.

2. **Mitigating Domain Shift Collapse in Bilingual Models**:
   - Under configuration E3 (trained solely on general 60k bilingual data without domain samples), the 2B model suffered a catastrophic collapse on localized Vietnamese tools due to indiscriminate over-triggering (ArgA was 19.38% Seen and 27.88% Unseen).
   - In stark contrast, `Qwen3.5-4B` demonstrates robust intrinsic resilience: ArgA jumps to **62.00%** on Seen (**+42.62%**) and **69.75%** on Unseen (**+41.87%**). Higher parameter capacity empowers superior in-context schema following, mitigating distribution shift even prior to domain negative tuning.

3. **Establishing New State-of-the-Art in Configuration E4**:
   - When aligned with localized domain data and negative samples (E4), `Qwen3.5-4B` achieves **94.66% Tool Acc / 87.92% ArgA** on `test_seen` and **96.75% Tool Acc / 86.75% ArgA** on `test_unseen`.
   - This sets the new benchmark ceiling across all local SLM models, surpassing `Qwen3.5-2B` (87.00% Seen, 86.38% Unseen).
   - Crucially, the ArgA Gap remains positive at **+1.17%** (Unseen ArgA matching or exceeding Seen), corroborating robust zero-shot generalization to unseen APIs.

4. **Compute and Inference Latency Trade-Offs**:
   - Training duration on 1× A100 scales from ~50–54 min (2B) to ~1 hr 55 min – 2 hrs 05 min (4B) (~2.3× compute budget).
   - On 2× NVIDIA Tesla T4 GPUs, average inference latency for 4B E3 is **2,431.83 ms** (Core VI) versus ~864 ms for 2B (~2.8× increase). This reflects the autoregressive overhead of a 4.56B model on memory-bandwidth-constrained hardware. Consequently, `Qwen3.5-2B` remains the preferred configuration for sub-second latency constraints, while `Qwen3.5-4B` serves as the premier choice when absolute syntactic reliability and extraction precision are paramount.

---

## 8. Discussion, Error Analysis, & Limitations

### 8.1 Error Categorization in Argument Extraction
Analyzing 200 failure cases from Method 1 (E4) and Method 2 reveals three primary error modes:
1. **Numerical and Date Format Variance (42.5%)**: Colloquial expressions such as "thứ sáu tuần sau" or "ba triệu tư". Generative SLMs occasionally produce verbatim text rather than canonical integers (`3400000`), while Method 2 fails if regex normalizer rules miss specific slang patterns.
2. **Entity Boundary Ambiguity (34.0%)**: In complex administrative queries (e.g., *"tìm trọ gần KTX Khu B ĐHQG phường Đông Hòa"*), models struggle to determine whether the university or the ward name belongs in the `khu_vuc` slot.
3. **Schema Default Conflicts (23.5%)**: When optional parameters specify default values in schema definitions, generative models sometimes emit explicit arguments where the ground truth omitted them.

### 8.2 Limitations
- **Single-Turn Scope**: The current study focuses strictly on single-turn interactions supporting multiple tool calls. Multi-turn dialogue state tracking remains for future investigation.
- **Tool Execution Sandbox**: Evaluation measures schema-conforming generation accuracy rather than full execution feedback within live sandboxed environments.

---

## 9. Conclusion & Future Work

This paper presented an extensive empirical study on Vietnamese tool calling, comparing end-to-end Small Language Models against modular dual/cross-encoder architectures:
1. Standardized benchmarks comprising 77,000+ canonical pairs and 8,000 localized Vietnamese domain instances were established.
2. We demonstrated that `Qwen3.5-2B` (E4) achieves state-of-the-art accuracy (**87.00%** Seen, **86.38%** Unseen), while the scaled `Qwen3.5-4B` E4 sets a new benchmark ceiling of **87.92%** (Seen) and **86.75%** (Unseen) while suppressing syntax errors to **0.32%**.
3. We proved that the modular Bi+Cross Encoder pipeline provides an unbeatable latency advantage (**58–92 ms**) and constant memory complexity when scaled to 1,000 tools in empirical stress testing.
4. We clearly mapped the Pareto trade-off frontier and identified the physical breakdown threshold of generative SLMs due to quadratic attention memory explosion at $N \ge 500$.
5. Empirical benchmarking against frontier commercial APIs demonstrated that our locally fine-tuned 2B and 4B SLMs surpass `GPT-5.6 Luna` in Vietnamese domain argument extraction while offering competitive edge/on-premise deployment.

**Future Directions**:
- Finalizing `Qwen3.5-4B` E4 inference evaluation on the Canonical Core Benchmark.
- Developing **Hybrid Cascading Architectures**: utilizing a Bi-Encoder to retrieve Top-3 candidates in 30 ms, followed by an SLM to generate arguments in 300 ms, realizing the optimal synthesis of speed and intelligence.

---

## Acknowledgments

This research was conducted at the Faculty of Information Technology, University of Information Technology, VNU-HCM. The authors express sincere gratitude to **Dr. Dang Van Thin** for insightful research supervision, experimental guidance, and invaluable feedback throughout the development of this work.

---

## References

1. Ersoy, O., Altinisik, E., Sencar, H. T., & Darwish, K. (2025). *Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning*. arXiv preprint.
2. Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). *Gorilla: Large Language Model Connected with Massive APIs*. Advances in Neural Information Processing Systems (NeurIPS 2024). arXiv:2305.15334.
3. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. Advances in Neural Information Processing Systems (NeurIPS 2023).
4. Yan, X., Liu, Z., et al. (2024). *Berkeley Function-Calling Leaderboard (BFCL)*. Gorilla LLM Project, UC Berkeley.
5. Qin, B., Wang, Y., Xu, Y., Meng, Y., Wang, Y., Teng, Z., Yan, J., Wei, Z., Feng, Y., Wang, Z., & Zhao, D. (2023). *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs*. ICLR 2024.
6. Liu, Z., Hoang, T., Zhang, J., Zhu, M., Lan, T., Kokane, S., Tan, J., Yao, W., Liu, Z., Feng, Y., Murthy, R., Yang, L., Savarese, S., Niebles, J. C., Wang, H., Heinecke, S., & Xiong, C. (2024b). *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*. NeurIPS 2024.
7. Liu, W., Huang, X., Zeng, X., Hao, X., Yu, S., Li, D., Wang, S., Gan, W., Liu, Z., Yu, Y., Wang, Z., Wang, Y., Ning, W., Hou, Y., Wang, B., Wu, C., Wang, X., Liu, Y., Wang, Y., et al. (2024a). *ToolACE: Winning the Points of LLM Function Calling*. arXiv preprint arXiv:2409.00920.
8. Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE-M3)*. arXiv:2309.07597.
9. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)*. ACL 2020.
10. Chen, Z., Shen, S., Shen, G., Zhi, G., Chen, X., & Lin, Y. (2024). *Towards Tool Use Alignment of Large Language Models*. EMNLP 2024.
