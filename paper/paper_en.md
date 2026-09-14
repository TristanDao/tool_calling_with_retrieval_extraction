# Vietnamese Tool Calling: A Comparative Study Between End-to-End Small Language Models and Specialized Bi-Encoder + Cross-Encoder Architecture

**Authors**:  
Dao Phuoc Thinh$^1$, Ha Quang Dat$^1$, Dang Van Thin$^1$ (Supervisor)  

$^1$Faculty of Information Technology, University of Information Technology, Vietnam National University, Ho Chi Minh City (VNU-HCM), Vietnam  
*Email*: {21521469, 21521925}@gm.uit.edu.vn, thindv@uit.edu.vn  

**Date**: September 2026  
**Type**: Academic Research Paper / Engineering Thesis  

---

## Abstract

Tool calling (function calling) is a transformative capability that empowers Large Language Models (LLMs) to interface with external APIs, databases, and computational services, serving as the foundational building block for modern autonomous agents. While substantial progress has been achieved for high-resource languages such as English, enabling robust, low-latency, and zero-shot tool calling for lower-resource languages like Vietnamese remains severely underexplored. Conventional cloud-based Large Language Models (e.g., GPT-4o, Gemini-1.5) suffer from crippling drawbacks in production: heavy autoregressive latency (800–2,000 ms), high token costs, syntax invalidity, quadratic degradation when scaling to large tool catalogs, and enterprise privacy vulnerabilities.

In this work, we present the first systematic comparative investigation addressing four fundamental research questions in Vietnamese tool calling:
1. **RQ1 (Cross-Lingual Knowledge Transfer)**: Can pre-trained multilingual models transfer tool-calling competence from high-resource English datasets to Vietnamese without requiring native training data?
2. **RQ2 (The Over-Triggering Pathology & Negative Calibration)**: Does general bilingual instruction tuning introduce destructive trigger biases on non-tool conversational queries, and does in-domain negative calibration eliminate it?
3. **RQ3 (Zero-Shot Generalization on Unseen Tools)**: How does a generative Small Language Model (SLM) compare against a modular discriminative pipeline (Bi-Encoder + Cross-Encoder) when confronted with entirely novel, culturally specialized tools (`test_unseen`)?
4. **RQ4 (Pareto Efficiency: Latency vs. Accuracy Frontier)**: Can a modular, non-autoregressive dual-encoder pipeline achieve competitive argument accuracy while delivering an order-of-magnitude latency reduction over generative SLMs?

To answer these questions, we curate and standardize two benchmarks: the **Canonical Core Benchmark** (77,028 deduplicated paired English-Vietnamese samples spanning 4,421 unique tools) and **CustomTools-VI** (8,000 samples grounded in 10 Vietnamese domains across 40 tools, strictly partitioned into 20 seen and 20 zero-shot unseen tools with a 50% negative ratio). We perform extensive empirical evaluation across two contrasting paradigms:
- **Method 1 (End-to-End SLM)**: Fine-tuning `unsloth/Qwen3.5-2B` (and scaling to `4B`) via QLoRA on native structured XML tool representations with response-only cross-entropy loss across five controlled data splits (E0: Zero-shot baseline, E1: Monolingual English, E2: Monolingual Vietnamese, E3: Balanced Bilingual, E4: Bilingual + Domain Adaptation).
- **Method 2 (Modular Discriminative Pipeline)**: Decoupling tool selection via a Bi-Encoder (`BAAI/bge-m3` trained with two-round Cached Multiple Negatives Ranking Loss) and schema-aware parameter extraction via a Cross-Encoder (`xlm-roberta-base` with multi-task hierarchical heads: binary existence, extractive span, schema enum, and boolean classification) coupled with a rule-based Vietnamese Value Normalizer.

**Key Empirical Findings**:
- On `CustomTools-VI`, Method 1 (E4) establishes a new state of the art, achieving an Argument Accuracy (ArgA / Exact Match) of **87.00%** on seen tools and **86.38%** on unseen tools, outperforming Method 2 (**67.75%** on seen and **22.75%** on unseen — a massive **+63.63%** margin in zero-shot generalization).
- We identify a severe "over-triggering" vulnerability in general bilingual models (E3): lacking localized negative examples, Non-FC Recall collapses to **2.00%–3.00%**, causing ArgA to crash to 19.38%–27.88%. Incorporating target-domain negative prompts in E4 completely cures this pathology, restoring Non-FC Recall to **100.00%**.
- Conversely, Method 2 achieves an overwhelming latency advantage, delivering a median (P50) wall-clock latency of **58.16–91.88 ms**, running **10–15× faster** than Method 1 (~860–970 ms) with zero syntax errors and minimal memory footprint (< 1.2 GB VRAM), proving ideal for latency-critical, edge, or on-premise industrial deployments.

---

## 1. Introduction

Autonomous language agents require the ability to interact with real-world digital ecosystems via Application Programming Interfaces (APIs). Tool calling—the capacity to recognize user intent, select appropriate tools from an external catalog, and extract structured parameters complying with strict schemas—has become a cornerstone of modern artificial intelligence.

```
       ┌────────────────────────────────────────────────────────┐
       │                 USER QUERY (VIETNAMESE)                │
       │    "Kiểm tra phạt nguội xe máy biển số 59P1-12345"     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                EXTERNAL TOOL REPOSITORY                │
       │  - tra_cuu_phat_nguoi(bien_so: str, loai_xe: str)      │
       │  - dat_ve_xe_khach(diem_di: str, diem_den: str)        │
       │  - thanh_toan_tien_dien(ma_khach_hang: str)            │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
 ═══════════════════════════════════════════════════════════════════════════
       ARCHITECTURAL EXECUTION PARADIGM COMPARISON
 ═══════════════════════════════════════════════════════════════════════════
       ┌───────────────────────────┴────────────────────────────┐
       ▼                                                        ▼
 【METHOD 1: END-TO-END SLM】              【METHOD 2: BI+CROSS ENCODER】
  Autoregressive XML tool call generation   Dense retrieval + Span/Enum classification
  <tool_call>                               Stage 1: Bi-Encoder BGE-M3
  <function=tra_cuu_phat_nguoi>              -> Select "tra_cuu_phat_nguoi" (58 ms)
  <parameter=bien_so>59P1-12345</parameter> Stage 2: Cross-Encoder XLM-R
  <parameter=loai_xe>xe_may</parameter>      -> bien_so = "59P1-12345" (has_value=True)
  </function>                                -> loai_xe = "xe_may" (enum class 1)
  </tool_call>                              Post-Processing: 100% Valid JSON
 ───────────────────────────────────────────────────────────────────────────
  Superior Accuracy, Zero-shot (86.38%)      Ultra-Low Latency (< 90 ms), Compact (< 1.2 GB)
```

Despite rapid advances spearheaded by proprietary Large Language Models (LLMs) such as OpenAI Function Calling and Google Gemini FC, widespread real-world deployment faces four severe limitations:
1. **Computational Overhead and Latency**: Autoregressive decoding over long tool definitions produces latency figures exceeding 800–2,000 ms per turn. This is unacceptable in latency-sensitive industrial applications such as telecommunications, financial customer care, and robotics.
2. **Context Window Exhaustion**: As tool repositories expand from dozens to hundreds or thousands of APIs, stuffing full schemas into the prompt degrades LLM reasoning ("needle in a haystack" phenomenon) and incurs quadratic token costs.
3. **Structured Format & Hallucination Vulnerability**: Generative models frequently hallucinate parameter values, emit invalid JSON strings, or invent fictitious APIs when confronted with domain-specific jargon.
4. **Scarcity of Vietnamese Tool-Calling Research**: The vast majority of function-calling benchmarks (e.g., Berkeley Function-Calling Leaderboard, ToolBench, Glaive) are exclusively in English. Vietnamese presents unique structural challenges: lack of morphological case markers, heavy reliance on tonality, distinct numerical and date formatting (e.g., "ngày 15 tháng 8 năm 2026", "hai triệu rưỡi"), and ubiquitous bilingual code-switching in technical queries.

### Research Contributions
1. **First Large-Scale Standardized Vietnamese Benchmark**: We construct and release the **Canonical Core Benchmark** (77,028 paired English–Vietnamese records across 4,421 unique tools) and **CustomTools-VI** (8,000 localized samples spanning 40 tools across 10 Vietnamese domains, featuring a strict 20-seen / 20-unseen zero-shot evaluation split).
2. **Systematic Cross-Lingual & Negative Calibration Analysis**: We demonstrate that while English tool-calling knowledge transfers surprisingly well to Vietnamese, general bilingual fine-tuning triggers catastrophic over-calling on conversational inputs (Non-FC Recall collapsing to 2.0%). We prove that in-domain negative data completely restores model calibration (100.0% Non-FC Recall).
3. **Establishing the Pareto Trade-Off Frontier**: We provide exhaustive empirical evidence establishing that fine-tuned SLMs (`Qwen3.5-2B` E4) dominate in accuracy and zero-shot tool generalization (**86.38%** ArgA), whereas modular Bi-Encoder + Cross-Encoder pipelines deliver unmatched deterministic inference speed (**58–92 ms** P50 latency), guiding architectural selection for production systems.

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
- **E4 (Bilingual + Domain Adaptation)**: 60,000 bilingual samples + 5,600 samples from `CustomTools-VI/train.jsonl` (total 65,600 training instances).

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
| **E0** | Qwen3.5-2B Zero-Shot | 39.25% | 56.75% | 97.75% | 40.75% | 62.50% | 97.25% | 11.00% |
| **E1** | Monolingual EN (60k) | 91.50% | 63.12% | 75.25% | 96.50% | 70.50% | 74.50% | 4.50% |
| **E2** | Monolingual VI (60k) | 91.00% | 51.50% | 67.50% | 96.25% | 59.62% | 67.75% | 7.18% |
| **E3** | Bilingual EN+VI (60k) | 92.25% | 19.38% | 3.00% | 96.00% | 27.88% | 2.00% | 4.31% |
| **E4** | Bilingual + Custom VI (65.6k)| **93.25%** | **87.00%** | **100.00%** | **97.00%** | **86.38%** | **100.00%** | **2.44%** |
| **Method 2**| Bi-Encoder + Cross-Encoder | 91.75%\* | 67.75%\* | 92.25%\* | 88.50%\* | 22.75%\* | 91.50%\* | **0.00%** |

*\*Note: Method 2 CustomTools scores were derived from an earlier independent notebook run. Re-evaluation under Method 1's unified evaluator script is pending to guarantee identical scoring criteria.*

---

**Table 3: Performance Evaluation on Canonical Core Benchmark (7,712 samples / language)**

| Exp | Training Setup | VI Test: Tool Acc (%) | VI Test: ArgA (%) | VI Test: Non-FC (%) | EN Test: Tool Acc (%) | EN Test: ArgA (%) | EN Test: Non-FC (%) | VI Latency (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Qwen3.5-2B Zero-Shot | 56.34% | 40.48% | 96.33% | 85.18% | 60.63% | 94.50% | 707 ms |
| **E1** | Monolingual EN (60k) | 93.67% | 65.57% | 94.30% | 94.07% | **73.66%** | 94.30% | 878 ms |
| **E2** | Monolingual VI (60k) | 90.25% | 64.90% | 94.30% | 93.45% | 71.36% | 93.08% | 200 ms* |
| **E3** | Bilingual EN+VI (60k) | **94.00%** | **69.76%** | 94.30% | **94.27%** | 73.22% | 94.30% | 864 ms |
| **E4** | Bilingual + Custom VI | 93.92% | 69.75% | 94.30% | 94.17% | 73.15% | 94.30% | 970 ms |
| **Method 2**| Bi-Encoder + Cross-Encoder | *[Re-eval pending]* | *[Re-eval pending]* | *[Re-eval pending]* | — | — | — | **58.16 ms** |

*\*Note: E2 latency was measured under a larger batch size. For Method 2, previous reported Core accuracy (ArgA 40.21%) was measured on a legacy 10,555 positive-only test split; inference must be re-run on Method 1's canonical 7,712-sample test set for strict comparability.*

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
*(Note: Method 2 accuracy scores are preliminary references from legacy notebooks; re-evaluation under Method 1's unified evaluator is pending to establish final standardized confrontation scores)*

| Dimension | Method 1: SLM Qwen3.5-2B (E4) | Method 2: Bi+Cross (BGE-M3 + XLM-R) | Engineering Recommendation |
|---|:---:|:---:|---|
| **Seen ArgA Accuracy** | **87.00%** | 67.75%\* (+19.25% for SLM) | SLM excels in complex argument extraction |
| **Unseen ArgA Accuracy**| **86.38%** | 22.75%\* (+63.63% for SLM) | **SLM decisively wins zero-shot transfer** |
| **Inference Latency (P50)**| ~860 – 970 ms | **58.16 – 91.88 ms** | **Method 2 is 10–15× faster** |
| **VRAM Footprint** | ~4.8 GB | **< 1.2 GB** | Method 2 is deployable on CPU / Edge devices |
| **Syntax Error Risk** | 2.44% | **0.00% (Guaranteed)** | Method 2 enforces 100% schema conformity |
| **Catalog Scalability** | Context grows quadratically | Fixed Vector Index, $O(1)$ lookup | Method 2 scales easily to 1,000+ tools |
| **Evaluation Status** | ✅ Completed (Unified Benchmark) | ⏳ **Re-evaluation pending under M1 protocol** | Rigorous standardization across both methods |

The Pareto boundary is sharply delineated:
- **Complex Open-World Agents, Dynamic Tool Insertion**: **Select Method 1 (SLM)**.
- **Real-Time Voicebots, Fixed Closed-World APIs, Cost-Sensitive Deployments**: **Select Method 2 (Bi+Cross)**.

---

## 7. Model Scaling Study: Qwen3.5-2B vs. Qwen3.5-4B (Planned)

To investigate parameter capacity effects on argument accuracy and syntax error rates, training runs for `unsloth/Qwen3.5-4B` under E3 and E4 are currently in progress.

**Table 5: Model Scaling Experimental Framework (Qwen3.5-2B vs. Qwen3.5-4B)**

| Configuration | Parameter Size | Core VI ArgA (%) | Custom Seen ArgA (%) | Custom Unseen ArgA (%) | Non-FC Recall (%) | Latency (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **E3 (Bilingual)** | Qwen3.5-2B | 69.76% | 19.38% | 27.88% | 2.00% | 863 ms |
| **E3 (Bilingual)** | Qwen3.5-4B | *[Training]* | *[Training]* | *[Training]* | *[Training]* | *[Pending]* |
| **E4 (Domain)** | Qwen3.5-2B | 69.75% | 87.00% | 86.38% | 100.00% | 970 ms |
| **E4 (Domain)** | Qwen3.5-4B | *[Training]* | *[Training]* | *[Training]* | *[Training]* | *[Pending]* |

*Hypothesis*: Scaling to 4B parameters is expected to suppress syntax errors below 1.0% and improve long administrative span extraction in Vietnamese. Empirical results will be incorporated upon evaluation completion.

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
2. We demonstrated that `Qwen3.5-2B` (E4) achieves state-of-the-art accuracy (**87.00%** Seen, **86.38%** Unseen) and extraordinary zero-shot generalization when trained with response-only loss and domain negative prompts.
3. We proved that the modular Bi+Cross Encoder pipeline provides an unbeatable latency advantage (**58–92 ms**) and negligible compute footprint for real-time applications.

**Future Directions**:
- Incorporating completed `Qwen3.5-4B` scaling benchmarks into Table 5.
- Performing **Stress Testing** across varying catalog sizes ($N \in [3, 10, 50, 100, 500, 1000]$) to quantify SLM accuracy degradation under context pressure.
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
