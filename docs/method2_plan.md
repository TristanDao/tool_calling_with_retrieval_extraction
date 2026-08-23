# Method 2 — Bi-Encoder + Cross-Encoder: Nghiên Cứu & Kế Hoạch Thực Hiện

> Tài liệu triển khai chi tiết cho §6 của `docs/experimental_plan.md`.
> Môi trường mục tiêu: **Kaggle Notebook, GPU T4 16GB** (fp16, không bf16).
> Ngày lập: 2026-08-23.

---

## 0. Tóm tắt quyết định đã chốt

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| Bi-Encoder backbone | `BAAI/bge-m3` (LoRA) | Đúng plan gốc; đa ngữ, 8192 ctx, mạnh cho VI |
| Cross-Encoder backbone | `xlm-roberta-base` (278M), fallback `intfloat/multilingual-e5-base` | BGE-M3 (568M) full-FT không vừa T4 với đủ epoch/ablation |
| Cross-Encoder I/O | Giữ per-parameter, **batch toàn bộ param của 1 tool trong 1 forward** | Không phải viết lại collator/heads; vẫn đạt latency thấp |
| Non-verbatim value | Span head + **normalizer hậu xử lý** (số/tiền/ngày VN) | Rẻ, không đổi kiến trúc |
| Type `array`/`object` | **Không hỗ trợ**, đánh dấu `unsupported`, báo cáo coverage | Chiếm nhỏ, thêm head sẽ làm loãng scope |
| Platform | Kaggle (2×T4, 30h/tuần, session 12h) | Job train dài, quota ổn định hơn Colab |

Ba quyết định còn **mở**, sẽ chốt sau Phase 2 (xem §9): ngưỡng abstention, chiến lược chọn số call, và có chạy 1 run BGE-M3-large cho Cross-Encoder hay không.

---

# PHẦN A — NGHIÊN CỨU PHƯƠNG PHÁP

## 1. Cơ sở phương pháp

### 1.1 Bi-Encoder — Semantic Tool Retrieval

Query và tool được mã hóa **độc lập** thành vector, ghép bằng cosine similarity:

```
sim(q, t) = cos(E(q), E(t))
```

Vì `E(t)` không phụ thuộc query, toàn bộ tool pool được **pre-compute một lần**. Tại inference chỉ tốn 1 forward pass cho query + 1 phép nhân ma trận. Đây chính là nguồn gốc lợi thế latency và scalability của Method 2 so với SLM: khi tool pool tăng từ 3 → 1000, chi phí của SLM tăng tuyến tính theo độ dài context, còn chi phí của Bi-Encoder gần như không đổi (chỉ là ma trận `1×d · d×N`).

**MultipleNegativesRankingLoss (MNRL)** — với batch gồm B cặp `(q_i, t_i)`, coi `t_j (j≠i)` là negative:

```
L = -1/B · Σ_i log[ exp(sim(q_i,t_i)/τ) / Σ_j exp(sim(q_i,t_j)/τ) ]
```

Đặc tính quan trọng: **chất lượng tỉ lệ thuận với batch size**, vì batch càng lớn thì càng nhiều in-batch negatives. Trên T4 16GB, batch lớn với BGE-M3 là bất khả thi nếu làm ngây thơ → dùng **GradCache** (`CachedMultipleNegativesRankingLoss` trong sentence-transformers), cho phép effective batch 256+ với bộ nhớ gần như không đổi bằng cách chia mini-batch và cache gradient của embedding.

**Hard negative mining**: sau vòng train đầu, dùng chính model retrieve top-k, lấy các tool sai nhưng xếp hạng cao làm negative tường minh. Với dataset này, nguồn hard negative tự nhiên rất tốt là `feature_group` — 40 tool của CustomTools-VI chia đều 10 nhóm × 4 tool, các tool cùng nhóm (vd `vi_search_restaurants` vs `vi_order_food`) là negative khó theo đúng nghĩa ngữ nghĩa.

### 1.2 Cross-Encoder — Schema-aware Parameter Extraction

Định dạng BERT-QA: **query là context, schema của parameter là question**.

```
[CLS] <query> [SEP] Param=<name>. Desc=<desc>. Type=<type>[. Enum=v1|v2|...] [SEP]
```

Điểm mấu chốt của thiết kế: **type routing lấy từ schema, không phải từ model**. Model không cần học "parameter này kiểu gì" — thông tin đó đã có sẵn trong JSON Schema. Model chỉ cần trả lời "giá trị nằm ở đâu trong query". Điều này khác hẳn generative: model không thể bịa ra tool name không tồn tại, và không thể sinh JSON sai cú pháp — JSON được **lắp ráp** từ schema chứ không được **sinh ra**. Đây là luận điểm chống hallucination của Method 2.

Bốn head:

| Head | Kích hoạt khi | Output |
|---|---|---|
| `has_value` | Luôn luôn | Sigmoid binary — parameter có được nhắc tới không |
| `span_start` / `span_end` | `has_value=1` ∧ `type ∈ {string, number, integer}` | Softmax trên token của query |
| `enum` | `has_value=1` ∧ có `enum` | Softmax N-way |
| `boolean` | `has_value=1` ∧ `type=boolean` | Softmax 2-way |

Gated multi-task loss:

```
L = BCE(has_value)
  + 1[has_value=1 ∧ type∈span]  · (CE_start + CE_end)
  + 1[has_value=1 ∧ type=enum]  · CE_enum
  + 1[has_value=1 ∧ type=bool]  · CE_bool
```

Head `has_value` là thành phần dễ bị đánh giá thấp nhưng quan trọng nhất: nó xử lý **optional parameter**. Đo trên CustomTools-VI train: 15,318 cặp `(query, param)` nhưng chỉ 12,420 được điền → **18.9% cặp có nhãn `has_value=0`**. Nếu head này kém, model sẽ điền bừa optional param và ArgA rớt thảm (ArgA yêu cầu **toàn bộ** argument đúng, kể cả không được thừa key).

### 1.3 Vì sao tách hai giai đoạn

| | SLM End-to-End | Bi + Cross |
|---|---|---|
| Cơ chế | Autoregressive generation | Retrieval + span prediction |
| Số forward pass | ~40-80 (theo số token sinh ra) | 1 (bi) + 1 batched (cross) |
| Chi phí khi N tool ↑ | Context dài tuyến tính | Pre-computed index, gần như hằng số |
| JSON invalid | Có thể | Không thể (lắp ráp từ schema) |
| Hallucinated tool | Có thể | Không thể (chỉ chọn trong pool) |
| Giá trị không có trong query | Có thể suy luận/sinh ra | **Không thể** (giới hạn cứng) |
| Multi-call cùng 1 tool | Được | **Không được** (giới hạn cứng) |

Hai ô "giới hạn cứng" ở cuối là nội dung chính của §3 — chúng quyết định plan này phải làm gì.

---

## 2. Hiện trạng repo

### 2.1 Đã có

| File | LOC | Trạng thái |
|---|---:|---|
| `src/models/crossencoder/model.py` | 86 | Wrapper BGE-M3 + heads, `save/from_pretrained`. Đã xử lý encoder không có `token_type_ids` (XLM-R) → **dùng lại được ngay với `xlm-roberta-base`** |
| `src/models/crossencoder/heads.py` | 58 | 4 head + mask span ngoài query. OK |
| `src/models/crossencoder/losses.py` | 103 | Gated multi-task loss. **Có bug**, xem §2.3 |
| `src/models/crossencoder/data_collator.py` | 157 | Tokenize BERT-QA, chuẩn hóa type. **Cần sửa padding/max_length** |
| `src/models/crossencoder/label_generator.py` | 167 | Sinh nhãn rule-based. **Cần sửa boolean + non-verbatim** |
| `src/models/crossencoder/inference.py` | 105 | Extract arguments. **Chưa batch, chưa normalize** |

### 2.2 Chưa có (phải viết mới)

- Toàn bộ Bi-Encoder: `src/models/biencoder/` — build tool pool, training pairs, train, index, retrieve, eval.
- Training loop cho Cross-Encoder (`train.py`, dataset, checkpoint/resume).
- Pipeline ghép Bi + Cross + validator.
- Module `src/evaluation/` — thư mục làm việc **chỉ còn `.pyc`**, không có source `.py`. Nguyên nhân: source nằm ở commit `79cab49 feat: add evaluation` trên nhánh **`origin/feature/evaluation`**, nhánh này tách ra từ `6a0f285` và **chưa bao giờ merge** vào lineage của HEAD hiện tại (`82253bd`). Không phải file bị xóa — chỉ là chưa merge. Cách lấy lại: `git checkout origin/feature/evaluation -- src/evaluation docs/evaluation.md docs/evaluation_methodology_thesis.md configs/eval/`. Commit đó có 14 module (`evaluator`, `matching`, `normalization`, `extraction_metrics`, `end_to_end_metrics`, `efficiency_metrics`, `compare`, `report`, `cli`, …) — **đọc `docs/evaluation_methodology_thesis.md` trước khi viết bất cứ metric nào cho Method 2** để không làm trùng.
- Config Hydra cho Method 2 (`configs/method2/`).
- `data/benchmark_vi/` chưa được build (config `configs/data/benchmark.yaml` đã có nhưng chưa chạy).

### 2.3 Bốn lỗi cần sửa trước khi train

**(a) `losses.py` — index sai (nghiêm trọng).**

```python
present_idx = (labels["has_value"] == 1).nonzero(...).tolist()   # index trong BATCH
schema_type_list = labels["schema_type"]                          # dài B
schema_type_present = [schema_type_list[i] for i in range(len(present_idx))]  # ← SAI
```

Dòng cuối lấy `schema_type[0..len(present_idx)-1]` thay vì `schema_type[present_idx]`. Khi số sample có `has_value=1` nhỏ hơn batch size (luôn luôn xảy ra), type bị gán lệch → span loss áp lên sample enum, enum loss áp lên sample string. Sửa:

```python
schema_type_present = [schema_type_list[i] for i in present_idx]
```

**(b) `data_collator.py` — lãng phí 4× compute.**
`max_length=1024`, `padding="max_length"`. Đo thực tế: query p50 = 18-24 từ, p95 = 34-43 từ, max = 253 từ (xLAM). Cộng schema question (~40 token) thì **max_length=256 với dynamic padding là đủ**. Sửa `padding="longest"` + `max_length=256` → giảm ~4× thời gian train.

**(c) `label_generator.py::_detect_boolean` — logic sai.**
Hàm quét cue từ ở cấp **toàn query**, không gắn với parameter cụ thể. Query "Tìm quán phở **không** cay, giao **có** hỗ trợ hóa đơn" sẽ trả cùng một nhãn cho cả hai boolean param. Cần: quét cue trong **cửa sổ ±N token quanh vị trí keyword của param** (khớp từ `description` của param), hoặc bỏ hẳn rule và để `has_value` + `boolean` head tự học từ nhãn gold (khuyến nghị — gold value đã có sẵn trong `function_calls`, không cần đoán cue).

> Thực tế: nhãn boolean **không cần rule** vì `function_calls[].arguments[param]` đã cho biết `true`/`false`. Rule cue chỉ nên dùng để *lọc* sample mà query không hề chứa manh mối (tránh train model đoán mò).

**(d) `inference.py` — decode span ngây thơ.**
`argmax(start)` và `argmax(end)` độc lập, rồi swap nếu `start > end`. Chuẩn SQuAD là chọn `(i,j)` maximize `logit_start[i] + logit_end[j]` với ràng buộc `i ≤ j ≤ i + max_answer_len`. Ngoài ra chưa có normalizer (§4.3) và chưa batch.

---

## 3. Phân tích dữ liệu — các con số quyết định thiết kế

Tất cả số dưới đây đo trực tiếp trên dữ liệu trong repo (không phải ước lượng).

### 3.1 Quy mô

| Dataset | Samples | Unique tool name | Unique schema signature | Query p50/p95/max (từ) |
|---|---:|---:|---:|---|
| `glaive_normalized_vi` | 45,593 | 768 | 10,500 | 18 / 34 / 108 |
| `glaive_negative_vi` | 15,141 | 479 | 4,812 | 13 / 18 / 49 |
| `xlam_normalized_vi` | 60,000 | 3,605 | 4,187 | 21 / 43 / 253 |
| `custom_vi/v1` (train+val+test) | 8,000 | 40 | 40 | 24 / 38 / 53 |

**Tool pool sau khi gộp và dedupe theo tên: 4,471 tool** (glaive 768 + glaive_negative 479 với 366 trùng + xLAM 3,605 + custom 40). Đủ lớn để stress test tới N=1000 mà không cần sinh distractor giả.

> ⚠️ **Glaive: 768 tên nhưng 10,500 signature.** Cùng một `tool_name` có tới hàng chục biến thể description/parameters (do dịch khác nhau giữa các sample). Bắt buộc phải canonicalize tool pool trước khi train Bi-Encoder, nếu không cùng một tool sẽ vừa là positive vừa là negative trong cùng batch → MNRL học nhiễu nặng.

### 3.2 Phân bố type parameter

| Dataset | string | integer | number | enum | boolean | array | object |
|---|---:|---:|---:|---:|---:|---:|---:|
| custom_vi train | 7,605 | 2,097 | 621 | 1,242 | 855 | 0 | 0 |
| glaive (10k đầu) | 8,968 | 1,079 | 6,416 | 94 | 31 | 882 | 397 |
| xLAM (10k đầu) | 15,956 | 6,801 | 1,527 | 0 | 341 | 1,639 | 70 |

→ **`array`+`object` chiếm 7.2% (glaive: 1,279/17,888) và 6.5% (xLAM: 1,709/26,336), 0% ở custom_vi.** Theo quyết định đã chốt: đánh dấu `unsupported`, loại khỏi train, và **báo cáo tường minh** trong bảng kết quả (một cột "coverage" hoặc một dòng error-class riêng). Không được im lặng bỏ qua — nếu không, ArgA của Method 2 trên glaive/xLAM sẽ bị thổi phồng.

### 3.3 Tỉ lệ verbatim — giới hạn trần của span head

Tỉ lệ giá trị gold xuất hiện **nguyên văn** trong query:

| Dataset | string verbatim | number verbatim |
|---|---|---|
| custom_vi train | 5,782 / 7,605 = **76.0%** | 1,476 / 2,718 = **54.3%** |
| custom_vi test_unseen | 484 / 505 = **95.8%** | 226 / 368 = **61.4%** |
| glaive (10k) | 5,900 / 8,968 = **65.8%** | 7,020 / 7,495 = **93.7%** |
| xLAM (10k) | 9,722 / 15,956 = **60.9%** | 6,786 / 8,328 = **81.5%** |

Nguyên nhân chính của phần non-verbatim, đọc mẫu thực tế:

| Query (VI) | Gold | Vì sao span thuần fail |
|---|---|---|
| "giá dưới **50.000 đồng**" | `max_price_vnd: 50000` | Dấu phân cách nghìn + đơn vị |
| "điểm đánh giá từ **1,5** trở lên" | `min_rating: 1.5` | Dấu phẩy thập phân VN |
| "tin tức **Hoa Kỳ**" | `country: "United States"` | Giá trị canonical bằng EN |
| "giao trong tối đa **30 phút**" | `max_delivery_minutes: 30` | Đơn vị bám sát số |

Hai nhóm đầu và nhóm bốn **normalizer sửa được**. Nhóm ba (VI→EN canonical) thì không — đây là error class `T` (Translation/language discrepancy) của Ersoy et al., và nó là **giới hạn kiến trúc** cần được báo cáo trung thực chứ không che giấu.

> **Hệ quả của quyết định "chỉ normalizer, không fuzzy alignment"**: các sample non-verbatim sẽ bị `SKIP` khi sinh nhãn train (label_generator không căn được span). Phải **đo và ghi lại tỉ lệ SKIP** ở Phase 3. Nếu SKIP > 30% trên tập train, quay lại xem xét fuzzy alignment (căn span trên chuỗi đã normalize) — đã ghi vào §9 như điểm quyết định.

### 3.4 Multi-call — giới hạn cứng

| Dataset | % positive multi-call | % positive gọi **lặp cùng 1 tool** |
|---|---:|---:|
| custom_vi train | 15.0% | **0.0%** |
| custom_vi test_seen | 15.0% | **0.0%** |
| glaive | 0.0% | **0.0%** |
| xLAM | 52.6% | **20.9%** |

Kiến trúc "1 tool → 1 bộ argument" **không thể** sinh 2 call cho cùng một tool (vd xLAM `live_giveaways_by_type(type="beta")` + `live_giveaways_by_type(type="game")`). Đây là **trần cứng 20.9% ArgA loss trên xLAM**, không liên quan gì đến chất lượng model.

**Xử lý:**
1. Tập đánh giá **chính** của Method 2 là `CustomTools-VI` (0% lặp) và `glaive` (0% lặp) → so sánh với Method 1 hoàn toàn công bằng.
2. Trên xLAM: báo cáo hai con số — ArgA thô, và ArgA trên tập con không-lặp (`n = 60,000 − 12,567`). Ghi rõ 20.9% là structural, kèm trong error analysis nhóm riêng.
3. Không cố "vá" bằng heuristic sinh nhiều call — sẽ làm bẩn kết luận.

### 3.5 CustomTools-VI — cấu trúc seen/unseen

- 40 tool tổng, 10 feature group × 4 tool.
- Train: 20 tool (vừa là gold vừa là candidate), 10 candidate/sample, **35.7% sample là negative** (`function_calls=[]`).
- `test_seen`: gold ∈ 20 tool đã train, candidate pool 30 tool.
- `test_unseen`: gold là 10 tool **chưa hề xuất hiện** trong train, candidate pool 30.

→ `test_unseen` là bài kiểm tra zero-shot đúng nghĩa cho Bi-Encoder. Đây là nơi Method 2 **kỳ vọng thắng** Method 1: Bi-Encoder chỉ cần embedding của description tool mới, không cần train lại. Cần nhấn mạnh điểm này trong Bảng D.

### 3.6 Số cặp (query, parameter) để train Cross-Encoder

Với oracle tool (chỉ param của tool đúng):

| Dataset | Gold calls | Cặp (query, param) |
|---|---:|---:|
| custom_vi train | 4,140 | 15,318 |
| glaive | 45,593 | 85,817 |
| xLAM | 100,011 | 203,497 |
| **Tổng** | 149,744 | **304,632** |

Sau khi loại `array`/`object` và các sample SKIP, ước tính còn **~230-260k cặp**.

---

## 4. Ràng buộc T4 và thiết kế theo ràng buộc

### 4.1 Ngân sách bộ nhớ T4 (15.0 GB khả dụng thực tế)

**Bi-Encoder — BGE-M3 (568M) + LoRA + GradCache:**

| Thành phần | VRAM |
|---|---|
| Weight fp16 (frozen base) | 1.14 GB |
| LoRA adapter (r=16, ~5M param) + grad + Adam fp32 | ~0.10 GB |
| Activation, mini_batch=8 @ len 192, có grad checkpointing | ~1.5 GB |
| Embedding cache (GradCache), effective batch 256 × 1024 dim | ~0.01 GB |
| Overhead CUDA/cuDNN | ~1.5 GB |
| **Tổng** | **~4.3 GB** ✅ rất thoải mái |

Nếu full fine-tune BGE-M3 thay vì LoRA: 1.14 (fp16) + 2.27 (fp32 master) + 2.27 (grad) + 4.55 (Adam m,v) ≈ **10.2 GB** trước activation → vẫn chạy được với mini_batch nhỏ nhưng không còn dư địa. **Khuyến nghị LoRA**, và nếu Recall@1 chưa đạt thì mới cân nhắc full-FT ở Phase 2b.

**Cross-Encoder — XLM-R base (278M), full fine-tune:**

| Thành phần | VRAM |
|---|---|
| Weight fp16 + fp32 master | 1.67 GB |
| Gradient fp32 | 1.11 GB |
| Adam (m, v) fp32 | 2.22 GB |
| Activation, batch=32 @ len 256, không cần grad checkpointing | ~2.5 GB |
| Overhead | ~1.5 GB |
| **Tổng** | **~9.0 GB** ✅ còn dư ~6 GB |

Nếu dùng BGE-M3 (568M) cho Cross-Encoder ở cùng setting: ~17.5 GB → **OOM**. Cần LoRA + grad checkpointing + batch 8 → chạy được nhưng chậm ~3.5× và ít dư địa ablation. Đây chính là lý do của quyết định backbone ở §0.

### 4.2 Ngân sách thời gian (Kaggle: 30h GPU/tuần, session tối đa 12h)

| Job | Ước lượng T4 | Ghi chú |
|---|---|---|
| Pre-compute embedding 4,400 tool (BGE-M3, len 256) | ~2 phút | Làm 1 lần, cache thành `.npy` |
| Bi-Encoder train, 1 epoch × 150k pair, LoRA, eff. batch 256 | ~50-70 phút | GradCache mini_batch 8 |
| Bi-Encoder 3 epoch + hard negative round 2 | **~4h** | Vừa 1 session |
| Cross-Encoder train, 1 epoch × 250k pair, XLM-R base, batch 32, len 256 | ~75-95 phút | Dynamic padding giúp giảm thêm ~20% |
| Cross-Encoder 3 epoch | **~4.5h** | Vừa 1 session |
| Eval full pipeline trên test (1,600 custom + subset glaive/xLAM) | ~20 phút | |
| Stress test 2,400 instance | ~40 phút | Bi-Encoder gần như miễn phí khi N tăng |
| **Tổng cho 1 vòng hoàn chỉnh** | **~10h** | ≈ 1/3 quota tuần → còn chỗ cho 2 vòng ablation |

Tất cả job đều **< 12h** → không cần chia session, nhưng vẫn phải **checkpoint mỗi 500 step** phòng Kaggle ngắt bất ngờ.

### 4.3 Ba tối ưu bắt buộc trên T4

1. **`max_length=256` + dynamic padding** thay vì `1024` + `padding="max_length"`. Riêng thay đổi này đã cho ~4× speedup ở Cross-Encoder (§2.3b).
2. **`torch.cuda.amp` fp16** (T4 không có bf16) + `GradScaler`. Chú ý: span logit bị mask bằng `-1e4` trong `heads.py` — đây là giá trị **đúng cho fp16** (`-1e9` sẽ tràn thành `-inf` → NaN). Giữ nguyên, đừng "sửa" thành `-inf`.
3. **GradCache cho MNRL**, không phải gradient accumulation. Gradient accumulation *không* làm tăng số in-batch negative — nó chỉ chia nhỏ update. GradCache mới thực sự cho effective batch 256.

### 4.4 Lựa chọn backbone Cross-Encoder

| Model | Params | Nhận xét |
|---|---:|---|
| `xlm-roberta-base` | 278M | **Chọn.** Đa ngữ, fast tokenizer có `offset_mapping` (cần cho căn span), không có `token_type_ids` → `model.py` đã handle |
| `intfloat/multilingual-e5-base` | 278M | Fallback. Cùng kiến trúc XLM-R base, đã pretrain thêm cho retrieval → có thể tốt hơn ở `has_value` |
| `vinai/phobert-base-v2` | 135M | **Không dùng.** Yêu cầu word-segmentation VnCoreNLP; query chứa nhiều identifier/entity tiếng Anh (`"United States"`, `"Shopee"`) mà PhoBERT tokenizer xử lý kém |
| `BAAI/bge-m3` | 568M | Chỉ chạy **1 run so sánh** ở Phase 6 nếu còn quota, để trả lời "backbone lớn hơn có đáng không" |

Bi-Encoder cân nhắc thêm ở Phase 2b nếu BGE-M3 chưa đủ: `AITeamVN/Vietnamese_Embedding` (fine-tune từ chính BGE-M3 trên ~300k triplet VI, báo cáo vượt BGE-M3 gốc trên Legal Zalo 2021) — thay checkpoint là xong, không đổi code.

---

## 5. Hai lỗ hổng của plan gốc cần bổ sung

`experimental_plan.md` §6 mô tả Bi-Encoder và Cross-Encoder nhưng **thiếu hai cơ chế** mà nếu không có thì Method 2 không thể điền được Bảng C/D.

### 5.1 Cơ chế `<no_tool_call>` (abstention)

Bi-Encoder **luôn** trả về top-k tool, kể cả khi query không cần tool nào. Nhưng 35.7% sample CustomTools-VI và 15,141 sample Glaive là negative. Không có abstention thì Negative Recall = 0% và Method 2 thua trắng ở §8.1.

**Thiết kế (primary):** ngưỡng similarity `τ` hiệu chỉnh trên validation.

```
if max_i sim(q, t_i) < τ:  →  <no_tool_call>
else:                      →  chọn tool
```

`τ` chọn bằng cách maximize Macro-F1 (coi `no_tool_call` là một class) trên `val_seen ∪ val_unseen ∪ glaive_val_negative`. **Bắt buộc chốt `τ` trên val và freeze trước khi chạy test** — nếu tune trên test thì kết quả vô hiệu.

**Ablation (V2):** thêm head `should_call` (binary) vào Cross-Encoder, dùng cùng forward pass với `(query, top-1 tool schema)`. Chi phí ~0, và học được quan hệ query↔tool sâu hơn ngưỡng cosine. Nếu V2 thắng rõ, dùng làm kết quả chính.

### 5.2 Cơ chế chọn số lượng call (multi-call)

15% CustomTools-VI và 52.6% xLAM là multi-call. Cần quyết định gọi bao nhiêu tool.

**Thiết kế:** chọn tất cả tool có `sim ≥ τ_call`, giới hạn tối đa `k_max=3` (bao phủ 100% custom_vi, ~99% xLAM sau khi loại lặp). `τ_call` hiệu chỉnh riêng, độc lập với `τ` abstention.

**Ablation:** so với chiến lược "gap-based" — chọn top-1, rồi thêm top-2 nếu `sim_1 − sim_2 < δ`. Thường ổn định hơn ngưỡng tuyệt đối.

---

# PHẦN B — KẾ HOẠCH THỰC HIỆN

## 6. Cấu trúc thư mục mục tiêu

```
src/models/
  biencoder/
    __init__.py
    tool_pool.py          # canonicalize + dedupe tool → tool pool thống nhất
    pairs.py              # sinh (query, positive_tool, hard_negatives)
    train.py              # sentence-transformers + CachedMNRL + LoRA
    index.py              # pre-compute embedding, save .npy + tool_id map
    retrieve.py           # query → top-k + score, có abstention
    evaluate.py           # Recall@{1,3,5,10}, MRR, NDCG@10
  crossencoder/
    ... (đã có, sửa theo §2.3)
    dataset.py            # NEW — Dataset trả (query, param, labels)
    train.py              # NEW — training loop, fp16, checkpoint/resume
    normalize.py          # NEW — normalizer số/tiền/ngày VN (§7.4)
    evaluate.py           # NEW — metric §6.2 experimental_plan
  pipeline/
    method2.py            # NEW — Bi → Cross → validator → function_calls JSON
    validator.py          # NEW — jsonschema validate + coerce type

configs/method2/
  tool_pool.yaml  biencoder.yaml  crossencoder.yaml  pipeline.yaml  stress.yaml

notebooks/
  method2_kaggle_biencoder.ipynb
  method2_kaggle_crossencoder.ipynb
  method2_kaggle_eval.ipynb

data/method2/
  tool_pool.json            # ~4,400 tool canonical
  biencoder/{train,val}.jsonl
  crossencoder/{train,val}.jsonl
  index/tool_embeddings.npy

artifacts/method2/
  biencoder/{run_id}/       crossencoder/{run_id}/
results/method2/
  predictions/  metrics/  latency/  stress/
```

## 7. Các phase

### Phase 0 — Chuẩn bị (local, không cần GPU) — ~1 ngày

| # | Việc | Done khi |
|---|---|---|
| 0.1 | Lấy `src/evaluation/` từ nhánh chưa merge: `git checkout origin/feature/evaluation -- src/evaluation docs/evaluation.md docs/evaluation_methodology_thesis.md configs/eval/` rồi rà lại xem có khớp §8 `experimental_plan.md` không | `pytest tests/` xanh, `import src.evaluation` chạy được |
| 0.2 | Chạy `scripts/data/run_benchmark.sh` để sinh `data/benchmark_vi/` | Có `train/val/test.jsonl` + `tool_pool.json` |
| 0.3 | Freeze dataset + manifest SHA-256 (§3.2 experimental_plan) | `data/method2/manifest.json` |
| 0.4 | Sửa 4 bug §2.3 | Có unit test cho từng bug, đặc biệt test loss với batch trộn type |
| 0.5 | Viết `configs/method2/*.yaml` | `python -m src.models.biencoder.train --cfg job` in ra config đúng |

> 0.4 là **blocker cứng**. Bug loss index sẽ làm mọi kết quả Cross-Encoder vô nghĩa mà không có triệu chứng rõ ràng (loss vẫn giảm). Viết test này trước tiên:
> ```python
> # batch cố ý: [string(hv=1), enum(hv=0), boolean(hv=1), string(hv=0)]
> # assert span loss chỉ áp lên index 0, boolean loss chỉ áp lên index 2
> ```

### Phase 1 — Tool pool + training pairs (local) — ~1 ngày

| # | Việc | Chi tiết |
|---|---|---|
| 1.1 | `tool_pool.py`: canonicalize | Group theo `name`; chọn description **xuất hiện nhiều nhất**; parameters = union theo key, giữ type của biến thể phổ biến nhất; log số biến thể bị gộp (glaive: 10,500 → 768) |
| 1.2 | Gán `feature_group` cho tool chưa có | `src/data/feature_group_classify.py` đã có sẵn, dùng lại |
| 1.3 | `pairs.py`: sinh cặp Bi-Encoder | Positive = tool trong `function_calls`. Hard negative round-0 = tool cùng `feature_group` (custom_vi) hoặc BM25 top-10 loại positive (glaive/xLAM). 4 hard negative/positive |
| 1.4 | Sinh cặp Cross-Encoder | Mỗi (gold_call, param của tool đó) → 1 cặp. Bao gồm cả param **không** có trong arguments (`has_value=0`). Loại `array`/`object`, gắn cờ `unsupported` |
| 1.5 | **Đo tỉ lệ SKIP** | Ghi vào `data/method2/label_stats.json`: %SKIP theo type, theo dataset. **Nếu >30% → dừng và xem lại §9-Q2** |

Kiểm tra bắt buộc trước khi qua Phase 2:
- Không có tool `test_unseen` nào lọt vào tool pool dùng để train Bi-Encoder? — **Không**, tool unseen *phải* có trong pool để retrieve được (đó là điểm của zero-shot), nhưng **không được xuất hiện làm positive trong training pairs**. Assert điều này.
- Không có query nào của test set nằm trong training pairs (dedupe theo hash query).

### Phase 2 — Train Bi-Encoder (Kaggle T4) — ~4h GPU

```yaml
# configs/method2/biencoder.yaml
model: BAAI/bge-m3
lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: [query, key, value, dense]}
loss: CachedMultipleNegativesRankingLoss
scale: 20.0            # = 1/τ, mặc định của sentence-transformers
batch_size: 256        # effective, nhờ GradCache
mini_batch_size: 8     # thực tế đưa vào GPU
max_seq_length: 192
epochs: 3
lr: 2.0e-5
warmup_ratio: 0.1
scheduler: cosine
fp16: true
gradient_checkpointing: true
eval_steps: 500
save_steps: 500
seed: 42
```

Quy trình:
1. **Round 1** — train với in-batch negative + hard negative round-0 (feature_group/BM25).
2. **Mine hard negatives** — dùng checkpoint round 1 retrieve top-20 trên train, lấy các tool sai xếp hạng cao (bỏ top-1 để tránh false negative), 4 negative/positive.
3. **Round 2** — train lại từ base với hard negative mới.
4. Pre-compute embedding toàn bộ tool pool → `data/method2/index/tool_embeddings.npy`.
5. **Hiệu chỉnh `τ` (abstention) và `τ_call` trên validation.** Lưu vào `artifacts/method2/biencoder/{run_id}/thresholds.json` và **freeze**.

Gate để qua Phase 3:

| Metric | Tập | Ngưỡng tối thiểu |
|---|---|---|
| Recall@1 | custom `val_seen` | ≥ 0.90 |
| Recall@1 | custom `val_unseen` | ≥ 0.75 |
| Recall@5 | custom `val_unseen` | ≥ 0.92 |
| Negative Recall @ `τ` | custom val negative | ≥ 0.80 |

Không đạt → thử theo thứ tự: (a) đổi document text (thêm param name vào text tool), (b) tăng số hard negative lên 8, (c) đổi checkpoint sang `AITeamVN/Vietnamese_Embedding`, (d) full fine-tune thay LoRA.

### Phase 3 — Train Cross-Encoder (Kaggle T4) — ~4.5h GPU

```yaml
# configs/method2/crossencoder.yaml
model: xlm-roberta-base
max_length: 256
padding: longest          # KHÔNG dùng max_length
batch_size: 32
grad_accum: 2             # effective 64
epochs: 3
lr: 3.0e-5
head_lr: 1.0e-4           # head mới, lr cao hơn encoder
warmup_ratio: 0.06
weight_decay: 0.01
fp16: true
loss_weights: {has_value: 1.0, span: 1.0, enum: 1.0, boolean: 1.0}
max_enum_size: 20         # đủ: enum lớn nhất đo được = 12
max_answer_len: 30        # ràng buộc decode span
seed: 42
save_steps: 500
```

Thứ tự train (curriculum, giúp hội tụ ổn định hơn trên T4):
1. **Warm-up trên glaive+xLAM** (dữ liệu lớn, học kỹ năng span tổng quát) — 2 epoch.
2. **Fine-tune trên custom_vi train** (domain đích) — 2 epoch, lr giảm còn `1e-5`.

Gate để qua Phase 4 (chế độ **oracle retrieval**, tool đúng được đưa thẳng vào):

| Metric | Tập | Ngưỡng tối thiểu |
|---|---|---|
| `has_value` F1 | custom val | ≥ 0.90 |
| Span EM | custom val | ≥ 0.80 |
| Enum accuracy | custom val | ≥ 0.90 |
| Boolean accuracy | custom val | ≥ 0.85 |
| Argument EM (per-call) | custom val | ≥ 0.70 |

### Phase 4 — Normalizer + Validator (local) — ~1 ngày

`normalize.py` phải xử lý, theo đúng thứ tự:

| Loại | Ví dụ input span | Output | Điều kiện áp dụng |
|---|---|---|---|
| Số có phân cách nghìn | `"50.000"`, `"50,000"` | `50000` | `type ∈ {integer, number}` |
| Thập phân VN | `"1,5"` | `1.5` | `type = number` |
| Đơn vị tiền | `"50.000 đồng"`, `"50k"`, `"2 triệu"` | `50000`, `50000`, `2000000` | tên param chứa `price`/`vnd`/`amount`/`fee` |
| Đơn vị thời gian | `"30 phút"`, `"2 tiếng"` | `30`, `120` | tên param chứa `minutes`/`duration` |
| Ngày tháng | `"ngày 15/3"`, `"mai"` | `"2026-03-15"` | `format: date` trong schema |
| Boolean phủ định | span `"không cay"` | `false` | `type = boolean` |
| Whitespace/Unicode | | NFC normalize, strip | mọi type |

Quy tắc bất di bất dịch: **normalizer chỉ được chạy trên span đã predict, và chỉ khi schema cho phép.** Không dùng synonym mapping, không dùng LLM — §8.4 `experimental_plan.md` cấm điều đó cho metric chính.

`validator.py`: jsonschema validate → nếu required thiếu thì thử lại với `has_value_threshold` thấp hơn (0.3) cho riêng param đó → nếu vẫn thiếu thì đánh dấu `incomplete` (error class `I`).

### Phase 5 — Full pipeline + đánh giá (Kaggle) — ~1h GPU

Hai chế độ bắt buộc theo §6.3 `experimental_plan.md`:

| Chế độ | Input Cross-Encoder | Trả lời câu hỏi |
|---|---|---|
| **Oracle retrieval** | Tool gold | Extraction tốt đến đâu, độc lập retrieval |
| **Full pipeline** | Bi-Encoder top-k + abstention | Hiệu năng hệ thống thật |

`ArgA_oracle − ArgA_pipeline` = phần lỗi do retrieval. Con số này phải xuất hiện tường minh trong báo cáo.

Đo latency **tách 4 giai đoạn** (§10.1):

```
t_query_embed  |  t_retrieve  |  t_cross_encode  |  t_validate
```

Ghi riêng `t_index_build` (pre-compute tool embedding), **không cộng vào latency/query**.

Điền các bảng: C (so sánh 4 method), D (seen/unseen generalization), và phần Bi+Cross của E.

### Phase 6 — Ablation (Kaggle, tùy quota) — ~3h GPU

Thứ tự ưu tiên nếu quota có hạn:

1. **`should_call` head vs ngưỡng cosine** (§5.1) — ảnh hưởng trực tiếp Negative Recall.
2. **Có/không normalizer** — định lượng đóng góp của §4 Phase.
3. **Document text của tool**: `name+desc` vs `name+desc+param_names`.
4. **Số hard negative**: 0 / 4 / 8.
5. **Backbone Cross-Encoder**: `xlm-roberta-base` vs `multilingual-e5-base` vs `bge-m3`.

### Phase 7 — Stress test — ~1h GPU

N ∈ {3, 10, 50, 100, 500, 1000} × {random, same_domain} × 200 query = 2,400 instance.
Distractor lấy từ tool pool ~4,400 tool thật (không sinh giả). `same_domain` = cùng `feature_group`.

Điểm cần chứng minh: **latency của Method 2 gần như phẳng theo N**, trong khi SLM và API tăng tuyến tính. Vẽ 2 biểu đồ: accuracy-vs-N và latency-vs-N (log scale trục x).

---

## 8. Notebook Kaggle — khung sườn

```python
# ===== Cell 1: env =====
!pip install -q sentence-transformers>=3.0 peft transformers accelerate \
                jsonschema rank_bm25 datasets
import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())
# Kỳ vọng: Tesla T4, ~15.0 GB free

# ===== Cell 2: mount code + data =====
# Đẩy repo lên Kaggle Dataset (private) rồi:
!cp -r /kaggle/input/toolcalling-vi-src/src /kaggle/working/
!cp -r /kaggle/input/toolcalling-vi-data/data /kaggle/working/data
%cd /kaggle/working

# ===== Cell 3: resume-safe training =====
CKPT = "/kaggle/working/artifacts/method2/crossencoder/run01"
import os, glob
resume = sorted(glob.glob(f"{CKPT}/checkpoint-*"))[-1] if os.path.exists(CKPT) else None
!python -m src.models.crossencoder.train \
    --config-name crossencoder \
    resume_from={resume} output_dir={CKPT}

# ===== Cell 4: lưu artifact =====
# Kaggle chỉ giữ /kaggle/working (20GB). Nén checkpoint để tải về / làm Dataset mới.
!tar czf /kaggle/working/crossencoder_run01.tar.gz -C {CKPT} .
```

Ba quy tắc sống còn trên Kaggle:

1. **Bật "Save & Run All (Commit)"** cho job dài — session tương tác bị ngắt sau ~20 phút không tương tác, commit run chạy nền đủ 12h.
2. **Checkpoint mỗi 500 step** vào `/kaggle/working`, và **luôn** hỗ trợ `resume_from`.
3. **Cache model HuggingFace** thành Kaggle Dataset (`BAAI/bge-m3` ~2.3GB) thay vì tải lại mỗi session — vừa nhanh vừa tránh rate-limit.

---

## 9. Điểm quyết định còn mở

| # | Câu hỏi | Chốt khi nào | Dữ liệu để quyết |
|---|---|---|---|
| Q1 | Abstention: ngưỡng cosine hay `should_call` head? | Sau Phase 6.1 | Negative Recall + Macro F1 trên val |
| Q2 | Nếu %SKIP nhãn > 30% thì có thêm fuzzy span alignment không? | Sau Phase 1.5 | `data/method2/label_stats.json` |
| Q3 | Có chạy Cross-Encoder BGE-M3-large để so sánh không? | Sau Phase 5 | Quota GPU còn lại + gap accuracy so với XLM-R base |
| Q4 | Multi-call: ngưỡng tuyệt đối hay gap-based? | Sau Phase 2 | Multi-call F1 trên custom val |
| Q5 | Bi-Encoder: giữ LoRA hay chuyển full-FT? | Sau Phase 2 gate | Recall@1 `val_unseen` |

## 10. Rủi ro

| Rủi ro | Xác suất | Tác động | Giảm thiểu |
|---|---|---|---|
| Bug loss index (§2.3a) không được sửa | — | **Toàn bộ kết quả Cross-Encoder vô nghĩa** | Unit test bắt buộc ở Phase 0.4 |
| Trần verbatim làm ArgA thấp hơn SLM | Cao | Luận điểm "Method 2 cạnh tranh accuracy" yếu đi | Đã lường trước: normalizer + báo cáo trần lý thuyết theo §3.3 làm reference line |
| xLAM 20.9% same-tool-repeat | Chắc chắn | ArgA xLAM bị trần cứng | Báo cáo 2 con số, tập đánh giá chính là custom_vi + glaive |
| `src/evaluation/` mất source | Chắc chắn | Chặn mọi việc đo đạc | Phase 0.1, ưu tiên cao nhất |
| Kaggle ngắt session giữa chừng | Trung bình | Mất ~2h GPU | Checkpoint 500 step + resume |
| Glaive 10,500 schema variant làm nhiễu MNRL | Cao nếu bỏ qua | Recall thấp bất thường | Canonicalize ở Phase 1.1, log số biến thể gộp |
| OOM khi eval (batch lớn hơn train) | Trung bình | Crash cuối job | Eval batch = 1/2 train batch, `torch.no_grad` |

## 11. Định nghĩa "xong" (theo §13 experimental_plan)

Mỗi run phải sinh đủ:

- [ ] `config.yaml` đã resolve + commit hash + seed
- [ ] Checkpoint + `thresholds.json` (τ, τ_call) đã freeze trên val
- [ ] `raw_predictions.jsonl` (score retrieval, logit từng head)
- [ ] `parsed_predictions.jsonl` (function_calls JSON cuối)
- [ ] `metrics.json` — retrieval, extraction, end-to-end, cả strict và normalized
- [ ] `latency.json` — tách 4 giai đoạn, P50/P95, peak VRAM
- [ ] `label_stats.json` — %SKIP, %unsupported type, coverage
- [ ] `errors.jsonl` — phân loại theo W/T/P/I (§9 experimental_plan)
- [ ] Chạy lại được trên cùng snapshot dataset

## 12. Lịch trình gợi ý (5 tuần)

| Tuần | Nội dung | GPU |
|---|---|---|
| 1 | Phase 0 + Phase 1 (khôi phục evaluation, sửa bug, tool pool, pairs) | 0h |
| 2 | Phase 2 (Bi-Encoder, 2 round + threshold) | ~6h |
| 3 | Phase 3 (Cross-Encoder, curriculum 2 giai đoạn) | ~6h |
| 4 | Phase 4 + 5 (normalizer, validator, full eval, Bảng C/D) | ~2h |
| 5 | Phase 6 + 7 (ablation, stress test, error analysis, viết báo cáo) | ~5h |

Tổng ~19h GPU, nằm gọn trong quota Kaggle của một tuần — dư địa cho ít nhất một lần chạy lại toàn bộ.

---

## Nguồn tham khảo

- [BGE-M3 fine-tuning và ràng buộc VRAM](https://unsloth.ai/docs/basics/embedding-finetuning) — Unsloth embedding fine-tuning docs
- [sentence-transformers: Cached MNRL / GradCache](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/unsloth/README.md)
- [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding) — BGE-M3 fine-tune cho VI, 72.74% Acc@1 trên Legal Zalo 2021
- [bkai-foundation-models/vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) — PhoBERT-base-v2 backbone
- [hiieu/halong_embedding](https://huggingface.co/hiieu/halong_embedding) — multilingual-e5-base fine-tune cho VI
- [VN-MTEB: Vietnamese Massive Text Embedding Benchmark](https://arxiv.org/pdf/2507.21500)
- [ViRanker: BGE-M3 Cross-Encoder cho Vietnamese Reranking](https://arxiv.org/pdf/2509.09131)
- [Advancing Vietnamese Information Retrieval with Learning Objective and Benchmark](https://arxiv.org/pdf/2503.07470)
