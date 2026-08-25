# Method 2 — Hướng dẫn vận hành

Tài liệu vận hành **duy nhất** cho Method 2: code đã có, cách chạy ở local, và
toàn bộ quy trình Kaggle. Hai tài liệu bổ trợ, không trùng nội dung:

- `docs/method2_plan.md` — nghiên cứu phương pháp, ngân sách VRAM/thời gian,
  các quyết định thiết kế và điểm còn mở.
- `docs/evaluation.md` — định nghĩa metric và prediction contract dùng chung cho
  cả bốn method.

| Mục | Nội dung |
|---|---|
| [1](#1-bản-đồ-module) | Module nào làm gì |
| [2](#2-nguồn-dữ-liệu) | Ba nguồn dữ liệu và cách chia split |
| [3](#3-quy-trình-chạy) | Lệnh chuẩn bị dữ liệu → train → đánh giá |
| [4](#4-những-ràng-buộc-đã-mã-hoá-trong-code) | Ràng buộc đã mã hoá, kèm lý do |
| [5](#5-decontamination-theo-query-giữa-các-split) | Chống leakage giữa các split |
| [6](#6-run-manifest-audit-sau-mỗi-lần-train) | Audit sau mỗi lần train |
| [7](#7-chạy-trên-kaggle) | Đóng gói, upload, attach, cấu hình đã chốt, cắt giảm quota, xử lý sự cố |
| [8](#8-việc-còn-lại) | Việc chưa làm và quyết định đang treo |

---

## 1. Bản đồ module

```
src/models/
  sources.py               # nguồn dữ liệu, loader, decontamination, manifest SHA-256
  preflight.py             # cổng fail-closed trước training
  run_manifest.py          # gom artifact audit của một run
  biencoder/
    tool_pool.py           # canonicalize + dedupe tool → tool pool thống nhất
    pairs.py               # sinh (query, positive, hard_negatives)
    train.py               # sentence-transformers + CachedMNRL + LoRA; mine hard negatives
    index.py               # pre-compute embedding → .npy + tool_ids.json
    retrieve.py            # query → top-k, abstention (τ), chọn số call (τ_call / gap)
    evaluate.py            # Recall@k, MRR, NDCG@10 + hiệu chỉnh ngưỡng trên val
  crossencoder/
    model.py heads.py losses.py     # kiến trúc 4 head + gated multi-task loss
    data_collator.py                # BERT-QA, dynamic padding, routing type
    label_generator.py              # nhãn rule-based, char span, SkipLabel có lý do
    dataset.py                      # sinh cặp (query, param) + Dataset train
    train.py                        # training loop fp16, curriculum, resume-safe
    normalize.py                    # normalizer số/tiền/thời gian/ngày VN
    inference.py                    # batch theo tool, decode span SQuAD, normalize
    evaluate.py                     # metric theo head + gate Phase 3
  pipeline/
    validator.py           # jsonschema + coerce + phục hồi required → incomplete
    method2.py             # Bi → Cross → Validator → predictions.jsonl

configs/method2/   tool_pool.yaml  biencoder.yaml  crossencoder.yaml
                   pipeline.yaml   stress.yaml
notebooks/         method2_kaggle_{run0_preflight,biencoder,crossencoder,eval}.ipynb
scripts/method2/   make_notebooks.py
```

`src/models/sources.py` là bổ sung so với cây thư mục ở `method2_plan.md` §6:
cả Bi-Encoder lẫn Cross-Encoder đều đọc chung ba nguồn dữ liệu, để chỗ khác thì
một trong hai package phải import ngược sang package kia.

---

## 2. Nguồn dữ liệu

| Nguồn | File | Vai trò |
|---|---|---|
| glaive + xLAM positive | `data/benchmark_vi/{train,val,test}.jsonl` | 105,539 sample, dựng từ `src/data/build_benchmark.py` |
| glaive negative | `data/translations/glaive_negative_vi.jsonl` | 15,141 sample no-call, chia split theo hash **normalized query** |
| CustomTools-VI | `data/custom_vi/v1/{train,val_seen,val_unseen,test_seen,test_unseen}.jsonl` | 8,000 sample, có sẵn seen/unseen |

Split của Method 2 (`train`/`val`/`test`) khai báo trong `DEFAULT_SOURCES`;
`glaive_negative` chưa có split nên được chia xác định 80/10/10 theo SHA-256 của
**normalized query**. Chia theo `id` là sai bản chất: đơn vị cần cô lập là
query instance chứ không phải id nguồn — file đó lặp lại cùng một query dưới
hàng chục id, chia theo id thì query nằm ở cả ba split (đo được: negative dùng
được tụt từ 18,334 xuống 6,469 sau khi dedupe).

---

## 3. Quy trình chạy

### 3.1 Chuẩn bị dữ liệu (local, không cần GPU)

```bash
# Bộ 2 benchmark từ Bộ 1 translations
python -m src.data.build_benchmark --config configs/data/benchmark.yaml --no-classify

# Tool pool canonical (~4,464 tool, gộp từ ~23k biến thể)
python -m src.models.biencoder.tool_pool

# Cặp huấn luyện Bi-Encoder (~20 phút; tự build decontamination index nếu chưa có)
python -m src.models.biencoder.pairs

# Cặp (query, parameter) cho Cross-Encoder + label_stats.json
# Dùng LẠI decontamination.json của bước trên — hai stage phải chia split y hệt.
python -m src.models.crossencoder.dataset

# Freeze snapshot: SHA-256 nguồn + artefact dẫn xuất + commit hash
python -m src.models.sources
```

Kiểm tra bắt buộc sau bước này:

- `data/method2/biencoder/pairs_stats.json` → `split_overlap_after` phải bằng 0 ở
  **cả ba cặp** và `unseen_tools_leaked_into_train_positives` phải rỗng. Cả hai
  script đều tự `raise` nếu không đạt, nên không thể lỡ tay train trên dữ liệu bẩn.
- `data/method2/label_stats.json` → `skip_rate`. **Vượt 30% thì dừng** và xem lại
  quyết định Q2 (§9 `method2_plan.md`): có thêm fuzzy span alignment hay không.

### 3.2 Bi-Encoder (Kaggle T4, ~4.1h — xem §7.8)

```bash
python -m src.models.biencoder.train train  --config configs/method2/biencoder.yaml
python -m src.models.biencoder.train mine   --config configs/method2/biencoder.yaml \
                                            --model artifacts/method2/biencoder/run01/final
# sửa train_path → train_mined.jsonl rồi train lại TỪ BASE (không train tiếp)
python -m src.models.biencoder.index    --config configs/method2/biencoder.yaml \
                                        --model artifacts/method2/biencoder/run02/final
python -m src.models.biencoder.evaluate calibrate \
    --config configs/method2/biencoder.yaml \
    --model artifacts/method2/biencoder/run02/final \
    --pairs data/method2/biencoder/val.jsonl
```

Bước `calibrate` chỉ được chạy trên **val**. `thresholds.json` sinh ra phải được
freeze trước khi chạy test — tune trên test thì kết quả vô hiệu.

### 3.3 Cross-Encoder (Kaggle T4, ~4.5h)

```bash
python -m src.models.crossencoder.train --config configs/method2/crossencoder.yaml
python -m src.models.crossencoder.evaluate \
    --config configs/method2/crossencoder.yaml \
    --model artifacts/method2/crossencoder/run01/final \
    --pairs data/method2/crossencoder/val.jsonl
```

Training loop tự tìm checkpoint mới nhất trong `output_dir` nếu không truyền
`--resume-from`, nên Kaggle ngắt session giữa chừng thì chỉ cần chạy lại cell.

### 3.4 Pipeline + đánh giá

```bash
# Hai chế độ bắt buộc §6.3 experimental_plan
python -m src.models.pipeline.method2 --gold data/custom_vi/v1/test_seen.jsonl --mode pipeline
python -m src.models.pipeline.method2 --gold data/custom_vi/v1/test_seen.jsonl --mode oracle

python -m src.evaluation.cli evaluate \
    --gold data/custom_vi/v1/test_seen.jsonl \
    --predictions results/method2/predictions/predictions.jsonl \
    --oracle-predictions results/method2/predictions/oracle_predictions.jsonl \
    --slice metadata.tool_split \
    --output-dir results/evaluation/method_2
```

Method 2 **không tự tính** N-FCEM/ArgEM. Nó chỉ xuất `predictions.jsonl` đúng
prediction contract; mọi metric end-to-end đi qua `src/evaluation` để bốn method
dùng chung một normalization và một rule so khớp.

---

## 4. Những ràng buộc đã mã hoá trong code

| Ràng buộc | Ở đâu | Vì sao |
|---|---|---|
| Nhãn lưu **char span**, token index tính lúc train | `label_generator.py`, `dataset.py` | Đổi backbone ở ablation §6.5 không phải sinh lại 250k cặp nhãn |
| Non-verbatim → `SkipLabel("non_verbatim")` | `label_generator.py` | Bản cũ gán span (0,0) tức dạy model trỏ vào `[CLS]` |
| `array`/`object` → `SkipLabel("unsupported_type")`, đếm riêng | `dataset.py` | §3.2: phải báo cáo coverage, không được im lặng bỏ qua |
| Span logit mask bằng `-1e4` chứ không phải `-inf` | `heads.py` | fp16 trên T4: `-1e9` tràn thành `-inf` → NaN |
| Span decode maximize `start[i]+end[j]`, `i ≤ j < i+30` | `inference.py` | argmax độc lập rồi swap cho span sai cả hai đầu |
| Normalizer chỉ chạy trên span đã predict, chỉ khi schema cho phép | `normalize.py` | §8.4 experimental_plan cấm synonym mapping/LLM cho metric chính |
| Key ngoài schema bị loại | `validator.py` | ArgA tính sai nếu thừa key |
| Thiếu required → hạ ngưỡng riêng param đó xuống 0.3 rồi mới `incomplete` | `validator.py` | Error class `I` của §9 |
| Query trùng giữa các split bị dồn về split ưu tiên cao nhất (`test > val > train`) | `sources.py::DecontaminationIndex` | Chống leakage; xem §5 |
| Tool `test_unseen` **có** trong pool nhưng **không** làm positive ở train | `pairs.py` (assert) | Đó chính là bài kiểm tra zero-shot |

---

## 5. Decontamination theo query giữa các split

`data/benchmark_vi` chia split theo **sample** chứ không theo query, nên cùng một
query xuất hiện ở nhiều split. Đo trên dữ liệu thật, theo normalized query
(NFC + gộp whitespace + casefold), **1,718 query** bị trùng:

| Cặp | Số query |
|---|---:|
| `test ∩ train` | 574 |
| `train ∩ val` | 572 |
| `test ∩ train ∩ val` | 542 |
| `test ∩ val` | 30 |

`test ∩ val` (30 + 542 = 572 query) là rủi ro phương pháp luận nặng nhất: dù
không train trên query đó, việc chọn checkpoint/hyperparameter bằng val vẫn làm
metric test lạc quan hơn thực tế.

**Cách xử lý.** `sources.py::DecontaminationIndex` gán mỗi normalized query đúng
**một** split theo thứ tự ưu tiên `test > val > train`, rồi `iter_samples()` bỏ
mọi bản sao ở split thấp hơn. Hệ quả:

- **Tập test không mất sample nào** (13,819 positive + 1,270 negative giữ
  nguyên) — bốn method vẫn được đánh giá trên đúng cùng một tập.
- Val bỏ 3,432 dòng, train bỏ 28,908 dòng.
- `train ∩ val = train ∩ test = val ∩ test = 0`, cả hai script tự `raise` nếu
  không đạt.
- CustomTools-VI **không bị ảnh hưởng** (230/230/460/460 positive nguyên vẹn),
  nên gate Phase 2 đo trên custom val vẫn hợp lệ. Toàn bộ contamination nằm
  trong glaive/xLAM.

Hai stage dùng **chung** `data/method2/decontamination.json`. Nếu Bi-Encoder và
Cross-Encoder chia split khác nhau thì val của stage này lại là test của stage kia.

**Không sửa benchmark gốc.** `data/benchmark_vi/*` giữ nguyên để tái lập được;
decontamination chỉ diễn ra ở tầng dataset của Method 2 và số sample bị loại được
ghi vào `pairs_stats.json::decontamination`. Câu báo cáo tương ứng:

> Benchmark gốc có overlap query giữa các split. Method 2 gán mỗi query đúng một
> split theo thứ tự ưu tiên test > val > train, loại 3,432 dòng khỏi val và
> 28,908 dòng khỏi train; tập test giữ nguyên.

---

## 6. Run manifest — audit sau mỗi lần train

`python -m src.models.run_manifest --run-dir <RUN> --config <CFG> --stage biencoder
--report retrieval=<eval.json>` gom vào `<RUN>/run_manifest.json`: commit SHA kèm
cờ dirty, config YAML nguyên văn + SHA-256, fingerprint dataset và tool pool,
query counts + overlap theo split, số positive/negative pair, checkpoint, best
step và metric đã dùng để chọn, VRAM peak, thời lượng train, Recall@1/@5/@10 và
MRR. Trường `audit_complete.missing` liệt kê mục còn thiếu; notebook `assert`
trên trường này nên không thể train xong mà không audit được.

Tên metric của `InformationRetrievalEvaluator` đổi theo phiên bản
sentence-transformers nên `train.metric_for_best_model` để `null` ở lần chạy đầu;
`train_report.json::checkpoint_selection.available_metrics` sẽ liệt kê tên thật
để khai chính xác cho lần sau. Việc chọn checkpoint hiện chỉ **báo cáo**, chưa tự
nạp lại — bật `load_best_model_at_end` khi đã biết chắc tên metric, để một tên sai
không làm hỏng job sau nhiều giờ GPU.

---

## 7. Chạy trên Kaggle

### 7.1 Dựng gói upload

```bash
python scripts/method2/build_kaggle_upload.py --hf-cache
```

Script copy vào `kaggle_upload/`, kiểm tra `decontamination.json` có mặt, và
verify SHA-256 mọi artefact khớp `manifest.json`. Không đạt thì exit khác 0 —
đừng upload khi đó.

| Kaggle Dataset | Nội dung | Dung lượng |
|---|---|---|
| `toolcalling-vi-src` | `src/`, `configs/{method2,eval}/`, `notebooks/` | 809 KB |
| `toolcalling-vi-data` | `method2/` (192 MB), `custom_vi/v1/` (15 MB), `benchmark_vi/test.jsonl` (14 MB) | 223 MB |
| `toolcalling-vi-hf-cache` | `hub/models--BAAI--bge-m3/`, `hub/models--xlm-roberta-base/` | 3.2 GB |

Tách ba dataset vì vòng đời khác nhau: code đổi mỗi commit, data đổi khi rebuild
pairs, cache model gần như không đổi. Sửa code thì không phải upload lại 3 GB.

`custom_vi/v1/train.jsonl` (38 MB) **không** nằm trong gói: Method 2 train từ
cặp đã sinh sẵn trong `data/method2/`, không đọc lại dữ liệu thô.

### 7.2 Ba chỗ dễ sai khi đóng gói

**`decontamination.json` nằm trong `.gitignore`** nên không đi theo `git clone`.
Thiếu nó thì Run 0 fail — đúng thiết kế, nhưng dễ quên khi đóng gói bằng tay.

**Layout cache HuggingFace.** Đúng là `hub/models--BAAI--bge-m3/…`, không phải
`BAAI/bge-m3/…`. `HF_HOME` phải trỏ vào thư mục **chứa** `hub/`; thiếu tầng đó
thì biến bị bỏ qua trong im lặng và model vẫn tải lại từ Hub mỗi session.
`snapshot_download(cache_dir=X)` đặt thẳng `models--*` vào `X` (layout của
`HF_HUB_CACHE`) nên script tự thêm `hub/`.

**Có thư mục model chưa đủ.** `BAAI/bge-m3` chỉ có `pytorch_model.bin`, không có
safetensors — loại nhầm định dạng đó thì cache tải xong mà không có trọng số
nào, và lỗi chỉ lộ ra lúc nạp model trên Kaggle. `verify_hf_cache()` bắt buộc
mỗi model có file trọng số > 100 MB.

### 7.3 Upload

Qua web: New Dataset → kéo thả từng thư mục con của `kaggle_upload/`.

Qua CLI:

```bash
pip install kaggle          # cần ~/.kaggle/kaggle.json
kaggle datasets init -p kaggle_upload/toolcalling-vi-src
# sửa dataset-metadata.json: "title" và "id" = "<username>/toolcalling-vi-src"
kaggle datasets create -p kaggle_upload/toolcalling-vi-src --dir-mode zip
```

`--dir-mode zip` giữ nguyên cấu trúc thư mục con; thiếu cờ này Kaggle trải phẳng
và notebook không tìm thấy `method2/…`.

Cập nhật về sau (dataset tạo qua web thì phải kéo metadata trước):

```bash
kaggle datasets metadata -p kaggle_upload/toolcalling-vi-src <user>/toolcalling-vi-src
kaggle datasets version -p kaggle_upload/toolcalling-vi-src --dir-mode zip -m "mô tả thay đổi"
```

### 7.4 Attach vào notebook

Notebook Editor → sidebar phải → **Input** → **Add Input** → Add lần lượt ba
dataset. Bật GPU T4.

**Notebook ghim theo version dataset, không tự nhảy sang bản mới.** Sau khi bump
version, phải bấm cập nhật ở sidebar Input (hoặc remove rồi Add lại), nếu không
Kaggle vẫn chạy code cũ và bạn sẽ thấy đúng lỗi như trước — dễ tưởng là chưa sửa
được. Cách xác nhận nhanh: Cell 0 in ra ba dòng `SRC:` / `DATA:` / `HF:`.

### 7.5 Bốn cell đầu của notebook

| Cell | Việc | Vì sao tách riêng |
|---|---|---|
| 0 | dò `SRC_ROOT` / `DATA_ROOT` / `HF_HOME` theo **marker file**, set `HF_HOME` + `HF_HUB_OFFLINE` + `TRANSFORMERS_OFFLINE` | Kaggle mount thành `/kaggle/input/datasets/<user>/<ds>/<ds>/`, số tầng đổi theo cách upload. Phải chạy **trước mọi import transformers** vì thư viện chốt cache lúc import |
| 1 | `pip install` bản đã pin | |
| 2 | copy `src/`, `configs/`, `data/` sang `/kaggle/working` | dataset chỉ đọc, mà code ghi checkpoint và dùng đường dẫn tương đối. Dataset data bắt đầu thẳng bằng `method2/`, **không** có tầng `data/` |
| 3 | kiểm 14 file bắt buộc + `import src.models.preflight` | tách khỏi preflight để phân biệt "copy hỏng" với "dữ liệu sai" |

Đã kiểm tra logic dò trên 4 layout (phẳng, lồng 1 tầng, namespace, namespace +
lồng) — đều tìm ra; không thấy thì notebook dừng kèm marker đã thử.

### 7.6 Run 0 — Pre-flight, 0 giờ GPU training

```bash
python -m src.models.preflight --config configs/method2/biencoder.yaml --require-gpu T4
```

Chuỗi fail-closed, dừng ngay tại bước đầu tiên không đạt:

```
decontamination.json tồn tại → SHA-256 == manifest → overlap == 0
    → unseen leakage == 0 → version khớp bản pin → CHO PHÉP TRAIN
```

`decontamination.json` là **artefact bắt buộc**. Thiếu file hoặc hash lệch thì
job dừng (exit 1), **không rebuild tự động** — nếu experiment chính tự dựng lại
index từ dữ liệu đang có trên máy thì ta mất đúng thứ cần đảm bảo: bằng chứng
model được train trên đúng split đã kiểm định. Rebuild là lệnh preprocessing
riêng, chạy ở local rồi upload lại:

```bash
python -m src.models.sources decontaminate
python -m src.models.sources manifest
```

### 7.7 Cấu hình đã chốt từ benchmark

Lần chạy đầu trên T4 cho **475 s/step** — một epoch 49 giờ, trong khi plan dự
toán 50-70 phút. Nguyên nhân: `CachedMNRL` gọi model ~384 lần mỗi step (batch
256 × 6 văn bản ÷ mini_batch 8, nhân 2 pha GradCache), mỗi lần chỉ 8×192 token
— quá nhỏ để lấp đầy T4; cộng DataParallel giữa 2 GPU thì nhân lên tiếp.

```bash
python scripts/method2/benchmark_biencoder.py --steps 5
```

| Case | batch | mini | ckpt | s/step | step/epoch | h/epoch | VRAM |
|---|---|---|---|---|---|---|---|
| A | 256 | 8 | on | 64.9 | 307 | 5.53 | 2.6 GB |
| B | 256 | 16 | on | 44.1 | 307 | 3.76 | 2.9 GB |
| C | 256 | 32 | on | 47.7 | 307 | 4.07 | 3.6 GB |
| D | 128 | 32 | on | **24.8** | 613 | 4.22 | 3.5 GB |
| **E** | 256 | 32 | **off** | 36.4 | 307 | **3.10** | 10.9 GB |

**Xếp hạng theo `h/epoch`, không theo `s/step`.** D có s/step thấp nhất nhưng
effective batch 128 nên gấp đôi số step mỗi epoch — chọn theo s/step sẽ lấy
đúng cấu hình vừa chậm hơn vừa yếu hơn (MNRL mạnh lên theo số in-batch
negative). `benchmark_biencoder.py` in cảnh báo riêng cho đúng cái bẫy này.

Cảnh báo VRAM của E: 10.9 GB đo khi **tắt eval**, và `max_memory_allocated()`
không tính phần allocator giữ lại. OOM thì bật `gradient_checkpointing: true`
rồi chạy lại — training tự resume từ checkpoint gần nhất.

### 7.8 Cắt giảm để vừa quota

Plan gốc (E, `n_hard_negatives` 4, 3 epoch, có Round 2) tốn **18.6 h** chỉ
riêng Bi-Encoder, cộng ~3.9 h cho phần còn lại của Method 2 → 22.5 h. Cấu hình
hiện tại trong `configs/method2/biencoder.yaml`:

| Cắt gì | Từ → đến | Tiết kiệm | Ảnh hưởng |
|---|---|---|---|
| `epochs` | 3 → 2 | 3.1 h | ít |
| `n_hard_negatives` | 4 → 2 | 3.1 h | **đổi chất lượng** — ablation §6.4 |
| `mining.enabled` (Round 2) | true → false | 4.1 h | không có hard negative đã mine |

Còn **4.1 h** cho Bi-Encoder, tổng Method 2 khoảng **8 h**.

`n_hard_negatives` chỉ là số cột đọc từ file pairs — file vẫn có sẵn 4 negative
mỗi dòng, nâng lại chỉ cần sửa config, không phải sinh lại dữ liệu.

Ba khoản cắt này **phải xuất hiện trong báo cáo**: kết quả Bi-Encoder là của
Round 1, 2 epoch, 2 hard negative — không phải cấu hình trong plan gốc.

### 7.9 Smoke run (đã chạy, không còn trong luồng)

Smoke 100 → resume 200 đã chạy và pass phần chức năng:

- `resume từ .../checkpoint-100 (global_step 100)` — đúng mốc;
- loss 4.606 → 4.297 → 3.971 → 3.626 → 2.931, giảm đều, không NaN;
- checkpoint ghi và đọc lại được;
- lấy đủ tên metric của `InformationRetrievalEvaluator`.

Số **180 s/step** của lần smoke đó **không** đo cấu hình E: log ghi
`Currently using DataParallel (DP)` và `mini_batch` còn là 8. Đừng dùng nó để
ngoại suy.

Bật lại smoke khi đổi backbone hoặc đổi stack: `SMOKE_CELLS` vẫn nằm trong
`scripts/method2/make_notebooks.py`, chỉ cần nối lại vào `BIENCODER_CELLS`.

### 7.10 Full training

Chỉ chạy sau khi Run 1 pass. Gate trước khi sang Cross-Encoder: Recall@1 seen
≥ 0.90 · Recall@1 unseen ≥ 0.75 · Recall@5 unseen ≥ 0.92 · Negative Recall
≥ 0.80. Không đạt thì xử lý retrieval trước, chưa train Cross-Encoder.

### 7.11 Version pin

`configs/method2/pinned_versions.json` — `transformers`, `sentence-transformers`
và `peft` pin tuyệt đối vì chúng quyết định API training **và** tên metric của
`InformationRetrievalEvaluator`; preflight fail nếu lệch. `torch` chỉ ghi nhận:
Kaggle cài sẵn bản CUDA riêng, ép cài lại vừa chậm vừa dễ lệch CUDA runtime.

### 7.12 Xử lý sự cố

| Triệu chứng | Nguyên nhân |
|---|---|
| `ConnectionError … xet-read-token … 404` lúc tải model | backend `xet` của `huggingface_hub`; script đã tự đặt `HF_HUB_DISABLE_XET=1`, tải tay thì export biến đó trước |
| `SHA-256 … không có trong manifest` | manifest sinh trên Windows dùng `\`; đã sửa bằng `preflight.posix_key()`, nếu vẫn gặp thì đang chạy code cũ |
| `commit SHA — không đọc được git` | Kaggle không có `.git`; preflight rơi về `manifest.json::git_commit` |
| Cell 0 báo không tìm thấy marker | chưa Add đủ dataset, hoặc upload thiếu `--dir-mode zip` khiến Kaggle trải phẳng thư mục |
| Notebook chạy code cũ dù đã bump version | chưa cập nhật version dataset ở sidebar Input |
| Train chậm bất thường (hàng trăm s/step) | DataParallel + mini_batch quá nhỏ; chạy §7.7 để chốt cấu hình |

---

## 8. Việc còn lại

- `feature_group` của glaive/xLAM giữ nguyên `Khác` ở baseline này (quyết định
  đã chốt: không chờ `ALIBABA_API_KEY`). Hard negative của chúng dùng BM25 đúng
  như plan. Slice `same_domain` của Phase 7 đánh dấu **temporarily unavailable —
  pending feature-group enrichment**. Có key thì chỉ cần chạy lại
  `python -m src.models.biencoder.tool_pool`, không phải build lại benchmark.
- Thứ tự đã chốt: Bi-Encoder train → evaluate → Cross-Encoder train →
  normalize/calibrate → full Method 2 evaluation → **sau đó** mới quyết định
  ablation (Phase 6) và stress test (Phase 7). Hai phase đó chưa có code;
  `configs/method2/stress.yaml` đã khai báo sẵn tham số.
- **Method 1 (SLM/SFT)**: `src/data/convert_to_instruction.py` giờ áp cùng
  `decontamination.json` (mặc định bật, `--no-decontamination` để tắt kèm cảnh
  báo). Trước thay đổi này nó đọc thẳng `data/benchmark_vi/{train,val,test}.jsonl`
  nên nếu đã train thì đã có train→test leakage. **Còn một khác biệt chưa xử lý**:
  Method 1 chỉ dùng `benchmark_vi`, không có `custom_vi` và `glaive_negative`,
  trong khi Method 2 train trên cả ba. Hai method đang học trên corpus khác nhau
  — cần chốt trước khi so sánh, nằm ngoài phạm vi đã duyệt nên chưa sửa.
- Chưa train: mọi số trong plan là ước lượng cho tới khi chạy thật trên T4.
