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
| [7](#7-chạy-trên-kaggle) | Đóng gói, upload, attach, cấu hình đã chốt, cắt giảm quota, xử lý sự cố, **hướng dẫn chạy lại Phase 6/7 (§7.16)** |
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
    stress.py              # Phase 7: haystack N tool → accuracy/latency theo N

configs/method2/   tool_pool.yaml  biencoder.yaml  crossencoder.yaml
                   pipeline.yaml   stress.yaml
notebooks/         method2_kaggle_{run0_preflight,biencoder,crossencoder,eval}.ipynb
scripts/method2/
  make_notebooks.py        # sinh 4 notebook Kaggle từ một nguồn
  build_kaggle_upload.py   # đóng gói dataset src/data để upload
  benchmark_biencoder.py   # đo giờ/epoch trước khi tiêu quota
  replay_selection.py      # phát lại bước chọn tool offline (chẩn đoán, §7.10c)
  plot_stress.py           # vẽ accuracy-vs-N và latency-vs-N từ stress_report.json
  calibrate_should_call.py # dò ngưỡng abstention của head should_call (§7.17)
  ablation.py              # sinh config nhánh Phase 6 + gộp kết quả (§7.18)
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

### 3.2 Bi-Encoder (Kaggle T4 — Run 1 §7.8, Run 2 §7.9)

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

### 7.8 Run 1 — cắt giảm để vừa quota (đã chạy xong)

Plan gốc (case E, `n_hard_negatives` 4, 3 epoch, có Round 2) tốn **18.6 h** chỉ
riêng Bi-Encoder. Run 1 cắt ba khoản để vừa 4.1 h:

| Cắt gì | Từ → đến | Tiết kiệm | Ảnh hưởng |
|---|---|---|---|
| `epochs` | 3 → 2 | 3.1 h | ít |
| `n_hard_negatives` | 4 → 2 | 3.1 h | **đổi chất lượng** — ablation §6.4 |
| `mining.enabled` (Round 2) | true → false | 4.1 h | không có hard negative đã mine |

Kết quả Run 1 (`run01`, 4.484 h, 614 step, peak 10,885 MB):

| Slice | positive | gold | micro R@1 | **trần** |
|---|---|---|---|---|
| `custom_val_seen` | 200 | 230 | 0.8696 | **0.8696** |
| `custom_val_unseen` | 200 | 230 | 0.8696 | **0.8696** |
| `benchmark_val` | 7,121 | 9,386 | 0.7535 | 0.7586 |

`micro_recall@1 = hits@1 / tổng_gold`, mà sample 2 gold tool ở k=1 chỉ lấy được
1 — nên **0.8696 là trần toán học**, không phải điểm số. Cả 200/200 sample
custom đều có gold ở hạng 1 (`mrr = 1.0`). Gate §Phase 2 ghi ngưỡng ≥ 0.90 theo
giả định mỗi sample 1 gold; **phải sửa lại thành `full_recall@1` hoặc ghi chú
trần**, nếu không người đọc tưởng là trượt gate.

Cảnh báo diễn giải: mọi số trên đo ở `--scope candidates` (pool 1–8 tool cho
benchmark, đúng 10 cho custom), nên `recall@10 = 1.0` là tất yếu. Số pool-scope
duy nhất đang có là `eval_custom_val_cosine_ndcg@10 = 0.7146` từ
`InformationRetrievalEvaluator` (corpus = cả 4,464 tool) — và nó đạt đỉnh ở
**step 614 = step cuối**, tức model vẫn đang lên khi training dừng.

### 7.9 Run 2 — trả về đúng plan trong 11 h

Chìa khoá: plan §Phase 2 bước 3 train Round 2 **từ base**, nên Round 1 chỉ còn
vai trò *máy đào negative*. `run01` đã có → không train lại, tiết kiệm 9.7 h.

```yaml
n_hard_negatives: 4        # = plan
epochs: 3                  # = plan
mining.enabled: true       # = plan
```

| Bước | Ước tính |
|---|---|
| Mine top-20 từ `run01/final` | ~0.3 h |
| Round 2 từ base: 921 step × ~38 s | ~9.7 h |
| Index + calibrate + evaluate | ~0.2 h |
| **Tổng** | **~10.2 h** |

**Điều kiện tiên quyết**: `/kaggle/working` không sống qua session. Lấy lại
checkpoint từ tab Output của notebook cũ — `kaggle kernels output
<user>/<slug> -p ./out` — rồi zip thư mục
`out/artifacts/method2/biencoder/run01/final` (giữ nguyên tên `final`) và
upload thành Kaggle Dataset thứ tư. Đừng mount thẳng Output của notebook cũ:
nó chứa cả `src/` `configs/` `data/` cũ và `find_root` ở Cell 0 có thể vớ phải
bản code cũ. Cell Round 1 dò theo marker
`final/modules.json` — không thấy thì nó train lại Round 1 và **ngân sách vỡ**.

Còn lệch plan đúng một chỗ: `--single-gpu` (plan §0 ghi 2×T4). DDP qua
`torchrun --nproc_per_node=2` về lý thuyết giảm còn ~5.5 h vì
`per_device_train_batch_size=config.batch_size` cho effective batch 512, nhưng
GradCache + DDP là tổ hợp chưa kiểm chứng trong repo này — và nó cũng là một sai
lệch mới so với plan (`batch_size` 256).

Hai cell tốn thời gian đã tắt mặc định: `RUN_BENCHMARK = False` (grid A–E mất
~35 phút và `yaml.safe_dump` của nó xoá sạch comment trong config), và
`HAVE_ROUND1` chặn train lại Round 1.

### 7.10 Smoke run (đã chạy, không còn trong luồng)

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

### 7.10b Bug tìm thấy khi đọc kết quả Phase 5

Kết quả `custom_seen` đầu tiên: Tool Set Accuracy chỉ 0.65 dù đây là tập
"seen" dễ nhất, và `custom_val_seen` lúc calibrate cho `mrr = 1.0` (top-1
luôn đúng). Chênh lệch quá lớn để coi là bình thường — đào ra một bug thật:

**`configs/method2/biencoder.yaml` hardcode `thresholds.strategy: absolute`.**
`calibrate_thresholds()` ([evaluate.py](../src/models/biencoder/evaluate.py))
chỉ đọc `selection["winner"]` (chiến lược thắng theo F1 — thường là `gap`)
khi giá trị config đúng bằng chuỗi `"auto"`. Với `"absolute"` cứng, file
`thresholds.json` luôn ghi `strategy: absolute` **dù chính calibration của nó
đã tính ra `gap` tốt hơn** (F1 0.9543 vs 0.9475, tool_set_accuracy 0.8947 vs
0.8802 trên val) — `winner: gap` nằm ngay trong `metrics.call_selection`
nhưng bị bỏ qua.

Hậu quả với CustomTools-VI: hard negative được thiết kế **cùng
`feature_group`**, nên các candidate có cosine similarity rất gần nhau.
Ngưỡng `absolute` (chọn mọi tool ≥ τ_call) dễ chọn thừa 2–3 tool khi chỉ
đúng 1 — đúng kiểu lỗi mà `gap` (chỉ mở rộng khi điểm runner-up gần top-1)
được thiết kế để tránh.

**Vá:**
1. `configs/method2/biencoder.yaml`: `strategy: absolute` → `strategy: auto`
   — áp dụng cho lần calibrate **sau này**.
2. `reconcile_strategy()` mới trong `evaluate.py`: sửa `thresholds.json`
   **đã có sẵn** bằng chính dữ liệu đã tính trong file đó (`gap_delta` đã
   đúng, chỉ field `strategy` sai) — **không cần GPU, không cần calibrate
   lại**. Notebook Phase 5 gọi hàm này ngay sau khi nạp `thresholds.json`,
   in rõ khi có sửa, và chỉ ghi đè bản copy trong `/kaggle/working` — không
   đụng tới Kaggle Dataset gốc.

5 test mới trong `tests/models/biencoder/test_retrieve_and_calibrate.py`,
gồm một test tái hiện đúng cấu trúc `run02/thresholds.json` đã gặp bug.

**Kết quả sau khi vá** (chạy lại Phase 5, log có dòng
`SỬA strategy: 'absolute' -> 'gap'`):

| Tool Set Accuracy | `absolute` (bug) | `gap` (đã vá) |
|---|---|---|
| `custom_seen` | 0.6500 | **0.8550** |
| `custom_unseen` | 0.5850 | **0.7625** |
| `benchmark` | 0.7949 | **0.8028** |

Số tool chọn trung bình trên `custom_seen` giảm 1.54 → 1.29, đúng cơ chế
chọn thừa đã mô tả. `benchmark` gần như không đổi (+0.008) vì nó không có
hard negative cùng `feature_group` — xác nhận bug này đánh vào riêng
CustomTools-VI.

Negative recall giữ nguyên 0.7450 / 0.7275 ở cả hai chiến lược. Đúng như
code: abstention so `ranked[0][1] < tau` và chạy **trước** `select_tools()`,
nên `strategy` không đổi việc có gọi tool hay không, chỉ đổi số tool.

Không cần train lại, không cần build lại index; chỉ 6 lần chạy
pipeline/oracle + evaluate (~40 phút).

### 7.10c `gap_delta` đang tối ưu cho benchmark — quyết định GIỮ NGUYÊN

`scripts/method2/replay_selection.py --sweep` phát lại bước chọn tool từ
`raw_predictions_pipeline.jsonl` (đã lưu `ranked_tools` kèm score), nên quét
ngưỡng là **tính lại offline**, không tốn GPU. Bảng trên test:

| `gap_delta` | `custom_seen` | `custom_unseen` | `benchmark` |
|---|---|---|---|
| 0.10 | 0.9825 | **0.8875** | 0.7620 |
| 0.15 | **0.9900** | 0.8850 | 0.7879 |
| **0.20 (đang dùng)** | 0.8550 | 0.7625 | 0.8028 |
| 0.25 | 0.7675 | 0.6450 | **0.8074** |

Hai domain muốn hai giá trị khác nhau: custom tối ưu ở ~0.10–0.15, benchmark
ở ~0.25. Giá trị 0.20 là điểm thoả hiệp bị kéo về phía benchmark, vì
`data/method2/biencoder/val.jsonl` dùng để calibrate chủ yếu là row
benchmark.

**Quyết định: giữ 0.20.** Hai thay đổi này khác bản chất:

- Sửa `absolute → gap` là **khôi phục giá trị val đã chọn sẵn** —
  `winner: gap` nằm trong `metrics.call_selection` của chính
  `thresholds.json`, code chỉ không đọc nó. Vá bug, không phải tune.
- Đổi `gap_delta` là **chọn giá trị mới sau khi đã nhìn test**. Kể cả khi
  lấy số từ val của từng domain (`custom_vi/v1/val_seen.jsonl` có sẵn,
  preflight xác nhận `test∩val = 0`), động cơ thay đổi vẫn đến từ quan sát
  trên test — đúng cái pattern làm hỏng tính tin cậy của kết quả.

Thêm nữa, luận văn so sánh 4 method trên cùng benchmark. Cho riêng Method 2
ngưỡng theo domain trong khi 3 method kia dùng ngưỡng chung sẽ làm bảng so
sánh mất công bằng.

Dư địa ~0.13 trên `custom_seen` ghi nhận thành **hạn chế đã biết**, đưa vào
phần thảo luận: *ngưỡng hiệu chỉnh trên val gộp bị benchmark áp đảo nên chưa
tối ưu cho CustomTools-VI*. Bảng sweep chỉ chứng minh dư địa tồn tại, **không
phải nguồn để lấy ngưỡng**.

### 7.10d Extraction không tổng quát hoá sang tool chưa thấy

Đây là phát hiện chính của Phase 5, và nó **không** nằm ở retrieval.

| | `custom_seen` | `custom_unseen` |
|---|---|---|
| Recall@5 | 1.0000 | 1.0000 |
| Tool Set Accuracy | 0.8550 | 0.7625 |
| ArgEM \| ORACLE tool | 0.6425 | **0.1925** |
| N-FCEM positive | 0.5350 | **0.0750** |

Vì cột ArgEM là chế độ **oracle** (tool đúng được đưa thẳng vào
Cross-Encoder), sụt từ 0.6425 xuống 0.1925 không thể đổ cho Bi-Encoder.
Retrieval tổng quát hoá tốt; extraction thì không.

Con số tự kiểm chứng qua `errors_pipeline.jsonl` — đếm lỗi mức **tham số**:

| nhãn (§9) | nghĩa | seen | unseen | tỉ lệ |
|---|---|---|---|---|
| **W** | sai span dù giá trị có nguyên văn trong query | 4 | 123 | **×31** |
| **I** | thiếu tham số | 55 | 266 | ×4.8 |
| **P** | trùng một phần | 19 | 47 | ×2.5 |
| **T** | giá trị canonical, không có trong query | 120 | 129 | ×1.08 |

Kiểm tra tính nhất quán: seen có 1380 param gold, 198 lỗi → 14% sai →
`0.86³·⁰ ≈ 0.64`, đo được 0.6425. Unseen có 1204 param, 565 lỗi → 47% sai →
`0.53²·⁶ ≈ 0.24`, đo được 0.1925. Khớp — ArgEM 0.0984 là kết quả thật, không
phải lỗi đo.

Điều đáng chú ý nhất là **T gần như không đổi**. T là giới hạn kiến trúc của
span head (giá trị gold không xuất hiện nguyên văn nên head không thể sinh
ra), nó phụ thuộc dữ liệu chứ không phụ thuộc tool quen hay lạ — đúng như
kỳ vọng. Mọi nhãn còn lại đều nổ.

Kết luận: `has_value` head và span head đang bám vào **văn bản schema quen
thuộc** thay vì thực sự đọc `Desc=` / `Type=` để suy ra. Gặp tên tham số và
mô tả chưa từng thấy, chúng vừa bỏ sót tham số (I) vừa trỏ nhầm span (W)
ngay cả khi câu trả lời nằm nguyên văn trong query.

### 7.10e Abstention chặn trần Overall Success

Negative recall 0.7450 (`custom_seen`) / 0.7275 (`custom_unseen`): 25% trong
400 sample âm vẫn bị gọi tool — 190 record `hallucinated_call` ≈ 102 sample
× ~1.9 tool. `Overall Success` của slice `none` **bằng đúng** negative recall
nên đây là trần cứng cho nửa âm của tập.

τ cũng hiệu chỉnh trên cùng val bị benchmark áp đảo — cùng gốc vấn đề với
§7.10c, và cũng giữ nguyên vì cùng lý do.

### 7.10f Đóng nốt §11 — run manifest cho Phase 5 và biến thể strict

Hai mục của §11 `method2_plan.md` mà Phase 5 chưa đạt, cả hai bù được mà
không cần GPU.

**1. Notebook eval không sinh run manifest.** Hai notebook train đều gọi
`src.models.run_manifest`, notebook eval thì không — Phase 5 vì thế thiếu
commit + config đã resolve + seed gom về một chỗ.

Không gọi thẳng được vì `_audit_complete()` chỉ có bộ check của run **train**:
nó đòi `checkpoint`, `checkpoint_selection`, `peak_vram`, `duration`,
`retrieval_metrics` — thứ một run đánh giá vốn không có. Chạy nguyên vẹn sẽ
báo thiếu giả và làm `missing` mất hết ý nghĩa cảnh báo.

`_audit_complete()` giờ phân nhánh theo stage. Với `--stage evaluation`, bộ
check là: `predictions`, `raw_predictions`, `errors_classified`,
`latency_stages`, `peak_vram`, `seed`, `metrics_normalized`, `metrics_strict`
— ánh xạ 1-1 với checklist §11. Seed lấy từ `config.seed` trong report của
evaluator: đó là seed thật sự có ảnh hưởng ở Phase 5 (bootstrap CI), còn
pipeline suy luận bằng argmax nên bản thân nó không có seed để ghi.

**2. Evaluator chỉ xuất bản normalized.** Thêm
`NormalizationConfig.strict()` (tắt `collapse_whitespace`, `casefold_strings`,
`coerce_numbers`, `coerce_booleans`, `normalize_dates`, bỏ alias; giữ
`unicode_form=NFC` vì đó là vệ sinh mã hoá chứ không phải nới lỏng ngữ nghĩa).
Report có thêm `strict_extraction`, `strict_oracle_extraction`,
`strict_end_to_end`; `summary.md` có thêm dòng *Strict ArgEM given tool*.

Ví dụ kiểm chứng: gold `{"amount": 5000, "note": "Gấp"}` vs dự đoán
`{"amount": "5000", "note": "gấp"}` → normalized **1.0**, strict **0.0**.

**Đính chính 2026-09-06: chênh lệch normalized − strict chỉ đo normalization khi chấm, không phải ablation inference của
normalizer (Phase 4)** — tức là hạng mục #2 trong danh sách ablation của
Phase 6 (`Có/không normalizer`) đo được ngay từ predictions đã có, không cần
chạy lại trên GPU.

9 test mới: `tests/evaluation/test_strict_metrics.py` (5),
`tests/models/test_run_manifest_evaluation.py` (4).

### 7.11 Phase 3 — Cross-Encoder

Cấu hình **đúng plan §Phase 3, không cắt gì**: `xlm-roberta-base` full
fine-tune, `max_length` 256 + dynamic padding, batch 32 × grad_accum 2 = 64
effective, lr 3e-5 / head_lr 1e-4, curriculum warm-up glaive+xLAM 2 epoch →
fine-tune custom_vi 2 epoch @ 1e-5.

Vòng train **viết tay** (batch mang `schema_type` dạng `list[str]` để route
loss nên `transformers.Trainer` không dùng được) → chạy thẳng trên `cuda:0`,
**không có DataParallel**, không cần `--single-gpu`.

Bốn lỗi đã vá trước khi chạy:

| Lỗi | Hậu quả |
|---|---|
| Resume dựng model mới, chỉ nạp optimizer state | Mất sạch trọng số đã train; optimizer state khớp với bộ trọng số không còn tồn tại |
| `trainer_state.pt` không ghi vị trí curriculum | Resume luôn quay về đầu warm-up, train lại phần đã xong |
| `find_last_checkpoint` sắp theo tên | `checkpoint-1000` < `checkpoint-500`; `stage-warmup` ném `ValueError` |
| `train_report.json` thiếu 3 khoá | `run_manifest` báo audit chưa đủ, assert nổ **sau khi** đã tiêu hết giờ GPU |
| `truncation="only_first"` chỉ cắt query | Param có `description` 883 ký tự (461 token) → cắt query về 0 vẫn không lọt 256 → `Truncation error`. Nổ giữa epoch, sau khi smoke đã pass |
| `dataset` căn span theo `max_length`, collator chỉ chừa `max_length − n_question − 4` | **Hỏng nhãn im lặng**: `_collate_labels._clip` kẹp nhãn vượt biên về vị trí hợp lệ nhưng SAI → span head học nhãn rác |

Gate §Phase 3 giờ đo trên **`by_source["custom_vi"]`**, không phải `overall` —
plan chốt gate ở custom val, mà `overall` bị xLAM chi phối (14,452/17,769 cặp).

Hai lỗi cuối được vá bằng cách cho `data_collator` cắt schema question xuống
`max_question_tokens: 96` và mở API `query_token_budget(param)`; `dataset`
căn span theo đúng ngân sách đó. Đo trên dữ liệu thật: **3 dòng bị loại thay
vì 1** (thêm 2/142,321 = 0.0014%), ngân sách query còn 156–224 token.

**Ngân sách đĩa** là ràng buộc mới so với Bi-Encoder: full fine-tune nên mỗi
checkpoint = 1.04 GB trọng số + 2.07 GB AdamW = **3.11 GB**.

```
run01: 2×checkpoint-* + stage-warmup + stage-finetune + final = 13.4 GB
smoke_run01: 2 checkpoint                                     =  6.2 GB
                                              /kaggle/working =   20 GB
```

Nên notebook **xoá `smoke_run01` ngay sau khi đọc report**, và trước khi đóng
gói thì xoá `trainer_state.pt` của mọi checkpoint (train xong không cần nữa).

Luồng cell: preflight → đọc `label_stats.json` (**không** sinh lại dataset —
sinh lại làm lệch SHA và `benchmark_vi/train.jsonl` cố tình không nằm trong
dataset upload) → smoke 50 step đo `estimated_hours_full_run` → train →
gate → run manifest → dọn + đóng gói.

### 7.12 Full training

Chỉ chạy sau khi Run 1 pass. Gate trước khi sang Cross-Encoder: Recall@1 seen
≥ 0.90 · Recall@1 unseen ≥ 0.75 · Recall@5 unseen ≥ 0.92 · Negative Recall
≥ 0.80. Không đạt thì xử lý retrieval trước, chưa train Cross-Encoder.

### 7.13 Version pin

`configs/method2/pinned_versions.json` — `transformers`, `sentence-transformers`
và `peft` pin tuyệt đối vì chúng quyết định API training **và** tên metric của
`InformationRetrievalEvaluator`; preflight fail nếu lệch. `torch` chỉ ghi nhận:
Kaggle cài sẵn bản CUDA riêng, ép cài lại vừa chậm vừa dễ lệch CUDA runtime.

### 7.14 Xử lý sự cố

| Triệu chứng | Nguyên nhân |
|---|---|
| `ConnectionError … xet-read-token … 404` lúc tải model | backend `xet` của `huggingface_hub`; script đã tự đặt `HF_HUB_DISABLE_XET=1`, tải tay thì export biến đó trước |
| `SHA-256 … không có trong manifest` | manifest sinh trên Windows dùng `\`; đã sửa bằng `preflight.posix_key()`, nếu vẫn gặp thì đang chạy code cũ |
| `commit SHA — không đọc được git` | Kaggle không có `.git`; preflight rơi về `manifest.json::git_commit` |
| Cell 0 báo không tìm thấy marker | chưa Add đủ dataset, hoặc upload thiếu `--dir-mode zip` khiến Kaggle trải phẳng thư mục |
| Notebook chạy code cũ dù đã bump version | chưa cập nhật version dataset ở sidebar Input |
| Train chậm bất thường (hàng trăm s/step) | DataParallel + mini_batch quá nhỏ; chạy §7.7 để chốt cấu hình |

### 7.15 Phase 7 — Stress test theo số lượng tool

Luận điểm bán hàng của Method 2 (§10.2 `experimental_plan.md`): SLM và API phải
nhét cả N tool vào context nên latency tăng tuyến tính theo N, còn Method 2 chỉ
thêm một phép nhân `1×d · d×N` ở Bi-Encoder — Cross-Encoder không đổi vì chỉ
chạy trên tool **đã chọn**.

`src/models/pipeline/stress.py` chạy 200 query của `test_seen` (100 positive,
100 negative — phân tầng để mẫu số của negative recall không đổi theo N) qua
6 mức N, dùng lại đúng `Method2Pipeline` của Phase 5. Chạy cuối notebook eval để
tái dùng index 4.464 tool đã dựng ở đó; tách ra notebook riêng thì phải dựng lại
index ấy lần nữa.

Ba quyết định làm đường cong đọc được:

1. **Haystack lồng nhau.** Mỗi query có đúng một thứ tự distractor cố định, N chỉ
   là độ dài prefix — haystack N=10 chứa trọn haystack N=3. Bốc lại ngẫu nhiên
   cho từng N sẽ trộn "nhiều distractor hơn" với "distractor khác đi", và dao
   động thu được không quy được về N.
2. **Gold luôn ở trong haystack** (`gold ∪ prefix(distractor)`, đúng N phần tử).
   Gold bị bốc rơi thì accuracy tụt vì không có đáp án chứ không phải vì
   retrieval khó lên.
3. **Schema gold lấy từ chính sample, schema distractor lấy từ pool.** Pool gộp
   biến thể trùng tên (`n_variants`) nên schema pool có thể lệch với schema mà
   argument gold được viết theo — lấy nhầm là tự chế thêm lỗi extraction không
   liên quan tới N.

Đã kiểm khô trên dữ liệu thật (không cần GPU): 200 query × 6 mức N, mọi haystack
đúng kích thước, lồng nhau và chứa đủ gold.

**Đọc `t_retrieve` như chặn trên.** `ToolRetriever.score` gom hàng từ index dùng
chung 4.464 tool bằng một vòng lặp Python theo tên, nên phần đó tăng tuyến tính
theo N vì lý do Python thuần tuý; hệ thống thật dựng index đúng N tool thì chỉ
còn một matmul. Khẳng định "phẳng" đọc ở `t_cross_encode` (thành phần chi phối)
và ở `total`. Cố tình **không** fork `run_sample` cho stress test: đo một bản
sao đã tối ưu riêng thì con số không còn nói về hệ thống đang chạy.

#### `same_domain` vì sao vẫn tắt — giờ có số

Config cũ chỉ ghi "sẽ suy biến thành random". Đo thật thì suy biến nhanh hơn dự
đoán rất nhiều, vì mỗi `feature_group` chỉ có 4 tool (1 gold + 3 distractor):

| N | 3 | 10 | 50 | 100 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| purity | 1.000 | 0.359 | 0.069 | 0.034 | 0.007 | 0.003 |

Từ N ≥ 50 slice này **chính là** `random`, chỉ khác cái tên. Bật nó lúc này là
báo cáo hai đường cong trùng nhau như thể chúng đo hai thứ khác nhau. Runner vẫn
cài sẵn mode và ghi `same_domain_purity` vào report; bật lại chỉ là sửa một dòng
config sau khi có nhãn `feature_group` thật.

#### Sản phẩm

`results/method2/stress/` — `stress_report.json`, `stress_summary.md`,
`predictions_{mode}_n{N}.jsonl`, cùng 2 biểu đồ do
`scripts/method2/plot_stress.py` vẽ (accuracy-vs-N, latency-vs-N, trục x log).
Script vẽ tách riêng vì chỉ cần file JSON đã tải về: đổi nhãn, đổi màu, xuất lại
hình cho báo cáo không nên phải đụng tới quota GPU. Trục y của biểu đồ latency
để tuyến tính — log trục y sẽ bóp phẳng cả những đường thật sự tăng tuyến tính,
tức là làm hỏng đúng thứ cần chứng minh.

#### Một bug bắt được khi dựng Phase 7

Cell `run_manifest` (thêm ở §7.10f) nằm **trước** cell định nghĩa `GOLD`: đúng
cú pháp, đủ khối, qua hết mọi kiểm tra đang có, và sẽ chết bằng `NameError` ở
phút đầu của "Save & Run All". Đã chuyển xuống sau phần đánh giá và thêm
`tests/scripts/test_notebook_cell_order.py`: một tên được gán trong notebook thì
lần đọc đầu tiên không được nằm ở cell trước cell gán nó. Kiểm cú pháp từng cell
không thấy được lớp lỗi này vì nó nằm ở quan hệ giữa các cell.

### 7.16 Hướng dẫn chạy lại — Phase 7 và Phase 6

#### A. Phase 7 — chỉ cần một lần Save & Run All

Không train gì. Notebook eval đã có sẵn 3 cell Phase 7 ở cuối, dùng lại chính
index 4.464 tool và 2 checkpoint mà Phase 5 nạp.

1. **Đóng gói lại `src` — bắt buộc.** Code đã đổi (`stress.py`, `plot_stress.py`,
   và cell `run_manifest` đã đổi vị trí). Dataset `toolcalling-vi-src` trên
   Kaggle vẫn là bản cũ; chạy notebook mới trên dataset cũ thì cell Phase 7 gọi
   một module không tồn tại.

   ```bash
   python scripts/method2/build_kaggle_upload.py
   kaggle datasets version -p kaggle_upload/toolcalling-vi-src --dir-mode zip -m "phase 7 stress"
   ```

   `data` và `hf-cache` **không** đổi — Phase 7 không sinh artefact dữ liệu mới.

2. Import lại `notebooks/method2_kaggle_eval.ipynb` vào Kaggle (file trong
   dataset chỉ để lưu trữ; bản chạy là bản trong Kaggle editor).
3. Attach đủ 5 dataset: `src` · `data` · `hf-cache` · `biencoder-run02` ·
   `crossencoder-run01`. **Bump version xong phải bấm cập nhật ở sidebar Input**
   (§7.4) — quên bước này là chạy lại đúng code cũ.
4. Save & Run All. Ước tính ~55 phút: Phase 5 chạy lại 6 lượt (~40 phút, vì
   `/kaggle/working` rỗng ở mỗi commit) + stress 1.200 lượt (~15 phút).

Sản phẩm nằm trong `results/method2/stress/` và đã được `SAVE_CELL` đóng gói.

#### B. Phase 6 — chi phí thật, và vì sao không chạy hết được

Giờ GPU quy từ benchmark case E (§7.7): 307 step/epoch, và s/step **ngoại suy
tuyến tính từ hai điểm đo** (24,3 s ở `n_neg=2`; 36,4 s ở `n_neg=4`) →
`s/step ≈ 6,05 × (2 + n_neg)`. Hai điểm là ít, hãy coi cột giờ là bậc độ lớn.

| # | Ablation | Đụng vào | Cần code mới? | Giờ GPU (giao thức đầy đủ) |
|---|---|---|---|---|
| 1 | `should_call` head vs ngưỡng cosine | Cross-Encoder + abstention | **Có** — head thứ 5 | ~6 h + 1–2 ngày code |
| 3 | doc text `name+desc` vs `+param_names` | Bi-Encoder | Không (1 cờ config) | ~10 h |
| 4 | số hard negative 0/4/8 | Bi-Encoder | Không (1 cờ config) | ~30 h |
| 5 | backbone Cross-Encoder | Cross-Encoder | Không (1 cờ config) | ~13,5 h |

Tổng theo giao thức đầy đủ ≈ 60 h. Quota Kaggle 30 h/tuần, session tối đa 12 h —
và riêng `n_neg=8` chạy đủ 3 epoch là 15,5 h, **không lọt một session**. Chạy hết
Phase 6 đúng như plan là không khả thi.

#### C. Giao thức rút gọn — cách duy nhất để Phase 6 vừa quota

Ablation trả lời câu hỏi **so sánh** ("X có giúp không?"), không phải câu hỏi về
mức tuyệt đối. Nên mọi nhánh chạy 1 epoch, một vòng, không mining:

| `n_neg` | s/step | 1 epoch |
|---:|---:|---:|
| 0 | 12,1 | 1,03 h |
| 4 | 36,3 | 3,10 h |
| 8 | 60,5 | 5,16 h |

- **#4** (0 / 4 / 8) = 9,3 h + 3 × 0,2 h index/calibrate/evaluate ≈ **9,9 h**
- **#3** = thêm đúng một nhánh `n_neg=4, include_param_names=false` ≈ **3,3 h**,
  dùng chung nhánh `n_neg=4` của #4 làm đối chứng.

**#3 + #4 ≈ 13 h — vừa một tuần quota**, thay vì 40 h.

Cái bẫy phải tránh: **không** so nhánh rút gọn với `run02` (3 epoch + Round 2).
Làm vậy là đo số epoch chứ không đo ablation. Nhánh đối chứng phải được chạy lại
ở đúng giao thức rút gọn, kể cả khi đã có một model "tốt hơn" nằm sẵn đó.

Báo cáo cũng phải nói rõ: các số Phase 6 đo ở giao thức 1-epoch, không so trực
tiếp được với bảng Phase 5.

#### D. Fail-closed và ablation — chỗ dễ làm sai nhất

`preflight` khoá SHA-256 của bốn artefact (§7.6), trong đó có
`data/method2/tool_pool.json` và `data/method2/biencoder/train.jsonl`. Ablation
**#3 và #4 sinh ra phiên bản khác của đúng hai file đó**. Sửa cờ trong config
gốc rồi rebuild tại chỗ là ghi đè baseline và bị Run 0 chặn — nó chặn đúng.

**Bản sửa so với lần viết đầu của mục này.** Ban đầu tôi ghi ở đây là phải
rebuild artefact tại chỗ, chạy lại `sources manifest`, rồi upload thành một
dataset data riêng cho mỗi nhánh. Cách đó chạy được nhưng sai hướng: nó ghi đè
artefact baseline rồi đi vá lại cổng, và giữ được baseline chỉ nhờ kỷ luật của
người thao tác. Cách đúng là **không đụng vào file bị khoá ngay từ đầu**.

`scripts/method2/ablation.py` sinh config trong đó mọi output của nhánh trỏ vào
`data/method2/ablation/<nhánh>/` và `artifacts/method2/ablation/<nhánh>/`. Bốn
artefact bị khoá không bị ghi lên, nên:

- `manifest.json` **không đổi**, không phải chạy lại `sources manifest`;
- preflight vẫn pass, không phải vá gì;
- baseline còn nguyên **do cấu trúc**, không phải do nhớ đừng ghi đè;
- không cần dataset data riêng — chỉ bump version dataset data hiện có, vì
  `build_kaggle_upload.py` đã copy nguyên `data/method2/` nên thư mục `ablation/`
  đi kèm sẵn.

```bash
python scripts/method2/ablation.py configs     # sinh + kiểm "khác đúng 1 knob"
# chỉ những nhánh cần: xem dòng "CẦN dựng lại:" mà lệnh trên in ra
python -m src.models.biencoder.tool_pool --config configs/method2/ablation/doc_name_desc.tool_pool.yaml
python -m src.models.biencoder.pairs     --config configs/method2/ablation/nneg8.biencoder.yaml
python scripts/method2/build_kaggle_upload.py
kaggle datasets version -p kaggle_upload/toolcalling-vi-data --dir-mode zip -m "ablation arms"
```

`tool_pool` và `pairs` phải chạy ở **local**: chúng đọc
`data/benchmark_vi/train.jsonl`, file cố tình không nằm trong dataset upload.
Trên Kaggle chỉ còn train → index → đánh giá.

`control` không có dòng "CẦN dựng lại" nào: nó dùng thẳng
`data/method2/biencoder/` vì cùng `n_hard_negatives`, cùng seed, cùng nguồn thì
bộ pair sinh ra là y hệt. Sinh lại cho nó là mở cửa cho một control "giống nhưng
không y hệt" baseline.

Đừng tìm cờ để bỏ qua preflight. Cổng đó tồn tại vì đã có lần artefact lệch mà
không ai biết cho tới khi tiêu hết giờ GPU.

Riêng **#5** (backbone Cross-Encoder) không đụng artefact nào: chỉ đổi
`model.name` trong `configs/method2/crossencoder.yaml`, upload lại `src`. Nhưng
`bge-m3` (568M) ở batch 32 × len 256 là ~17,5 GB → OOM trên T4. Phải hạ
`batch_size: 16` **và** nâng `grad_accum: 4` để giữ effective batch 64 — đổi
effective batch là đổi luôn quá trình tối ưu, so sánh backbone khi đó không còn
sạch. Ước tính nhánh này ~9 h, sát trần session.

#### E. Thứ tự đề xuất

**#1 trước, dù nó là mục duy nhất cần code mới** — code đó **đã viết xong**,
xem §7.17. §7.10e đã cho thấy abstention là thứ chặn trần Overall Success
(negative recall 0,745 / 0,728), mà #1 nhắm thẳng vào đó. Ba mục còn lại chỉnh retrieval hoặc extraction — hai chỗ đang
không phải nút thắt. Plan xếp #1 đầu bảng vì lý do khác (§5.1), nhưng số đo của
Phase 5 dẫn tới cùng một kết luận.

Sau đó: **#4** (rẻ, đã có sẵn cờ, trả lời câu hỏi thiết kế đã đặt từ §Phase 1.3)
→ **#3** (thêm 3,3 h khi đã dựng xong bộ #4) → **#5** cuối cùng, và bỏ hẳn nếu
quota không đủ: đổi backbone là câu hỏi ít gắn với đóng góp của luận văn nhất.

Mục #2 của plan (có/không normalizer) **không cần chạy lại** — chênh lệch
`strict` − `normalized` trong report Phase 5 chính là nó (§7.10f).

### 7.17 Ablation §6.1 — head `should_call` thay ngưỡng cosine τ

Mục ưu tiên số một của Phase 6, và §7.10e là lý do: abstention chính là thứ chặn
trần Overall Success (negative recall 0,745 / 0,728). Ba ablation còn lại chỉnh
retrieval hoặc extraction — hai chỗ hiện không phải nút thắt.

Cơ chế: một head nhị phân cấp **tool** trên Cross-Encoder, chạy trên
`(query, tool top-1 của Bi-Encoder)`. Nó thay đúng một thứ — ngưỡng τ — chứ
không thay việc chọn tool.

#### Thiết kế dữ liệu, và một thứ cố tình KHÔNG làm

Hàng cấp tool đi chung file, chung batch với hàng cấp parameter, phân biệt bằng
`schema_type="should_call"`; `HierarchicalLoss` tách hai loại ra trước khi tính
bất cứ head nào. Đây là chỗ dễ hỏng âm thầm nhất: hàng cấp tool không có nhãn
span/enum/boolean, để nó lọt vào các head đó là dạy nhãn rác mà không có gì báo.
Có test riêng cho cả hai chiều.

**Negative chỉ lấy từ sample no-call.** Sinh thêm negative kiểu "query cần tool,
nhưng không phải tool NÀY" từ candidate sai của sample positive là làm được và
đã bị loại bỏ có chủ đích: head này thay τ, mà τ chỉ trả lời "query có cần gọi
tool không". Trộn thêm câu hỏi "có phải tool này không" là đổi luôn thứ head
phải học, và ablation không còn so một-đổi-một với baseline nữa. Chọn đúng tool
đã là việc của Bi-Encoder.

**Cân bằng lớp là bắt buộc, không phải tinh chỉnh.** Train có 59.130 sample
positive so với 16.199 no-call; để nguyên thì head học được cách luôn trả 1 mà
loss vẫn thấp — hỏng đúng lớp mà §7.10e chỉ ra là nút thắt. Downsample positive
về bằng negative, giữ trọn lớp hiếm.

Sinh thật trên toàn corpus (local, không GPU):

| Split | hàng param | + hàng tool | tổng | positive/negative |
|---|---:|---:|---:|---|
| train | 142.321 | 32.398 | 174.719 (**+22,8%**) | 16.199 / 16.199 |
| val | 17.769 | 1.730 | 19.499 (+9,7%) | 865 / 865 |
| test | 23.810 | 2.540 | 26.350 (+10,7%) | 1.270 / 1.270 |

→ CE run ≈ 4,5 h × 1,228 ≈ **5,5 h**, cộng val + test pipeline ≈ **6 h** tổng.
Đúng bằng ước tính ở §7.16.

#### Hai điểm yếu đã biết, ghi ra trước khi có số

**Tool ghép với negative lúc train không phải tool sẽ gặp lúc chạy.** Lúc train,
negative được ghép với candidate của chính sample đó (train chỉ có ~2,11
candidate/sample no-call nên gần như là ghép cố định). Lúc chạy thật, tool được
hỏi là top-1 của Bi-Encoder — tức là tool *dễ nhầm nhất*. Đây là proxy, và nếu
head chạy kém hơn τ thì đây là chỗ phải soi đầu tiên.

**Val chỉ có 865 sample no-call** để dò ngưỡng. Đủ để chọn một ngưỡng, không đủ
để tin vào chữ số thứ ba. Ngưỡng chọn ra nên được đọc kèm đường cong macro-F1
trong `should_call_threshold.json` — nếu đỉnh phẳng thì con số cụ thể không quan
trọng bằng khoảng.

#### Không bao giờ quay lặng lẽ về τ

`abstention: should_call` trên checkpoint không có head thì pipeline **dừng**.
Quay về τ trong im lặng nghĩa là báo cáo số baseline dưới tên ablation — đúng
lớp lỗi đã xảy ra với `strategy` ở §7.10b và chỉ lộ ra sau khi đã tiêu hết giờ
GPU. Cùng lý do, `pairs.should_call.enabled` và `model.enable_should_call` phải
bật cùng nhau: bật một bên là có nhãn mà không có head, hoặc có head mà không có
gì dạy nó.

Khi tắt, head **không được tạo** — state dict còn đúng 5 tensor như `run01`, nên
checkpoint baseline vẫn nạp bằng `load_state_dict` strict. `from_pretrained`
dựng lại `HeadConfig` từ dict lưu trong checkpoint; dict cũ không có khoá này nên
nhận default `False`.

#### Quy trình chạy

Ngưỡng được hiệu chỉnh **trên val rồi FREEZE**, y hệt τ, và bằng cùng một hàm
mục tiêu (Macro-F1 giữa {call, no_tool_call}) — so hai cơ chế abstention mà chọn
ngưỡng theo hai tiêu chí khác nhau là so hai thứ khác nhau.

Nhánh này ghi pair sang thư mục riêng, **không** đụng
`data/method2/crossencoder/train.jsonl` — cùng nguyên tắc với ablation
Bi-Encoder (§7.16 mục D): artefact bị khoá còn nguyên nên `manifest.json` không
đổi, preflight vẫn pass, và baseline chạy lại được do cấu trúc chứ không do nhớ.
Bốn giá trị phải đổi cùng nhau đã ghi sẵn trong `crossencoder.yaml`.

Nhánh này là một arm của `ablation.py` (§7.18), nên config được **sinh ra** chứ
không sửa tay — bốn giá trị phải đổi cùng lúc (hai cờ + ba đường dẫn) là đúng
loại việc mà sửa tay hay sót một chỗ.

```bash
# 1. Local: sinh config rồi dựng pair cho nhánh.
python scripts/method2/ablation.py configs
python -m src.models.crossencoder.dataset \
  --config configs/method2/ablation/should_call.crossencoder.yaml
python scripts/method2/build_kaggle_upload.py
kaggle datasets version -p kaggle_upload/toolcalling-vi-data --dir-mode zip -m "should_call pairs"

# 2. Kaggle: notebook crossencoder, đặt
#      CE_CONFIG = 'configs/method2/ablation/should_call.crossencoder.yaml'
#    → checkpoint có head.

# 3. Kaggle: notebook eval, đặt RUN_SHOULD_CALL = True.
#    Cell tự chạy val → hiệu chỉnh → freeze → test dưới CẢ HAI cơ chế → in bảng.
```

Sinh pair phải chạy ở local: `benchmark_vi/train.jsonl` cố tình không nằm trong
dataset upload (§7.11).

Ngưỡng dò được **offline** từ `metadata.should_call_prob` mà pipeline ghi vào
từng prediction — cùng mẹo đã cứu §7.10b: chạy GPU một lần trên val rồi quét
ngưỡng bao nhiêu lần cũng được mà không tốn thêm quota.

#### Phép so phải cùng checkpoint

Bản đầu của cell này so `should_call` (checkpoint mới) với bảng Phase 5 (`run01`).
Đó là phép so **hỏng**: hai bên khác nhau cả cơ chế abstention lẫn checkpoint —
`run01` train trên bộ pair khác (+22,8% dòng ở bản mới), nên chênh lệch gộp hai
nguyên nhân và không tách ra được nữa.

Cách đúng: chạy **cùng một checkpoint** dưới cả hai cơ chế —
`--abstention tau` và `--abstention should_call`. Checkpoint mới vẫn dùng được τ
bình thường vì head `should_call` chỉ là một head thêm, không đụng gì tới ranking
của Bi-Encoder. Tốn thêm một lượt pipeline mỗi tập (~7 phút) và đổi lại là một
phép so sạch, chỉ khác đúng một biến.

Bảng Phase 5 vẫn nên đọc kèm, nhưng như **mốc thứ ba** (checkpoint khác), không
phải như đối chứng của ablation này.

### 7.18 Trình chạy ablation Phase 6 — `scripts/method2/ablation.py`

Ablation hỏng theo đúng một kiểu, và nó không tự lộ ra: nhánh khác baseline ở
**hai** chỗ thay vì một, vì lần sửa YAML trước còn sót lại. Con số vẫn ra, vẫn
"hợp lý", và kết luận rút ra từ nó thì sai.

Nên config từng nhánh được **sinh ra**, không sửa tay. Mỗi nhánh khai đúng knob
nó đổi; `verify_arms()` đối chiếu ngược — config đã resolve phải khác `control`
đúng bằng tập knob đã khai, không hơn. Lệch là `SystemExit` ngay ở local. Có test
dựng lại đúng tai nạn đó để chứng minh chốt chặn thật sự chặn.

Tám nhánh, chia **hai họ có control riêng**. Bi-Encoder rút gọn được xuống
1 epoch; Cross-Encoder giữ nguyên curriculum của Phase 3 vì §6.1 cần head học
đến nơi thì con số abstention mới nói lên điều gì. Một control chung cho cả hai
là so nhầm, nên `verify_arms()` đối chiếu mỗi nhánh với control **của họ nó**.

| Họ | Nhánh | Knob | Câu hỏi |
|---|---|---|---|
| Bi | `control` | — | mốc so sánh, chạy lại ở giao thức rút gọn |
| Bi | `nneg0` | `pairs.n_hard_negatives = 0` | §6.4 — hard negative có đáng số giờ nó tốn? |
| Bi | `nneg8` | `pairs.n_hard_negatives = 8` | §6.4 — gấp đôi có còn cải thiện? |
| Bi | `doc_name_desc` | `include_param_names_in_doc = false` | §6.3 — tên param trong doc_text có giúp? |
| CE | `ce_control` | — | mốc so sánh của họ Cross-Encoder |
| CE | `should_call` | `pairs.should_call.enabled`, `model.enable_should_call` | §6.1 — head có hơn τ? |
| CE | `backbone_e5` | `model.name = multilingual-e5-base` | §6.5 — backbone cùng cỡ |
| CE | `backbone_bge` | `model.name = bge-m3`, `batch_size = 16`, `grad_accum = 4` | §6.5 — backbone 568M |

`backbone_bge` là nhánh duy nhất đổi nhiều hơn một knob, và hai knob thừa là để
vừa VRAM chứ không thuộc câu hỏi. Chốt chặn không cấm — nó bắt phải **khai đủ**;
`caveat` của nhánh thì bắt phải **công bố**, và nó được in ra thành một mục riêng
trong báo cáo ("Nhánh không sạch một biến"). Giữ effective batch 64 y control,
nhưng số micro-batch đổi nên bước tối ưu không còn giống hệt — người đọc bảng
cần biết điều đó mà không phải đi đọc config.

Bốn thứ script tự lo mà nếu làm tay thì dễ quên:

1. **Nhánh đổi `doc_text` phải dựng lại cả pool lẫn index.** Quên index thì nhánh
   train trên doc_text mới nhưng được đánh giá bằng embedding cũ — số ra vẫn
   trong khoảng hợp lý nên không ai nhận ra.
2. **Nhánh chỉ đổi knob training thì không sinh lại pair**, dùng thẳng bộ của
   baseline (xem §7.16 mục D). Đổi backbone Cross-Encoder cũng vậy: nhãn ghi ra
   đĩa là char span, tokenizer-free — đúng lý do §1.4 chọn char span thay vì
   token index, và đây là lần nó trả công.
3. **Giao thức được ghi vào chính file report**, nên người đọc bảng thấy ngay đây
   là số 1-epoch chứ không phải số so được với Phase 5.
4. **Metric của họ Cross-Encoder đọc trên lát `custom_vi`**, đúng lát mà gate
   §Phase 3 dùng — `overall` bị xLAM chi phối (14.452/17.769 cặp). Hai nơi đọc
   hai lát khác nhau về cùng một run là cách chắc chắn để tự mâu thuẫn.

Nhánh Bi-Encoder chạy trong notebook Bi-Encoder qua `ARMS_TO_RUN`; nhánh
Cross-Encoder chạy trong notebook Cross-Encoder qua `CE_CONFIG` — trỏ vào file
config mà `ablation.py configs` sinh ra, **không** sửa `crossencoder.yaml` gốc
(sửa gốc thì lần sau không còn biết nhánh nào đã chạy).

Ngân sách: 4 nhánh Bi-Encoder ≈ 13,2 h, **vượt trần 12 h/session**, nên chia ba
lượt: `control`+`nneg0` (~4,4 h), `nneg8` (~5,4 h), `doc_name_desc` (~3,4 h).
Họ Cross-Encoder tính riêng: `should_call` ~6 h (§7.17), mỗi nhánh backbone
~4,5 h, riêng `backbone_bge` ~9 h vì phải hạ micro-batch.

Tải kết quả từng lượt về rồi gộp ở local:

```bash
python scripts/method2/ablation.py report   # → results/method2/ablation/
```

---

## 8. Việc còn lại

- `feature_group` của glaive/xLAM giữ nguyên `Khác` ở baseline này (quyết định
  đã chốt: không chờ `ALIBABA_API_KEY`). Hard negative của chúng dùng BM25 đúng
  như plan. Slice `same_domain` của Phase 7 đánh dấu **temporarily unavailable —
  pending feature-group enrichment**. Có key thì chỉ cần chạy lại
  `python -m src.models.biencoder.tool_pool`, không phải build lại benchmark.
- Thứ tự đã chốt: Bi-Encoder train → evaluate → Cross-Encoder train →
  normalize/calibrate → full Method 2 evaluation → **sau đó** mới tới ablation
  (Phase 6) và stress test (Phase 7). Phase 7 đã có code và đã nằm trong notebook
  eval (§7.15); Phase 6 còn lại 4/5 mục.
- **Method 1 (SLM/SFT)**: `src/data/convert_to_instruction.py` giờ áp cùng
  `decontamination.json` (mặc định bật, `--no-decontamination` để tắt kèm cảnh
  báo). Trước thay đổi này nó đọc thẳng `data/benchmark_vi/{train,val,test}.jsonl`
  nên nếu đã train thì đã có train→test leakage. **Còn một khác biệt chưa xử lý**:
  Method 1 chỉ dùng `benchmark_vi`, không có `custom_vi` và `glaive_negative`,
  trong khi Method 2 train trên cả ba. Hai method đang học trên corpus khác nhau
  — cần chốt trước khi so sánh, nằm ngoài phạm vi đã duyệt nên chưa sửa.
- Phase 0–5 đã chạy thật trên T4; số trong §7.10b–e là số đo, không còn ước
  lượng. **Phase 7 đã có code** (`src/models/pipeline/stress.py` +
  `scripts/method2/plot_stress.py`, §7.15) nhưng **chưa chạy trên GPU** — cần
  một lần Save & Run All của notebook eval để có số thật.
- **Phase 6 (ablation)**: mục #2 "có/không normalizer" cần hai lần inference trên cùng checkpoint. Chênh lệch `strict` − `normalized` không thay thế ablation này. Xem `method2_completion.md`. Mục **#1 `should_call` head đã có
  đủ code** (head + loss + dữ liệu + pipeline + hiệu chỉnh + notebook, §7.17)
  nhưng **chưa train** — cần ~6 h GPU. Ba mục còn lại (#3 document text, #4 số
  hard negative, #5 backbone) chỉ là cờ config nhưng đều phải train lại. **Cả bốn
  mục giờ đều là arm khai báo sẵn trong `ablation.py`** (§7.18) — config sinh ra
  và có chốt "khác đúng một knob", chưa mục nào được train. Còn thiếu ~13,2 h GPU
  cho họ Bi-Encoder và ~20 h cho họ Cross-Encoder. Chi phí thật và giao thức rút
  gọn để vừa 30 h/tuần: **§7.16**.
- Hai mục hụt của §11 `method2_plan.md` đã đóng, không tốn GPU — xem §7.10f.


### Method 2 completion — 2026-09-06

User chọn strict unseen. Các run cũ có negative exposure được giữ nguyên. Workflow mới, thứ tự notebook và kiểm định chất lượng ở [method2_completion.md](method2_completion.md). Phase 6 vẫn là tùy quota; pure same-domain cần đủ tool cùng nhóm, không âm thầm thay bằng random.
