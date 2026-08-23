# Method 2 — Hướng dẫn vận hành

Tài liệu này mô tả **code đã có và cách chạy**. Phần nghiên cứu phương pháp,
ngân sách VRAM/thời gian và các quyết định thiết kế nằm ở `docs/method2_plan.md`.
Định nghĩa metric và contract đánh giá nằm ở `docs/evaluation.md`.

---

## 1. Bản đồ module

```
src/models/
  sources.py               # nguồn dữ liệu, loader, decontamination, manifest SHA-256
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
notebooks/         method2_kaggle_{biencoder,crossencoder,eval}.ipynb
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

### 3.2 Bi-Encoder (Kaggle T4, ~4h)

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

## 7. Việc còn lại

- `feature_group` của glaive/xLAM giữ nguyên `Khác` ở baseline này (quyết định
  đã chốt: không chờ `ALIBABA_API_KEY`). Hard negative của chúng dùng BM25 đúng
  như plan. Slice `same_domain` của Phase 7 đánh dấu **temporarily unavailable —
  pending feature-group enrichment**. Có key thì chỉ cần chạy lại
  `python -m src.models.biencoder.tool_pool`, không phải build lại benchmark.
- Thứ tự đã chốt: Bi-Encoder train → evaluate → Cross-Encoder train →
  normalize/calibrate → full Method 2 evaluation → **sau đó** mới quyết định
  ablation (Phase 6) và stress test (Phase 7). Hai phase đó chưa có code;
  `configs/method2/stress.yaml` đã khai báo sẵn tham số.
- Chưa train: mọi số trong plan là ước lượng cho tới khi chạy thật trên T4.
