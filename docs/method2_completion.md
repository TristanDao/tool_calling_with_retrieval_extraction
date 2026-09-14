# Hoàn thiện Method 2 — 2026-09-06

## Protocol

User chọn strict unseen: tất cả gold tools của CustomTools test_unseen phải vắng mặt trong mọi positive/negative dùng tối ưu Bi-Encoder và trong pool đào hard negatives. Round 1 và round 2 mới đều khởi tạo từ base; round 1 mới là mining teacher. Các schema chỉ được thêm vào index đánh giá. Validation được dùng chọn cấu hình, không dùng test sweep. Giữ nguyên `output_train` gốc và test gold.

Run cũ có negative exposure; vẫn giữ để audit và báo cáo tiến độ. XLM-R base là backbone CE thực tế; BGE-M3 + LoRA + CachedMNRL là Bi-Encoder thực tế. BGE-M3 CE và các ablation Phase 6 tùy quota, không phải điều kiện cứng của nghiệm thu chính.

## Các bước

1. Sửa và kiểm tra evaluator/error analysis, tính lại report từ predictions cũ ở thư mục mới.
2. Dựng snapshot strict unseen, audit mọi negative và mining, tạo cấu hình hai round riêng.
3. Chạy oracle trên CustomTools validation; gate thiếu metric phải ghi incomplete, không tự pass. Điều tra enum/boolean/numeric trước quyết định train CE tiếp.
4. Chạy hai round Bi-Encoder strict trên Kaggle và hiệu chỉnh threshold trên val; đánh giá oracle/pipeline bằng checkpoint đã freeze.
5. Chạy stress random và kiểm tra tính khả thi same-domain. Không gọi mixed distractors là pure same-domain. Ablation normalizer phải chạy inference bật/tắt trên cùng checkpoint.
6. Xuất bảng, biểu đồ và review lỗi; chỉ đánh dấu nghiệm thu khi có output chạy thực tế.

Máy local hiện dùng torch CPU. GPU jobs sẽ có notebook và hướng dẫn; không đánh dấu chúng đã chạy khi mới tạo script.

## Định nghĩa metric

Strict vẫn cho phép Unicode NFC; không đổi case, whitespace, type hay alias. Normalized dùng evaluator chung. Cả hai chấm trên output cuối đã qua postprocessing. Chênh lệch hai metric chỉ đo tác động của normalization khi chấm, không phải đóng góp của normalizer trong inference.

Oracle–pipeline so trên cùng mẫu số positive; không lấy oracle ArgEM trừ ArgEM conditional trên tập đã đúng tool. Cost chưa có đơn giá GPU là unavailable, không phải 0. Throughput 1000/mean latency ghi rõ là estimated sequential QPS. W/T/P/I tự động là gợi ý cần review; đếm event và unique sample riêng.

## Đã thực hiện trên local

- Tái chấm 12.155 query từ run cũ, đủ strict/normalized, oracle gap, bootstrap CI và slices; xuất bảng Markdown/JSON, hình PNG/PDF, danh sách lỗi để review. Output: `artifacts/method2_completion/legacy_review/`.
- Snapshot mới giữ 94.634 dòng Bi-Encoder pairs, loại 905 lượt negative exposure của 20 tool val/test-unseen; audit sau lọc bằng 0. CE train không có các tool heldout. Test gold giữ nguyên từng byte.
- Guard kiểm tra training pairs và loại heldout khỏi corpus trước khi encode để mining. Cả hai round khởi tạo mới; teacher mining chỉ được lấy từ strict round 1 cùng bundle.
- Sửa strict whitespace/type, matching repeated calls, tránh đếm thiếu argument hai lần, gate thiếu metric không tự pass, lưu logits CE và bỏ cost=0 giả định. Sửa wrapper CE giữ cờ `should_call` khi bật ablation.
- Tạo sáu notebook, cấu hình strict hai round, cấu hình CE custom4 dự phòng, runner kiểm tra hash checkpoint/index/threshold và tự xuất báo cáo sau inference.

Không coi các kết quả legacy là kết quả strict unseen. Không suy ra chất lượng mới từ test unit.

## Bước tiếp theo ngay bây giờ: notebook 01

1. Giải nén `artifacts/method2_completion/method2_completion_kaggle.zip`, upload nội dung thành một Kaggle Dataset riêng. Dataset phải chứa đúng một `completion_manifest.json`; không trộn bundle v1/v2 hay output cũ vào dataset này.
2. Import `notebooks/method2_completion_01_validation.ipynb` vào Kaggle. Chọn GPU T4; notebook giới hạn GPU 0 để giữ chế độ đo một GPU. Bật Internet cho cài dependency/tải base model, hoặc chuẩn bị sẵn wheel/cache phù hợp.
3. Add Input gồm dataset bundle ở bước 1 và **một** output Cross-Encoder cũ. Output CE cần thư mục `final/` đầy đủ: `crossencoder_heads.pt`, encoder weights/config và tokenizer. Nguồn local hiện tại: `output_train/output_crossencoder/artifacts/method2/crossencoder/run01/final/`.
4. Run All rồi Save Version với output. Kiểm tra `results/method2/completion/validation/validation_gate.json`. File có `passed`, `complete`, `missing_metrics`, các metric thành phần và accuracy giá trị theo enum/boolean/integer/number/required.
5. Dùng validation để quyết định freeze CE hay train tiếp. Giữ output này để notebook 04 dùng đúng checkpoint đã kiểm định. Có thể gửi `validation_gate.json` và report validation để phân tích bước tiếp theo.

Marker `validation_completed.json` chỉ xác nhận chạy xong. Nếu `passed=false`, đó vẫn là quality gate chưa đạt. Notebook có thể kết thúc thành công để bảo toàn output chẩn đoán; không được ghi gate đạt trong báo cáo.

## Thứ tự các notebook còn lại

Tất cả dùng **cùng một bundle**. Mỗi notebook chạy trong session mới. Dùng Add Input → notebook output đã Save Version của bước trước, giữ nguyên cây thư mục `artifacts/`, `results/`, `data/method2/index/`. Archive có hậu tố `_reports.tar.gz` chỉ tiện tải báo cáo, không chứa model/index và không đủ để làm input bước sau.

| Notebook | Input ngoài bundle | Việc thực hiện | Output chính |
|---|---|---|---|
| 01 validation | CE final cũ hoặc CE ứng viên đã chọn để kiểm định | Oracle CustomTools val + component validation | `validation/validation_gate.json`, oracle raw/predictions, report |
| 02 round1 | Cache BGE-M3 nếu có | Train Bi-Encoder round 1 strict từ base | `artifacts/method2/biencoder/strict_round1/`, marker round1 |
| 03 round2 | Toàn bộ output 02 | Mining bằng teacher strict mới; train từ base; index; calibration chỉ val; retrieval val candidates/pool | `strict_round2/`, `thresholds.json`, index, `retrieval_val_*.json`, marker round2 |
| 04 evaluation | Output 03 **và** output 01 của CE đã freeze | Pipeline + oracle trên ba test sets; tái chấm; error review queue; bảng/hình | `results/method2/completion/evaluation/` |
| 05 stress | Toàn bộ output 04 | 200 anchors × 6 N × random = 1.200 instances | `completion/stress/` gồm report, raw, predictions, anchors, hình |
| 06 normalizer_ablation | Toàn bộ output 04 | Tắt normalizer trong inference, giữ checkpoint/threshold/test | `completion/normalizer_ablation/` với cùng bộ bảng/hình |

02 và 03 là hai lần train thật, cần quota GPU riêng. Run Bi-Encoder cũ mất khoảng 10,65 giờ cho 3 epoch; đây chỉ là mốc tham khảo, thời gian round mới còn phụ thuộc GPU, mining, index và evaluation. Không chạy cả hai round trong cùng session nếu quota không đủ. Cache base model giúp tránh tải lại; notebook không thay thế môi trường có CUDA bằng CPU.

### Input notebook 03 sau round 1 đã hoàn thành

Output 02 ngày 2026-09-06 đã kiểm tra: đủ 3 epoch/921 step, 9,033 giờ, 114 bundle hashes và 10 checkpoint hashes đều khớp. Dùng notebook `notebooks/method2_completion_03_round2.ipynb` bản local mới (có fix torchao).

Trong Add Input, chọn **chỉ toàn bộ output notebook 02 đã Save Version**, vì output này đã chứa `completion_manifest.json`, data/code/config và checkpoint teacher. Không gắn thêm dataset bundle cũ hay output 01: setup yêu cầu đúng một completion manifest. Có thể thêm cache model gốc BGE-M3; bật Internet để cài dependencies và tải base nếu thiếu cache.

Nếu upload từ máy, dùng `artifacts/method2_completion/round2_input_from_round1.zip` thay cho output 02. Gói giữ nguyên bundle, final adapter/tokenizer/pooling, train reports và marker `round1_completed.json`, bỏ các checkpoint trung gian. Giữ nguyên đường dẫn trong zip. Chỉ chọn một trong hai: output Kaggle 02 hoặc dataset tạo từ zip này. Archive `_reports.tar.gz` riêng không đủ.

Notebook 03 tự mining negatives bằng strict round1/final, train round2 từ BGE-M3 base, tạo index và calibrate/evaluate trên val. Dòng `CHƯA ĐỦ: ['retrieval_metrics']` trong manifest round1 phản ánh chưa có report retrieval riêng; bước 03 sẽ tạo report đó, không cần train lại round1. Sau 03 giữ toàn bộ output và kiểm tra `round2_completed.json`, `retrieval_val_candidates.json`, `retrieval_val_pool.json`, `strict_round2/thresholds.json`. CE vẫn cần xử lý validation gate trước khi chốt notebook 04.

06 là ablation tùy quota theo Phase 6. Khi so 04 với 06 phải dùng cùng định nghĩa normalized metric; khoảng cách strict-vs-normalized của 04 không phải hiệu quả bật/tắt normalizer.

## Khi validation CE không đạt

Cập nhật 2026-09-07: đã chuẩn bị thí nghiệm sửa nhãn CE bằng argument_mentions. Chạy **notebook `method2_ce_repair_01b.ipynb`** theo [hướng dẫn CE repair](method2_ce_repair.md) trước khi thử tăng epoch custom4 bên dưới. Addon giữ nguyên các file bundle và checkpoint strict BI; không phải train lại notebook 02/03.

Đọc enum/boolean errors, logits và per-type accuracy trong validation trước. Không chọn cấu hình từ test_unseen. Run cũ có enum validation thấp hơn gate 90%; tăng epoch là một ứng viên cần thử, chưa phải bảo đảm cải thiện.

Bundle có `configs/method2/ce_custom4.yaml`: cùng XLM-R base, warmup 2 epoch, custom finetune 4 epoch, output riêng. Trong session GPU sau cell setup của notebook 01, có thể chạy:

```python
subprocess.run([sys.executable, "-m", "src.models.crossencoder.train",
                "--config", "configs/method2/ce_custom4.yaml"], check=True)
subprocess.run([sys.executable, "scripts/method2/validation_gate.py",
                "--config", "configs/method2/ce_custom4.yaml",
                "--model", "artifacts/method2/crossencoder/custom4/final",
                "--output", "results/method2/completion/validation_custom4"], check=True)
```

So với run01 chỉ trên validation, chọn một checkpoint theo protocol đã định. Nếu chọn custom4, tạo input chỉ chứa thư mục final của checkpoint đó rồi chạy **notebook 01 trong session mới**, không gắn CE run01 đồng thời. Notebook copy checkpoint đã chọn vào đường dẫn chuẩn run01/final để 04–06 dùng thống nhất; hash lưu danh tính weights thật. Lưu riêng train report/config custom4 để mô tả đúng nguồn gốc checkpoint.

Nếu vẫn không đạt, báo cáo chính xác các gate chưa đạt và phân tích lỗi; không tự hạ gate để tuyên bố hoàn thành. Baseline run01 vẫn có thể được báo cáo như thực nghiệm có hạn chế. Runner 04 ghi `quality_gate_passed` để phân biệt chạy xong và đạt tiêu chí.

## Stress same-domain còn thiếu gì

Pool hiện có 4.424 tool nhãn `Khác`; 40 tool CustomTools chia 10 nhóm, mỗi nhóm 4 tool. Vì vậy chưa thể dựng pure same-domain đủ N=10…1000 theo plan. `same_domain_feasibility.json` ghi thống kê này. Phải phân loại thêm tool và kiểm tra số distractor hợp lệ **cho từng anchor** trước khi bật mode; gán nhãn không đảm bảo mỗi nhóm có đủ 1.000 tool.

Runner mặc định chỉ chạy random; yêu cầu pure same-domain vượt capacity sẽ dừng trước inference. Nếu muốn nghiên cứu mixed distractors, phải khai báo `allow_mixed_domain=true`, ghi nhãn `same_domain_first_mixed` và purity; đây là một protocol khác, không bù được 1.200 instances same-domain còn thiếu của plan. Việc mở rộng pool hoặc sửa mục tiêu N cần quyết định nghiên cứu sau khi xem capacity, không tự sinh thêm tool giả.

## Bộ output dùng viết báo cáo

Cập nhật notebook06 (2026-09-08): kiểm tra 114 bundle + 29 artifact hashes, cùng weights/index/thresholds với04, không lỗi cell, đủ IDs cho cả pipeline/oracle/raw trên12.155 query. OFF normalized ArgA benchmark20,99%, seen33,00%, unseen7,00%; ON cải thiện19,22/34,75/15,75 điểm phần trăm. Không có query đổi ranking hoặc chuỗi tên tool; 4.510/271/216 query đổi arguments. Báo cáo tổng hợp PDF/Markdown và evidence tại `reports/method2_20260908/`. Không cần chạy lại01–06. Còn review189 event, CE gate false, same-domain và cost chưa đủ số đo. Lưu ý latency OFF cao hơn ON ở session khác, không suy ra overhead nhân quả của normalizer.

Cập nhật sau notebook05 (2026-09-07): random stress đã hoàn thành đủ 1.200 instances (cùng 200 test_seen anchors, 100 positive/100 negative, trên sáu mức N). Kiểm tra 114 bundle hashes + 29 artifact hashes, giữ đúng checkpoint/threshold notebook04; không lỗi cell, không thiếu/trùng prediction IDs, không cắt gold. Audit: `artifacts/method2_completion/stress05_verification.json`.

| N tools | ArgA positive (strict = normalized) | Tool set accuracy positive | Negative recall | Total P50 ms | Total P95 ms |
|---|---:|---:|---:|---:|---:|
| 3 | 81% | 100% | 100% | 58,16 | 84,50 |
| 10 | 79% | 98% | 100% | 57,15 | 87,38 |
| 50 | 78% | 96% | 96% | 55,82 | 87,68 |
| 100 | 77% | 95% | 95% | 58,55 | 93,83 |
| 500 | 62% | 78% | 76% | 87,44 | 144,97 |
| 1000 | 54% | 68% | 57% | 91,88 | 160,15 |

Đối chiếu predictions: ở N1000, cả 100 positive anchors vẫn có đủ gold tools trong top3, nhưng 32 query chọn thêm tool thừa; 43/100 negative anchors phát sinh call. Không kết luận retrieval hoàn toàn thất bại hay latency phẳng: P50 tăng khoảng58%, P95 khoảng90% từ N3→N1000. Đây là một lượt đo T4 trên random distractors/test_seen, không đại diện test_unseen hoặc so sánh với Method1/API. CE gate vẫn false, pure same-domain vẫn pending. Không tune threshold theo kết quả stress test. Không cần chạy lại05; bước06 tùy quota vẫn dùng toàn bộ output04. Sau đó đối chiếu normalizer, review lỗi và tổng hợp báo cáo.

Cập nhật sau notebook 04 (2026-09-07): output `output_train/output_method2-completion-04-evaluation` đã chạy đủ pipeline/oracle cho 12.155 query, kiểm tra IDs không thiếu/trùng, 114 bundle hashes + 29 artifact hashes khớp checkpoint BI/CE đã chốt. Không có lỗi cell. Audit: `artifacts/method2_completion/evaluation04_verification.json`. Normalized ArgA trên positive queries: benchmark 40,21%, custom_seen 67,75%, custom_unseen 22,75%; CE gate vẫn false.

Bước tiếp: import **file local** `notebooks/method2_completion_05_stress.ipynb`, Add Input **chỉ toàn bộ output04**, GPU T4 + Internet, session mới → Run All → Save Version với toàn bộ output. Giữ cấu trúc `artifacts/`, `data/`, `configs/`, `src/`, `scripts/`, `results/`, `completion_manifest.json`; archive reports riêng không đủ. Không cần input03/01b/bundle riêng. Notebook05 chạy random 200 anchors × N=[3,10,50,100,500,1000] = 1.200 instances; pure same-domain vẫn pending capacity. Gửi output05, notebook đã chạy, marker `results/method2/completion/stress_completed.json` và `method2_completion_stress_reports.tar.gz` để review.

Notebook06 `notebooks/method2_completion_06_normalizer_ablation.ipynb` tùy quota; cũng dùng **output04**, không cần chờ hoặc dùng output05, chạy trong session riêng. Không phải chạy lại 01–04. Giữ checkpoint/threshold cố định khi đã xem test; báo cáo gate chưa đạt, review `human_review_queue.jsonl` và chưa tuyên bố hoàn thành pure same-domain theo plan gốc.

Cập nhật sau CE repair 01b (2026-09-07): đã kiểm tra checkpoint candidate được chọn; đề xuất tiếp notebook 04 với **output 03 + output 01b**, thay output 01 cũ. Không cần chạy lại training. Validation Argument EM=61,96%, enum=74,49% nên gate vẫn false; evaluation phải giữ và báo trạng thái này. Chi tiết kết quả, tradeoff và các input chính xác tại `method2_ce_repair.md`, phần “Bước sau”. Các ghi chú chưa có CE mới ở phần lịch sử bên dưới mô tả trạng thái trước run01b.

Trong output 04, thư mục `results/method2/completion/evaluation/report/` có:

- `report.md`, `tables.json`: bảng ba tập test, strict/normalized ArgA, oracle gap, Arg F1, latency.
- `figures/oracle_pipeline.png` và `.pdf`: so sánh pipeline/oracle, CI pipeline.
- Mỗi dataset: report tổng hợp JSON/Markdown, per-sample metrics, slices/CI, predictions, provenance hash, `errors_reclassified.jsonl`, `human_review_queue.jsonl`.
- Raw predictions gốc cùng timing/logits nằm cạnh report trong thư mục từng dataset của `evaluation/`.

Thêm train report/config/manifest của hai round và CE, validation gate, threshold đã freeze, stress summary/figures và kết quả ablation nếu chạy. Khi viết phần thực nghiệm, nêu rõ candidate-scope test so với full-pool retrieval validation; không diễn giải latency candidate pool nhỏ thành số đo với 4.464 tool.

Review lỗi thủ công: mở `human_review_queue.jsonl`, kiểm tra query/gold/prediction tương ứng, điền `human_label` và `review_note`, giữ nhãn heuristic gốc. Chỉ chọn ví dụ W/T/P/I có bằng chứng; không mặc định mọi giá trị sai là lỗi ngôn ngữ. Báo cáo số event riêng với số query lỗi. Chi phí USD chưa đo đơn giá GPU để `unavailable`.

## Chạy lại và khôi phục

Bản notebook cập nhật 2026-09-07 xử lý duplicate marker: chỉ nhận cây output có đủ weights khớp hash, bỏ bản reports-only (ví dụ archive được giải nén thành cây thứ hai), chấp nhận nhiều bản giống hệt cùng checkpoint. Nếu có nhiều checkpoint khác nhau, notebook liệt kê đường dẫn và yêu cầu chỉ giữ run đã chọn. Bundle giống nhau được nhận diện bằng manifest; bundle khác nhau vẫn bị chặn. Giữ input cũ và import notebook local mới; không phải train lại round1 hoặc upload lại bundle.

Nếu notebook 02 báo `Found an incompatible version of torchao ... 0.10.0 ... 0.16.0`: PEFT gặp package quantization cài sẵn không tương thích. Run hiện tại dùng LoRA thông thường, không dùng torchao. Notebook completion bản local đã thêm `python -m pip uninstall -y torchao` sau cài dependencies và kiểm tra LoRA bằng encoder nhỏ trong subprocess trước khi train. Import lại notebook 02 bản mới, giữ nguyên dataset bundle/cache và chạy trong session mới. Không cần upload bundle lại hay chạy lại 01. Lỗi này xảy ra trong `build_model`, chưa có optimizer step để resume. Nếu sửa notebook trực tiếp trên Kaggle, thêm lệnh uninstall vào cuối cell setup rồi chạy lại cell train; train chạy trong subprocess mới nên không giữ cache import của process vừa lỗi.

Các script chuẩn bị/report từ chối ghi đè thư mục có sẵn. Dùng đường dẫn mới khi tái chạy. Bundle chứa cấu hình YAML tương thích loader dataclass hiện tại, không dùng Hydra YAML composing.

```powershell
.venv/Scripts/python.exe scripts/method2/prepare_completion.py --destination artifacts/method2_completion/kaggle_bundle_new
.venv/Scripts/python.exe scripts/method2/review_outputs.py --output artifacts/method2_completion/legacy_review_new
python scripts/method2/plot_completion.py --tables artifacts/method2_completion/legacy_review_new/tables.json --output artifacts/method2_completion/legacy_review_new/figures
```

Nếu GPU bị ngắt: giữ checkpoint và log mới nhất của **đúng strict round**, không resume từ run02 legacy. Lệnh train hỗ trợ `--resume-from <checkpoint-N>`. Runner theo stage hiện không tự tiếp tục giữa các subcommand; đừng Run All vào thư mục chạy dở vì có thể bắt đầu train lại. Cần khôi phục checkpoint của đúng stage và chạy phần còn lại có kiểm tra manifest. Chỉ marker do runner tạo sau khi tất cả subcommand thành công mới dùng để nối bước sau; không tự tạo marker giả.

## Điều kiện chốt Method 2

Trạng thái 2026-09-07: notebook 03 đã hoàn thành và kiểm tra đủ checkpoint/index/calibration, không cần chạy lại 02/03. Xem `artifacts/method2_completion/round2_verification.json`. CE validation vẫn chưa đạt enum/argument gate. Kiểm tra component pairs cho thấy 250 gold argument occurrences trên CustomTools val bị thiếu (72 integer, 47 number, 131 string); đây là coverage của tập chấm component, không phải toàn bộ gold của oracle inference. Cần kiểm tra cơ chế skip/align, numeric units và has_value trước khi quyết định retrain. Chưa có CE mới để chạy lại notebook 01 hoặc freeze cho notebook 04. Sau khi CE được chốt, notebook 04 dùng output 03 và output validation 01 của CE được chọn; không gắn thêm bundle riêng nếu các output đã chứa bundle.

Code và gói chạy đã chuẩn bị; nghiệm thu thực nghiệm còn cần: validation CE thực tế (và xử lý gate chưa đạt), hai round strict mới, evaluation đã freeze, stress theo phạm vi được chốt, review lỗi và kết luận từ số đo. Pure same-domain theo plan gốc vẫn là mục pending riêng. Method 1 và API baselines nằm ngoài phần việc này.
