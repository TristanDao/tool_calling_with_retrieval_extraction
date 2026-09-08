# CE repair v1 — chạy notebook 01b

## Đã sửa gì

Label generator cũ tìm `str(gold_value)` trong query. Cách này bỏ các số có format/đơn vị khác gold và có thể trỏ vào lần xuất hiện sai của chuỗi ngắn. CustomTools đã có annotation `argument_mentions`, nên bản sửa dùng đúng `(call_index, parameter_path, start, end)` sau khi kiểm tra surface và canonical value. Bao gồm span đầy đủ như `100 triệu`, không gán nhãn chỉ `100` cho gold `100000000`.

| Phần | Pairs trước | Pairs sau | Khôi phục | Sửa ranh giới span |
|---|---:|---:|---:|---:|
| Custom train | 12.253 | 15.318 | 3.065 | 693 |
| Custom validation | 1.434 | 1.684 | 250 | 109 |

Giữ nguyên 3.600 sample train đã được giữ trong run trước, 400 sample positive validation, các dòng benchmark, split và test gold. Negative optional parameters vẫn có `has_value=0`; không tự hạ threshold để giữ mọi giá trị. Training CE không có 20 tool heldout. Tất cả 17.002 custom pairs mới đã kiểm tra bằng tokenizer XLM-R thực tế: không bị loại vì ngoài cửa sổ và không có nhãn token nằm ngoài segment query.

Một số annotated surfaces không chuyển được sang canonical string bằng normalizer hiện tại: 642 train và 44 validation, ví dụ “Sài Gòn” → “TP. Hồ Chí Minh”. Các trường hợp này được dùng để học surface span và has_value, nhưng canonical Argument EM vẫn chấm theo gold gốc. Không thêm synonym map hay dùng gold/argument_mentions ở inference. Span EM chỉ đo surface extraction, không thay thế Argument EM.

## Input cần gắn

1. Import `notebooks/method2_ce_repair_01b.ipynb` bản mới từ repo.
2. Add Input: **toàn bộ output notebook 01 validation đã chạy**. Output này phải có `completion_manifest.json`, `validation_completed.json` và CE final đầy đủ weights, heads, tokenizer. Đây là baseline CE được dùng để đối chiếu.
3. Upload `artifacts/method2_completion/ce_repair_v1.zip` thành một dataset và Add Input dataset đó. Giữ nguyên các đường dẫn trong zip. Dataset addon chỉ có `ce_repair_manifest.json`, không thay bundle gốc.
4. Chọn GPU T4, bật Internet để cài dependencies và tải `xlm-roberta-base` gốc. Chạy session mới, Run All rồi Save Version kèm toàn bộ output.

Không cần gắn output 02/03 vào notebook 01b. Giữ chúng trên Kaggle để dùng sau. Nếu chỉ có bundle cũ và checkpoint CE thô mà không có marker validation, dùng output 01 đã Save Version đầy đủ thay vì tự tạo marker. Notebook có xử lý duplicate marker từ archive reports được giải nén.

## Notebook thực hiện gì

1. Kiểm tra hashes của bundle, CE baseline và addon; addon không được ghi đè file trong bundle đã freeze.
2. Đánh giá lại checkpoint cũ trên component validation đã sửa và oracle validation gold gốc.
3. Train CE mới **từ XLM-R base**, giữ curriculum warmup benchmark 2 epoch + CustomTools 2 epoch, cùng hyperparameters cũ. Thí nghiệm đầu chỉ thay nhãn, không đồng thời tăng lên custom4.
4. Đánh giá checkpoint mới trên cùng validation. Chọn checkpoint có Argument EM cao hơn, tie-break bằng enum accuracy; nếu cả hai bằng nhau thì giữ baseline.
5. Copy checkpoint được chọn sang đường dẫn chuẩn `artifacts/method2/crossencoder/run01/final`, đo validation lại tại đường dẫn đó và ghi marker kèm hashes. Model mới chưa chắc được chọn hoặc đạt gate.

Run CE cũ mất khoảng 0,67 giờ train; đây chỉ là mốc tham khảo. Notebook mới còn có ba lượt validation và thêm training pairs nên cần dự trù thêm thời gian. Không dùng thời gian train Bi-Encoder 9–11 giờ làm ước lượng cho CE.

## Gửi lại gì sau khi chạy

- `results/method2/ce_repair/comparison.json`: baseline/candidate scores, checkpoint được chọn, gate status.
- `results/method2/completion/validation/validation_gate.json`: gate của checkpoint được chọn.
- `method2_ce_repair_reports.tar.gz`: gói báo cáo gồm raw predictions và chi tiết validation để điều tra nếu gate còn fail.
- Giữ toàn bộ output để còn weights, train report/config, repaired pairs và addon manifest. Archive reports không chứa weights.

Checkpoint ứng viên mới nằm ở `artifacts/method2/crossencoder/repair_v1/final`; checkpoint cũ được giữ ở `baseline_for_repair/final`. Không đánh đồng “run hoàn tất” với “gate đạt”. Notebook không dùng test để chọn model và không tự đổi gate.

## Bước sau

Đã review output `output_train/output_method2-ce-repair-01b` ngày 2026-09-07: notebook không có cell error; 114 bundle hashes, 7 addon hashes và 5 selected checkpoint hashes khớp. Selected checkpoint giống candidate `repair_v1/final`. Audit tại `artifacts/method2_completion/ce_repair_01b_verification.json`.

| Validation metric | Baseline | Candidate |
|---|---:|---:|
| Oracle Argument EM, 460 gold calls | 45,65% (210/460) | 61,96% (285/460) |
| Custom span EM trên cùng repaired pairs | 75,64% | 93,64% |
| Custom enum accuracy, component | 75,71% | 74,49% |
| Integer value accuracy, oracle | 65,37% | 100% |

Candidate thắng theo tiêu chí chọn đã định trước (Argument EM, rồi enum). Gate vẫn **false**: enum <90%, Argument EM <70%. Candidate cũng giảm span EM trên Glaive 96,73%→85,59% và xLAM 77,63%→71,90%; không kết luận tốt hơn trên mọi miền. Đây là component validation, chưa phải kết quả end-to-end test.

Khuyến nghị chốt checkpoint được chọn cho lượt đánh giá hiện tại và tiếp notebook 04 để hoàn thiện số đo, giữ nguyên trạng thái gate chưa đạt. Không cần chạy lại 01/01b/02/03. Đây không phải xác nhận Method 2 đã đạt mọi tiêu chí nghiệm thu; không hạ gate và không dùng kết quả test sắp chạy để tune lại model.

1. Import file local `notebooks/method2_completion_04_evaluation.ipynb` vào notebook Kaggle mới.
2. Add Input đúng hai output đầy đủ: **`output_method2-completion-03-round2` + `output_method2-ce-repair-01b`**. Có thể chọn output notebook đã lưu hoặc upload hai thư mục thành dataset; giữ nguyên cấu trúc thư mục.
3. Không gắn output 01 cũ, addon zip riêng hoặc bundle riêng. Hai output trên đã đủ; chỉ archive reports sẽ thiếu weights/index. Loader local đã được kiểm tra chọn đúng hai marker trên các output thực tế.
4. GPU T4, Internet bật, session mới → Run All → Save Version kèm toàn bộ output.
5. Gửi output 04 và notebook đã chạy; giữ `results/method2/completion/evaluation_completed.json` và `method2_completion_evaluation_reports.tar.gz`. `quality_gate_passed=false` là kết quả chất lượng đã biết, không phải lỗi cần chạy lại training.

Sau khi 04 hoàn tất, 05 stress dùng toàn bộ output 04. Notebook 06 normalizer ablation cũng dùng output 04, là thí nghiệm tùy quota. Stress hiện chỉ có random; pure same-domain còn thiếu capacity, phải ghi rõ trong báo cáo.

Bundle gốc và hai round Bi-Encoder không đổi. Addon có manifest riêng để ghi nguồn nhãn/config/code bổ sung; marker validation vẫn liên kết đúng bundle parent của notebook 03.

Nếu job bị ngắt, giữ output dở. Runner 01b không tự resume; không Run All đè vào thư mục có kết quả dở. Cần khôi phục đúng checkpoint của repair_v1 và giữ provenance trước khi tiếp tục.
