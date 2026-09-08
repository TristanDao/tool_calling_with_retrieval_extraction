# Đối chiếu output train với experimental_plan.md

Ngày kiểm tra: 2026-09-06. Phạm vi: ba notebook có output và toàn bộ thư mục `output_train`. Đối chiếu chính với `docs/experimental_plan.md`; dùng `docs/method2_plan.md` để giải thích cấu hình và quality gate riêng của Method 2. Không chạy lại training/inference, không sửa dữ liệu hay checkpoint.

## 1. Kết luận

**Đủ để báo cáo tiến độ và kết quả thực nghiệm bước đầu của Method 2. Chưa hoàn tất experimental plan, chưa đủ cho báo cáo so sánh cuối cùng của luận văn.**

Không thể dùng bộ này để kết luận dữ liệu VI tốt hơn EN, bilingual tốt hơn monolingual, CustomTools fine-tuning tạo ra mức cải thiện nào, hoặc Bi+Cross tốt hơn SLM/API về accuracy, latency và cost. Những kết luận đó cần các thí nghiệm đối chứng còn thiếu.

Kết quả thấp vẫn có giá trị báo cáo nếu mô tả đúng thiết lập và giới hạn. Không cần đạt một accuracy tùy ý mới được viết báo cáo; nhưng phải xử lý hoặc công khai khác biệt về protocol, metric và unseen exposure.

## 2. Mức đáp ứng kế hoạch

| Hạng mục trong experimental plan | Bằng chứng hiện có | Đánh giá |
|---|---|---|
| Dataset snapshot, SHA-256, split audit | Manifest, decontamination, preflight; kiểm tra trực tiếp 5 source files và 8 derived files trong output evaluation đều khớp hash | Có nền tảng; chưa chứng minh toàn bộ duplicate scenario và strict unseen exclusion |
| Parser, validator, evaluator chung | Có code, parsed predictions, report và parse coverage 100% cho Method 2 | Đã dùng cho Method 2; chưa có bằng chứng chạy contract trên SLM/API; thiếu bộ kết quả strict tách riêng |
| Method 1: E0/E1/E2/E4/E5 | Không tìm thấy output tương ứng | Chưa có bằng chứng thực nghiệm; E3 là optional theo điều kiện của plan |
| Bi-Encoder | Round 1 checkpoint; round 2 hard-negative training; validation candidate/pool; threshold calibration | Đã train và đánh giá |
| Cross-Encoder | Checkpoint warmup/finetune/final, train report, validation từng head | Đã train; enum gate chưa đạt |
| Oracle và full pipeline | Đủ hai chế độ trên benchmark, custom_seen, custom_unseen | Đã chạy, với retrieval scope = candidates |
| OpenAI và Gemini | Không tìm thấy raw response/prediction/metrics | Chưa có bằng chứng thực nghiệm |
| Latency/resource/cost | Latency tổng và từng stage, GPU memory, thời gian build index | Có latency Method 2; cost đang ghi cứng 0, throughput là ước lượng tuần tự |
| Stress: 200 × 6 × 2 | Có config, không thấy kết quả 2.400 instances | Chưa hoàn thành |
| Error analysis | Có errors JSONL, W/T/P/I tự động và lỗi tool | Có sơ bộ; cần rà soát nhãn, đơn vị đếm và multi-call |
| Bảng A–E và final comparison | Có summaries Method 2 và bảng trong notebook | Có vật liệu cho phần Method 2 của C/D; B và E chưa có, A cần cập nhật theo snapshot |

Không quy đổi thành một phần trăm hoàn thành vì plan không gán trọng số và các hạng mục có khối lượng rất khác nhau.

## 3. Kết quả có thể trích dẫn ngay, kèm giới hạn

Nguồn số liệu: `output_evaluation/results/evaluation/method_2_{benchmark,custom_seen,custom_unseen}/report.json`.

| Chỉ số | Benchmark | Custom seen | Custom unseen |
|---|---:|---:|---:|
| Tổng mẫu | 10.555 | 800 | 800 |
| Positive / negative | 10.555 / 0 | 400 / 400 | 400 / 400 |
| Tool-set accuracy trên positive | 80,28% | 85,50% | 76,25% |
| Argument Pair F1 | 61,51% | 85,13% | 58,60% |
| N-FCEM positive | 39,84% | 53,50% | 7,50% |
| Số positive đúng hoàn toàn | 4.205 | 214 | 30 |
| Oracle argument exact match trên positive | 48,58% | 64,25% | 19,25% |
| Overall success, gồm negative | 39,84% | 64,00% | 40,13% |
| Negative recall | Không có mẫu | 74,50% | 72,75% |
| Parse validity | 100% | 100% | 100% |
| Schema validity trên predicted calls | 82,20% | 80,65% | 67,99% |
| P50 latency, ms | 56,304 | 58,661 | 58,413 |
| P95 latency, ms | 79,595 | 89,312 | 93,280 |

N-FCEM positive là exact match toàn bộ lời gọi sau normalization, tính trên positive samples. Có thể mô tả là normalized ArgA theo công thức trong plan; chưa có bảng strict ArgA riêng. Tool-set accuracy ở đây cũng chỉ tính positive, không phải accuracy toàn bộ tập có thêm class no-call.

Overall success 40,13% của unseen gồm 291 negative đúng và 30 positive đúng; không được diễn giải thành 40,13% lời gọi unseen chính xác. Multi-call unseen đạt 0/60 positive đúng hoàn toàn.

Bootstrap CI 95% hiện có cho N-FCEM positive: benchmark 38,96–40,82%; seen 48,50–58,01%; unseen 5,00–10,25%. Đây là bootstrap trên mẫu của một run, không phải độ biến thiên qua nhiều training seeds.

So oracle và pipeline trên cùng mẫu số positive: mức chênh tương ứng khoảng 8,74; 10,75; 11,75 điểm phần trăm. Extraction vẫn là hạn chế lớn, đặc biệt trên unseen. Không dùng phép trừ trực tiếp `oracle ArgEM` và `ArgEM given correct tool` trong notebook để quy lỗi retrieval: hai đại lượng đó dùng hai tập/mẫu số khác nhau.

## 4. Những điểm cần giải quyết trước báo cáo cuối

### 4.1 Unseen protocol chưa khớp experimental plan

`experimental_plan.md` §3.2 yêu cầu tool unseen không xuất hiện trong tool schema của training. Tuy nhiên `method2_plan.md` dùng điều kiện yếu hơn: không xuất hiện làm positive.

Kiểm tra trực tiếp 10 gold tools của `test_unseen`:

- Bi-Encoder train positives: không có tool nào trong 10 tool này.
- Cross-Encoder train tool names: không có tool nào trong 10 tool này.
- Bi-Encoder train hard negatives ban đầu: có đủ 10 tool, tổng 604 lượt.
- Bi-Encoder `train_mined.jsonl` dùng ở round 2: có đủ 10 tool, tổng 477 lượt.

Vì Bi-Encoder encode negative descriptions trong training, các tool này đã có exposure. Đây không phải bằng chứng test query/gold answer bị đưa vào train, nhưng không đáp ứng cách hiểu “chưa hề thấy tool schema”.

Hai hướng hợp lệ: (a) báo cáo rõ thiết lập hiện tại là chưa có positive supervision cho unseen tools nhưng có negative exposure; hoặc (b) giữ protocol strict của experimental plan, loại test-unseen tools khỏi mọi negative/mining/in-batch training exposure và chạy lại training liên quan. Chỉ đưa các tool đó vào index ở inference. Không âm thầm đổi tên thiết lập thành strict zero-shot.

### 4.2 Retrieval scope và kích thước candidate

Pipeline thực tế dùng `retrieval_scope: candidates`. Benchmark có 1–8 candidate tools, trong đó 6.053/10.555 mẫu chỉ có 1 candidate; CustomTools có 10 candidate/mẫu. Recall@5 gần 100% không chứng minh retrieval trên toàn pool hoặc khả năng mở rộng.

Validation full pool riêng đã có: micro Recall@5 = 86,44%, full Recall@5 = 84,46%, MRR = 0,833. Phải giữ nhãn validation và pool-scope khi trích dẫn, không trộn với test candidate-scope.

### 4.3 Cross-Encoder và quality gate

Model thực chạy: Bi-Encoder BGE-M3 + LoRA + CachedMultipleNegativesRankingLoss; Cross-Encoder `xlm-roberta-base`, không phải BGE-M3 như mô tả cũ trong AGENTS/experimental plan. Backbone này đã được giải thích ở `method2_plan.md`; cần đồng bộ tài liệu báo cáo.

Bi round 2 hoàn tất 3 epoch/921 steps, khoảng 10,647 giờ, peak VRAM 10.888,9 MB. CE hoàn tất curriculum warmup + finetune, 4.446 steps, `stopped_early=false`, khoảng 0,667 giờ, peak VRAM 7.703,3 MB. Model được xuất là final; best Bi validation checkpoint được ghi ở step 900 nhưng không tự nạp lại. Cần ghi chính xác checkpoint đã dùng, không gọi final là best.

CE validation: has_value F1 97,37%, span EM 80,95%, enum accuracy 74,54%, boolean accuracy 92,90%. Enum trên CustomTools val 75,71%, thấp hơn gate 90%. Gate per-call Argument EM ≥70% trong method2 plan chưa được chứng minh bằng báo cáo validation này. Không lấy oracle test thay cho validation gate.

Cần kiểm tra lỗi enum, numeric, missing required và boolean: validation CustomTools boolean đạt 100%, nhưng test unseen oracle chỉ đúng 7/53 gold boolean values. Đây là dấu hiệu cần điều tra phân phối nhãn/đường inference, chưa đủ để khẳng định nguyên nhân là một bug cụ thể.

### 4.4 Metric, cost và phân tích lỗi

- `cost_usd=0.0` được ghi cứng trong pipeline. Chỉ phản ánh không gọi API trả phí; chưa đo compute cost. Không dùng để kết luận chi phí hệ thống bằng 0 hoặc rẻ hơn baseline.
- Latency đã có trên Tesla T4; cần chốt warmup, cold/warm, batch/concurrency và cùng cách tính percentile trước so sánh. `latency_pipeline.json` và evaluator dùng hai cách tính percentile nên có sai khác nhỏ; bảng trên thống nhất dùng evaluator `report.json`.
- Throughput hiện là `1000 / mean_latency`, tức estimated sequential throughput, chưa phải benchmark tải đồng thời.
- Error labels W/T/P/I là heuristic. “Gold text không có trong query” được gán T, “thiếu argument” được gán I; không tự động đồng nghĩa lỗi dịch hoặc thiếu ngữ cảnh như phân loại học thuật.
- `classify_errors` dùng dictionary theo tool name nên có thể gộp nhiều call cùng tool; cần matching theo từng call cho error analysis multi-call. Error log cũng không có cùng đơn vị đếm với sample-level error rate.
- Có sweep threshold trên test trong notebook để chẩn đoán. Không thấy bằng chứng sweep này đã thay đổi predictions hiện có, nhưng không dùng test sweep để chọn lại threshold rồi gọi kết quả là held-out test. Strategy gap hiện dùng được lấy từ validation winner.

## 5. Output hiện có để làm báo cáo

Các đường dẫn dưới đây nằm trong `output_train`:

| Output | Vị trí | Cách dùng |
|---|---|---|
| Metrics test tổng hợp, CI, slices | `output_evaluation/results/evaluation/method_2_*/report.json` | Bảng Method 2, seen/unseen, single/multi-call, nguồn dữ liệu |
| Tóm tắt đọc nhanh | Cùng thư mục, `summary.md` | Tra cứu, không thay thế định nghĩa metric |
| Kết quả từng mẫu | Cùng thư mục, `per_sample.jsonl` | Chọn case, kiểm tra lỗi, phân tích paired results |
| Parsed pipeline/oracle | `output_evaluation/results/method2/predictions/{benchmark,custom_seen,custom_unseen}/` | `predictions.jsonl`, `oracle_predictions.jsonl` |
| Raw pipeline/oracle và lỗi | Cùng thư mục predictions | `raw_predictions_*.jsonl`, `errors_*.jsonl` |
| Latency theo stage và GPU memory | Cùng thư mục predictions | `latency_pipeline.json`, `latency_oracle.json` |
| Retrieval validation candidate/pool | `output_biencoder/results/method2/metrics/` | Recall@1/3/5/10, MRR, NDCG |
| CE validation theo head/source | `output_crossencoder/results/method2/metrics/crossencoder_val.json` | Has-value, span, enum, boolean và gate |
| Training history/config/commit | `output_biencoder/artifacts/method2/biencoder/run02/`, `output_crossencoder/artifacts/method2/crossencoder/run01/` | `train_report.json`, `run_manifest.json`, checkpoints |
| Data snapshot và provenance | Các thư mục `data/method2/`, `results/method2/preflight.json` | SHA-256, pairs, labels, decontamination |
| Nhật ký chạy có output | Ba notebook `method2-kaggle-*.ipynb` | Evidence train/evaluation và bảng console |

Đã kiểm tra số dòng và IDs: trên cả ba test sets, mỗi file parsed/raw của cả pipeline và oracle có đủ đúng IDs gold, không thiếu/thừa/trùng ID. Tổng 12.155 mẫu mỗi chế độ. Báo cáo hash khớp 13 file hiện có; output evaluation không kèm 4 source files training/validation gốc được manifest tham chiếu, nên cần giữ bản snapshot nguồn để dựng lại dữ liệu từ đầu.

Bộ trình bày nên xuất tiếp: bảng thống kê dataset; bảng training/config; bảng retrieval candidate/pool; bảng CE từng head; bảng oracle/full pipeline; bảng seen/unseen; bảng latency từng stage; đường learning curve; biểu đồ N-FCEM theo test set với CI; bảng case lỗi đã review. Bảng so sánh bốn methods và biểu đồ stress chỉ hoàn thiện sau khi có các run còn thiếu.

## 6. Thứ tự làm tiếp

1. Chốt mô tả unseen exposure và backbone thực chạy; lưu bản kết quả hiện tại như một run có provenance rõ ràng. Đồng bộ experimental plan, architecture và memory khi quyết định protocol được chốt.
2. Kiểm tra/sửa các lỗi triển khai hoặc báo cáo có thể xác định độc lập: strict/normalized metrics, error matching multi-call, cách so oracle/pipeline, cost/latency labels. Tính lại metric từ predictions có sẵn khi có thể; chưa cần train lại chỉ để dựng bảng.
3. Dùng validation để điều tra CE enum/boolean/numeric/required fields và gate chưa đạt. Nếu giữ strict unseen protocol, cần train lại Bi-Encoder với negatives được lọc và kiểm tra toàn bộ đường mining.
4. Hoàn tất SLM 1.5B E0/E1/E2/E4/E5; E3 theo điều kiện có general instruction dataset. Sau đó chạy các experiment chính trên 0.5B theo plan.
5. Chạy OpenAI/Gemini trên cùng test/candidates, lưu response, version, latency, tokens và cost; chuẩn hóa điều kiện so sánh.
6. Chạy stress 2.400 instances cho bốn methods, hoàn thiện latency/resource/cost và error analysis. Chạy thêm seed nếu tài nguyên cho phép; plan khuyến nghị ít nhất 3 seed cho kết quả chính, không phải điều kiện cứng hiện đã chốt.
7. Xuất Bảng A–E, biểu đồ và kết luận bốn câu hỏi nghiên cứu. Chỉ khi đó mới gọi là hoàn tất toàn bộ experimental plan.

Nếu cần báo cáo với giảng viên ngay, tiêu đề phù hợp là “Kết quả bước đầu Method 2 và phân tích hạn chế tổng quát hóa trên CustomTools-VI”, kèm bảng kết quả ở mục 3 và các phần việc còn thiếu. Không cần chờ mọi baseline để báo cáo tiến độ.
