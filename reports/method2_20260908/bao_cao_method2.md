# Báo cáo tổng hợp thực nghiệm Method 2

Ngày tổng hợp: 08/09/2026

## Báo cáo tổng hợp thực nghiệm Method 2

Tool calling tiếng Việt bằng Bi-Encoder và Cross-Encoder

Ngày tổng hợp 08/09/2026 • Bộ kết quả output_train • Phạm vi nghiên cứu Method 2

### Kết luận chính

Chuỗi thực nghiệm Method 2 đã hoàn thành hai round Bi-Encoder theo strict unseen, sửa và train lại Cross-Encoder, đánh giá pipeline và oracle, random stress test và ablation normalizer. Notebook 06 chạy thành công, đủ prediction cho 12.155 query và giữ đúng checkpoint, index, threshold của notebook 04.

Có thể sử dụng bộ kết quả để viết phần thực nghiệm Method 2. Tuy nhiên, chưa đủ căn cứ tuyên bố đáp ứng toàn bộ plan: CE còn trượt gate enum và Argument EM, pure same-domain stress chưa chạy, và review lỗi thủ công chưa hoàn tất. Chưa có kết quả Method 1/API để kết luận Method 2 thắng về accuracy, latency hoặc chi phí.

| Kết quả chính | Giá trị |
| --- | --- |
| ArgA normalized trên test | 40,21% benchmark; 67,75% seen; 22,75% unseen |
| Lợi ích normalizer trên ArgA normalized | +19,22 / +34,75 / +15,75 điểm phần trăm |
| Random stress từ 3 đến 1.000 tool | ArgA 81% → 54%; latency P50 58,16 → 91,88 ms |
| CE validation gate | Argument EM 61,96% <70%; enum 74,49% <90% |
| Strict training | 20 tool heldout bị loại; zero identifier exposure trong pairs gốc/mined đã kiểm tra |

### Cách đọc báo cáo

Phần 1–5 mô tả dữ liệu, cấu hình và lịch sử train. Phần 6–11 trình bày định nghĩa metric, kết quả test, ablation và stress. Phần 12–16 phân tích lỗi, kiểm định tái lập, đối chiếu plan và chỉ dẫn dùng bộ báo cáo. Các số hiệu [Sxx] trỏ tới file nguồn ở phần cuối.

ArgA trong báo cáo dùng mẫu số positive query; CE gate Argument EM dùng mẫu số gold call. Hai metric này không thể thay thế trực tiếp cho nhau. Số phần trăm là kết quả đo, còn nhận định nguyên nhân được phân biệt với giả thuyết cần kiểm chứng.

## 1 Danh mục các run và nguồn kết quả

output_train có 10 thư mục output và 10 notebook đã lưu. Các thư mục giai đoạn sau chứa bản sao của bundle, checkpoint và báo cáo giai đoạn trước; không cộng chúng thành các lần train độc lập. Inventory đầy đủ, số file, dung lượng và hash notebook được lưu trong evidence.json.

| Output | Vai trò trong báo cáo |
| --- | --- |
| output_biencoder | Legacy Bi-Encoder run02; chỉ dùng lịch sử |
| output_crossencoder | CE run01 cũ; baseline của CE repair |
| output_evaluation | Legacy test; dùng bản tái chấm, gắn nhãn exposure |
| completion 01 validation | Kiểm định CE cũ và phát hiện gate chưa đạt |
| completion 02 round1 | Strict BI round1, teacher để mining |
| completion 03 round2 | Strict BI round2, index và calibration trên val |
| ce repair 01b | Sửa supervision; train và chọn CE bằng val |
| completion 04 evaluation | Kết quả chính khi bật normalizer |
| completion 05 stress | 200 anchors × 6 N, random distractors |
| completion 06 normalizer ablation | Đánh giá khi tắt normalizer; đối chiếu với 04 |

### Phân biệt legacy và strict

Run legacy có 905 negative exposures của 20 tool heldout trong snapshot pairs trước khi sửa. Dù test query không được dùng trực tiếp để train, run đó không thỏa protocol strict unseen đã chọn. Vì vậy, bảng legacy chỉ mô tả tiến trình; không được trình bày như bằng chứng zero-shot strict. [S01, S02]

Chuỗi được dùng cho kết luận hiện tại là strict round1 → mining → strict round2, kết hợp CE repair_v1 được chọn bằng validation → notebook04 → notebook05/06. CE được copy sang tên đường dẫn run01/final để tương thích loader, nhưng hash weights xác nhận đó là candidate mới.

## 2 Dữ liệu và phạm vi đánh giá

| Tập test | Query | Positive / negative | Gold calls | Multi-call |
| --- | --- | --- | --- | --- |
| Benchmark | 10555 | 10555 / 0 | 14539 | 3174 |
| Custom seen | 800 | 400 / 400 | 460 | 60 |
| Custom unseen | 800 | 400 / 400 | 460 | 60 |

Benchmark gồm 4.447 Glaive và 6.108 xLAM query. Có 1.274 query gọi lặp cùng một tool; đây là nhóm khó đối với pipeline hiện tại vì retrieval/chọn tool hoạt động theo tên tool. Không loại nhóm này khỏi mẫu số test. Hai tập CustomTools đều có 60 multi-call positive và không có repeated-tool query. [S06, S10]

Validation CustomTools gồm val_seen 400 và val_unseen 400 query, mỗi tập 200 positive/200 negative. Tổng 460 gold calls positive được dùng cho CE gate. Bi-Encoder validation retrieval có 7.521 positive query và 9.846 gold tools; đây là tập khác với 12.155 query test.

### Training representation

Strict BI pairs có 94.634 rows; 78.435 training examples thực sự được đưa vào loss. Không diễn giải 94.634 là số positive MNRL examples. CE repair giữ 3.600 CustomTools training sample IDs đã có trong protocol, mở rộng supervision từ 12.253 lên 15.318 custom pairs. Sau collator, train report ghi 145.383 CE pairs và 3 dòng unalignable bị loại.

### Tool pool và strict unseen

Pool canonical có 4.464 tool. 20 tool validation/test-unseen được loại khỏi training positives, negatives và mining. Scan lại 94.634 pairs train và 94.634 pairs mined không thấy identifier của 20 tool này. Danh sách và audit trước/sau nằm ở excluded_tools.json. Điều này kiểm tra các representation đã train, không chứng minh backbone chưa từng gặp công cụ tương tự trong pretraining.

Tập train master đầy đủ không được copy hết vào mọi output evaluation. Vì thế báo cáo dùng counts của representation và test/val thực sự có trong artifact; không dùng số 105k raw ban đầu làm số mẫu optimizer đã thấy.

## 3 Kiến trúc và cấu hình đã chạy

Query được mã hóa bởi Bi-Encoder BGE-M3, so điểm với tool embeddings đã tiền tính, rồi chọn tool và quyết định no-call bằng threshold. Với từng tool được chọn, CE nhận query và schema của từng parameter; has_value xác định có giá trị, schema route sang span, enum hoặc boolean. Normalizer chuyển surface sang dạng giá trị phù hợp, sau đó validator kiểm tra cấu trúc/schema.

| Thành phần | Cấu hình thực tế |
| --- | --- |
| Bi-Encoder | BAAI/bge-m3; sentence-transformers CachedMNRL; LoRA r=16, alpha=32, dropout=0,05 |
| BI input và optimizer | Max length 192; batch 256; cached mini-batch 32; LR 2e-5; cosine; warmup 0,1; FP16; seed 42 |
| BI rounds | Mỗi round 3 epoch, từ base; round1 là teacher mining, không phải weights khởi tạo round2 |
| Cross-Encoder | xlm-roberta-base; max length 256; dynamic padding; max enum size 20; has_value + span/enum/boolean |
| CE optimizer | Batch 32 × grad accumulation 2; head LR 1e-4; FP16; seed 42; dropout 0,1 |
| CE curriculum | Warmup Glaive/xLAM 2 epoch, LR 3e-5; CustomTools 2 epoch, LR 1e-5 |
| CE inference | has_value 0,5; fallback 0,3; max answer span 30; batch 64; schema question tối đa 96 token |
| No-call và selection | abstention tau=0,35; gap delta=0,21; k_max=3; CE should_call head tắt |

Thiết kế đầu dự kiến BGE-M3 cho cả hai encoder và FlagEmbedding. Implementation được train thực tế dùng XLM-R base cho CE, sentence-transformers + LoRA cho BI. Mô tả luận văn cần cập nhật theo implementation này; không gọi checkpoint CE là BGE-M3. [S01, S03, S04]

Config CE có trường train.epochs=3 nhưng curriculum đang bật nên số epoch thực tế là 2+2 theo từng stage. Final checkpoint là checkpoint cuối; các best_step trong train report chỉ là thống kê và không đồng nghĩa đã reload best.

## 4 Huấn luyện Bi Encoder và calibration

| Run | Examples | Steps | Giờ train | Peak MB |
| --- | --- | --- | --- | --- |
| Legacy run02 | 78435 | 921 | 10,647 | 10.888,9 |
| Strict round1 | 78435 | 921 | 9,033 | 10.888,9 |
| Strict round2 | 78435 | 921 | 11,310 | 10.888,9 |

Strict round1 và round2 đều hoàn thành 3 epoch. Round2 dùng hard negatives được mining từ teacher strict round1 với top_k=20, skip_top=1 và 4 negatives. Scan training xác nhận không có heldout identifier. Thời gian trên là train report, không bao gồm toàn bộ cài đặt, mining, index, đánh giá và tải output. [S03]

| Validation retrieval round2 | Candidate pool | Full pool |
| --- | --- | --- |
| micro_recall@1 | 75,96% | 57,17% |
| full_recall@1 | 72,46% | 53,21% |
| micro_recall@3 | 99,38% | 79,88% |
| full_recall@3 | 99,20% | 77,48% |
| micro_recall@5 | 99,99% | 85,99% |
| full_recall@5 | 99,99% | 83,96% |
| micro_recall@10 | 100,00% | 91,73% |
| full_recall@10 | 100,00% | 90,19% |
| mrr | 99,66% | 83,12% |

Micro recall chia theo tổng gold tools; full recall yêu cầu đủ mọi gold tool trong top-k cho một positive query. MRR đo vị trí gold đầu tiên. Không đặt các chỉ số này ngang với accuracy@k trong trainer pair-level mà không nói rõ đơn vị đánh giá. Round1 trainer ghi R@1=55,35%, R@5=85,13%, MRR@10=0,6804; round2 trainer ghi 57,15%, 86,00%, 0,6961.

Calibration chỉ dùng val: chọn gap thay absolute threshold vì selection F1=0,9552 so với 0,9482; tool-set accuracy=0,8972. Abstention macro F1=0,9082, negative recall=0,8636. Không recalibrate bằng test/stress sau khi biết kết quả. [S05]

## 5 Cross Encoder và thí nghiệm sửa supervision

Label generator cũ dò str(gold_value) trong query, nên bỏ sót số có format/đơn vị khác và có thể chọn nhầm occurrence của chuỗi ngắn. Repair dùng argument_mentions, kiểm tra call_index, parameter_path, offset và canonical value để supervise đúng surface. Annotation chỉ dùng xây nhãn training/validation, không đi vào inference. [S04]

| Custom pairs | Trước | Sau | Sửa span |
| --- | --- | --- | --- |
| Train | 12253 | 15318 | 693 |
| Validation | 1434 | 1684 | 109 |

Giữ nguyên benchmark rows và sample IDs của protocol. 17.002 custom pairs mới đã qua tokenizer check không rơi ngoài query window. Có 642 train và 44 val surface cần alias/canonical mapping mà normalizer chưa hỗ trợ; model học surface nhưng gold EM vẫn giữ canonical gốc.

| CE run | Train pairs sau collator | Steps thực tế | Giờ |
| --- | --- | --- | --- |
| Baseline run01 | 142318 | 4446 | 0,667 |
| Repair v1 | 145383 | 4542 | 0,685 |

Repair chạy fresh XLM-R base, cùng curriculum 2+2. Report ghi planned 4.546 steps, observed 4.542, stopped_early=false. Code tính budget bằng ceil nhưng chỉ optimizer.step khi đủ 2 micro-batches, không flush nhóm lẻ cuối epoch. Đây là hạn chế gradient accumulation cần sửa cho run tương lai; không phải job bị ngắt. Chưa định lượng tác động lên chất lượng. Peak memory 7.703,3 MB.

| Validation trên cùng repaired pairs | Baseline | Candidate | Gate |
| --- | --- | --- | --- |
| Custom has_value F1 | 96,31% | 97,18% | ≥90% |
| Custom span EM | 75,64% | 93,64% | ≥80% |
| Custom enum accuracy | 75,71% | 74,49% | ≥90% |
| Custom boolean accuracy | 100% | 100% | ≥85% |
| Oracle Argument EM | 45,65% | 61,96% | ≥70% |

Candidate thắng theo quy tắc định trước: Argument EM, rồi enum accuracy; baseline được giữ nếu hòa. Đúng call tăng 210/460 lên 285/460. Gate vẫn false. Custom integer value accuracy tăng 65,37%→100%; number 79,66%→91,53%. Glaive span EM giảm 96,73%→85,59%, xLAM 77,63%→71,90%; không kết luận cải thiện đồng đều trên mọi miền.

## 6 Protocol đánh giá và ý nghĩa metric

| Chỉ số | Định nghĩa dùng trong báo cáo |
| --- | --- |
| Strict ArgA | Positive query đúng toàn bộ tên tool, arguments, số call theo so khớp nghiêm ngặt; không phạt thứ tự các call độc lập |
| Normalized ArgA | Cùng predictions; scorer chuẩn hóa Unicode/whitespace/case, số/ngày/boolean theo config; không có synonym/LLM judge |
| Oracle ArgA | CE nhận gold tools; vẫn dùng cùng positive query denominator; không phải kết quả pipeline triển khai |
| Oracle gap | Oracle normalized ArgA trừ pipeline normalized ArgA trên cùng mẫu số; đo ảnh hưởng tổng hợp selection/no-call/multi-call |
| Argument F1 | Micro F1 của cặp key-value sau alignment; không yêu cầu cả query đúng nên có thể cao khi ArgA thấp |
| CE gate Argument EM | Tỷ lệ gold calls có arguments đúng trong oracle Custom val; 460 calls, khác mẫu số ArgA query |
| Negative recall | Số no-call query được dự đoán không gọi chia tổng negative queries |
| Schema validity | Tỷ lệ predicted calls hợp schema; khác JSON parse validity |

Evaluation chính chạy candidate-scope trên danh sách tool được cấp cho từng query. Full-pool retrieval chỉ có số đo validation riêng. Vì vậy, không gọi latency notebook04 là latency tìm kiếm toàn bộ 4.464 tool trên mọi test query. Notebook05 mới kiểm soát N trong candidate haystack. [S05–S08]

Normalizer ở inference và normalization ở scorer là hai lớp khác nhau. Notebook06 tắt lớp inference; scorer vẫn xuất cả strict và normalized. So sánh normalized 04 với normalized 06 đo ảnh hưởng inference normalizer; khoảng cách strict-normalized trong riêng một notebook không phải ablation.

JSON được dựng có cấu trúc nên parse validity cao không đủ để kết luận call hợp lệ hoặc đúng nghĩa. Các mảng/object không có extraction head đầy đủ và repeated-tool calls vẫn nằm trong gold; không loại bỏ test khó để tăng ArgA.

## 7 Kết quả chính trên ba tập test

| Tập | Strict ArgA | Norm ArgA | Oracle | Gap pp | Arg F1 |
| --- | --- | --- | --- | --- | --- |
| Benchmark | 39,76% | 40,21% | 42,97% | 2,77 | 61,94% |
| Custom seen | 67,75% | 67,75% | 82,00% | 14,25 | 91,28% |
| Custom unseen | 22,75% | 22,75% | 36,50% | 13,75 | 67,67% |

![ArgA normalized trên positive query (%)](figures/pipeline_oracle.png)

| Tập | Positive đúng / tổng | Tool set accuracy | CI 95% ArgA |
| --- | --- | --- | --- |
| Benchmark | 4244/10555 | 80,47% | 39,30% – 41,19% |
| Custom seen | 271/400 | 85,25% | 63,50% – 72,25% |
| Custom unseen | 91/400 | 74,50% | 18,50% – 26,76% |

Các CI là bootstrap trên query, 1.000 resamples, seed 42; chúng phản ánh biến thiên mẫu test, không phải độ ổn định qua nhiều training seeds. Không có multi-seed training để kết luận phương sai do khởi tạo. [S06]

Custom seen tốt hơn unseen 45,00 điểm phần trăm ArgA. Ngay cả oracle unseen chỉ đạt 36,50%, cho thấy extraction vẫn là nút thắt lớn; chỉ cải thiện retrieval không đủ giải quyết. Oracle gap là chênh lệch thực nghiệm, không phải phép phân rã nhân quả độc lập cho từng component.

## 8 Detection và các nhóm test khó

| Tập | TP / FN | TN / FP | Negative recall | Detection F1 |
| --- | --- | --- | --- | --- |
| Benchmark | 10318 / 237 | 0 / 0 | N/A | 98,86% |
| Custom seen | 400 / 0 | 300 / 100 | 75,00% | 88,89% |
| Custom unseen | 399 / 1 | 287 / 113 | 71,75% | 87,50% |

Custom seen có 100/400 negative query phát sinh call; unseen có 113/400. Đây là số query, khác số error events hallucinated_call trong log (195 và 232) vì một query có thể phát sinh nhiều call thừa. Benchmark không có negative query, nên không suy ra năng lực abstention từ riêng benchmark.

| Output validity | Benchmark | Seen | Unseen |
| --- | --- | --- | --- |
| JSON parse validity ON | 100% | 100% | 100% |
| Call schema validity ON | 81,54% | 66,67% | 68,75% |
| Call schema validity OFF | 53,46% | 50,70% | 45,70% |

| Tập | Số call | Positive | Norm ArgA | Tool set acc |
| --- | --- | --- | --- | --- |
| Benchmark | single | 7381 | 51,55% | 96,26% |
| Benchmark | multi | 3174 | 13,83% | 43,76% |
| Custom seen | single | 340 | 69,71% | 85,29% |
| Custom seen | multi | 60 | 56,67% | 85,00% |
| Custom unseen | single | 340 | 26,47% | 80,59% |
| Custom unseen | multi | 60 | 1,67% | 40,00% |

Benchmark single-call đạt 51,55% ArgA nhưng multi-call chỉ 13,83%. Source slice Glaive đạt 57,99%, xLAM 27,26%; cần trình bày cùng cấu trúc dữ liệu, không quy toàn bộ chênh lệch cho ngôn ngữ. Có 1.274 query benchmark gọi lặp cùng tool, trong khi pipeline chọn tool theo tên và trích xuất một bộ arguments cho mỗi tool.

Theo per-type evaluator, Custom unseen boolean value accuracy trên gold chỉ 15,09% (8/53), integer 96,99% (354/365); number chỉ có 3 trường hợp nên không suy rộng tỷ lệ 33,33%. Enum được scorer gộp vào underlying type string, còn CE component gate có enum head riêng: không dùng accuracy string của test làm enum accuracy. [S06]

## 9 Ablation normalizer và kiểm tra notebook 06

Notebook06 đã hoàn thành: 114 file bundle và 29 file checkpoint/index/threshold khớp hash; checkpoint hash dictionary giống notebook04. Đủ pipeline, oracle và raw predictions trên 10.555 + 800 + 800 query, không thiếu/trùng IDs. Không có cell error trong notebook đã lưu. [S07]

| Tập | Norm ArgA ON | Norm ArgA OFF | Lợi ích ON pp |
| --- | --- | --- | --- |
| Benchmark | 40,21% | 20,99% | 19,22 |
| Custom seen | 67,75% | 33,00% | 34,75 |
| Custom unseen | 22,75% | 7,00% | 15,75 |

![Bật và tắt inference normalizer (%)](figures/normalizer_ablation.png)

| Tập | Strict OFF | Oracle OFF | Arg F1 OFF | Calls thay đổi |
| --- | --- | --- | --- | --- |
| Benchmark | 20,65% | 22,84% | 38,40% | 4510 |
| Custom seen | 11,25% | 45,00% | 79,35% | 271 |
| Custom unseen | 7,00% | 20,25% | 46,75% | 216 |

Đối chiếu từng query cho thấy 0 thay đổi ranked_tools và 0 thay đổi chuỗi tên tool trên cả ba tập. Có 4.510/271/216 query thay đổi nội dung calls; vì selection giữ nguyên, chênh lệch accuracy này gắn với xử lý argument sau extraction trong cấu hình đã kiểm định.

Ở Custom seen, OFF strict=11,25% nhưng OFF normalized=33,00%: scorer vẫn cứu một phần khác biệt biểu diễn. Dù vậy, ON normalized=67,75% vẫn hơn OFF normalized 34,75 điểm; lợi ích không chỉ là đổi cách chấm. Giữ normalizer ON cho kết quả chính đã chốt, không chọn lại checkpoint bằng test.

## 10 Latency throughput và tài nguyên

| Tập | P50 ON ms | P95 ON ms | P50 OFF ms | QPS ON ước tính |
| --- | --- | --- | --- | --- |
| Benchmark | 54,68 | 77,11 | 68,60 | 17,36 |
| Custom seen | 57,71 | 87,55 | 72,38 | 17,06 |
| Custom unseen | 57,54 | 92,80 | 71,54 | 16,91 |

Notebook06 tắt normalizer nhưng latency cao hơn notebook04. Hai run diễn ra ở session khác nhau, nên không diễn giải chênh lệch này thành normalizer làm mô hình nhanh hơn. Cần benchmark xen kẽ ON/OFF cùng session, cùng warmup và nhiều lặp nếu muốn ước lượng overhead normalizer. Không cần chạy lại training để hoàn thiện phép đo này.

Các QPS là 1.000 chia mean latency ms trong chế độ tuần tự, không phải throughput server đo dưới tải đồng thời. Thời gian pipeline gồm query embedding, retrieval, cross encoding và validation; không bao gồm cài dependencies, download weights hoặc toàn bộ chi phí tiền tính tool embeddings. [S06, S07]

| Khoản tài nguyên | Số đo có thể báo cáo |
| --- | --- |
| Thiết bị | Tesla T4; torch 2.10.0+cu128; BI/CE FP16 |
| Strict BI round1 + round2 | 20,343 giờ train; không tính mining/index/evaluation |
| CE repair mới | 0,685 giờ train; tổng 3 job train strict BI + CE repair =21,028 giờ |
| Peak training memory | BI 10.888,9 MB; CE 7.703,3 MB |
| Peak stress memory | Khoảng 3.279,5–3.351,1 MB; peak CUDA có thể tích lũy qua các N trong cùng run |
| USD trên 1.000 query | Chưa có đơn giá GPU và phạm vi hạch toán; để unavailable, không ghi 0 USD |

Không cộng các train_report được copy từ round1 vào output03/04/05/06 nhiều lần. Legacy BI run02 10,647 giờ và CE cũ 0,667 giờ thuộc lịch sử chi phí; chưa có đầy đủ log thời gian cho mọi job cũ để tính tổng ngân sách GPU cả dự án.

## 11 Random stress test khi số tool tăng

Notebook05 dùng cùng 200 test_seen anchors (100 positive, 100 negative), seed 42, pool 4.464 tool, N=[3,10,50,100,500,1000]. Có đủ 1.200 predictions và raw records; không cắt gold. Checkpoint/index/threshold giống notebook04. [S08]

| N | ArgA | Tool acc | Neg recall | P50 ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| 3 | 81,00% | 100,00% | 100,00% | 58,16 | 84,50 |
| 10 | 79,00% | 98,00% | 100,00% | 57,15 | 87,38 |
| 50 | 78,00% | 96,00% | 96,00% | 55,82 | 87,68 |
| 100 | 77,00% | 95,00% | 95,00% | 58,55 | 93,83 |
| 500 | 62,00% | 78,00% | 76,00% | 87,44 | 144,97 |
| 1000 | 54,00% | 68,00% | 57,00% | 91,88 | 160,15 |

![Stress accuracy trên random distractors (%)](figures/stress_accuracy.png)

Từ N3 đến N1000, ArgA giảm 27 điểm phần trăm, P50 tăng khoảng58%, P95 tăng khoảng90%. Vì vậy kết quả không chứng minh đường latency phẳng trên toàn dải N. Retrieval P50 tăng 0,079→2,829 ms; query embedding và CE cũng tăng trong lượt đo, nên chưa tách được ảnh hưởng tải/session khỏi ảnh hưởng thuật toán.

Ở N1000, 100/100 positive query vẫn có đủ gold tools trong top3; 32 query chọn thêm tool thừa, 43/100 negative query phát sinh call. Đây là dấu hiệu selection/abstention kém ổn định khi haystack lớn, dù top-k retrieval vẫn giữ tool đúng. Không dùng kết quả test này để tune threshold.

Chỉ random stress đã hoàn tất. Pure same-domain N1000 chưa khả thi: 40 CustomTools có nhãn ở 10 nhóm ×4 tool; 4.424 tool còn lại là Khác. Không gộp Khác thành một domain hoặc gọi mixed distractors là pure same-domain.

## 12 Phân tích lỗi và giới hạn coverage

| Error events heuristic | Benchmark | Seen | Unseen |
| --- | --- | --- | --- |
| wrong_tool | 2355 | 59 | 101 |
| missed_call | 341 | 0 | 1 |
| hallucinated_call | 0 | 195 | 232 |
| extra_argument | 395 | 0 | 43 |
| W | 1052 | 10 | 193 |
| T | 2444 | 64 | 19 |
| P | 908 | 8 | 59 |
| I | 3477 | 0 | 33 |

Bảng đếm events, không phải query độc lập. W là sai giá trị, T là lệch ngôn ngữ/canonical, P là khác diễn đạt, I là thiếu ngữ cảnh/argument theo heuristic. Các nhãn này chưa được reviewer xác nhận. Queue chính có 70 + 48 + 71 =189 event cần review; hiện không có human_label được điền. [S09]

### Các ví dụ đọc trực tiếp từ query và prediction

xlam_29775: query có ‘30 từ dễ’, gold difficulty='easy' nhưng prediction='dễ'. Đây là ứng viên lỗi canonical/ngôn ngữ; normalizer số không giải quyết mapping này. Không dùng synonym tự thêm để nâng điểm test.

custom_vi_v1_006644: gold location='huế', prediction='cố đô huế'. Model lấy rộng surface hơn gold; có thể mô tả lỗi ranh giới span, dù heuristic gắn P. custom_vi_v1_006433 trích địa điểm của một call khác trong query multi-call; nhãn heuristic T chưa đủ bằng chứng vì còn khả năng trộn ngữ cảnh giữa các call.

custom_vi_v1_007592 có gold boolean false cho family_friendly_only nhưng model bỏ parameter. Query có ‘không giới hạn theo tiêu chí gia đình’; đây là ví dụ has_value/boolean ở tool unseen cần phân tích. Không quy mọi missing argument thành thiếu thông tin người dùng.

### Loại dữ liệu chưa hỗ trợ và repeated calls

Đếm trực tiếp top-level arguments trong gold benchmark: 1.527 array và 230 object trên tổng25.121 argument occurrences, khoảng6,99%. Đây là inventory theo schema gốc, khác evaluator có flatten/matching và type inference. Test gold vẫn giữ những argument này; CE hiện không có head cấu trúc tương ứng. Span F1 chuyên biệt chưa có bảng đo trong bộ tổng hợp, không được thay bằng Argument F1.

Cùng một tên tool được gọi nhiều lần và query có nhiều ngữ cảnh là giới hạn cấu trúc đáng chú ý. Các ví dụ ở đây là review sơ bộ của trợ lý, không được coi là hoàn thành 189 nhãn review thủ công hoặc là causal attribution đã kiểm chứng.

## 13 So sánh với lịch sử legacy

| Tập | Legacy norm ArgA | Strict hiện tại | Chênh lệch pp |
| --- | --- | --- | --- |
| Benchmark | 39,84% | 40,21% | 0,37 |
| Custom seen | 53,50% | 67,75% | 14,25 |
| Custom unseen | 7,50% | 22,75% | 15,25 |

Bản legacy ở bảng này được tái chấm bằng evaluator sửa strict type/whitespace và matching multi-call. Số query giữ nguyên, nhưng giữa legacy và hiện tại đã đổi cả Bi-Encoder protocol lẫn CE supervision/checkpoint. Vì vậy chênh lệch không phải ablation riêng của strict unseen hay CE repair; chỉ là mô tả hai cấu hình trong lịch sử. [S02, S06]

### Các sửa chữa ảnh hưởng tính hợp lệ thực nghiệm

Strict data loại heldout khỏi negatives trước training và mining; BI được train lại từ base. Evaluator bổ sung strict matching, oracle gap cùng mẫu số và error alignment cho multi-call. CE validation không tự pass khi thiếu metric; raw outputs/logits được lưu để điều tra.

Kaggle từng lỗi PEFT/torchao không tương thích trước optimizer: setup bỏ torchao vì run không dùng quantization, rồi smoke test LoRA. Loader cũng sửa cách chọn input khi Kaggle giải nén archive reports làm trùng marker; chỉ nhận artifact đầy đủ khớp hash và cùng bundle identity. Đây là sửa môi trường/loader, không phải thay kết quả training đã hoàn thành.

CE repair được vận chuyển như addon riêng có parent manifest để không sửa bundle strict đã freeze. Lưu output01b làm nguồn gốc repaired labels/config và baseline/candidate comparison, dù các notebook evaluation sau không cần toàn bộ training pairs. Không chỉ giữ final CE ở output06 rồi xóa nguồn01b.

### Mức độ chắc chắn

Hash audit chứng minh file không đổi giữa các stage đã kiểm tra; nó không thay thế audit chất lượng nhãn, xác nhận không có scenario overlap, hoặc tái lập độc lập từ clean environment. Chưa có run nhiều seed và chưa có benchmark so sánh bốn phương pháp.

## 14 Đối chiếu mức hoàn thành với plan

| Hạng mục Method 2 | Trạng thái và giới hạn |
| --- | --- |
| Strict unseen BI training | Hoàn thành 2 rounds; train/mining scan zero heldout identifier |
| CE training và kiểm định | Hoàn thành baseline + repair + selection; quality gate chưa đạt |
| Oracle và pipeline test | Hoàn thành 3 test sets, đủ IDs và provenance |
| Strict và normalized scoring | Đã có; tách rõ scorer normalization và inference normalizer |
| Normalizer ablation | Notebook06 đã hoàn thành, cùng weights/threshold với04 |
| Random stress | Hoàn thành 1.200 instances |
| Pure same-domain stress | Chưa hoàn thành; cần nhãn domain và kiểm tra capacity |
| Phân tích lỗi | Có thống kê + ví dụ; 189 event queue chưa human review |
| Latency và tài nguyên | Có timing/peak training; thiếu lặp cùng session và concurrent load |
| Chi phí USD và Span F1 | Chưa đủ số đo; không suy diễn từ token hoặc Argument F1 |
| Các ablation khác | Theo quota, chưa có đầy đủ; không phải điều kiện cứng của scope đã chốt |
| So sánh Method1 và API | Ngoài scope riêng Method2; chưa thể chốt plan toàn đề tài |

Có thể báo cáo kết quả thực nghiệm với giới hạn trên. Không đánh dấu tất cả checkbox trong experimental_plan.md là done, không hạ gate để tuyên bố nghiệm thu. Các số đo chưa đạt vẫn là kết quả nghiên cứu hợp lệ, nhưng khác với đạt mục tiêu chất lượng.

Nếu cần cải tiến enum/boolean, repeated-tool hoặc abstention, đó phải là experiment tiếp theo với protocol và validation được định trước; không chọn hyperparameter dựa trên các test vừa xem. Bộ strict hiện tại được giữ như một cấu hình đã freeze để so sánh sau này.

## 15 Công việc còn lại và cách dùng bộ báo cáo

### Không cần chạy lại notebook 01 đến 06

Tất cả stage chính đã có output thực tế. Giữ nguyên notebook04 làm baseline ON, notebook06 là OFF và notebook05 là random stress. Không dùng output05 thay input04 cho một ablation mới. Không sửa checkpoint hoặc threshold trong các output đã freeze.

### Hoàn thiện luận văn từ kết quả hiện tại

Ưu tiên review 189 event trong human_review_queue.jsonl của notebook04: mở query, gold, predicted arguments và alignment; điền human_label/review_note trong bản sao riêng, giữ heuristic gốc. Chú ý phân biệt sai ngôn ngữ, span quá rộng, trộn call, unsupported type và mismatch kiểu số. Sau review, cập nhật bảng nguyên nhân và mô tả cách lấy mẫu.

Đối với pure same-domain, cần quyết định phạm vi nghiên cứu: bổ sung nhãn/candidate pool đủ cho từng N hoặc công bố phần này chưa khả thi. Không tự tạo tool giả và không điền random vào chỗ same-domain. Với chi phí, chốt đơn giá GPU cùng phạm vi train/inference/index trước khi tính USD.

Bộ PDF này trình bày số đo và kết luận. Bản Markdown là nội dung chỉnh sửa; evidence.json giữ các metric đầy đủ, inventory và audit; figures chứa biểu đồ vector và PNG. File source_index.json lưu đường dẫn và SHA-256 của các nguồn chính để truy vết.

### Cách viết kết luận được hỗ trợ bởi dữ liệu

Có thể viết: ‘Method 2 đạt ArgA normalized 40,21% trên benchmark và 67,75%/22,75% trên CustomTools seen/unseen. Inference normalizer cải thiện ArgA ở cả ba tập. Khi tăng random candidate tools từ3 lên1.000, latency P50 tăng từ58,16 lên91,88 ms và ArgA giảm từ81% xuống54%. Khả năng tổng quát hóa extraction và chọn/no-call với pool lớn còn hạn chế.’

Chưa thể viết: ‘Method 2 vượt SLM/API’, ‘latency không đổi theo N’, ‘zero-shot chưa từng thấy tool trong pretraining’, hoặc ‘tất cả gate đã đạt’. Không dùng 0 USD cho dữ liệu cost unavailable và không trình bày ví dụ heuristic như nhãn của reviewer.

### Lưu trữ

Giữ đầy đủ output01b,02,03,04,05,06 cùng các notebook đã chạy. Report archives giúp chia sẻ số liệu nhưng không có đầy đủ weights/index. Chỉ dọn bản sao checkpoint khi đã xác nhận bản lưu khác đầy đủ hash; không xóa source training manifests/configs để tiết kiệm dung lượng một cách không kiểm chứng.

## 16 Danh mục nguồn và provenance

Các đường dẫn dưới đây tính từ thư mục dự án C:/Users/Admin/Desktop/KLTN/tool_calling_with_retrieval_extraction. Số liệu được đọc từ file kết quả, không lấy từ log tiến độ cũ. evidence.json và source_index.json đi kèm chứa giá trị đầy đủ và hash nguồn.

[S01] Protocol và scope: docs/experimental_plan.md

[S01b] Hướng dẫn completion: docs/method2_completion.md

[S02] Legacy tái chấm: artifacts/method2_completion/legacy_review/tables.json

[S03] BI round1 train: output_train/output_method2-completion-02-round1/artifacts/method2/biencoder/strict_round1/train_report.json

[S03b] BI round2 train: output_train/output_method2-completion-03-round2/artifacts/method2/biencoder/strict_round2/train_report.json

[S04] CE repair selection: output_train/output_method2-ce-repair-01b/results/method2/ce_repair/comparison.json

[S04b] CE gate candidate: output_train/output_method2-ce-repair-01b/results/method2/ce_repair/candidate/validation_gate.json

[S05] Calibration: output_train/output_method2-completion-03-round2/artifacts/method2/biencoder/strict_round2/thresholds.json

[S05b] Full pool retrieval: output_train/output_method2-completion-03-round2/results/method2/completion/retrieval_val_pool.json

[S06] Evaluation ON: output_train/output_method2-completion-04-evaluation/results/method2/completion/evaluation/report/tables.json

[S07] Evaluation OFF: output_train/output_method2-completion-06-normalizer-ablation/results/method2/completion/normalizer_ablation/report/tables.json

[S08] Stress: output_train/output_method2-completion-05-stress/results/method2/completion/stress/stress_report.json

[S09] Error queue ví dụ: output_train/output_method2-completion-04-evaluation/results/method2/completion/evaluation/report/custom_seen/human_review_queue.jsonl

[S10] Audit và số đếm trực tiếp: artifacts/method2_completion/report_20260908/evidence.json

Bundle parent SHA-256: d562aa4b38bbba20eb28adfc1702df42c193add4795c7415528a62bde945c908

CE model weights SHA-256: f90c604b061d01d35eb4d848f09f3482dd891240accf283ad2659e0233b19549

Threshold SHA-256: 64df229509df03ccddd2cecf7ec9cc03e3b46101cb236373cef1d992dfad138e
