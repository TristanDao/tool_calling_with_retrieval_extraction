# Báo cáo stress test Shared E4 — Method 2 và đối chiếu source Thịnh

Ngày kiểm chứng:20/09/2026. Run notebook06 hoàn tất; không cần chạy lại stress random này. Model BI/CE và thresholds giống hệt final05. Đây là kết quả mới, thay thế vai trò tham chiếu của stress legacy07/09 trong báo cáo Shared E4.

## 1. Thiết kế và bằng chứng hoàn tất

Sáu mức N=3,10,50,100,500,1000, cùng200 query CustomSeen (100 positive+100 negative), tổng1.200 instances. Dùng đúng bytes stress dataset Thịnh cung cấp; candidate là prefix lồng nhau, gold giữ trong haystack. Toàn bộ1.200 predictions và raw records đủ ID, không trùng hoặc mất. Notebook không có cell error. Hashes: {'bundle': 95, 'round2': 17, 'ce': 7, 'calibration': 28, 'evaluation': 28, 'stress': 27, 'addon': 8}. Tái chấm counts, reference/strict metrics và latency mean/P50/P95 khớp summary; hash model/threshold khớp output05.

## 2. Kết quả Method 2 mới

| N | Tool Acc % | ArgA % | Non-FC % | P50 ms | P95 ms | Peak allocated MiB | Peak reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 89.00 | 87.50 | 100.00 | 55.13 | 84.92 | 3277.14 | 3388.00 |
| 10 | 89.00 | 87.50 | 100.00 | 55.05 | 84.74 | 3281.76 | 3484.00 |
| 50 | 89.00 | 87.50 | 100.00 | 56.07 | 86.46 | 3278.02 | 3670.00 |
| 100 | 89.00 | 87.00 | 99.00 | 61.01 | 85.09 | 3283.04 | 3772.00 |
| 500 | 87.00 | 85.00 | 97.00 | 92.27 | 128.82 | 3283.04 | 3772.00 |
| 1000 | 87.00 | 84.00 | 95.00 | 107.68 | 148.21 | 3283.09 | 3772.00 |

Tool Acc tính trên100 positive theo danh sách tên tool có thứ tự; ArgA tính trên200mẫu, gồm negative; Non-FC Recall tính trên100 negative. Syntax errors0% ở cả sáu mức, không đồng nghĩa mọi arguments đúng schema/gold. Peak VRAM trong bảng là PyTorch allocated/reserved tuyệt đối của inference, gồm model resident, không phải incremental allocation hay tổng nvidia-smi. 3.283MiB xấp xỉ3,21 GiB; không còn cơ sở dùng tuyên bố “<1,2GB” từ bài cũ.

![Stress curves](stress_curves.png)

## 3. P50, P95 và bộ nhớ: protocol đo

Kaggle, một Tesla T4, batch 1, Python3.12.13, torch2.10.0+cu128/CUDA12.8. Mỗi N chạy subprocess riêng,3 warmup, CUDA synchronize trước/sau từng query. Timer gồm candidate mapping, query embedding, retrieval, extraction và validation; không gồm ghi file, model load/index build. Index tool được cache; thời gian load/index nằm trong metrics của từng N. CPU RAM/index không thuộc VRAM được đo.

Peak CUDA reset sau warmup; full-run peak bằng max giai đoạn load/index/warmup và inference, ghi riêng trong audit. Telemetry memory đã kiểm hash và quan hệ số học; không tái đo GPU độc lập từ máy local. P95 là phân vị trên200 query của một run, không phải confidence interval hay latency SLA. Có phân vị positive/negative riêng trong audit.

## 4. Phân tích theo N

ArgA giảm87,50→84,00% (3,50điểm), Tool Acc89,00→87,00% (2điểm), Non-FC100→95%. P50 tăng55,13→107,68ms, khoảng1.95lần; P95 tăng84,92→148,21ms. Do đó latency không phẳng trên toàn dảiN, dù bộ nhớ allocated gần ổn định quanh3,2 GiB và không OOM ởN1000.

ỞN1000, 100/100 positive có đủ gold tool trong top3; 2/100 positive chọn thêm tool ngoài gold; 5/100 negative phát sinh call. Positive exact còn73/100; tổng exact168/200 gồm95negative đúng. Tool multiset accuracy95% cao hơn ordered87%, cho thấy thứ tự calls là một phần lỗi. Không thay metric sau khi xem test và không tune tau/delta theo stress.

## 5. Bảng đối chiếu theo yêu cầu Thịnh

| N | SLM Tool Acc %¹ | M2 Tool Acc % | SLM ArgA %¹ | M2 ArgA % | SLM P50 ms¹ | M2 P50 ms | M2 P95 ms | SLM P95 / VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 94.0% | 89.00 | 85.5% | 87.50 | 1,258.94 ms | 55.13 | 84.92 | Chưa có |
| 10 | 93.0% | 89.00 | 84.5% | 87.50 | 1,604.68 ms | 55.05 | 84.74 | Chưa có |
| 50 | 93.0% | 89.00 | 85.5% | 87.50 | 4,703.39 ms | 56.07 | 86.46 | Chưa có |
| 100 | 92.0% | 89.00 | 83.0% | 87.00 | 9,318.21 ms | 61.01 | 85.09 | Chưa có |
| 500 | *OOM* | 87.00 | *OOM* | 85.00 | *OOM* | 92.27 | 128.82 | Chưa có |
| 1000 | *OOM* | 87.00 | *OOM* | 84.00 | *OOM* | 107.68 | 148.21 | Chưa có |

¹ Các số SLM/OOM trích bảng stress trong [paper_vi.md của Thịnh](<C:/Users/Admin/Desktop/KLTN/tool_calling_with_retrieval_extraction của Thịnh/tool_calling_with_retrieval_extraction-main/paper/paper_vi.md>), chưa có raw output SLM để kiểm độc lập hay chứng minh cùng checkpoint/data revision trong lần audit này. M2 dùng kết quả mới đã xác minh; không sao chép cột Method2 cũ. Thiếu P95/VRAM SLM được ghi thiếu, không nội suy.

Source worker SLM chạy hai GPUworkers, batch8/4/2/1 theo N, lấy batch generation time chia số mẫu; chưa thấy peakVRAM logging, warmup hoặc CUDA sync tương ứng. Có truncation prompt32768tokens. Vì vậy bảng này chỉ là đối chiếu mô tả, không tính speedup trực tiếp, không coi SLM OOM là speedup vô hạn, và chưa xác nhận đủN tool thực sự vào context SLM ở mỗi mức. Nên đề nghị Thịnh gửi raw metrics/predictions và bổ sung cùng protocol latency/memory để hoàn thiện bảng chính thức.

## 6. Kết luận và việc còn lại

Stress random Shared E4 đã hoàn tất đủ200×6; không cần train lại hay chạy lại notebook06 cho bộ số này. Có thể gửi Thịnh bảng Method2 với P50/P95/peakVRAM ngay. Method2 chạy được tớiN1000 trong run này với ArgA84%, P50 107,68ms, P95 148,21ms và peak allocated3.283,09 MiB. Validation gate từ04 vẫn false; stress hoàn tất không đồng nghĩa quality gate đã đạt. Pure same-domain và normalizer ablation trên checkpoint mới chưa được thực hiện trong run này.

Artifacts: [audit.json](audit.json), [bảng để ghép bài](tables_for_thinh.md). Dữ liệu gốc nằm tại `output_train/output_method2-shared-06-stress/shared_e4/shared_runs/stress`. Audit tái lập bằng `scripts/method2/audit_shared_stress_results.py`.
