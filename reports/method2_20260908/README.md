# Báo cáo Method 2 ngày 08/09/2026

- `bao_cao_method2.pdf`: báo cáo tổng hợp 17 trang, có bảng và biểu đồ.
- `bao_cao_method2.md`: nội dung tương ứng để chỉnh sửa, copy vào luận văn.
- `evidence.json`: số liệu đầy đủ, inventory 10 output folders, audit notebook/checkpoint, paired predictions và các reports thành phần.
- `source_index.json`: file nguồn và SHA-256. Đường dẫn tính từ root repository.
- `figures/`: biểu đồ PNG 180 DPI và PDF vector.

Notebook06 hoàn thành, cùng weights/index/threshold với04. Normalizer ON cải thiện ArgA normalized 19,22 / 34,75 / 15,75 điểm phần trăm trên benchmark / Custom seen / Custom unseen.

Không cần chạy lại01–06 để báo cáo cấu hình này. CE quality gate vẫn false; pure same-domain, 189 event human review và cost/Span F1 còn thiếu. Không dùng test để tune lại model.

Nguồn output gốc vẫn được giữ nguyên trong output_train. Gói này là báo cáo và số liệu tổng hợp, không chứa toàn bộ weights hay raw predictions.
