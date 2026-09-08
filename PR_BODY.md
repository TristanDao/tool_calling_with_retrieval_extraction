Method 2 cần loại tool heldout khỏi mọi negative training và dùng cùng checkpoint đã kiểm định cho evaluation. Thay đổi này bổ sung protocol strict unseen, sửa CE supervision từ argument_mentions và hoàn thiện chuỗi notebook Kaggle 01–06 với hash provenance.

## Thay đổi

- Loại 20 tool heldout khỏi training/mining; giữ legacy run riêng để audit.
- Sửa strict/normalized scoring, matching multi-call, oracle gap và error classification.
- Thêm CE repair, validation gate, loader xử lý duplicate marker và môi trường LoRA Kaggle.
- Lưu notebook đã chạy, audit nhỏ, báo cáo PDF/Markdown 17 trang, bảng số liệu và biểu đồ.
- Bỏ qua datasets/checkpoints/bundle lớn và QA scratch trong Git; thêm matplotlib vào dev dependencies cho test biểu đồ.

## Kết quả và giới hạn

Pipeline normalized ArgA: benchmark40,21%, Custom seen67,75%, Custom unseen22,75%. Normalizer ON hơn OFF19,22/34,75/15,75 điểm phần trăm. Random stress hoàn thành1.200 instances; ArgA81%→54% khi N3→1000.

CE gate vẫn false (enum74,49%, Argument EM61,96%). Pure same-domain và189 event human review còn pending; không tuyên bố nghiệm thu toàn bộ plan hoặc so sánh thắng Method1/API. Lưu kết quả đã freeze, không tune bằng test.

## Kiểm tra

Các test evaluation, BI, CE, pipeline và completion notebooks đã chạy; test biểu đồ chạy lại thành công sau khi cài dependency matplotlib còn thiếu. Kiểm tra diff whitespace và patterns credential trên các file thay đổi không phát hiện vấn đề. Notebook06 xác minh114 bundle hashes,29 artifact hashes và đủ12.155 query với weights/index/threshold giống notebook04.

Báo cáo: reports/method2_20260908/bao_cao_method2.pdf và bao_cao_method2.md.
