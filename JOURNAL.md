# JOURNAL.md — Project History & Technical Decisions Log

> **Mục đích**: Lưu trữ toàn bộ nhật ký thay đổi, lịch sử các quyết định kiến trúc đã đóng, quá trình thực nghiệm pilot và các giải pháp khắc phục hạ tầng qua từng giai đoạn của đề tài Khóa luận tốt nghiệp.
> File này tách biệt khỏi [AGENTS.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/AGENTS.md) để giữ cho bộ nhớ vận hành của AI agent luôn gọn gàng, súc tích và cập nhật.

---

## 1. Change Log (Nhật ký thay đổi chi tiết)

| Ngày | Thay đổi |
|---|---|
| 2026-07-25 | Khởi tạo repo, chốt tech stack ban đầu, tạo khung thư mục (skeleton) và các file `.md` ban đầu. |
| 2026-07-25 | Bổ sung Phase 7 (Stress Test) và cấu trúc thư mục mở rộng. |
| 2026-07-26 | Đóng quyết định: BGE-M3 base cho cả 2 model. Cross-Encoder thiết kế Hierarchical heads, định dạng input BERT-QA. |
| 2026-07-26 | Hoàn tất khung code Cross-Encoder (6/6 files). Tạm dừng triển khai model để tập trung pipeline dữ liệu. |
| 2026-07-26 | Thu thập dữ liệu EN: Glaive (45,593 positive first-turn), xLAM (28,461 single-call/multi-call). |
| 2026-07-28 | Thử nghiệm pivot sang multi-turn + multi-call (sau đó đã revert để bảo toàn tính tập trung vào single-turn). |
| 2026-08-03 | **Pivot chiến lược**: Chuyển sang so sánh 2 phương pháp. Method 1: SLM End-to-End (theo hướng Ersoy et al.). Method 2: Bi-Encoder + Cross-Encoder. Schema master canonical single-turn + multi-call. Cập nhật tài liệu đối chuẩn 4 phương pháp. |
| 2026-08-04 | Xác nhận Bộ 1 dịch giữ format gần raw; xây dựng lại benchmark theo schema master: 209 samples, 395 tools; tạo instruction format cho 3 split. |
| 2026-08-04 | Thêm smoke test `feature_group` không ghi cache; xử lý lỗi quota API Alibaba (`403 insufficient_quota`). |
| 2026-08-04 | Sửa cấu hình `feature_group` dùng biến môi trường `${ALIBABA_MODEL}`; smoke test thành công với `qwen3.7-flash`. |
| 2026-08-04 | Chạy pilot translation thêm 25 mẫu/dataset: Glaive 24/25, xLAM 25/25; vượt qua toàn bộ quy tắc QA. |
| 2026-08-04 | Giảm batch dịch từ 25 xuống 10; commit output + checkpoint trước `feature_group` để lỗi classifier không làm mất batch dịch. |
| 2026-08-04 | Xây dựng chuỗi fallback 3 tầng và smoke test dịch 1 sample không ghi đè output/checkpoint. |
| 2026-08-04 | Cấu hình `${ALIBABA_URL}`; smoke test dịch thuật và gán nhóm chức năng đều thành công và trả JSON hợp lệ. |
| 2026-08-04 | Lọc tập Glaive raw giữ 45,593 mẫu positive first-turn sạch; reset output pilot cũ, chạy pilot mới 10+10 đạt 100% QA. |
| 2026-08-04 | **Multi-model fallback chain**: Thiết lập danh sách 74 models Alibaba xếp theo tier chất lượng (loại bỏ thinking/OCR/video). Primary: `qwen-mt-plus`. Concurrency 8→12. Daily budget ~74M tokens. |
| 2026-08-10 | Thêm luồng dịch trực tiếp `data/normalized_en/glaive_normalized.jsonl` với dataset `glaive_normalized`, output/checkpoint/QA riêng biệt; không ghi đè Bộ 1 raw. |
| 2026-08-10 | Chuyển entrypoint dịch Glaive và xLAM sang normalized schema; raw outputs giữ lại để kiểm toán. |
| 2026-08-11 | **CustomTools-VI Strategy**: Thống nhất kế hoạch xây dựng 8,000 samples (40 tools / 10 nhóm) thuần ngữ cảnh Việt Nam, split 70/10/20. Ban hành `docs/custom_vi_dataset_plan.md`. |
| 2026-08-20 | **Cập nhật backbone model Method 1**: Nâng cấp mô hình SLM từ Qwen2.5 (0.5B/1.5B) sang **Qwen3.5 (2B/4B)**. Đồng bộ toàn bộ tài liệu và kế hoạch thực nghiệm. |
| 2026-08-27 | **Materialize Experiment Data**: Thêm `prepare_experiments.py` tạo các tập `e0` đến `e4` với manifest và train-only native data; validation/test dùng shared frozen revision. |
| 2026-08-27 | Tự động hóa upload dataset lên Kaggle qua `kagglehub` và token `~/.kaggle/access_token`. |
| 2026-08-31 | **Cập nhật Trainer Method 1**: Chốt dùng **Unsloth** cho QLoRA/SFT checkpoint `unsloth/Qwen3.5-2B` và `unsloth/Qwen3.5-4B`. Chat template native từ checkpoint, response-only loss. |
| 2026-09-02 | **Frozen canonical benchmark revision**: Khóa revision `2026-09-02-full-dedup-seed42` gồm 77,028 mẫu paired (61,615 train / 7,701 val / 7,712 test) với 4,421 unique tools. Lưu trữ pilot vào `data/legacy/`. |
| 2026-09-03 | **Hạ tầng Kaggle DDP**: Xây dựng hướng dẫn chạy phân tán trên 2 GPU T4 qua `torchrun` và Unsloth DDP, `effective_batch_size = 16`. |
| 2026-09-03 | **Kaggle Timeout Mitigation**: Tách quy trình đánh giá batched/resumable (E0) và huấn luyện (E1–E4) có checkpointing định kỳ 250 steps vượt giới hạn 12 giờ. |
| 2026-09-04 | **RAM Mitigation**: Huấn luyện Unsloth đọc streaming JSONL, tokenize bằng generator và chỉ lưu token tensor Arrow; tắt dataloader workers tránh OOM RAM máy chủ. |
| 2026-09-04 | Bổ sung hướng dẫn Colab single-GPU A100 với Drive-sync checkpointing. |
| 2026-09-08 | Hoàn tất thực nghiệm Method 2 Round 1 & Round 2 (BGE-M3 + XLM-R base) do Đạt chạy, nhập báo cáo về `reports/method2_20260908/`. |
| 2026-09-13 | Hoàn tất bộ số liệu Method 2 Shared E4 (VI Tool Acc 59.20%, ArgA 30.26%; Custom Seen 85.38%, Unseen 60.25%) tại `reports/method2_20260913/`. |
| 2026-09-15 | **Stress Test & Training Telemetry Update**: Hoàn thành đối đầu trực diện 2B_E4 vs Method 2 ($N = 3 \to 1000$). Ghi nhận sụp đổ CUDA OOM của SLM tại $N \ge 500$ trên T4 16GB. Method 2 duy trì latency 55–108 ms, VRAM 3.2 GiB. Ghi nhận thời gian huấn luyện thực tế trên A100: 2B (49.5 phút), 4B (2 giờ 05 phút, loss 0.0103). Đồng bộ vào paper và khóa luận. |
| 2026-09-20 | Đồng bộ toàn bộ số liệu thực nghiệm, phân tích đối chứng Oracle, bảng phân tích ca lỗi thực tế vào bản thảo Khóa luận tốt nghiệp UIT và bộ đôi Paper EN/VI. |

---

## 2. Lịch sử các Quyết định Kỹ thuật đã đóng (Archived Decisions)

### 2.1 Thiết kế Scope & Master Data Schema (Đóng 2026-07-28 & 2026-08-03)
- **Định dạng hội thoại**: Quyết định chỉ lấy lượt đầu tiên (first turn) từ Glaive và giữ cấu trúc multi-call từ xLAM. Loại bỏ ý định mở rộng multi-turn để tập trung giải quyết bài toán cốt lõi: Semantic Retrieval và Parameter Extraction theo JSON Schema.
- **Master Schema format**: Thống nhất dùng chung một cấu trúc JSON canonical cho toàn bộ hệ thống (`id`, `source`, `query`, `function_calls[]`, `tools[]`). Đối với Method 1, chuyển đổi sang native chat messages (`tools`, `tool_calls`). Đối với Method 2, bóc tách trực tiếp thành cặp truy hồi `(query, tool)` và cặp trích xuất `(query, param_schema)`.

### 2.2 Kiến trúc Method 1 — SLM End-to-End (Đóng 2026-08-20 & 2026-08-31)
- **Backbone**: Thay thế đề xuất sơ khởi Qwen2.5 (0.5B/1.5B) bằng `unsloth/Qwen3.5-2B` và `unsloth/Qwen3.5-4B` để tận dụng khả năng suy luận ngôn ngữ tự nhiên và cấu trúc JSON vượt trội.
- **Framework huấn luyện**: Chuyển hoàn toàn sang Unsloth QLoRA (NF4, LoRA rank 16, alpha 32, target all-linear). Không sử dụng LLaMA-Factory hay ShareGPT trung gian nhằm kiểm soát tuyệt đối chat template native và loss mask (chỉ tính loss trên phản hồi của assistant).
- **Phản hồi tiêu cực (Negative samples)**: Sử dụng văn bản trả lời tự nhiên của assistant (từ chối do thiếu công cụ phù hợp), tuyệt đối không dùng token giả `<no_tool_call>`.

### 2.3 Kiến trúc Method 2 — Bi-Encoder + Cross-Encoder (Đóng 2026-07-26 & 2026-08-20)
- **Bi-Encoder (Retrieval)**: Backbone `BAAI/bge-m3` tinh chỉnh bằng `CachedMultipleNegativesRankingLoss` (CachedMNRL) qua 2 vòng (Round 1 huấn luyện teacher, Round 2 khai phá hard negatives). Thiết lập ngưỡng hiệu chuẩn động $\tau = 0.35, \delta = 0.21$.
- **Cross-Encoder (Extraction)**: Sử dụng backbone `xlm-roberta-base` kết hợp kiến trúc phân cấp (Hierarchical Heads) gồm 1 head nhị phân `has_value` phát hiện tham số rỗng và 3 sub-heads định hướng theo kiểu dữ liệu (Span Head, Enum Head, Boolean Head). Thiết kế BERT-QA nhận context là user query và question là parameter schema.
- **Bộ chuẩn hóa giá trị (Value Normalizer)**: Bổ sung bộ chuẩn hóa hậu xử lý dựa trên luật kết hợp từ điển số và ngày tháng tiếng Việt để chuyển đổi các chuỗi bề mặt tự nhiên sang kiểu dữ liệu đích (integer, float, boolean, enum canonical).

### 2.4 Dữ liệu Thử nghiệm CustomTools-VI (Đóng 2026-08-11)
- Xây dựng 8,000 mẫu dữ liệu đặc thù Việt Nam (gồm 40 tools trải rộng trên 10 lĩnh vực: Giao thông, Hành chính công, TMĐT, Tài chính, Y tế, Giáo dục,...).
- Tách thành 5,600 train + 800 val + 1,600 test (chia đôi thành 800 Seen Tools và 800 Unseen Tools để kiểm tra năng lực tổng quát hóa zero-shot của cả 2 phương pháp).

---

## 3. Lịch sử Khắc phục Sự cố & Tối ưu Hạ tầng (Telemetry & Bugfixes)

### 3.1 Vấn đề Dịch thuật tự động và Chuỗi Fallback (Tháng 08/2026)
- **Vấn đề**: API Alibaba thường xuyên gặp biến động về quota (`403 insufficient_quota`) và một số mô hình trả định dạng không tuân thủ strict JSON.
- **Giải pháp**: Xây dựng chuỗi dự phòng 74 mô hình (`ALIBABA_BACKUP_MODELS`), cơ chế thử lại exponential backoff tối đa 3 lần cho mỗi mẫu, lưu checkpoint atomic theo từng batch 10 mẫu kèm fsync để bảo toàn dữ liệu khi có sự cố mạng.

### 3.2 Vấn đề Giới hạn Thời gian và Bộ nhớ trên Kaggle/Colab (Tháng 09/2026)
- **Giới hạn 12 giờ trên Kaggle**: Thiết kế pipeline huấn luyện lưu checkpoint mỗi 250 bước, có khả năng resume tự động liền mạch qua nhiều notebook sessions.
- **OOM RAM máy chủ khi tokenize DDP**: Khi chạy 2 GPU T4 trên Kaggle, việc nạp toàn bộ dataset vào bộ nhớ gây crash kernel. Giải pháp: Sử dụng generator đọc streaming JSONL, tokenize on-the-fly và chuyển đổi sang Arrow Dataset chỉ chứa cột token IDs, tắt đa tiến trình của Dataloader.
- **Sự cố CUDA OOM trong Stress Test**: Trên GPU NVIDIA T4 (16GB), mô hình SLM Qwen3.5-2B bị tràn bộ nhớ tại $N \ge 500$ do ma trận self-attention của SDPA vượt quá 32GB dung lượng tính toán khi context chứa hàng trăm tool definitions. Ngược lại, Method 2 duy trì ổn định bộ nhớ VRAM ở mức ~3.21 GiB và hoàn thành toàn bộ 1,000 tools.
