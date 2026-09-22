# AGENTS.md — Operational Memory for AI Assistants

> **Mục đích**: File này là "bộ nhớ vận hành cốt lõi" (operational memory) giữa các phiên làm việc với AI.
> Mọi agent (Antigravity, Claude, GPT, Cursor, v.v.) **phải đọc file này đầu tiên** trước khi thực hiện bất kỳ hành động nào trong repo.
> Lịch sử thay đổi (Changelog), các quyết định kỹ thuật đã đóng và nhật ký thử nghiệm được lưu tại [JOURNAL.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/JOURNAL.md).

---

## 1. Định danh Đề tài (Project Identity)

- **Tên đề tài Khóa luận tốt nghiệp chính thức (Đăng ký tại UIT)**:
  - Tiếng Việt: **Nghiên cứu phương pháp Tool Calling dựa trên truy hồi ngữ nghĩa và trích xuất tham số theo Tool Schema**
  - Tiếng Anh: **Research on Tool Calling using Semantic Retrieval and Schema-aware Parameter Extraction**
- **Đơn vị đào tạo**: Khoa Khoa học Máy tính — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
- **Ngành đào tạo**: Cử nhân ngành Trí tuệ Nhân tạo
- **Giảng viên hướng dẫn**: TS. Đặng Văn Thìn
- **Sinh viên thực hiện**:
  - Đào Phước Thịnh (MSSV: 25210038)
  - Hà Quang Đạt (MSSV: 25210008)
- **Thời gian thực hiện**: 15/07/2026 – 23/09/2026
- **Trạng thái hiện tại**: **Phase 6/7 — Hoàn thành 100% thực nghiệm toàn diện**. Đồng bộ hóa 100% nội dung vào Khóa luận tốt nghiệp UIT ([thesis/khoa_luan_tot_nghiep.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md), [latex/uit_thesis/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/latex/uit_thesis/)) và bộ đôi bài báo khoa học ([paper/paper_vi.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_vi.md), [paper/paper_en.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_en.md), [latex/paper_en/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/latex/paper_en/)).

---

## 2. Bài toán & Hai phương pháp Nghiên cứu

### 2.1 Bài toán
Xây dựng và đánh giá giải pháp Tool Calling tối ưu cho tiếng Việt, khắc phục các nhược điểm của các mô hình sinh thương mại lớn (chi phí cao, độ trễ autoregressive lớn, nguy cơ sinh ảo cú pháp JSON, và suy giảm hiệu năng khi không gian công cụ $N$ mở rộng).

### 2.2 Đối tượng so sánh (4 Phương pháp)
1. **Method 1: SLM End-to-End**
   - Backbone: `unsloth/Qwen3.5-2B` và `unsloth/Qwen3.5-4B`
   - Phương pháp: Huấn luyện tinh chỉnh QLoRA (NF4, rank 16, alpha 32) với Unsloth trên định dạng chat native; hàm mất mát tính riêng biệt trên phản hồi của assistant (`response-only loss`).
2. **Method 2: Bi-Encoder + Cross-Encoder**
   - **Semantic Retrieval**: Backbone `BAAI/bge-m3` tinh chỉnh bằng `CachedMultipleNegativesRankingLoss` (2 rounds với teacher hard negatives mining); ngưỡng kích hoạt động $\tau = 0.35, \delta = 0.21$.
   - **Parameter Extraction**: Backbone `xlm-roberta-base` kết hợp cấu trúc phân cấp (Hierarchical Heads): 1 binary head `has_value` phân biệt tham số xuất hiện vs rỗng, và 3 sub-heads định hướng theo kiểu dữ liệu (Span / Enum / Boolean). Bổ sung module `Value Normalizer` chuẩn hóa chuỗi bề mặt tiếng Việt về kiểu canonical.
3. **Frontier API Baseline 1**: OpenAI Function Calling (`gpt-4o-mini` / `GPT-5.6 Luna`).


### 2.3 Phạm vi nghiên cứu
- ✅ Single-turn + multi-call (một câu truy vấn có thể kích hoạt nhiều công cụ song song).
- ✅ Tập dữ liệu đối chuẩn tiếng Việt: Core Benchmark (`7,712` test samples) và CustomTools-VI (`1,600` test samples chia đều Seen / Unseen).
- ✅ Thử nghiệm ứng suất độ bền (Stress Test) đối đầu với số lượng công cụ gây nhiễu tăng dần ($N = 3 \to 1000$).
- ❌ Ngoài phạm vi: Multi-turn chat state, thực thi code tool thực tế, multi-agent orchestration.

---

## 3. Cấu trúc Dữ liệu Master Canonical

Toàn bộ hệ thống dùng chung một schema JSON duy nhất (`data/benchmark_core/<revision>/{en,vi}/*.jsonl`):

```json
{
  "id": "glaive_00042",
  "source": "glaive",
  "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
  "function_calls": [
    {
      "name": "search_tutors",
      "arguments": {"subject": "Toán", "location": "Hà Nội"}
    }
  ],
  "tools": [
    {
      "name": "search_tutors",
      "description": "Tìm gia sư theo môn học và khu vực.",
      "feature_group": "Tìm kiếm & Kết nối",
      "parameters": {
        "type": "object",
        "properties": {
          "subject": {"type": "string", "description": "Môn học cần tìm"},
          "location": {"type": "string", "description": "Thành phố hoặc khu vực"}
        },
        "required": ["subject", "location"]
      }
    }
  ]
}
```

### Quy ước bắt buộc
- `id`: Định danh duy nhất `<source>_<index>` (`glaive_00042`, `xlam_00123`, `custom_00001`).
- `query` & `tools[].description`: Theo ngôn ngữ của split (EN hoặc VI).
- `function_calls[].name` & `tools[].name`: snake_case tiếng Anh, không dấu.
- `parameters`: Tuân thủ chuẩn JSON Schema (`string`, `integer`, `number`, `boolean`, `array`, `object`).
- Negative samples (không gọi tool): `function_calls: []`. Trong Method 1, assistant trả lời hội thoại từ chối tự nhiên (tuyệt đối không dùng `<no_tool_call>`).
- Frozen Benchmark Revision hiện hành: `data/benchmark_core/2026-09-02-full-dedup-seed42/` (gồm 77,028 mẫu paired: 61,615 train / 7,701 val / 7,712 test, 4,421 unique tools).

---

## 4. Bản đồ Thư mục & Tài liệu Tham chiếu Chính

```
tool_calling_with_retrieval_extraction/
├── AGENTS.md                  # Bộ nhớ vận hành AI cốt lõi (file này)
├── JOURNAL.md                 # Nhật ký thay đổi & lịch sử quyết định kỹ thuật
├── configs/                   # Hydra Structured Configs (Python @dataclass)
├── src/
│   ├── data/                  # Pipeline thu thập, dịch, chuẩn hóa, phân chia dataset
│   ├── models/
│   │   ├── slm/               # Method 1: Qwen3.5 Unsloth SFT & inference
│   │   ├── biencoder/         # Method 2: BGE-M3 Retrieval
│   │   ├── crossencoder/      # Method 2: XLM-R Hierarchical Extraction
│   │   └── baselines/         # OpenAI & Gemini API runners
│   └── evaluation/            # Metrics (Tool Acc, ArgA), Latency, Stress Test
├── data/
│   ├── benchmark_core/        # Frozen revisions EN/VI (gitignored)
│   ├── custom_vi/             # 8,000 mẫu CustomTools-VI (train/val/test)
│   └── legacy/                # Lưu trữ pilot cũ
├── results/                   # Bảng số liệu JSON, CSV và biểu đồ thực nghiệm
├── reports/                   # Báo cáo thực nghiệm chi tiết của Method 2
├── thesis/                    # Bản thảo Markdown Khóa luận tốt nghiệp UIT
├── latex/
│   ├── uit_thesis/            # LaTeX Khóa luận tốt nghiệp UIT chuẩn quy chế
│   └── paper_en/              # LaTeX bài báo khoa học tiếng Anh
├── paper/                     # Bản thảo Paper Markdown (paper_vi.md & paper_en.md)
└── docs/                      # Tài liệu kỹ thuật chuyên sâu (architecture, methodology, guides)
```

### Các tài liệu kỹ thuật trọng yếu
- [JOURNAL.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/JOURNAL.md): Tra cứu lịch sử thay đổi chi tiết, các quyết định đã đóng và kỹ thuật tối ưu phần cứng.
- [docs/architecture.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/architecture.md): Kiến trúc đường ống chi tiết của Method 1 và Method 2.
- [docs/methodology.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/methodology.md): Cơ sở toán học, hàm mất mát và thiết kế thí nghiệm.
- [docs/paper_vi_method2_shared.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/docs/paper_vi_method2_shared.md): Ground truth số liệu Method 2 Shared E4.
- [paper/paper_vi.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_vi.md): Ground truth số liệu Method 1, Frontier Baselines và Stress Test đối đầu.

---

## 5. Quy chuẩn Bắt buộc cho AI Agents

### 5.1 Quy tắc Lập trình
1. **Không thêm comment vào code trừ khi user yêu cầu**.
2. **Type hints đầy đủ** cho toàn bộ public functions và classes.
3. **Hydra config**: Luôn sử dụng Python Structured Config (`@dataclass`), không dùng YAML lồng ghép tùy tiện.
4. **Reproducibility**: Thiết lập seed chặt chẽ qua `utils/seed.py`.
5. **Tuyệt đối không commit dữ liệu, checkpoints, file nặng hoặc token bảo mật**.

### 5.2 Quy chuẩn Văn phong Học thuật & Trình bày KLTN UIT (BẮT BUỘC TUÂN THỦ)
Khi hỗ trợ viết hoặc chỉnh sửa tài liệu Khóa luận ([thesis/khoa_luan_tot_nghiep.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md), [latex/uit_thesis/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/latex/uit_thesis/)):
1. **Thông tin định danh học thuật**:
   - Trường: **Trường Đại học Công nghệ Thông tin – ĐHQG-HCM**
   - Khoa: **KHOA KHOA HỌC MÁY TÍNH** *(tuyệt đối không ghi Khoa Công nghệ Thông tin)*
   - Ngành: **CỬ NHÂN NGÀNH TRÍ TUỆ NHÂN TẠO** *(tuyệt đối không ghi tắt CNTT/TTNT)*
   - Giảng viên hướng dẫn: **TS. Đặng Văn Thìn**
   - Sinh viên: **ĐÀO PHƯỚC THỊNH** (25210038), **HÀ QUANG ĐẠT** (25210008)
2. **Văn phong & Giọng văn**:
   - **Ngôi xưng**: Nhất quán sử dụng *"chúng em"* hoặc *"nghiên cứu / nhóm tác giả"*. Tuyệt đối **không** xưng *"chúng con"* hay *"tôi"*.
   - **Tính liền mạch (Narrative Flow)**: Diễn đạt bằng các đoạn văn nghị luận học thuật hoàn chỉnh, có câu chủ đề và liên từ mạch lạc; **hạn chế tối đa các gạch đầu dòng vụn vặt và tách đoạn vụn vặt**.
   - **Tránh sáo rỗng & khẩu ngữ**: Không dùng văn mẫu cảm tính sến súa (*"tri ân vô hạn"*, *"điểm tựa tinh thần"*, *"bế tắc"*) hay khẩu ngữ (*"nhồi thêm"*, *"kích hoạt mù quáng"*). Thay bằng thuật ngữ chuẩn mực (*"thiên kiến kích hoạt quá mức (over-triggering)"*, *"bổ sung công cụ gây nhiễu"*).
3. **Quy chế cấu trúc KLTN Đại học UIT**:
   - **Không có mục "Lời cam đoan"** (quy chế UIT chỉ áp dụng cho bậc Thạc sĩ / Tiến sĩ).
   - **Không có mục "Abstract" tiếng Anh riêng ở giữa tài liệu** (Abstract chỉ nằm ở phần mở đầu).
   - **Tính thống nhất toàn đoàn**: Khóa luận là công trình khoa học chung; tuyệt đối không đưa phân chia nhiệm vụ cá nhân hay tranh luận nội bộ vào tài liệu nộp Hội đồng.
   - **Tài liệu tham khảo**: Trình bày theo chuẩn IEEE, trích dẫn dạng số trong ngoặc vuông (`[1]`, `[2]`).
   - **Dung lượng**: Thân bài khóa luận từ 50 đến 100 trang khổ A4 theo quy chế UIT.
4. **Ký hiệu toán học & Số liệu**:
   - Số hàng nghìn trong LaTeX / Math mode phải viết là **`$1000$`** hoặc **`$1{,}000$`**, tuyệt đối không dùng `$1.000$`.
   - Bảng số liệu Train / Val / Test phải khớp số học 100% từng mẫu và luôn có cột tập Val.
   - `docs/tables_for_paper.md` đã lỗi thời; ground truth số liệu Method 1 & API lấy từ `paper/paper_vi.md`, Method 2 lấy từ `docs/paper_vi_method2_shared.md`.
   - Diễn giải kết quả khách quan, định lượng; không quy chụp quan hệ nhân quả tuyệt đối nếu thiếu kiểm chứng cùng protocol.

### 5.3 Quy chuẩn Định dạng Văn bản Word (.docx) khi Xuất bản
Khi thực hiện chuyển đổi hoặc biên tập tài liệu sang Word (`.docx`), cần tuân thủ quy chuẩn định dạng KLTN UIT:
- **Font chữ**: Times New Roman (hoặc font Unicode chuẩn tương đương).
- **Cỡ chữ**: 13 pt (thân văn bản).
- **Dãn dòng (Line spacing)**: 1.5 lines; Spacing: Before 0 pt, After 4 pt.
- **Căn lề (Margins)**:
  - Lề trên (Top): 3.0 cm
  - Lề dưới (Bottom): 3.5 cm
  - Lề trái (Left / Gutter side): 3.5 cm (để đóng gáy)
  - Lề phải (Right): 2.0 cm
- **Phân cấp Heading**:
  - **Chương (Heading 1)**: Tên Chương (Chương 1, Chương 2, ...), 14 pt, Đậm, căn giữa hoặc căn trái.
  - **Mục (Heading 2)**: 13 pt, Đậm (ví dụ: 3.1.), thụt lề 1 tab (0.6 cm).
  - **Tiểu mục (Heading 3)**: 13 pt, Đậm (ví dụ: 3.1.1.), thụt lề 2 tab (1.2 cm).
- **Bảng biểu (Tables)**: Tiêu đề bảng đặt **phía trên** bảng, căn giữa hoặc căn trái. Các cột căn lề hợp lý (văn bản căn trái, số liệu căn giữa hoặc căn phải).
- **Hình ảnh (Figures)**: Tiêu đề hình đặt **phía dưới** hình, căn giữa, hình ảnh rõ nét, không vỡ nét; chỉ sử dụng các sơ đồ diagram chuẩn (Mermaid / Kiến trúc hệ thống), không sử dụng các biểu đồ thô sinh từ thư viện đồ họa khi chưa chuẩn hóa ấn bản.

---

## 6. Trạng thái Thực nghiệm & Checklist Hành động

### 6.1 Trạng thái các Hạng mục (Đã hoàn thành 100% thực nghiệm)
- ✅ **Method 1 (Qwen3.5-2B E0→E4 & Qwen3.5-4B E3, E4)**: Đã hoàn thành 100% đánh giá trên cả Core Benchmark 7,712 mẫu EN/VI và CustomTools-VI 1,600 mẫu Seen/Unseen. Kết quả chuẩn hóa lưu tại `results/slm/`.
- ✅ **Method 2 (Bi-Encoder BGE-M3 + Cross-Encoder XLM-R)**: Hoàn tất 100% đánh giá trên toàn bộ các tập đối chuẩn và lưu tại `reports/method2_20260908/`, `reports/method2_20260913/`.
- ✅ **Stress Test Đối đầu ($N = 3 \to 1000$)**: Hoàn tất đối đầu trực diện giữa Method 1 (2B_E4) và Method 2. Method 2 duy trì ổn định Tool Acc > 87%, ArgA > 84%, độ trễ 55–108 ms và VRAM 3.2 GiB. SLM gặp CUDA OOM ở $N \ge 500$ trên T4 16GB do giới hạn SDPA attention.
- ✅ **Frontier API Baselines**: Hoàn tất đánh giá GPT-5.6 Luna và Gemini 3.8 Flash trên toàn bộ 1,600 mẫu CustomTools-VI.
- ✅ **Đồng bộ Tài liệu**: Đồng bộ toàn diện dữ liệu vào [thesis/khoa_luan_tot_nghiep.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md), [latex/uit_thesis/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/latex/uit_thesis/), [paper/paper_vi.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/paper/paper_vi.md) và [latex/paper_en/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/latex/paper_en/).

### 6.2 Các công việc cần tập trung hiện tại
1. [ ] **Rà soát & Hoàn thiện Bản thảo Khóa luận**: Đọc kiểm tra chính tả, câu từ, căn chỉnh bảng biểu trong [latex/uit_thesis/](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/latex/uit_thesis/) và [thesis/khoa_luan_tot_nghiep.md](file:///home/thinh/project/UIT/tool_calling_with_retrieval_extraction/thesis/khoa_luan_tot_nghiep.md) chuẩn bị in nộp.
2. [ ] **Đóng gói mã nguồn & Benchmark**: Kiểm tra tính độc lập của mã nguồn, scripts chạy thực nghiệm, manifest và checksum dữ liệu trước khi công bố repository.
