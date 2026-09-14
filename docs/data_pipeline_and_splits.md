# Quy Trình Xây Dựng Dữ Liệu Và Phân Bổ Ngân Sách Huấn Luyện (E0 – E4)

Tài liệu này chuẩn hóa toàn bộ vòng đời dữ liệu trong đề tài **Tool Calling tiếng Việt**: từ thu thập nguồn thô, chuẩn hóa cấu trúc, dịch thuật, kiểm định chất lượng (QA), khử trùng lặp kịch bản (scenario deduplication), đóng băng phiên bản benchmark (frozen canonical revision), cho đến công thức phân chia ngân sách huấn luyện cho các thực nghiệm **E0 – E4** (Method 1) và biểu diễn cặp dữ liệu (Method 2).

---

## 1. Tổng quan Vòng đời Dữ liệu

```
[1. Thu thập dữ liệu thô]
   ├── Glaive Function Calling v2: 112,960 records
   └── xLAM-60k (Salesforce)     :  60,000 records
                 │
                 ▼
[2. Lọc single-turn & Chuẩn hóa Schema]
   ├── Glaive positive (first-turn) : 45,593 records
   ├── Glaive negative (no-call)    : 15,141 records
   └── xLAM positive (single-turn)  : 60,000 records
                 │
                 ▼
[3. Dịch thuật & QA chất lượng]
   ├── API Qwen-MT (Alibaba Cloud) với cơ chế Retry & Fallback Chain
   ├── Rule-based Validation (JSON syntax, param keys, schema consistency)
   └── LLM Judge QC trên 5% mẫu ngẫu nhiên
                 │
                 ▼
[4. Khử trùng lặp & Đóng băng phiên bản (Frozen Revision)]
   ├── Khử trùng lặp user query và scenario overlap giữa các split
   ├── Ghép cặp song ngữ 1-1 (EN - VI) dựa trên `id`
   └── Frozen Canonical Revision: `2026-09-02-full-dedup-seed42` (77,028 records)
                 │
                 ▼
[5. Phân chia Train / Val / Test (80 / 10 / 10)]
   ├── Train : 61,615 records (80%)
   ├── Val   :  7,701 records (10%)
   └── Test  :  7,712 records (10%) (7,221 positive + 491 negative)
                 │
        ┌────────┴──────────────────────────┐
        ▼                                   ▼
[Method 1: SLM Generative]          [Method 2: Bi+Cross Encoder]
- Main Controlled Track (60k)       - Bi-Encoder: 78,435 train pairs
- Native Chat Template              - Cross-Encoder: 145,383 param pairs
- Response-only Loss                - Threshold Calibration trên Val
```

---

## 2. Thu Thập Dữ Liệu Thô (Raw Data)

Đề tài sử dụng hai nguồn dữ liệu tool-calling quy mô lớn chuẩn quốc tế:

1. **Glaive Function Calling v2** (`112,960` records):
   - Đặc điểm: Dạng hội thoại multi-turn (chiếm ~74%), chỉ có 1 tool định nghĩa trong `system prompt`.
   - Xử lý: Áp dụng `filter_single_turn.py`, chỉ giữ lại lượt tương tác đầu tiên (first-turn) của user và assistant.
   - Kết quả: Thu được **45,593 mẫu positive** (có gọi hàm) và **15,141 mẫu negative** (hội thoại thông thường hoặc câu hỏi từ chối gọi hàm / no-call).
2. **Salesforce xLAM-60k** (`60,000` records):
   - Đặc điểm: Dữ liệu phẳng, đa cuộc gọi (multi-call chiếm ~53%), cung cấp toàn bộ tool pool cho mỗi mẫu.
   - Xử lý: Giữ nguyên 60,000 mẫu positive single-turn.

Tổng số mẫu đầu vào được đưa vào chuẩn hóa: $45,593 + 15,141 + 60,000 = \mathbf{120,734}$ **mẫu**.

---

## 3. Chuẩn Hóa Schema Master (Canonical Schema)

Toàn bộ dữ liệu được chuẩn hóa về định dạng Schema Master JSON thống nhất:

```json
{
  "id": "xlam_00123",
  "source": "xlam",
  "query": "Tìm chuyến bay từ Hà Nội đi Đà Nẵng vào ngày mai.",
  "function_calls": [
    {
      "name": "search_flights",
      "arguments": {
        "departure": "Hà Nội",
        "destination": "Đà Nẵng",
        "date": "2026-09-13"
      }
    }
  ],
  "tools": [
    {
      "name": "search_flights",
      "description": "Tìm kiếm các chuyến bay nội địa và quốc tế theo ngày.",
      "feature_group": "Đặt vé & Du lịch",
      "parameters": {
        "type": "object",
        "properties": {
          "departure": {"type": "string", "description": "Thành phố khởi hành"},
          "destination": {"type": "string", "description": "Thành phố đến"},
          "date": {"type": "string", "description": "Ngày bay (YYYY-MM-DD)"}
        },
        "required": ["departure", "destination", "date"]
      }
    }
  ]
}
```

### Quy tắc chuẩn hóa kiểu dữ liệu
- Mọi kiểu dữ liệu không chuẩn từ xLAM (ví dụ: `str`, `int`, `float`, `list`, `List[int]`, `bool, optional`) đều được ánh xạ về chuẩn **JSON Schema Draft 7**: `string`, `integer`, `number`, `boolean`, `array`, `object`.
- Hậu tố `, optional` được tách thành mảng `required: []` cấp tool (thuộc tính không nằm trong `required` được hiểu là tùy chọn).
- Thuộc tính liệt kê (Enum) dùng `type: "string"` kèm mảng `enum: [...]`.

---

## 4. Pipeline Dịch Thuật Tự Động & QA (Translation Pipeline)

### 4.1 Quy tắc Dịch thuật Bất biến (`docs/translation_guidelines.md`)
Nhằm bảo đảm tính tương thích tuyệt đối cho lời gọi hàm khi thực thi máy tính:
* **CHỈ DỊCH**:
  * `query`: Câu hỏi / yêu cầu của người dùng.
  * `tools[].description`: Mô tả công dụng của công cụ.
  * `tools[].parameters.properties[].description`: Mô tả tham số.
  * `function_calls[].arguments` values: Chỉ dịch nếu giá trị là ngôn ngữ tự nhiên thông thường (ví dụ: câu trích dẫn, từ khóa tìm kiếm tiếng Việt).
* **TUYỆT ĐỐI KHÔNG DỊCH**:
  * `function_calls[].name` và `tools[].name`: Tên hàm định danh (snake_case).
  * `function_calls[].arguments` keys và `properties` keys: Tên tham số.
  * Các định danh kỹ thuật, mã tiền tệ (USD, VND), tên hàm mã hóa, URL, UUID, giá trị enum cố định.

### 4.2 Cơ sở Hạ tầng & Fallback Chain
- **Engine chính**: Alibaba Cloud OpenAI-compatible API (`qwen3.7-flash` hoặc `qwen3.7-max`).
- **Batching & Concurrency**: Lô $K=10$ mẫu, 8 worker bất đồng bộ (`asyncio`).
- **Dung sai lỗi (Retry & Fallback Chain)**:
  - Tối đa 3 lần thử lại với hàm số mũ ngắt quãng (exponential backoff).
  - Tự động chuyển đổi giữa danh sách 74 model dự phòng (`models.txt`) nếu gặp giới hạn quota (RateLimit) hoặc lỗi máy chủ 5xx.
- **QA & Phê duyệt**:
  - Tự động kiểm tra cú pháp JSON, tính nguyên vẹn của danh sách tham số và định danh tiếng Anh.
  - LLM Judge đánh giá chất lượng ngữ nghĩa trên 5% mẫu ngẫu nhiên.

---

## 5. Khử Trùng Lặp & Đóng Băng Phiên Bản (Frozen Canonical Revision)

Sau khi dịch thuật, dữ liệu trải qua bước kiểm định nghiêm ngặt:
1. **Scenario Deduplication**: Phát hiện và loại bỏ các mẫu có câu hỏi tương tự hoặc cùng kịch bản trùng lặp để tránh rò rỉ dữ liệu giữa tập train và tập test.
2. **Loại bỏ mẫu hỏng**: 944 mẫu không thỏa mãn schema hoặc lỗi cú pháp đối số bị đào thải.
3. **Ghép cặp song ngữ (Bilingual Pairing)**: Mỗi bản ghi tiếng Việt được ghép cặp 1-1 với bản ghi tiếng Anh tương ứng qua trường `id`.
4. **Đóng băng phiên bản (Freeze Revision)**:
   - Tên phiên bản: `2026-09-02-full-dedup-seed42` (lưu tại `data/benchmark_core/2026-09-02-full-dedup-seed42/`).
   - Tổng số bản ghi ghép cặp hợp lệ: **77,028 records** (`18,210` Glaive + `58,818` xLAM).
   - Tổng số mẫu negative (no-call): **4,817 records**.
   - Tổng số công cụ duy nhất (unique tools): **4,421 tools**.

### Phân chia Split 80 / 10 / 10 (Cố định Seed 42)
Bản ghi mapping được ghi bất biến trong file `split_manifest.json`:

| Tập | Tỷ lệ | Số lượng mẫu (EN) | Số lượng mẫu (VI) | Ghi chú |
|---|:---:|:---:|:---:|---|
| **Train** | 80% | 61,615 | 61,615 | Dùng để trích xuất tập huấn luyện |
| **Validation** | 10% | 7,701 | 7,701 | Dùng cho Model Selection / Diagnostic |
| **Test** | 10% | 7,712 | 7,712 | 7,221 positive + 491 negative |

---

## 6. Thiết Kế Ngân Sách Huấn Luyện Main Controlled Track (E0 – E4)

### 6.1 Cơ sở lý thuyết: Controlled Experiment (Ersoy et al., 2025)
Trong các nghiên cứu khoa học về đa ngôn ngữ và chuyển giao tri thức (cross-lingual transfer), nếu số lượng mẫu huấn luyện giữa các cấu hình thay đổi thì **hiệu năng tăng lên có thể do dung lượng dữ liệu lớn hơn chứ không phải do ngôn ngữ**. 

Do đó, đề tài thiết lập **Main Controlled Track với ngân sách cố định 60,000 mẫu** cho E1, E2 và E3:
$$\text{Budget}(E1) = \text{Budget}(E2) = \text{Budget}(E3) = 60,000 \text{ examples}$$

Từ 61,615 mẫu trong `train.jsonl`, thuật toán `_take_quota` (`src/data/prepare_experiments.py`) trích xuất chính xác 60,000 mẫu bằng seed cố định (`seed=42`).

### 6.2 Chi tiết 5 Thực nghiệm (E0 – E4)

```
E0: Base Model (Zero-shot) ──────────────────────────► 0 train samples
E1: Monolingual EN         ──► 60,000 mẫu EN ─────────► SFT baseline
E2: Monolingual VI         ──► 60,000 mẫu VI (cùng ID) ► So sánh EN vs VI
E3: Bilingual EN-VI        ──► 30k EN + 30k VI ───────► Đánh giá song ngữ
E4: Domain Adaptation      ──► 60k E3 + 5.6k CustomVI ─► Đánh giá miền VN
```

| Experiment | Tổng mẫu | Thành phần ngôn ngữ | Phân bổ nhãn | Nguồn dữ liệu |
|---|:---:|---|---|---|
| **E0** | 0 | Không huấn luyện | — | Base model `unsloth/Qwen3.5-2B` hoặc `4B` |
| **E1** | 60,000 | 100% Tiếng Anh | 56,151 positive<br>3,849 negative | Glaive: 10,712 pos<br>xLAM: 45,439 pos |
| **E2** | 60,000 | 100% Tiếng Việt | 56,151 positive<br>3,849 negative | Dùng **chính xác các sample ID của E1** nhưng ở bản dịch VI |
| **E3** | 60,000 | 50% EN + 50% VI<br>(30k EN + 30k VI) | Mỗi ngôn ngữ gồm:<br>27,000 positive<br>3,000 negative | Cùng seed trích xuất 30,000 cặp ID song ngữ |
| **E4** | 65,600 | 60,000 mẫu của E3<br>+ 5,600 CustomTools-VI | 60k song ngữ E3<br>+ 3,600 pos Custom<br>+ 2,000 neg Custom | Dành riêng cho khảo sát thích ứng ngữ cảnh Việt Nam |

---

## 7. Bộ Dữ Liệu Miền Đặc Thù: `CustomTools-VI`

Để kiểm tra khả năng gọi công cụ trong đời sống thực tế tại Việt Nam và năng lực tổng quát hóa zero-shot với công cụ chưa từng thấy (unseen), đề tài xây dựng bộ dữ liệu độc lập `CustomTools-VI`:
* **Quy mô**: 8,000 mẫu được xây dựng thủ công và kiểm duyệt chuyên gia (4,800 positive + 3,200 negative).
* **Cơ cấu**: 40 công cụ thuộc **10 nhóm chức năng đặc trưng Việt Nam** (Ví dụ: Tra cứu phạt nguội giao thông, xem lịch âm, đặt xe công nghệ, kiểm tra số dư VietQR, tra cứu giá vàng SJC, ...).
* **Phân chia Seen vs. Strict Unseen**:
  * **Seen Tools (20 tools)**: Đã xuất hiện trong tập huấn luyện của E4.
  * **Unseen Tools (20 tools)**: Hoàn toàn bị loại khỏi tập huấn luyện (zero exposure trong cả training và negative mining).

| Phân vùng dữ liệu | Số lượng mẫu | Phân bổ nhãn | Mục đích sử dụng |
|---|:---:|:---:|---|
| `train.jsonl` | 5,600 | 3,600 pos + 2,000 neg | Huấn luyện domain-specific cho E4 (Method 1) và Custom pairs (Method 2) |
| `val_seen.jsonl` | 400 | 200 pos + 200 neg | Diagnostic / Validation cho tool đã gặp |
| `val_unseen.jsonl` | 400 | 200 pos + 200 neg | Diagnostic / Validation cho tool chưa gặp |
| `test_seen.jsonl` | 800 | 400 pos + 400 neg | **Đánh giá chính thức hiệu năng trên tool tiếng Việt đã gặp** |
| `test_unseen.jsonl` | 800 | 400 pos + 400 neg | **Đánh giá chính thức năng lực Zero-shot trên Unseen Tools** |

---

## 8. So Sánh Định Dạng Biểu Diễn: Method 1 vs. Method 2

Cùng xuất phát từ Schema Master, hai phương pháp chuyển đổi dữ liệu sang định dạng đầu vào khác nhau:

### 8.1 Method 1: Generative SFT (Qwen3.5)
- Dữ liệu được chuyển đổi qua `src/data/convert_to_instruction.py` thành cấu trúc hội thoại đa lượt native của OpenAI/Qwen:
  ```json
  {
    "messages": [
      {"role": "system", "content": "Bạn là trợ lý AI có khả năng sử dụng công cụ."},
      {"role": "user", "content": "Query của người dùng..."},
      {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}]}
    ],
    "tools": [...]
  }
  ```
- **Hàm mất mát (Loss)**: Áp dụng `AssistantOnlyCollator`, mặt nạ nhãn (label masking) toàn bộ system prompt, user prompt và tool definitions, **chỉ tính Cross-Entropy Loss trên các token do assistant sinh ra** (khớp với thiết kế của Unsloth).

### 8.2 Method 2: Modular Pipeline (Bi-Encoder + Cross-Encoder)
- **Bi-Encoder (BGE-M3 Retrieval)**:
  - Chuyển đổi thành các cặp truy xuất ngữ nghĩa: `(query, tool_description)`.
  - Áp dụng `CachedMultipleNegativesRankingLoss` (CachedMNRL) với 2 round (Round 1 làm giáo viên khai thác hard negatives cho Round 2). Huấn luyện trên **78,435 training pairs**.
- **Cross-Encoder (XLM-RoBERTa-base Parameter Extraction)**:
  - Chuyển đổi thành các bộ ba: `(query, parameter_schema, target_value)`.
  - Áp dụng cấu trúc BERT-QA: query làm ngữ cảnh (context), mô tả tham số làm câu hỏi (question).
  - Huấn luyện trên **145,383 parameter pairs** với curriculum learning.
