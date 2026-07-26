# Translation Guidelines — Tool Calling VI

> Quy tắc dịch **EN → VI** cho dataset Tool Calling, dùng với **Qwen-MT (Alibaba, 1M token context)** qua DashScope API.
> Mục tiêu: bảo toàn cấu trúc JSON, bảo vệ identifier kỹ thuật, chỉ dịch phần natural language.

---

## 1. Nguyên tắc tổng quát

Dịch toàn bộ dataset Tool Calling từ tiếng Anh sang tiếng Việt, **bảo toàn tuyệt đối**:

- Cấu trúc JSON (không thêm/bớt key, không đổi thứ tự logic).
- Identifier kỹ thuật (function name, argument keys, brand, proper noun).
- Định dạng gốc (whitespace, quote, indent).

---

## 2. Quy tắc BẮT BUỘC

### 2.1 KHÔNG dịch (giữ nguyên 100%)

| Loại | Ví dụ |
|---|---|
| Function name (snake_case) | `search_tutors`, `book_tutoring_session`, `get_user_profile` |
| Argument keys (snake_case) | `subject`, `location`, `date`, `max_price` |
| Identifier rõ ràng | UUID, mã sản phẩm, mã đơn hàng |
| Tên thương hiệu, sản phẩm, người | `OpenAI`, `ChatGPT`, `iPhone`, `Nguyễn Văn A` |
| Mã tiền tệ | `USD`, `VND`, `EUR` |
| Mã thành phố / quốc gia chuẩn hóa | `Hà Nội`, `TP.HCM`, `US`, `VN` |
| Protocol, library, framework | `HTTP`, `REST`, `JSON`, `OAuth2` |

### 2.2 CHỈ dịch (natural language)

| Loại | Ví dụ |
|---|---|
| User query | `"Tôi muốn tìm gia sư Toán ở Hà Nội."` ← `"I want to find a Math tutor in Hanoi."` |
| Tool description | `"Tìm gia sư theo môn học và khu vực."` ← `"Search for tutors by subject and location."` |
| Argument values (nếu là natural language) | `"Toán"` ← `"Math"` |
| Parameter description (trong schema) | `"Môn học cần tìm"` ← `"Subject to search"` |

### 2.3 Argument values — quyết định tùy ngữ cảnh

- **Có dấu cách / là cụm từ tự nhiên** → dịch (VD: `"Math"` → `"Toán"`, `"Hanoi"` → `"Hà Nội"`).
- **Là identifier chuẩn** (mã, UUID, enum ID) → giữ nguyên (VD: `"USD"`, `"user_123"`).
- **Là số** → giữ nguyên (không đụng).
- **Enum list** → dịch từng phần tử, giữ key.

---

## 3. Ví dụ đầy đủ

### 3.1 Input (EN)

```json
{
  "name": "search_tutors",
  "description": "Search for tutors by subject and location.",
  "parameters": {
    "type": "object",
    "properties": {
      "subject": {
        "type": "string",
        "description": "Subject to search"
      },
      "location": {
        "type": "string",
        "description": "City or region"
      }
    },
    "required": ["subject", "location"]
  }
}
```

### 3.2 Output (VI) — áp dụng guidelines

```json
{
  "name": "search_tutors",
  "description": "Tìm gia sư theo môn học và khu vực.",
  "parameters": {
    "type": "object",
    "properties": {
      "subject": {
        "type": "string",
        "description": "Môn học cần tìm"
      },
      "location": {
        "type": "string",
        "description": "Thành phố hoặc khu vực"
      }
    },
    "required": ["subject", "location"]
  }
}
```

### 3.3 Sample đầy đủ

```json
{
  "query": "Tôi muốn tìm gia sư Toán ở Hà Nội.",
  "label": {
    "function_call": {
      "name": "search_tutors",
      "arguments": {
        "subject": "Toán",
        "location": "Hà Nội"
      }
    }
  },
  "tools_summary": [
    {
      "feature_group": "Tìm kiếm & Kết nối",
      "tools": [
        {"name": "search_tutors", "description": "Tìm gia sư theo môn và khu vực."},
        {"name": "view_tutor_details", "description": "Xem hồ sơ chi tiết của gia sư."}
      ]
    }
  ]
}
```

---

## 4. Trường hợp đặc biệt

### 4.1 Tên riêng Việt Nam

- Tên người Việt: giữ nguyên (không phiên âm, không dịch).
- Địa danh: ưu tiên tên chuẩn Việt (`Hà Nội`, `TP. Hồ Chí Minh`, `Đà Nẵng`).

### 4.2 Tên riêng nước ngoài

- Tên người: giữ nguyên (VD: `John Smith`, `Marie Curie`).
- Địa danh: phiên âm theo chuẩn (`New York` → `New York`, `Paris` → `Paris`).
- Thương hiệu: giữ nguyên (VD: `Google`, `Amazon`).

### 4.3 Câu dài chứa nhiều entity

- Dịch phần natural language, giữ nguyên entity.
- VD: `"Book a flight from Hanoi to Tokyo on Jan 15, 2025"` → `"Đặt vé máy bay từ Hà Nội đi Tokyo vào ngày 15/01/2025"`.

### 4.4 Compound identifier

- Identifier có dấu gạch dưới (`_`): giữ nguyên.
- Identifier có chữ số (`user_123`, `v2`): giữ nguyên.
- Identifier dạng camelCase: giữ nguyên (hiếm gặp trong tool schema, nhưng nếu có thì không dịch).

### 4.5 Unit / số đo

- `"5km"`, `"100USD"`, `"2 hours"`: giữ số, dịch unit (`"2 hours"` → `"2 giờ"`).
- `"25°C"`, `"50kg"`: giữ nguyên (đơn vị khoa học).

---

## 5. Quy trình QC tự động

Sau khi dịch, chạy `src/data/qa_translation.py` để kiểm tra:

1. **Rule-based check**:
   - Function name còn nguyên snake_case (regex `^[a-z][a-z0-9_]*$`).
   - Tất cả key trong JSON output khớp key trong JSON input.
   - JSON parse được.
   - Required fields không bị mất.

2. **LLM judge check** (mẫu nhỏ 5-10%):
   - Cho GPT-4/Gemini đánh giá ngữ nghĩa dịch: query, description, value.
   - Flag sample có vấn đề → sửa tay hoặc dịch lại.

3. **Spot check thủ công**: ~50 sample/batch để phát hiện pattern lỗi.

---

## 6. Khi nào cập nhật guidelines

- Khi phát hiện pattern lỗi dịch mới.
- Khi thêm loại dataset mới (multi-turn, complex schema).
- Khi đổi model dịch (hiện tại: Qwen-MT, có thể thay bằng GPT-4 nếu chất lượng không đủ).
- Khi có review từ chuyên gia tiếng Việt.

Mỗi lần cập nhật: sửa file này + cập nhật `AGENTS.md` Change Log.
