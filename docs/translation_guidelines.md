# Translation Guidelines — Tool Calling VI

> Quy tắc dịch **EN → VI** cho dataset Tool Calling, dùng với **Qwen-MT (Alibaba)** qua OpenAI-compatible API.
> Mục tiêu: bảo toàn cấu trúc JSON, bảo vệ identifier kỹ thuật, chỉ dịch phần natural language.

## 1. Kiến trúc 2 bộ (cập nhật 2026-08-03)

> **Quyết định**: Tách thành 2 bộ riêng biệt. Schema master là single-turn + multi-call.

| Bộ | Path | Format | Vai trò |
|---|---|---|---|
| **Bộ 1 (dịch)** | `data/translations/` | Gần raw (giữ `chat` text cho Glaive, `answers` JSON list cho xLAM); chỉ thay natural language | LLM dịch dễ, ít pre-processing |
| **Bộ 2 (task)** | `data/benchmark_vi/` | Single-turn schema master: `{id, source, query, function_calls[], tools[]}` | Train + eval Method 1 (SLM) + Method 2 (Bi+Cross) |

**Tại sao tách 2 bộ**:
- Bộ 1 giữ format gần raw → LLM dịch ít rủi ro corrupt JSON, ít mất ngữ nghĩa.
- Bộ 2 parse + restructure từ Bộ 1 → schema task riêng, độc lập hoàn toàn với raw.
- Bộ 1 hỏng → re-translate; Bộ 2 không bị ảnh hưởng.
- 2 codebase độc lập → dễ maintain, dễ test.

## 2. Nguyên tắc tổng quát

Dịch toàn bộ dataset Tool Calling từ tiếng Anh sang tiếng Việt, **bảo toàn tuyệt đối**:

- Cấu trúc JSON (không thêm/bớt key, không đổi thứ tự logic).
- Identifier kỹ thuật (function name, argument keys, brand, proper noun).
- Định dạng gốc (whitespace, quote, indent).

## 3. Quy tắc BẮT BUỘC

### 3.1 KHÔNG dịch (giữ nguyên 100%)

| Loại | Ví dụ |
|---|---|
| Function name (snake_case) | `search_tutors`, `book_tutoring_session`, `get_user_profile` |
| Argument keys (snake_case) | `subject`, `location`, `date`, `max_price` |
| Identifier rõ ràng | UUID, mã sản phẩm, mã đơn hàng |
| Tên thương hiệu, sản phẩm, người | `OpenAI`, `ChatGPT`, `iPhone`, `Nguyễn Văn A` |
| Mã tiền tệ | `USD`, `VND`, `EUR` |
| Mã thành phố / quốc gia chuẩn hóa | `Hà Nội`, `TP.HCM`, `US`, `VN` |
| Protocol, library, framework | `HTTP`, `REST`, `JSON`, `OAuth2` |
| Số, đơn vị khoa học | `25°C`, `50kg`, `3.14` |

### 3.2 CHỈ dịch (natural language)

| Loại | Ví dụ |
|---|---|
| User query | `"Tôi muốn tìm gia sư Toán ở Hà Nội."` ← `"I want to find a Math tutor in Hanoi."` |
| Assistant text (sau function call) | `"Tôi tìm được 2 gia sư phù hợp."` |
| Function response (text) | `"Có 5 chuyến bay từ Hà Nội đi Tokyo..."` |
| Tool description | `"Tìm gia sư theo môn học và khu vực."` |
| Parameter description | `"Môn học cần tìm"` |
| Argument values (nếu natural language) | `"Math"` → `"Toán"`, `"Hanoi"` → `"Hà Nội"` |

### 3.3 Argument values — quyết định tùy ngữ cảnh

- **Có dấu cách / cụm từ tự nhiên** → dịch (VD: `"Math"` → `"Toán"`, `"Hanoi"` → `"Hà Nội"`).
- **Là identifier chuẩn** (mã, UUID, enum ID) → giữ nguyên (VD: `"USD"`, `"user_123"`).
- **Là số** → giữ nguyên (không đụng).
- **Enum list** → dịch từng phần tử, giữ key.

## 4. Format Bộ 1 (input/output cho translate.py)

### 4.1 Glaive (Bộ 1)

**Input EN**:
```json
{
  "system": "SYSTEM: {\"name\": \"search_tutors\", \"description\": \"Search for tutors by subject and location.\", \"parameters\": {...}}",
  "chat": "USER: I want to find a Math tutor in Hanoi.\n\nA: <functioncall> {\"name\": \"search_tutors\", \"arguments\": '{\"subject\": \"Math\", \"location\": \"Hanoi\"}'} <|endoftext|>\n\nFUNCTION RESPONSE: [{...tutor 1...}, {...tutor 2...}]\n\nA: I found 2 tutors matching your criteria. <|endoftext|>"
}
```

**Output VI** (Bộ 1 — giữ format gần raw):
```json
{
  "system": "SYSTEM: {\"name\": \"search_tutors\", \"description\": \"Tìm gia sư theo môn học và khu vực.\", \"parameters\": {...}}",
  "chat": "USER: Tôi muốn tìm gia sư Toán ở Hà Nội.\n\nA: <functioncall> {\"name\": \"search_tutors\", \"arguments\": '{\"subject\": \"Toán\", \"location\": \"Hà Nội\"}'} <|endoftext|>\n\nFUNCTION RESPONSE: [...dữ liệu VI...]\n\nA: Tôi tìm được 2 gia sư phù hợp. <|endoftext|>"
}
```

→ Dịch: `system` JSON value (description), `chat` text (USER, A, FUNCTION RESPONSE).
→ Giữ: function name, argument key, JSON structure, `<functioncall>` tag, `<|endoftext|>` tag.

### 4.2 xLAM (Bộ 1)

**Input EN**:
```json
{
  "id": 0,
  "query": "Where can I find live giveaways for beta access and games?",
  "answers": "[{\"name\": \"live_giveaways_by_type\", \"arguments\": {\"type\": \"beta\"}}, ...]",
  "tools": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Retrieve live giveaways from the GamerPower API based on the specified type.\", \"parameters\": {\"type\": {\"description\": \"The type of giveaways to retrieve (e.g., game, loot, beta).\", \"type\": \"str\", \"default\": \"game\"}}}]"
}
```

**Output VI** (Bộ 1):
```json
{
  "id": 0,
  "query": "Tôi có thể tìm các chương trình tặng quà trực tiếp cho quyền truy cập beta và trò chơi ở đâu?",
  "answers": "[{\"name\": \"live_giveaways_by_type\", \"arguments\": {\"type\": \"beta\"}}, ...]",
  "tools": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Truy xuất các chương trình tặng quà trực tiếp từ GamerPower API theo loại được chỉ định.\", \"parameters\": {\"type\": {\"description\": \"Loại chương trình tặng quà cần truy xuất (VD: trò chơi, vật phẩm, beta).\", \"type\": \"str\", \"default\": \"game\"}}}]"
}
```

→ Dịch: `query` value, `tools` JSON value (description, parameter description).
→ Giữ: function name, argument key, enum value, JSON structure, `type: "str"`.

## 5. Translation pipeline (Bộ 1 → Bộ 1 VI)

### 5.1 Tham số

| Tham số | Value | Ghi chú |
|---|---|---|
| Model | `${ALIBABA_MODEL}` | Translation, qua `ALIBABA_URL` |
| Batch size K | 10 | Samples per request |
| Concurrency | 8 | asyncio.Semaphore |
| Retry | 3 | Exponential backoff (1s, 2s, 4s) |
| Validate | Per-sample | Rule check ngay khi response về |
| Output | Append JSONL | Flush per sample + `os.fsync()` |
| Resume | Atomic checkpoint | `data/translations/.checkpoint/<ds>.json` |

### 5.1.1 Input filtering and feature-group side-output

- Glaive: lọc trước từ `data/raw/glaive_raw.jsonl` thành `data/processed/glaive_single_turn_raw.jsonl`, giữ nguyên format `system/chat` và chỉ giữ positive first-turn samples.
- xLAM: giữ toàn bộ samples vì đã flat single-turn; giữ nguyên multi-call.
- Translation chạy với `feature_group.enabled: false`; feature group được classify một lần trên unique tool pool sau khi build Bộ 2.
- Lý do: không để classifier làm tăng token và không để lỗi classifier ảnh hưởng translation checkpoint.

- `translate.py` trước đây có thể pre-label `feature_group` ngay trong cùng job dịch.
- Nhãn này không nằm trong JSON dịch của sample, mà được ghi vào cache riêng `data/benchmark_vi/.cache/feature_group.json`.
- Mục tiêu: giảm số lần gọi LLM khi build benchmark và stress test.
- Nếu cache đã có tool name thì không gọi lại.

### 5.2 Prompt structure (gửi cho Qwen-MT)

```
SYSTEM: Bạn là chuyên gia dịch Anh-Việt cho dataset Tool Calling. Quy tắc:
  - Dịch natural language sang tiếng Việt tự nhiên
  - GIỮ NGUYÊN: function name, argument keys, JSON structure, identifier kỹ thuật
  - KHÔNG dịch: số, mã tiền tệ, enum values, brand names

USER: Translate the following JSON sample to Vietnamese. Keep all keys, identifiers, and JSON structure unchanged. Only translate natural language values.

```json
{ "system": "...", "chat": "..." }
```

Output ONLY the translated JSON. Do NOT add explanation.


### 5.3 Validate rule (per-sample, ngay khi response về)

1. **JSON parse được**: `json.loads(response)` không exception.
2. **Top-level keys preserved**: Glaive có `system`, `chat`; xLAM có `id`, `query`, `answers`, `tools`.
3. **Function name integrity**: Tất cả function name trong output match input (regex `^[a-z][a-z0-9_]*$`).
4. **Argument keys integrity**: Tất cả keys trong `arguments` object match input.
5. **No new keys added**: Output không có key lạ (so với input schema).
6. **Content không rỗng**: query/description không bị `""` hoặc `null`.

Nếu fail bất kỳ check nào → retry 3 lần → cuối cùng ghi `failed/<ds>_failed.jsonl`.

## 6. Quy trình QC

Sau khi dịch, chạy `src/data/qa_translation.py` để kiểm tra:

### 6.1 Rule-based check (100% sample)
- Function name còn nguyên snake_case (regex `^[a-z][a-z0-9_]*$`).
- Tất cả key trong JSON output khớp key trong JSON input.
- JSON parse được.
- Required fields không bị mất.

### 6.2 LLM judge check (5% sample random)
- Dùng `qwen3.7-max` đánh giá ngữ nghĩa dịch.
- Flag sample có vấn đề → sửa tay hoặc dịch lại.

### 6.3 Spot check thủ công
~50 sample/batch để phát hiện pattern lỗi.

## 7. Từ Bộ 1 → Bộ 2

`src/data/build_benchmark.py` parse Bộ 1 (VI) → schema master single-turn:

1. **Glaive Bộ 1** → parse first turn only:
   - Extract `USER: ...` → `query`
   - Extract `A: <functioncall> {...} <|endoftext|>` → parse JSON → `function_calls[]`
   - Parse `system` JSON → `tools[]` (description, parameters)
   - Bỏ qua FUNCTION RESPONSE và các turn sau.

2. **xLAM Bộ 1** → parse JSON lists:
   - `query` → `query`
   - `answers` JSON list → `function_calls[]`
   - `tools` JSON list → `tools[]` (giữ nguyên sau khi normalize type)

3. **Normalize xLAM type**: `src/data/normalize_schema.py`:
   - `str` → `string`
   - `int` → `integer`
   - `float` → `number`
   - `bool` → `boolean`
   - `list` / `List[T]` → `array` + `items: {type: T}`
   - Tách `, optional` suffix → `required: []`

4. **Feature group**: `src/data/feature_group_classify.py` LLM classify 1 lần/tool, cache.

5. **Split**: 80/10/10, seed=42.

6. **Convert cho Method 1**: `src/data/convert_to_instruction.py` chuyển schema master → LLaMA-Factory instruction format.

## 8. Trường hợp đặc biệt

### 8.1 Tên riêng Việt Nam
- Tên người Việt: giữ nguyên (không phiên âm, không dịch).
- Địa danh: ưu tiên tên chuẩn Việt (`Hà Nội`, `TP. Hồ Chí Minh`, `Đà Nẵng`).

### 8.2 Tên riêng nước ngoài
- Tên người: giữ nguyên (VD: `John Smith`).
- Địa danh: phiên âm theo chuẩn (`New York` → `New York`, `Paris` → `Paris`).
- Thương hiệu: giữ nguyên (VD: `Google`, `Amazon`).

### 8.3 Câu dài chứa nhiều entity
Dịch phần natural language, giữ nguyên entity.
VD: `"Book a flight from Hanoi to Tokyo on Jan 15, 2025"` → `"Đặt vé máy bay từ Hà Nội đi Tokyo vào ngày 15/01/2025"`.

### 8.4 Compound identifier
- Có dấu gạch dưới (`_`): giữ nguyên.
- Có chữ số (`user_123`, `v2`): giữ nguyên.
- camelCase: giữ nguyên (hiếm gặp).

### 8.5 Unit / số đo
- `"5km"`, `"100USD"`, `"2 hours"`: giữ số, dịch unit (`"2 hours"` → `"2 giờ"`).
- `"25°C"`, `"50kg"`: giữ nguyên (đơn vị khoa học).

## 9. Khi nào cập nhật guidelines
- Khi phát hiện pattern lỗi dịch mới.
- Khi thêm loại dataset mới (multi-turn, complex schema).
- Khi đổi model dịch (hiện tại: Qwen-MT, có thể thay bằng GPT-4 nếu chất lượng không đủ).
- Khi có review từ chuyên gia tiếng Việt.

Mỗi lần cập nhật: sửa file này + cập nhật `AGENTS.md` Change Log.
