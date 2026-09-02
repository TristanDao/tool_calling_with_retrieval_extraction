# Kế Hoạch Rebuild Data, Dọn Artifact Và Train Qwen3.5 Với Unsloth

## 1. Phạm Vi Và Mục Tiêu

Plan này thay thế flow data cũ đang bị chồng chéo giữa `data/benchmark_vi/` và `data/experiments/`.

Mục tiêu:

- Tạo lại benchmark từ nguồn dữ liệu đã QA, có cả positive và negative.
- Chỉ split core một lần, ghép cặp EN/VI theo cùng `id` và cùng split.
- Dùng một evaluation suite cố định cho mọi experiment và method.
- Xóa các artifact sinh lại, test copy và train data không cần thiết.
- Giữ dữ liệu nguồn cần thiết để audit, không xóa mù hoặc mất khả năng reproduce.
- Dùng native Qwen3.5 chat template với Unsloth, không dùng format tool-call tự chế theo paper.

Trạng thái: plan đã được thực thi ngày 2026-09-02. Frozen revision, active VI export,
cleanup archive và train-only experiment artifacts đã được materialize; bước còn phụ
thuộc môi trường GPU là template smoke test và training thực tế.

## 2. Quyết Định Đã Chốt

### 2.1 Benchmark cũ

Snapshot cũ không được sử dụng trong kết quả chính vì:

- Builder cũ chỉ nhận sample có `function_calls` khác rỗng.
- Snapshot cũ thiếu negative samples.
- Split cũ đã phát hiện duplicate query/scenario giữa các split.
- `prepare_experiments.py` phiên bản cũ không đọc snapshot này mà tự split lại từ normalized/translation inputs.
- Một số file train/val có thể thiếu hoặc không còn đồng bộ sau các lần rebuild trước.

Snapshot cũ đã được archive read-only dưới `data/legacy/pilot_20260902T000000Z/`,
kèm cleanup manifest để audit. Các file generated đã được dọn khỏi path active và
benchmark mới đã được build lại. `data/raw/`, `data/normalized_en/`, bản dịch đã QA
và `data/custom_vi/` được giữ nguyên.

### 2.2 Benchmark mới

Benchmark mới là một revision frozen, không append dồn vào file cũ và không random split lại cho từng E.

Revision được xây từ:

- Core English normalized data.
- Core Vietnamese translations tương ứng.
- Core negative data và bản dịch tương ứng.
- CustomTools-VI giữ thành dataset riêng, không trộn test vào core.

`data/benchmark_vi/` vẫn là active Vietnamese benchmark export. Nó không bị bỏ. Nó đã
được tạo lại từ revision đã freeze, thay vì là một split độc lập.

### 2.3 Method 1

Chốt checkpoint chính:

```text
unsloth/Qwen3.5-4B
```

Nếu chạy so sánh model size:

```text
unsloth/Qwen3.5-2B
```

Chốt trainer là Unsloth QLoRA/SFT. Không dùng LLaMA-Factory.

Training view chính là native `messages/role/content` và `tool_calls`, sau đó được
format bằng `apply_chat_template()` lấy trực tiếp từ checkpoint. ShareGPT không nằm
trong training path.

## 3. Kiểm Kê Data Experiment Hiện Tại

Inventory dưới đây ghi lại cấu trúc trước migration, làm cơ sở giải thích cleanup:

| Experiment | File hiện tạo | Số file hiện thấy | Vấn đề |
|---|---|---:|---|
| E0 | `val_en`, `val_vi`, `test_en`, `test_vi`, `instruction/val_*`, `custom_val_*`, `custom_test_*`, `manifest` | 12 | Zero-shot nhưng bị copy validation/test vào folder |
| E1 | `train_en`, `train`, `instruction/train_*`, `val_*`, `test_*`, `custom_*`, `manifest` | 16 | Chỉ cần train artifact + manifest |
| E2 | `train_vi`, `train`, `instruction/train_*`, `val_*`, `test_*`, `custom_*`, `manifest` | 16 | Chỉ cần train artifact + manifest |
| E3 | Giống E1 | 16 | Hiện chỉ là bản sao E1, không có general-SFT stage |
| E4 | `train_en`, `train_vi`, `train`, nhiều instruction files, `val_*`, `test_*`, `custom_*`, `manifest` | 18 | Có nhiều bản ghép/copy không cần thiết |
| E5 | E4 + `train_custom_vi`, `train_custom_vi_chat` và các copy eval | 20 | Chỉ CustomTools train cần nằm trong E5 |

Chi tiết các file hiện không cần giữ trong từng E:

- `test_en.jsonl`, `test_vi.jsonl`: test dùng chung từ frozen benchmark.
- `custom_test_seen.jsonl`, `custom_test_unseen.jsonl`: test dùng chung từ `data/custom_vi/`.
- `val_en.jsonl`, `val_vi.jsonl`: trainer đọc từ shared revision theo validation policy.
- `custom_val_seen.jsonl`, `custom_val_unseen.jsonl`: chỉ dùng model selection/diagnostic, không cần copy.
- `instruction/val_en_chat.jsonl`, `instruction/val_vi_chat.jsonl`, `instruction/val_chat.jsonl`: tạo on-demand hoặc đọc shared validation rồi format tại trainer.
- `instruction/train_en_chat.jsonl`, `instruction/train_vi_chat.jsonl`, `instruction/train_custom_vi_chat.jsonl`: chỉ giữ nếu cần audit từng nguồn; file train native hợp nhất mới là input chính.
- `train.jsonl`: giữ một file train native hợp nhất cho mỗi E; các file train theo ngôn ngữ chỉ là optional audit views.
- `manifest.json`: bắt buộc giữ.

## 4. Phân Loại Data Cần Giữ Và Dọn

### 4.1 Giữ làm nguồn hoặc audit

| Path | Quyết định | Lý do |
|---|---|---|
| `data/raw/` | Giữ | Nguồn tải gốc, cần cho audit/rebuild |
| `data/normalized_en/` | Giữ | Input normalized để tạo paired core data |
| `data/translations/` | Giữ | Kết quả dịch và checkpoint/QA cần thiết |
| `data/translations/failed/` | Giữ đến khi full translation kết thúc | Theo dõi sample lỗi và resume |
| `data/custom_vi/` | Giữ nguyên | Dataset độc lập có train/val/test riêng |
| `data/translations/qwen_mt_logs/` | Giữ | Audit chất lượng và model fallback |
| `data/translations/qa_samples/` | Giữ | Audit kết quả QA |

### 4.2 Archive hoặc xóa rồi sinh lại

| Path | Quyết định | Cách xử lý |
|---|---|---|
| `data/benchmark_vi/` cũ | Đã archive và export lại | Active path hiện là VI export của revision mới |
| `data/benchmark_vi/tool_schema/` cũ | Đã sinh lại | Phản ánh tool pool của frozen revision |
| `data/benchmark_vi/tool_pool.json` cũ | Đã sinh lại | Phủ toàn bộ core revision |
| `data/benchmark_vi/.cache/feature_group.json` cũ | Đã thay bằng cache revision | Cache hiện có cả giá trị fallback theo tool |
| `data/benchmark_vi/instruction/` cũ | Đã loại khỏi active path | Native train view nằm trong từng experiment |
| `data/experiments/e0...e5/` cũ | Đã archive và materialize lại | Chỉ còn E0 và E1/E2/E4/E5 theo policy mới |
| `data/statistics/` cũ | Đã archive | Thống kê mới lấy từ frozen revision |

Trước khi xóa phải tạo `cleanup_manifest.json` gồm path, file count, byte size, SHA-256 và lý do xử lý. Nếu cần giữ toàn bộ pilot, lưu archive read-only bên ngoài active benchmark path; archive không được dùng trong training/evaluation.

## 5. Cấu Trúc Đích

### 5.1 Frozen benchmark revision

```text
data/benchmark_core/<revision>/
  manifest.json
  split_manifest.json
  en/
    train.jsonl
    val.jsonl
    test.jsonl
  vi/
    train.jsonl
    val.jsonl
    test.jsonl
  tool_pool.json
  tool_schema/
```

`manifest.json` phải ghi input paths/hash, builder commit, dedup rule, split seed/tỷ lệ, counts positive/negative theo source/language, unique tools và hash output files.

`split_manifest.json` chứa mapping cố định:

```json
{
  "glaive_00042": "train",
  "xlam_00123": "test"
}
```

`data/benchmark_vi/` là active export của `vi/{train,val,test}.jsonl` trong revision. Không được chạy một split khác tại path này.

### 5.2 Shared evaluation

Evaluation không nằm trong từng E:

```text
data/benchmark_core/<revision>/en/{val,test}.jsonl
data/benchmark_core/<revision>/vi/{val,test}.jsonl
data/custom_vi/{val_seen,val_unseen,test_seen,test_unseen}.jsonl
```

Vai trò:

- `vi/test`: test chính cho bảng so sánh 4 methods.
- `en/test`: test phụ cho transfer EN -> VI của Method 1.
- `custom_vi/test_seen`: đánh giá tool đã gặp.
- `custom_vi/test_unseen`: đánh giá tool unseen.
- Stress-test anchors: chỉ lấy từ frozen `vi/test`.

## 6. Rebuild Benchmark

### Bước 1: Inventory

- Liệt kê toàn bộ file data hiện có.
- Đếm JSONL, kiểm tra duplicate IDs, query rỗng và parse errors.
- Xác định source index và pairing EN/VI.
- Xác định bản dịch đã QA, failed hoặc incomplete.
- Kiểm tra CustomTools schema và phân phối positive/negative.

### Bước 2: Tạo core canonical records

- Parse normalized English records.
- Ghép Vietnamese record bằng cùng sample ID.
- Giữ negative records với `function_calls=[]`.
- Không thêm `has_tool_call` như label độc lập; suy ra từ `bool(function_calls)`.
- Chuẩn hóa tool schema và parameter types.
- Bảo đảm function call name tồn tại trong candidate tools.

### Bước 3: Deduplication trước split

- Loại exact duplicate query.
- Loại duplicate scenario theo source/sample identity.
- Kiểm tra duplicate sau Unicode/whitespace normalization.
- Không để EN và VI counterpart của một scenario rơi vào hai split khác nhau.
- Không để cùng query/scenario xuất hiện giữa train/val/test.

### Bước 4: Split một lần

- Dùng seed cố định.
- Split theo scenario/group, không split độc lập EN và VI.
- Giữ tỷ lệ train/val/test đã đăng ký trước, mặc định 80/10/10.
- Stratify theo source và positive/negative nếu dữ liệu cho phép.
- Ghi mapping vào `split_manifest.json`.

### Bước 5: Validate trước khi promote

Build vào staging path, không ghi đè active benchmark ngay. Chỉ promote sau khi pass:

- JSON/schema validation 100%.
- Paired EN/VI ID validation.
- No cross-split ID/scenario/query leakage.
- Positive/negative count validation.
- Tool pool coverage validation.
- Tool schema và argument key validation.
- SHA-256 manifest validation.

## 7. Materialize Experiment Data

### 7.1 Nguyên tắc

`data/experiments/` chỉ chứa training input, native instruction output và manifest. Không chứa test copy hoặc validation copy.

Mỗi manifest phải ghi benchmark revision ID, checkpoint ID/revision, experiment ID, training sample IDs/hash, composition, validation source/rule, seed, target epoch và budget.

### 7.2 Experiment matrix

| Experiment | Training data | Validation | Trạng thái |
|---|---|---|---|
| E0 | Không có | Không selection | Chỉ manifest/config, không train file |
| E1 | 60k core EN | `core/en/val` | Giữ |
| E2 | Đúng 60k IDs của E1 nhưng VI | `core/vi/val` | Giữ |
| E3 | General SFT độc lập + 60k core EN | `core/en/val` | Optional, không tạo nếu thiếu prerequisite |
| E4 | 30k core EN + 30k paired core VI | Macro rule đăng ký trước trên EN/VI val | Giữ |
| E5 | E4 + `custom_vi/train.jsonl` 5.6k | Core val + CustomTools val theo rule đăng ký trước | Giữ |

E3 hiện không được materialize mặc định vì code cũ chỉ sao chép E1, không có general-SFT stage. Không gọi bản sao E1 là E3.

### 7.3 Cấu trúc sau cleanup

```text
data/experiments/
  e0/
    manifest.json
  e1/
    train.jsonl
    instruction/train_chat.jsonl
    manifest.json
  e2/
    train.jsonl
    instruction/train_chat.jsonl
    manifest.json
  e4/
    train.jsonl
    instruction/train_chat.jsonl
    manifest.json
  e5/
    train.jsonl
    instruction/train_chat.jsonl
    manifest.json
```

Các file `train_en.jsonl`, `train_vi.jsonl`, `train_custom_vi.jsonl` chỉ giữ như audit views nếu thực sự cần đối chiếu composition. Chúng không phải input bắt buộc nếu `train.jsonl` đã có field source/language.

Không tạo lại:

- `e*/test_en.jsonl`.
- `e*/test_vi.jsonl`.
- `e*/custom_test_*.jsonl`.
- `e*/val_*.jsonl`.
- `e*/custom_val_*.jsonl`.
- `e*/instruction/test_chat.jsonl`.
- `e*/instruction/val_chat.jsonl`.
- E0 `train.jsonl`.

## 8. Native Qwen3.5 Training Format

### 8.1 Structured training row

```json
{
  "messages": [
    {"role": "system", "content": "Bạn là trợ lý có khả năng sử dụng công cụ."},
    {"role": "user", "content": "Tìm gia sư Toán ở Hà Nội."},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "search_tutors",
            "arguments": {"subject": "Toán", "location": "Hà Nội"}
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_tutors",
        "description": "Tìm gia sư theo môn học và khu vực.",
        "parameters": {
          "type": "object",
          "properties": {
            "subject": {"type": "string"},
            "location": {"type": "string"}
          },
          "required": ["subject", "location"]
        }
      }
    }
  ]
}
```

### 8.2 Quy tắc formatting

- Dùng `apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)` để tạo training text.
- Không tự nối tool list vào system message khi đã truyền `tools=`.
- Không tự viết `<|im_start|>`, `<|im_end|>` hoặc `<think>`.
- Không đưa `<tool_call>{"name": ...}</tool_call>` vào assistant `content`.
- `tool_calls` là nguồn label có cấu trúc.
- `arguments` là object, không phải JSON string.
- Multi-call là nhiều phần tử trong cùng `tool_calls` list.
- Inference dùng `add_generation_prompt=True`; thiết lập thinking theo đúng template của checkpoint và ghi lại setting.
- Không copy template 4B sang 2B; mỗi checkpoint tự load template và được smoke-test riêng.

Qwen3.5 template phải được lấy trực tiếp từ exact checkpoint:

```text
https://huggingface.co/unsloth/Qwen3.5-4B/blob/main/chat_template.jinja
```

### 8.3 Negative samples

Native Qwen3.5 template không định nghĩa `<no_tool_call>`. Template hướng dẫn assistant trả lời bình thường khi không gọi tool.

Quy tắc mới:

- Master ground truth vẫn là `function_calls=[]`.
- Native training row negative không có `tool_calls`.
- Assistant `content` phải là câu trả lời tự nhiên/đã curate, không phải marker `<no_tool_call>`.
- Parser evaluation xác định negative khi output không chứa một tool call hợp lệ.

Core negative records hiện không mang assistant answer riêng. Converter dùng
assistant content nếu record có sẵn; nếu không, materialization dùng fallback xác định
theo ngôn ngữ và ghi policy này trong experiment manifest. Không dùng marker
`<no_tool_call>` hay custom serialization.

## 9. Trạng Thái Thực Thi Code

1. [x] Tạo inventory/cleanup manifest và staging output.
2. [x] Refactor benchmark builder để giữ negative, paired EN/VI và group-level dedup.
3. [x] Tạo frozen revision manifest và split manifest.
4. [x] Export active `data/benchmark_vi/` từ revision mới.
5. [x] Invalidate tool cache và rebuild tool pool/schema.
6. [x] Refactor `prepare_experiments.py` đọc frozen revision, không tự split lại.
7. [x] Không materialize E0 train; không tạo E3 nếu thiếu general-SFT prerequisite.
8. [x] Loại test/val copies khỏi experiment folders.
9. [x] Thay converter cũ bằng native Qwen3.5 training adapter.
10. [x] Tạo Unsloth training entrypoint/config cho `unsloth/Qwen3.5-4B` và tùy chọn 2B.
11. [x] Sửa Kaggle guide và uploader theo shared revision.
12. [ ] Chạy native template smoke test và training trong môi trường GPU/Internet.

## 10. Trạng Thái Đồng Bộ Documentation

| File | Nội dung đã đồng bộ |
|---|---|
| `AGENTS.md` | Model ID `unsloth/Qwen3.5-*`, native messages/tool_calls, E0 không train, benchmark revision và cleanup policy |
| `README.md` | Quick start mới: build/freeze benchmark -> materialize train -> Unsloth |
| `docs/architecture.md` | Shared frozen benchmark, E train-only artifacts và native Qwen flow |
| `docs/methodology.md` | Core VI, transfer EN, CustomTools và Unsloth native output |
| `docs/benchmark.md` | Negative support, dedup, paired split, revision manifest và active export |
| `docs/experimental_plan.md` | Nguồn truth duy nhất cho matrix E0-E5 và fixed evaluation |
| `docs/kaggle_notebook_guide.md` | Loại LLaMA-Factory; thêm Unsloth, native template và shared evaluation |
| `docs/custom_vi_dataset_plan.md` | Không merge validation vào train; chuẩn hóa field dẫn xuất |
| `docs/translation_guidelines.md` | Tách normalized main path khỏi raw legacy audit path |
| `data/README.md` | Phân biệt source, frozen benchmark, experiment train artifacts và evaluation |
| `scripts/README.md` | Đồng bộ cleanup/rebuild/materialization commands |

## 11. Tests Và Acceptance Criteria

### Data tests

- Mọi core ID có tối đa một split.
- EN/VI counterpart luôn cùng split.
- Không duplicate query/scenario giữa train/val/test.
- Revision có positive và negative theo counts manifest.
- Không CustomTools validation/test ID xuất hiện trong train.
- E1/E2 dùng cùng sample ID set.
- E4 đúng composition 30k EN + 30k VI.
- E5 đúng E4 + 5.6k CustomTools train.
- E0 không có train file.
- E3 bị từ chối nếu không có general-SFT prerequisite.
- Rebuild cùng input revision/seed tạo cùng hashes.

### Native Qwen tests

- Positive single-call render đúng `<tool_call><function=...>` native format.
- Multi-call render đúng nhiều block.
- Nested/list/number/boolean arguments render đúng.
- Tool schema truyền qua `tools=` không bị duplicate trong system content.
- Negative row không có `tool_calls` và không chứa marker ngoài native contract.
- Assistant-only loss mask không tính system/user tokens.
- 4B và 2B load template riêng và output smoke test pass.
- Parser đọc được native output và metric không phụ thuộc format của paper.

### Cleanup acceptance

- Không còn test/val copies trong `data/experiments/e*`.
- Không còn artifact instruction cũ được dùng bởi trainer.
- Không còn E3 duplicate được ghi là experiment độc lập.
- `data/benchmark_vi/metadata.json` trỏ tới revision mới.
- Mọi active benchmark file có hash trong manifest.
- Có cleanup manifest cho pilot cũ; không mất raw/normalized/translation/custom source data.

## 12. Thứ Tự Thực Thi Và Việc Còn Lại

1. [x] Duyệt plan, inventory và ghi cleanup manifest.
2. [x] Archive pilot và dọn generated benchmark/experiment artifacts.
3. [x] Rebuild, validate, freeze và promote active `data/benchmark_vi/`.
4. [x] Materialize E1, E2, E4, E5; giữ E0 manifest-only; không thêm E3 khi thiếu prerequisite.
5. [x] Generate native Qwen training views và đồng bộ documentation/Kaggle workflow.
6. [ ] Chạy template smoke test và Unsloth training trong môi trường GPU/Internet.
7. [ ] Chạy experiments chính sau khi smoke test và acceptance criteria còn lại pass.

## 13. Nguồn Tham Chiếu Kỹ Thuật

- Checkpoint/template: `unsloth/Qwen3.5-4B`.
- Native template: `chat_template.jinja` trong chính checkpoint.
- Unsloth guide: `https://unsloth.ai/docs/models/qwen3.5/fine-tune`.
- Paper Ersoy et al. chỉ tham khảo thiết kế ablation/metric; không quyết định serialization/chat template của Qwen3.5.
