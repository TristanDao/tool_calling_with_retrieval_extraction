# Hướng Dẫn Vận Hành Thực Nghiệm Trên Kaggle (Kaggle Execution Guide)

> **Tài liệu hợp nhất**: Hướng dẫn toàn diện quy trình huấn luyện (Training E1–E4) và đánh giá suy luận (Evaluation E0 & Post-training) cho mô hình **Qwen3.5 (2B/4B)** trên môi trường **Kaggle Dual GPU NVIDIA T4 (2 × 16GB VRAM)**.

---

## 1. Tổng Quan Hạ Tầng & Nguyên Tắc Tối Ưu

### 1.1 Cấu hình phần cứng mục tiêu
- **Môi trường**: Kaggle Notebook, Accelerator: **GPU T4 × 2** (30 giờ/tuần, giới hạn 12 giờ/phiên).
- **Nguyên tắc Batch Size**: Giữ nguyên `effective_batch_size = 16` của controlled track:
  $$\text{Per-device batch (2)} \times \text{Gradient accumulation (4)} \times \text{Số lượng GPU (2)} = 16$$

### 1.2 Các giải pháp chống nghẽn và timeout
1. **Khắc phục nghẽn I/O & Timeout 12h**:
   - Tắt thanh tiến trình `tqdm` dày đặc, chỉ ghi log mỗi 50 steps.
   - Tắt đánh giá toàn diện (full validation) giữa chừng khi huấn luyện 1 epoch để tiết kiệm thời gian.
   - Lưu checkpoint mỗi 250 steps và dừng có kiểm soát (`STOP_AFTER_STEP = 1000`) để tiếp tục phiên mới (resume) liền mạch.
2. **Khắc phục tràn RAM máy chủ (Host RAM OOM)**:
   - Đọc trực tiếp streaming từng dòng JSONL, tokenize qua generator và lưu dưới dạng Arrow Dataset chỉ chứa cột token IDs; không nạp toàn bộ danh sách Python vào bộ nhớ của từng tiến trình DDP.

---

# PHẦN A: HUẤN LUYỆN SLM (E1–E4 TRAINING VỚI UNSLOTH DDP)

### Cell 1: Cấu hình và Kiểm tra Toàn vẹn Dữ liệu (Preflight)
Thay `DATA_ROOT` bằng đúng đường dẫn Dataset trên Kaggle của bạn.

```python
from collections import Counter
import hashlib
import json
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/datasets/phcthnho/tool-calling-vi-experiments")
REVISION = "2026-09-02-full-dedup-seed42"
REVISION_DIR = DATA_ROOT / "benchmark_core" / REVISION
EXPERIMENT = "e1"  # e1, e2, e3, hoặc e4
EXPERIMENT_DIR = DATA_ROOT / EXPERIMENT
MODEL_ID = "unsloth/Qwen3.5-4B"
TRAIN_LIMIT = None
RUN_NAME = f"{EXPERIMENT}_{MODEL_ID.rsplit('/', 1)[-1].lower()}"
RUN_DIR = Path("/kaggle/working") / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert DATA_ROOT.is_dir(), DATA_ROOT
assert REVISION_DIR.is_dir(), REVISION_DIR
assert EXPERIMENT_DIR.is_dir(), EXPERIMENT_DIR
experiment_manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text(encoding="utf-8"))
revision_manifest = json.loads((REVISION_DIR / "manifest.json").read_text(encoding="utf-8"))
assert experiment_manifest["benchmark_revision"] == REVISION
assert revision_manifest["revision"] == REVISION
assert MODEL_ID in {
    experiment_manifest["checkpoint"]["primary"],
    experiment_manifest["checkpoint"]["alternative"],
}
for relative, expected in experiment_manifest["files"].items():
    assert sha256(EXPERIMENT_DIR / relative) == expected, relative

train_rows = read_jsonl(EXPERIMENT_DIR / "train.jsonl")
expected = {
    "e1": ({"en": 60000}, 60000),
    "e2": ({"vi": 60000}, 60000),
    "e3": ({"en": 30000, "vi": 30000}, 60000),
    "e4": ({"en": 30000, "vi": 35600}, 65600),
}
expected_languages, expected_total = expected[EXPERIMENT]
assert Counter(row["language"] for row in train_rows) == expected_languages
assert len(train_rows) == expected_total
assert len({row["id"] for row in train_rows}) == expected_total
assert len(read_jsonl(EXPERIMENT_DIR / "instruction/train_chat.jsonl")) == expected_total
print(f"{EXPERIMENT} preflight: PASS ({expected_total} train rows)")
```

### Cell 2: Cài đặt Dependency và Kiểm tra GPU

```python
%pip install -q "unsloth" "transformers>=5.2.0" "trl>=0.15" \
    "accelerate>=1.0" "peft>=0.14" "bitsandbytes>=0.43"
```

```python
import torch

assert torch.cuda.device_count() == 2, "Vui lòng bật chế độ 2 x T4 GPUs trong Settings"
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
```

### Cell 3: Kiểm thử Chat Template Native (Smoke Test)

```python
import unsloth
from transformers import AutoTokenizer

smoke_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
for row in read_jsonl(EXPERIMENT_DIR / "instruction/train_chat.jsonl")[:3]:
    rendered = smoke_tokenizer.apply_chat_template(
        row["messages"], tools=row["tools"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    expected_calls = sum(
        len(message.get("tool_calls", []))
        for message in row["messages"]
        if message.get("role") == "assistant"
    )
    assistant_part = rendered.rsplit("<|im_start|>assistant", 1)[-1]
    assert assistant_part.count("<tool_call>") == expected_calls, row["id"]
print("Native chat template smoke test: PASS")
```

### Cell 4: Thiết lập Cấu hình Huấn luyện & Ngắt Stage

```python
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
STOP_AFTER_STEP = 1000  # Ngắt nhịp để lưu checkpoint và giải phóng hạn mức 12h
RESUME_CHECKPOINT = None  # Hoặc trỏ tới đường dẫn checkpoint stage trước: e.g. "/kaggle/input/.../checkpoint-1000"

stage_config = {
    "experiment": EXPERIMENT,
    "model_id": MODEL_ID,
    "run_name": RUN_NAME,
    "run_dir": str(RUN_DIR),
    "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "stop_after_step": STOP_AFTER_STEP,
    "resume_checkpoint": RESUME_CHECKPOINT,
}
(RUN_DIR / "stage_config.json").write_text(json.dumps(stage_config, indent=2), encoding="utf-8")
print("Cấu hình stage đã sẵn sàng:", stage_config)
```

### Cell 5: Khởi chạy Huấn luyện Phân tán DDP qua `torchrun`

```python
# Chạy script huấn luyện độc lập qua torchrun để quản lý bộ nhớ tiến trình tốt nhất
!torchrun --nproc_per_node=2 scripts/train/train_unsloth_ddp.py \
    --config {RUN_DIR}/stage_config.json
```

---

# PHẦN B: ĐÁNH GIÁ VÀ SUY LUẬN (E0 & POST-TRAINING EVALUATION)

> **Lưu ý**: Đối với suy luận và sinh kết quả (generation), không dùng `torchrun`. Mô hình 4-bit sẽ tự động phân bổ đều trên 2 GPU T4 qua `accelerate` (`device_map="auto"`).

### Cell 1: Khởi tạo và Đọc Dữ liệu Đánh giá

```python
import json
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/datasets/phcthnho/tool-calling-vi-experiments")
REVISION = "2026-09-02-full-dedup-seed42"
TEST_FILE = DATA_ROOT / "benchmark_core" / REVISION / "vi" / "test.jsonl"
CUSTOM_SEEN_FILE = DATA_ROOT / "custom_vi" / "test_seen.jsonl"
CUSTOM_UNSEEN_FILE = DATA_ROOT / "custom_vi" / "test_unseen.jsonl"

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

vi_test_data = read_jsonl(TEST_FILE)
custom_seen_data = read_jsonl(CUSTOM_SEEN_FILE)
custom_unseen_data = read_jsonl(CUSTOM_UNSEEN_FILE)
print(f"Đã nạp: VI Core Test ({len(vi_test_data)}), Custom Seen ({len(custom_seen_data)}), Custom Unseen ({len(custom_unseen_data)})")
```

### Cell 2: Tải Mô hình Suy luận (FastLanguageModel Inference Mode)

```python
from unsloth import FastLanguageModel

# Sử dụng mô hình gốc (E0) hoặc Checkpoint LoRA đã huấn luyện
CHECKPOINT_PATH = "unsloth/Qwen3.5-4B"  # hoặc Path("/kaggle/working/checkpoint-final")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)
FastLanguageModel.for_inference(model)
print("Mô hình đã sẵn sàng cho chế độ suy luận.")
```

### Cell 3: Hàm Suy luận Theo Lô có Resume (Batched & Resumable Generation)

```python
from tqdm import tqdm

def run_evaluation(data_samples, output_file: Path, batch_size: int = 8):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if output_file.is_file():
        for line in output_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
    
    pending = [s for s in data_samples if s["id"] not in done_ids]
    print(f"Tổng mẫu: {len(data_samples)} | Đã hoàn thành: {len(done_ids)} | Còn lại: {len(pending)}")
    
    with output_file.open("a", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(pending), batch_size), desc="Đang đánh giá"):
            batch = pending[i : i + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["query"]}],
                    tools=item["tools"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for item in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.01,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            for item, out_ids, in_len in zip(batch, outputs, inputs["attention_mask"].sum(dim=1)):
                gen_text = tokenizer.decode(out_ids[in_len:], skip_special_tokens=False)
                res_record = {
                    "id": item["id"],
                    "query": item["query"],
                    "gold_calls": item.get("function_calls", []),
                    "model_output": gen_text,
                }
                out_f.write(json.dumps(res_record, ensure_ascii=False) + "\n")
                out_f.flush()

# Ví dụ chạy:
# run_evaluation(custom_seen_data, Path("/kaggle/working/eval_custom_seen.jsonl"))
```

### Cell 4: Chấm Điểm và Xuất Chỉ Số Đánh Giá (Metrics Evaluation)

```python
from src.evaluation.extraction_metrics import evaluate_predictions

# Sử dụng pipeline metrics chuẩn của repository để tính Tool Acc, ArgA, Syntax Error, Non-FC Recall
# metrics = evaluate_predictions("/kaggle/working/eval_custom_seen.jsonl")
# print(json.dumps(metrics, indent=2))
```
