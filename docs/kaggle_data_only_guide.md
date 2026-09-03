# Kaggle Data-Only Notebook Guide

Hướng dẫn này chạy Method 1 trên Kaggle khi Kaggle chỉ có Dataset data, không có
source code của repository. Notebook tự định nghĩa các helper cần thiết cho native
Qwen3.5, assistant-only loss, Unsloth training và đánh giá cơ bản.

## 1. Dataset Kaggle

Upload lại version mới nhất từ máy local:

```bash
python scripts/data/upload_experiments_to_kaggle.py \
  <kaggle-username>/tool-calling-vi-experiments \
  --benchmark-revision data/benchmark_core/2026-09-02-full-dedup-seed42 \
  --custom-data data/custom_vi \
  --version-notes "Rebuilt 2026-09-02; native Qwen3.5 E0-E4 data"
```

Trong Kaggle, attach Dataset version mới nhất vào notebook. Cấu trúc cần có:

```text
/kaggle/input/tool-calling-vi-experiments/
├── e0/ ... e4/
├── benchmark_core/2026-09-02-full-dedup-seed42/
└── custom_vi/
```

Không upload source code vào Dataset. Không sửa file bên dưới `/kaggle/input` vì
đây là thư mục read-only.

## 2. Quy Trình An Toàn

Chạy từng notebook riêng cho mỗi cặp `(experiment, model)`:

| Notebook | Mục đích |
|---|---|
| `01_preflight_e0.ipynb` | Kiểm tra revision, hash, composition và chạy E0 |
| `02_train_eval.ipynb` | Train/evaluate một E với một model |

Thứ tự:

1. Chạy preflight và E0.
2. Chạy thử E1 với `TRAIN_LIMIT = 256`.
3. Nếu loss và output hợp lệ, đặt `TRAIN_LIMIT = None` để train đầy đủ.
4. Chạy E2, E3, E4 lần lượt.
5. Lặp E1, E2, E3, E4 với Qwen3.5-2B.

E3 trong notebook này là bilingual tool-calling SFT, không phải general-SFT
experiment của bài Arabic.

## 3. Cell 1: Cấu Hình Và Đọc Data

Chạy cell này trong cả hai notebook.

```python
from collections import Counter
import hashlib
import json
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/datasets/phcthnho/tool-calling-vi-experiments")

REVISION = "2026-09-02-full-dedup-seed42"
REVISION_DIR = DATA_ROOT / "benchmark_core" / REVISION
CUSTOM_DIR = DATA_ROOT / "custom_vi"

EXPERIMENT = "e1"
MODEL_ID = "unsloth/Qwen3.5-4B"
TRAIN_LIMIT = 256
RUN_NAME = f"{EXPERIMENT}_{MODEL_ID.rsplit('/', 1)[-1].lower()}"
RUN_DIR = Path("/kaggle/working") / RUN_NAME
OUTPUT_DIR = RUN_DIR / "checkpoint"
RUN_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_ROOT.is_dir(), DATA_ROOT
assert REVISION_DIR.is_dir(), REVISION_DIR
assert (DATA_ROOT / EXPERIMENT / "manifest.json").is_file()

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

experiment_manifest = json.loads(
    (DATA_ROOT / EXPERIMENT / "manifest.json").read_text(encoding="utf-8")
)
revision_manifest = json.loads(
    (REVISION_DIR / "manifest.json").read_text(encoding="utf-8")
)
assert experiment_manifest["benchmark_revision"] == REVISION
assert revision_manifest["revision"] == REVISION
assert MODEL_ID in {
    experiment_manifest["checkpoint"]["primary"],
    experiment_manifest["checkpoint"]["alternative"],
}
print(json.dumps(experiment_manifest["training"], ensure_ascii=False, indent=2))
```

## 4. Cell 2: Kiểm Tra Manifest Và Hash

Cell này phải pass trước khi cài model hoặc train.

```python
experiment_dir = DATA_ROOT / EXPERIMENT
for relative, expected in experiment_manifest["files"].items():
    actual = sha256(experiment_dir / relative)
    assert actual == expected, (relative, actual, expected)

assert sha256(REVISION_DIR / "manifest.json") == experiment_manifest["benchmark_manifest_sha256"]
assert sha256(REVISION_DIR / "split_manifest.json") == experiment_manifest["split_manifest_sha256"]

metadata = json.loads((REVISION_DIR / "metadata.json").read_text(encoding="utf-8"))
assert metadata["n_unique_tools"] == 4421
assert metadata["counts"]["en"]["label"]["negative"] == 4817
assert metadata["split_counts"] == {
    "en": {"train": 61615, "val": 7701, "test": 7712},
    "vi": {"train": 61615, "val": 7701, "test": 7712},
}

print("manifest, hashes and frozen revision checks: PASS")
```

## 5. Cell 3: Kiểm Tra Composition Của Experiment

```python
train_rows = read_jsonl(experiment_dir / "train.jsonl") if EXPERIMENT != "e0" else []
counts = Counter(row.get("language") for row in train_rows)
sources = Counter(row.get("source") for row in train_rows)
labels = Counter("positive" if row.get("function_calls") else "negative" for row in train_rows)

expected = {
    "e1": ({"en": 60000}, 60000),
    "e2": ({"vi": 60000}, 60000),
    "e3": ({"en": 30000, "vi": 30000}, 60000),
    "e4": ({"en": 30000, "vi": 35600}, 65600),
}
if EXPERIMENT == "e0":
    assert not (experiment_dir / "train.jsonl").exists()
else:
    expected_languages, expected_total = expected[EXPERIMENT]
    assert dict(counts) == expected_languages
    assert len(train_rows) == expected_total
    assert len({row["id"] for row in train_rows}) == expected_total
    assert len(read_jsonl(experiment_dir / "instruction/train_chat.jsonl")) == expected_total

if EXPERIMENT in {"e1", "e2"}:
    other = "e2" if EXPERIMENT == "e1" else "e1"
    assert {row["id"] for row in train_rows} == {
        row["id"] for row in read_jsonl(DATA_ROOT / other / "train.jsonl")
    }

print("languages:", dict(counts))
print("sources:", dict(sources))
print("labels:", dict(labels))
print("experiment composition: PASS")
```

## 6. Cell 4: Cài Dependencies

Kaggle thường đã có PyTorch. Cài các package còn lại rồi restart kernel nếu Kaggle
yêu cầu:

```python
%pip install -q "unsloth" "transformers>=5.2.0" "trl>=0.15" \
    "accelerate>=1.0" "peft>=0.14" "bitsandbytes>=0.43"
```

Kiểm tra GPU (hiển thị toàn bộ GPU khả dụng, ví dụ 2 GPU T4):

```python
import torch
assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator first"
n_gpus = torch.cuda.device_count()
print(f"GPUs available: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

## 7. Cell 5: Native Qwen Helpers

Training data trong Dataset đã là native `messages`/`tools`/`tool_calls`. Cell này
chỉ dùng để smoke-test tokenizer và tạo validation view tạm thời.

```python
import re

SYSTEM_PROMPTS = {
    "en": "You are an AI assistant capable of using tools.",
    "vi": "Bạn là trợ lý AI có khả năng sử dụng công cụ.",
}

def format_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        },
    }

def format_call(call: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": call.get("arguments", {}),
        },
    }

def native_row(record: dict, language: str) -> dict:
    tools = [format_tool(tool) for tool in record["tools"]]
    calls = [format_call(call) for call in record["function_calls"]]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[language]},
        {"role": "user", "content": record["query"]},
    ]
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    else:
        fallback = {
            "en": "I cannot complete that request right now.",
            "vi": "Hiện tại tôi chưa thể thực hiện yêu cầu này.",
        }[language]
        messages.append({"role": "assistant", "content": record.get("assistant_content") or fallback})
    return {"id": record.get("id"), "messages": messages, "tools": tools}

def token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(values[0] if values and isinstance(values[0], list) else values)

def tokenize_assistant_only(tokenizer, row: dict, max_seq_length: int) -> dict[str, list[int]]:
    messages = row["messages"]
    prompt_row = {**row, "messages": messages[:-1]}
    prompt = tokenizer.apply_chat_template(
        prompt_row["messages"], tools=prompt_row["tools"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        messages, tools=row["tools"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    prompt_ids = token_ids(tokenizer, prompt)
    input_ids = token_ids(tokenizer, full)
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Prompt is not a prefix for {row.get('id')}")
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
    if len(input_ids) <= len(prompt_ids):
        raise ValueError(f"Assistant target was truncated for {row.get('id')}")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids):],
    }
```

## 8. Cell 6: Tokenizer Smoke Test

```python
import unsloth
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
train_chat_file = experiment_dir / "instruction/train_chat.jsonl"
if train_chat_file.exists():
    sample_rows = read_jsonl(train_chat_file)[:3]
else:
    # E0 là zero-shot (không có tập train), dùng 3 mẫu test để smoke-test tokenizer
    sample_rows = [native_row(record, "vi") for record in read_jsonl(REVISION_DIR / "vi/test.jsonl")[:3]]

for row in sample_rows:
    rendered = tokenizer.apply_chat_template(
        row["messages"], tools=row["tools"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    expected_calls = sum(
        len(message.get("tool_calls", []))
        for message in row["messages"]
        if message.get("role") == "assistant"
    )
    assistant_part = rendered.rsplit("<|im_start|>assistant", 1)[-1]
    actual_calls = assistant_part.count("<tool_call>")
    assert actual_calls == expected_calls, (
        f"tool_call mismatch for {row.get('id')}: "
        f"expected={expected_calls}, actual={actual_calls}"
    )
    tokenized = tokenize_assistant_only(tokenizer, row, 4096)
    assert any(label != -100 for label in tokenized["labels"])
print("native tokenizer smoke test: PASS")
```

## 9. Cell 7: Chạy E0 Zero-Shot

Chạy cell này trong notebook preflight với `EXPERIMENT = "e0"`. E0 không đọc
validation và không fine-tune.

```python
from unsloth import FastLanguageModel
import torch

zero_model, zero_tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=4096,
    load_in_4bit=True,
    load_in_16bit=False,
    full_finetuning=False,
)
zero_tokenizer = getattr(zero_tokenizer, "tokenizer", zero_tokenizer)
zero_model.eval()
zero_test = read_jsonl(REVISION_DIR / "vi/test.jsonl")[:20]
zero_outputs = []
for record in zero_test:
    row = native_row(record, "vi")
    inputs = zero_tokenizer.apply_chat_template(
        row["messages"][:-1], tools=row["tools"], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt",
    )
    if isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
        inputs = inputs["input_ids"]
    inputs = inputs.to(zero_model.device)
    with torch.no_grad():
        generated = zero_model.generate(
            inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=zero_tokenizer.eos_token_id,
        )
    output = zero_tokenizer.decode(
        generated[0][inputs.shape[-1]:], skip_special_tokens=False
    )
    zero_outputs.append({"id": record["id"], "output": output})

(RUN_DIR / "e0_outputs.json").write_text(
    json.dumps(zero_outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("E0 outputs:", len(zero_outputs))
```

Sau khi kiểm tra output mẫu, đánh giá toàn bộ `vi/test.jsonl`, `en/test.jsonl`,
`custom_vi/test_seen.jsonl` và `custom_vi/test_unseen.jsonl` trong các cell riêng.

## 10. Cell 8: Tạo Validation View Tạm Thời

Validation không được lưu vào Dataset experiment.

```python
validation_sources = {
    "e0": [],
    "e1": [(REVISION_DIR / "en/val.jsonl", "en")],
    "e2": [(REVISION_DIR / "vi/val.jsonl", "vi")],
    "e3": [
        (REVISION_DIR / "en/val.jsonl", "en"),
        (REVISION_DIR / "vi/val.jsonl", "vi"),
    ],
    "e4": [
        (REVISION_DIR / "en/val.jsonl", "en"),
        (REVISION_DIR / "vi/val.jsonl", "vi"),
        (CUSTOM_DIR / "val_seen.jsonl", "vi"),
        (CUSTOM_DIR / "val_unseen.jsonl", "vi"),
    ],
}

validation_rows = []
for path, language in validation_sources[EXPERIMENT]:
    validation_rows.extend(native_row(record, language) for record in read_jsonl(path))
EVAL_FILE = RUN_DIR / "validation_native.jsonl"
with EVAL_FILE.open("w", encoding="utf-8") as output:
    for row in validation_rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
print("validation rows:", len(validation_rows))
```

## 11. Cell 9: Train Một Experiment (Multi-GPU DDP với torchrun)

Không chạy cell này với `EXPERIMENT = "e0"`. Để chạy thử, giữ `TRAIN_LIMIT = 256`.
Để chạy đầy đủ, đổi thành `TRAIN_LIMIT = None` và restart runtime trước khi train.

Để tận dụng toàn bộ 2 GPU T4 trên Kaggle thông qua Distributed Data Parallel (DDP),
huấn luyện được tách thành 2 cell:
- **Cell 9a**: Ghi script huấn luyện độc lập `train_kaggle.py`.
- **Cell 9b**: Kích hoạt `torchrun` chạy song song trên cả 2 GPU (`--nproc_per_node=2`).

> **Cân bằng Batch Size**: Với 2 GPU và `per_device_train_batch_size = 1`,
> `gradient_accumulation_steps` được tự động gán bằng $16 / 2 = 8$, giúp giữ nguyên
> đúng ngân sách **`effective_batch_size = 16`** của các manifest E1–E4.

### Cell 9a: Tạo script huấn luyện

```python
%%writefile train_kaggle.py
import argparse
import json
import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]

def token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(values[0] if values and isinstance(values[0], list) else values)

def tokenize_assistant_only(tokenizer, row: dict, max_seq_length: int) -> dict[str, list[int]]:
    messages = row["messages"]
    prompt_row = {**row, "messages": messages[:-1]}
    prompt = tokenizer.apply_chat_template(
        prompt_row["messages"], tools=prompt_row["tools"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        messages, tools=row["tools"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    prompt_ids = token_ids(tokenizer, prompt)
    input_ids = token_ids(tokenizer, full)
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Prompt is not a prefix for {row.get('id')}")
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
    if len(input_ids) <= len(prompt_ids):
        raise ValueError(f"Assistant target was truncated for {row.get('id')}")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids):],
    }

class AssistantOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        return {
            key: torch.tensor(
                [item[key] + [pad_value] * (max_length - len(item[key])) for item in features],
                dtype=torch.long,
            )
            for key, pad_value in (
                ("input_ids", pad_id),
                ("attention_mask", 0),
                ("labels", -100),
            )
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/kaggle/input/tool-calling-vi-experiments")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="unsloth/Qwen3.5-4B")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    output_dir = run_dir / "checkpoint"
    experiment_dir = data_root / args.experiment

    train_native = read_jsonl(experiment_dir / "instruction/train_chat.jsonl")
    if args.train_limit is not None:
        train_native = train_native[:args.train_limit]

    eval_file = run_dir / "validation_native.jsonl"
    eval_rows = read_jsonl(eval_file) if eval_file.exists() else []

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_16bit=False,
        full_finetuning=False,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.config.use_cache = False

    train_features = [tokenize_assistant_only(tokenizer, row, args.max_seq_length) for row in train_native]
    eval_features = [tokenize_assistant_only(tokenizer, row, args.max_seq_length) for row in eval_rows]

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = max(1, 16 // world_size)

    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": 1.0,
        "learning_rate": 5e-7,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "seed": args.seed,
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    }
    if eval_features:
        training_kwargs["eval_strategy"] = "epoch"

    try:
        train_args = TrainingArguments(**training_kwargs)
    except TypeError:
        if eval_features:
            training_kwargs.pop("eval_strategy", None)
            training_kwargs["evaluation_strategy"] = "epoch"
        train_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=Dataset.from_list(train_features),
        eval_dataset=Dataset.from_list(eval_features) if eval_features else None,
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    trainer.train()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("training complete:", output_dir)

if __name__ == "__main__":
    main()
```

### Cell 9b: Khởi chạy huấn luyện phân tán với torchrun

```python
train_limit_arg = f"--train_limit {TRAIN_LIMIT}" if TRAIN_LIMIT is not None else ""
n_gpus = torch.cuda.device_count()

!torchrun --nproc_per_node={n_gpus} train_kaggle.py \
    --experiment {EXPERIMENT} \
    --model_id {MODEL_ID} \
    --run_dir {RUN_DIR} \
    --data_root {DATA_ROOT} \
    {train_limit_arg}
```

## 12. Cell 10: Lưu Run Manifest

```python
n_gpus = torch.cuda.device_count()
grad_accum = max(1, 16 // n_gpus)
train_native = read_jsonl(experiment_dir / "instruction/train_chat.jsonl")
if TRAIN_LIMIT is not None:
    train_native = train_native[:TRAIN_LIMIT]

run_manifest = {
    "experiment": EXPERIMENT,
    "model": MODEL_ID,
    "revision": REVISION,
    "train_limit": TRAIN_LIMIT,
    "train_rows_used": len(train_native),
    "validation_rows": len(validation_rows),
    "max_seq_length": 4096,
    "gradient_accumulation_steps": grad_accum,
    "effective_batch_size": 16,
    "n_gpus": n_gpus,
    "seed": 42,
    "assistant_only_loss": True,
    "native_chat_template": True,
}
(RUN_DIR / "run_manifest.json").write_text(
    json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
```

## 13. Cell 11: Kiểm Tra Generation

Nạp trực tiếp checkpoint đã fine-tune từ `OUTPUT_DIR` để kiểm tra checkpoint và tối ưu hóa tốc độ suy luận bằng `FastLanguageModel.for_inference`.

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(OUTPUT_DIR),
    max_seq_length=4096,
    load_in_4bit=True,
)
tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
FastLanguageModel.for_inference(model)

test_records = read_jsonl(REVISION_DIR / "vi/test.jsonl")[:20]
for record in test_records[:3]:
    row = native_row(record, "vi")
    prompt = tokenizer.apply_chat_template(
        row["messages"][:-1], tools=row["tools"], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt",
    )
    if isinstance(prompt, dict) or hasattr(prompt, "input_ids"):
        prompt = prompt["input_ids"]
    prompt = prompt.to(model.device)
    with torch.no_grad():
        generated = model.generate(
            prompt, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(generated[0][prompt.shape[-1]:], skip_special_tokens=False)
    print("QUERY:", record["query"])
    print("OUTPUT:", output[:1000])
    print("-" * 80)
```

Output positive phải có native `<tool_call><function=...>`. Negative phải là câu
trả lời bình thường, không phải `<no_tool_call>`.

## 14. Cell 12: Đánh Giá Toàn Diện & Báo Cáo Metric (Full Evaluation)

Cell này tính toán đầy đủ các chỉ số thực nghiệm của đề tài và đối sánh theo chuẩn
bài báo Ersoy et al. (2025):
- **Tool Selection Accuracy (Positive)**: Độ chính xác phát hiện đúng tên tool khi query cần gọi công cụ.
- **Non-FC Recall (Negative)**: Độ chính xác nhận diện câu không cần gọi tool (tránh hallucination).
- **ArgA (Exact Match End-to-End)**: Độ chính xác tuyệt đối (khớp cả tool name lẫn toàn bộ tham số).
- **Syntax / Parser Error Rate**: Tỷ lệ lỗi cú pháp XML/JSON không parse được.
- **Latency**: Độ trễ suy luận trung bình mỗi mẫu (ms).

Tự động lưu file kết quả chi tiết `eval_predictions_{split}.json` và bảng tổng kết
`eval_metrics.json` vào `RUN_DIR`.

```python
import json
import re
import time
from pathlib import Path
import torch

# 1. Trình phân tích cú pháp Tool Call (XML + JSON fallback)
TOOL_RE = re.compile(
    r"<tool_call>\s*<function\s*=\s*([^>\s]+)\s*>(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
PARAM_RE = re.compile(
    r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)

def parse_native_output(text: str) -> tuple[list[dict], list[str]]:
    calls = []
    errors = []
    open_tags = len(re.findall(r"<tool_call\b", text, re.IGNORECASE))
    close_tags = len(re.findall(r"</tool_call\s*>", text, re.IGNORECASE))
    if open_tags != close_tags:
        errors.append("unbalanced_tool_call_tags")

    for name, body in TOOL_RE.findall(text):
        arguments = {}
        for parameter, value in PARAM_RE.findall(body):
            parameter = parameter.strip()
            val_str = value.strip()
            try:
                arguments[parameter] = json.loads(val_str)
            except (json.JSONDecodeError, TypeError):
                arguments[parameter] = val_str
        calls.append({"name": name.strip(), "arguments": arguments})

    if not calls and open_tags > 0:
        errors.append("malformed_tool_call")
    return calls, errors

# 2. Hàm đánh giá trên một tập test
def evaluate_split(eval_model, eval_tokenizer, test_records: list[dict], language: str = "vi", split_name: str = "test"):
    results = []
    pos_samples = 0
    neg_samples = 0
    pos_tool_match = 0
    neg_correct = 0
    exact_match = 0
    syntax_errors = 0
    total_latency_ms = 0.0

    print(f"Bắt đầu đánh giá split [{split_name}] với {len(test_records)} mẫu...")
    start_time_all = time.time()

    for idx, record in enumerate(test_records, start=1):
        row = native_row(record, language)
        prompt = eval_tokenizer.apply_chat_template(
            row["messages"][:-1], tools=row["tools"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt",
        )
        if isinstance(prompt, dict) or hasattr(prompt, "input_ids"):
            prompt = prompt["input_ids"]
        prompt = prompt.to(eval_model.device)

        t0 = time.time()
        with torch.no_grad():
            generated = eval_model.generate(
                prompt, max_new_tokens=512, do_sample=False,
                pad_token_id=eval_tokenizer.eos_token_id,
            )
        latency = (time.time() - t0) * 1000.0
        total_latency_ms += latency

        output = eval_tokenizer.decode(generated[0][prompt.shape[-1]:], skip_special_tokens=False)
        pred_calls, errors = parse_native_output(output)
        gold_calls = record.get("function_calls", [])

        is_pos = len(gold_calls) > 0
        gold_names = [c["name"] for c in gold_calls]
        pred_names = [c["name"] for c in pred_calls]

        t_match = False
        n_correct = False
        e_match = (pred_calls == gold_calls)

        if is_pos:
            pos_samples += 1
            t_match = (gold_names == pred_names)
            if t_match:
                pos_tool_match += 1
        else:
            neg_samples += 1
            n_correct = (len(pred_calls) == 0 and not errors)
            if n_correct:
                neg_correct += 1

        if e_match:
            exact_match += 1
        if errors:
            syntax_errors += 1

        results.append({
            "id": record["id"],
            "query": record["query"],
            "gold": gold_calls,
            "predicted": pred_calls,
            "errors": errors,
            "tool_match": t_match if is_pos else n_correct,
            "exact_match": e_match,
            "latency_ms": round(latency, 2),
            "raw_output": output,
        })

        if idx % 50 == 0 or idx == len(test_records):
            print(f"  Đã xử lý {idx}/{len(test_records)} mẫu ({idx/len(test_records)*100:.1f}%)...")

    n_total = len(test_records)
    tool_acc = (pos_tool_match / pos_samples * 100.0) if pos_samples > 0 else 0.0
    non_fc_recall = (neg_correct / neg_samples * 100.0) if neg_samples > 0 else 0.0
    arga = (exact_match / n_total * 100.0) if n_total > 0 else 0.0
    err_rate = (syntax_errors / n_total * 100.0) if n_total > 0 else 0.0
    avg_latency = (total_latency_ms / n_total) if n_total > 0 else 0.0

    summary = {
        "split": split_name,
        "total_samples": n_total,
        "positive_samples": pos_samples,
        "negative_samples": neg_samples,
        "tool_accuracy_pos_pct": round(tool_acc, 2),
        "non_fc_recall_pct": round(non_fc_recall, 2),
        "arga_exact_match_pct": round(arga, 2),
        "syntax_error_rate_pct": round(err_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "elapsed_seconds": round(time.time() - start_time_all, 2),
    }

    print("\n" + "=" * 60)
    print(f"=== BÁO CÁO KẾT QUẢ ĐÁNH GIÁ: {split_name.upper()} ===")
    print(f"Tổng số mẫu:                   {n_total} (Dương tính: {pos_samples}, Âm tính: {neg_samples})")
    print(f"Tool Selection (Positive):     {pos_tool_match}/{pos_samples} ({tool_acc:.2f}%)")
    print(f"Non-FC Recall (Negative):      {neg_correct}/{neg_samples} ({non_fc_recall:.2f}%)")
    print(f"ArgA (Exact Match End-to-End): {exact_match}/{n_total} ({arga:.2f}%)")
    print(f"Tỷ lệ lỗi cú pháp (XML/JSON):   {syntax_errors}/{n_total} ({err_rate:.2f}%)")
    print(f"Độ trễ trung bình / mẫu:        {avg_latency:.2f} ms")
    print("=" * 60 + "\n")

    # Lưu chi tiết dự đoán
    (RUN_DIR / f"eval_predictions_{split_name}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary

# 3. Kích hoạt đánh giá (Tự động nhận diện 'model' sau khi train hoặc 'zero_model' khi chạy E0)
eval_target_model = model if "model" in globals() else zero_model
eval_target_tokenizer = tokenizer if "tokenizer" in globals() else zero_tokenizer

# Chọn tập test muốn chạy (có thể đặt EVAL_LIMIT = 100 để test nhanh)
EVAL_LIMIT = None
vi_test_data = read_jsonl(REVISION_DIR / "vi/test.jsonl")
if EVAL_LIMIT is not None:
    vi_test_data = vi_test_data[:EVAL_LIMIT]

metrics_summary = evaluate_split(
    eval_model=eval_target_model,
    eval_tokenizer=eval_target_tokenizer,
    test_records=vi_test_data,
    language="vi",
    split_name=f"{RUN_NAME}_vi_test",
)

(RUN_DIR / "eval_metrics.json").write_text(
    json.dumps(metrics_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("Đã lưu kết quả đánh giá vào:", RUN_DIR / "eval_metrics.json")
```

## 15. Lưu Kết Quả Kaggle

Đặt `RUN_DIR` làm output của notebook. Cuối notebook kiểm tra:

```python
assert (RUN_DIR / "run_manifest.json").exists()
print("Save or publish this folder as Kaggle Notebook Output:", RUN_DIR)
```

Kaggle Dataset input chỉ là data. Checkpoint, raw output, parsed output và metrics
phải lưu bằng Notebook Output hoặc Kaggle Model/Output riêng.

## 16. Checklist Trước Khi Báo Cáo

- [ ] Đã attach đúng Dataset version rebuild mới nhất.
- [ ] Revision là `2026-09-02-full-dedup-seed42`.
- [ ] Manifest/hash preflight pass.
- [ ] E0 chạy trước các experiment fine-tune.
- [ ] Native tokenizer smoke test pass.
- [ ] Trial 256 rows pass trước full training.
- [ ] E1/E2 giữ cùng sample IDs.
- [ ] E3 là `30k EN + 30k VI`.
- [ ] E4 là E3 cộng `5,600 CustomTools train`.
- [ ] Không dùng CustomTools validation/test để train.
- [ ] E3 được chạy với `30k EN + 30k VI`; E4 thêm `5,600 CustomTools train`.
- [ ] Lưu run manifest, raw output, parsed output và metrics.
