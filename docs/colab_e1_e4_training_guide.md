# Colab E1-E4 Training Guide

Guide nay dung de train Qwen3.5-2B hoac Qwen3.5-4B tren mot GPU Colab va
resume qua nhieu runtime. Colab runtime la tam thoi, vi vay data co the dat tren
Drive hoac Hugging Face, con checkpoint nen luu tren Drive.

## 1. Chon noi luu data

| Lua chon | Uu diem | Nhuoc diem | Khuyen nghi |
|---|---|---|---|
| Google Drive | De lam, khong can API key | I/O cham neu train truc tiep | Luu checkpoint; phu hop data nho/vua |
| Hugging Face Dataset | Download nhanh, tai lai de, phu hop nhieu runtime | Dataset private can HF token | Khuyen nghi cho data training co dinh |
| Kaggle Dataset | Data hien dang co san | Can Kaggle token va download lai moi runtime | Dung lam nguon backup |

Khong train truc tiep tu Drive. Hay copy experiment dang train vao `/content`
de doc data tren local disk. Chi copy checkpoint ve Drive sau moi stage.

## 2. Cai dat va mount Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
from pathlib import Path

EXPERIMENT = "e1"
MODEL_ID = "unsloth/Qwen3.5-2B"
DRIVE_ROOT = Path("/content/drive/MyDrive/tool_calling_vi")
DRIVE_DATA_ROOT = DRIVE_ROOT / "data"
DRIVE_RUN_DIR = DRIVE_ROOT / "runs" / f"{EXPERIMENT}_{MODEL_ID.rsplit('/', 1)[-1].lower()}"
LOCAL_DATA_ROOT = Path("/content/data")
LOCAL_RUN_DIR = Path("/content/run")
DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
```

Data tren Drive can co layout:

```text
MyDrive/tool_calling_vi/data/
└── e1/
    ├── manifest.json
    └── instruction/train_chat.jsonl
```

Copy data cua experiment dang train vao local:

```python
import shutil

shutil.copytree(
    DRIVE_DATA_ROOT / EXPERIMENT,
    LOCAL_DATA_ROOT / EXPERIMENT,
    dirs_exist_ok=True,
)
```

## 3. Tuy chon tai data tu Hugging Face

Neu data duoc upload vao Dataset repository, co the tai thang vao local disk.
Dataset public khong can token; dataset private can token cua Hugging Face.

```python
%pip install -q huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ThinhDao/tool-calling-vi-experiments",
    repo_type="dataset",
    local_dir=str(LOCAL_DATA_ROOT),
    allow_patterns=[f"{EXPERIMENT}/**", "benchmark_core/**"],
)
```

Neu repository private:

```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get("HF_TOKEN"))
```

Khong ghi token truc tiep vao notebook da chia se.

## 4. Tuy chon tai data tu Kaggle

Chi dung cach nay neu data chua duoc dua len Drive/Hugging Face.

```python
%pip install -q kagglehub
```

Can dat Kaggle token trong Colab Secrets voi ten `KAGGLE_API_TOKEN`, sau do:

```python
from google.colab import userdata
import os

os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
```

```python
import kagglehub

downloaded = kagglehub.dataset_download(
    "phcthnho/tool-calling-vi-experiments"
)
print(downloaded)
```

Dung thu muc dataset tra ve lam `DATA_ROOT`, hoac copy rieng `e1` vao
`/content/data/e1`. Kaggle download khong can thiet neu data da co tren Drive.

## 5. Cai dependency va kiem tra GPU

```python
%pip install -q "unsloth" "transformers>=5.2.0" "trl>=0.15" \
    "accelerate>=1.0" "peft>=0.14" "bitsandbytes>=0.43"
```

```python
import torch

print(torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
```

Colab Pro thuong chi co mot GPU. Khong dung `torchrun` va khong assert hai GPU.

## 6. Dinh nghia cac ham va logic training

Chay cell sau trong Colab de dinh nghia tokenizer, collator va ham `main`:

```python
import unsloth

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastLanguageModel


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(values[0] if values and isinstance(values[0], list) else values)


def tokenize_assistant_only(
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_length: int,
) -> dict[str, list[int]]:
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        messages,
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
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
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
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


class StopAtStep(TrainerCallback):
    def __init__(self, stop_after_step: int | None):
        self.stop_after_step = stop_after_step

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.stop_after_step is not None and state.global_step >= self.stop_after_step:
            control.should_training_stop = True
        return control


def make_training_arguments(kwargs: dict[str, Any]) -> TrainingArguments:
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs["group_by_length"] = kwargs.pop("train_sampling_strategy") == "group_by_length"
        return TrainingArguments(**kwargs)


def tokenized_dataset(
    tokenizer: Any,
    path: Path,
    max_seq_length: int,
    limit: int | None,
    num_proc: int = 2,
) -> Dataset:
    raw_dataset = Dataset.from_text(str(path))
    if limit is not None:
        raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))

    def process_row(example: dict[str, Any]) -> dict[str, list[int]]:
        row = json.loads(example["text"])
        return tokenize_assistant_only(tokenizer, row, max_seq_length)

    return raw_dataset.map(
        process_row,
        remove_columns=["text"],
        num_proc=num_proc,
        desc="Tokenizing dataset",
    )


def main(args_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-id", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=int, default=2)
    args = parser.parse_args(args_list)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch size and gradient accumulation must be >= 1")

    output_dir = Path(args.run_dir) / "checkpoint"
    train_path = Path(args.data_root) / args.experiment / "instruction/train_chat.jsonl"
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
    train_dataset = tokenized_dataset(
        tokenizer,
        train_path,
        args.max_seq_length,
        args.train_limit,
        num_proc=args.num_proc,
    )
    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": 1.0,
        "learning_rate": 5e-7,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "logging_first_step": True,
        "disable_tqdm": False,
        "save_strategy": "steps",
        "save_steps": 250,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "seed": args.seed,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "train_sampling_strategy": "group_by_length",
        "skip_memory_metrics": True,
    }
    trainer = Trainer(
        model=model,
        args=make_training_arguments(training_kwargs),
        train_dataset=train_dataset,
        data_collator=AssistantOnlyCollator(tokenizer),
        callbacks=[StopAtStep(args.stop_after_step)],
    )
    checkpoint = get_last_checkpoint(str(output_dir))
    effective_batch_size = (
        args.per_device_batch_size * args.gradient_accumulation_steps * 1
    )
    print(
        f"world_size=1 per_device_batch_size={args.per_device_batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"effective_batch_size={effective_batch_size} resume_from={checkpoint} "
        f"stop_after_step={args.stop_after_step}"
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("training stage complete:", output_dir)
```

Script nay doc file JSONL bang `Dataset.from_text()`, dung `json.loads` va `.map()` voi
`remove_columns=["text"]` va `num_proc=2` de tokenize thanh dataset chi gom
`input_ids`, `attention_mask`, `labels`, mmap ra dia giup tranh xung dot schema va tiet kiem RAM toi da.

## 7. Resume checkpoint tu Drive

Chay cell nay moi khi bat dau runtime moi:

```python
import shutil

if (DRIVE_RUN_DIR / "checkpoint").exists():
    shutil.copytree(DRIVE_RUN_DIR, LOCAL_RUN_DIR, dirs_exist_ok=True)
```

Khong copy checkpoint cua model khac vao cung `LOCAL_RUN_DIR`.

## 9. Chay training

Goi truc tiep ham `main()` voi cac tham so da cau hinh de bat dau training:

```python
PER_DEVICE_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1

TRAIN_LIMIT = None
STOP_AFTER_STEP = None

train_args = [
    "--experiment", EXPERIMENT,
    "--model-id", MODEL_ID,
    "--run-dir", str(LOCAL_RUN_DIR),
    "--data-root", str(LOCAL_DATA_ROOT),
    "--per-device-batch-size", str(PER_DEVICE_BATCH_SIZE),
    "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
    "--num-proc", "2",
]
if TRAIN_LIMIT is not None:
    train_args.extend(["--train-limit", str(TRAIN_LIMIT)])

if STOP_AFTER_STEP is not None:
    train_args.extend(["--stop-after-step", str(STOP_AFTER_STEP)])


main(train_args)
```

Trial dau tien nen dung `TRAIN_LIMIT = 256` va `STOP_AFTER_STEP = 100`.
Khi trial pass, dung `TRAIN_LIMIT = None`.

> **Luu y khi chay lai (re-run)**: Neu can chay lai cell hoac tiep tuc stage khac tren cung runtime, hay giai phong bo nho VRAM GPU truoc:
> ```python
> import gc, torch
> try:
>     del trainer, model
> except NameError:
>     pass
> gc.collect()
> torch.cuda.empty_cache()
> ```

## 10. Luu checkpoint len Drive

Chi chay sau khi training stage ket thuc:

```python
import subprocess

subprocess.run([
    "rsync", "-a", "--delete",
    f"{LOCAL_RUN_DIR}/",
    f"{DRIVE_RUN_DIR}/",
], check=True)
```

`--delete` dam bao Drive khong giu lai checkpoint cu da bi Trainer xoa boi
`save_total_limit`. Khong chi copy adapter; phai giu ca `optimizer.pt`,
`scheduler.pt`, `trainer_state.json` va `rng_state.pth` trong cac thu muc
`checkpoint/checkpoint-*`.

## 11. Cac stage resume

Voi E1/E2/E3:

```text
750 → 1500 → 2250 → 3000 → 3750
```

Voi E4:

```text
750 → 1500 → 2250 → 3000 → 3750 → 4100
```

`STOP_AFTER_STEP` la moc tuyet doi. Neu checkpoint moi nhat la 750 va dat
`STOP_AFTER_STEP = 1500`, trainer chi train tiep tu step 751 den 1500.

Moi experiment va moi model can mot `DRIVE_RUN_DIR` rieng. Khong resume Qwen3.5-2B
tu checkpoint Qwen3.5-4B.

## 12. Kiem tra sau moi stage

```python
from pathlib import Path

checkpoints = sorted(
    (LOCAL_RUN_DIR / "checkpoint").glob("checkpoint-*"),
    key=lambda path: int(path.name.rsplit("-", 1)[-1]),
)
print([path.name for path in checkpoints])
assert checkpoints
```

Khi runtime bi ngat truoc khi copy Drive, moi thay doi sau checkpoint gan nhat
se mat. Vi vay nen chon stage 500–750 steps thay vi dat sat gioi han runtime.

## 13. Evaluation tren tap Test Tieng Viet (benchmark_core)

Sau khi qua trinh training hoan tat, chay cac cell sau de danh gia mo hinh tren tap `vi/test.jsonl` (7.712 mau).

### Cell 13.1: Don VRAM va Nap Model cho Inference

```python
import gc, torch
from unsloth import FastLanguageModel
from transformers.trainer_utils import get_last_checkpoint

# Giai phong VRAM cua Trainer cu
try:
    del trainer, model
except NameError:
    pass
gc.collect()
torch.cuda.empty_cache()

# Nap checkpoint moi nhat vua train (hoac chi dinh duong dan checkpoint cu the)
CHECKPOINT_DIR = LOCAL_RUN_DIR / "checkpoint"
latest_ckpt = get_last_checkpoint(str(CHECKPOINT_DIR)) or str(CHECKPOINT_DIR)
print(f"Loading adapter from: {latest_ckpt}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=latest_ckpt,
    max_seq_length=4096,
    load_in_4bit=True,
    load_in_16bit=False,
)
tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
FastLanguageModel.for_inference(model)
model.eval()
print("Model loaded in inference mode: READY")
```

### Cell 13.2: Dinh nghia Helpers format Prompt

```python
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

def native_row(record: dict, language: str = "vi") -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[language]},
        {"role": "user", "content": record["query"]},
    ]
    calls = [format_call(call) for call in record.get("function_calls", [])]
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    else:
        fallback = "Hiện tại tôi chưa thể thực hiện yêu cầu này." if language == "vi" else "I cannot complete that request right now."
        messages.append({"role": "assistant", "content": record.get("assistant_content") or fallback})
    return {
        "id": record["id"],
        "messages": messages,
        "tools": [format_tool(tool) for tool in record.get("tools", [])],
    }

def prompt_text(record: dict, language: str = "vi") -> str:
    row = native_row(record, language)
    return tokenizer.apply_chat_template(
        row["messages"][:-1],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
```

### Cell 13.3: High-Throughput Generation (vLLM hoac PyTorch Toi Uu)

Co 2 lua chon de sinh ket qua danh gia:

#### Lua chon A: Dung vLLM (Khuyen nghi - Toc do cao nhat, ~2-3 phut tren A100)

vLLM su dung **Continuous Batching** va **PagedAttention** (tich hop san FlashAttention-2), cho toc do sinh vuot troi gap 5-10 lan so voi `model.generate` thong thuong.

Truoc tien, cai dat vLLM neu chua co:
```bash
pip install vllm
```

Chay cell inference voi vLLM:

```python
import json
import time
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers.trainer_utils import get_last_checkpoint

REVISION = "2026-09-02-full-dedup-seed42"
TEST_PATH = LOCAL_DATA_ROOT / "benchmark_core" / REVISION / "vi/test.jsonl"
PREDICTIONS_PATH = LOCAL_RUN_DIR / f"eval_predictions_{EXPERIMENT}_vi_test.jsonl"
CHECKPOINT_DIR = LOCAL_RUN_DIR / "checkpoint"
latest_ckpt = get_last_checkpoint(str(CHECKPOINT_DIR)) or str(CHECKPOINT_DIR)

# 1. Merge LoRA sang merged_16bit de vLLM doc native
MERGED_DIR = LOCAL_RUN_DIR / "merged_16bit"
if not MERGED_DIR.exists():
    print(f"Merging LoRA from {latest_ckpt} to {MERGED_DIR}...")
    model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_16bit")
    print("Merged completed!")

# 2. Doc danh sach test
with TEST_PATH.open(encoding="utf-8") as source:
    test_records = [json.loads(line) for line in source if line.strip()]

# 3. Khoi tao vLLM engine
llm = LLM(
    model=str(MERGED_DIR),
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    max_tokens=128,
    temperature=0.0,
    stop=["<|im_end|>", "<|endoftext|>"],
)

# 4. Chuan bi prompts
prompts = [prompt_text(record, "vi") for record in test_records]

print(f"Bat dau sinh ket qua cho {len(prompts)} mau test bang vLLM...")
t0 = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
total_time = time.perf_counter() - t0
print(f"Hoan thanh trong {total_time:.1f}s ({len(prompts) / total_time:.1f} samples/s)!")

# 5. Ghi ket qua ra file
with PREDICTIONS_PATH.open("w", encoding="utf-8") as output_file:
    for record, output in zip(test_records, outputs, strict=True):
        raw_output = output.outputs[0].text
        output_file.write(json.dumps({
            "id": record["id"],
            "query": record["query"],
            "gold": record.get("function_calls", []),
            "raw_output": raw_output,
            "latency_ms": round(total_time / len(prompts) * 1000.0, 2),
        }, ensure_ascii=False) + "\n")

print("Raw predictions saved to:", PREDICTIONS_PATH)
```

---

#### Lua chon B: Dung PyTorch + Unsloth (Khong can cai vLLM, toi uu batch 64 + Sort Do dai)

Neu khong muon cai them `vllm`, cell duoi day da duoc toi uu:
1. Săp xep du lieu theo do dai (`length-sorted`) de triet tieu padding thua.
2. Tang `BATCH_SIZE = 64` tan dung toi da GPU A100.
3. Chi dinh EOS `<|im_end|>` de ngat ngay khi goi xong tool call.
4. Ghi append va tu dong resume neu bi ngat.

```python
import json
import os
import time
from pathlib import Path

REVISION = "2026-09-02-full-dedup-seed42"
TEST_PATH = LOCAL_DATA_ROOT / "benchmark_core" / REVISION / "vi/test.jsonl"
PREDICTIONS_PATH = LOCAL_RUN_DIR / f"eval_predictions_{EXPERIMENT}_vi_test.jsonl"

BATCH_SIZE = 64      # 64 tren A100, 16-32 tren T4
MAX_NEW_TOKENS = 128
EVAL_LIMIT = None

def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as source:
        return {json.loads(line)["id"] for line in source if line.strip()}

with TEST_PATH.open(encoding="utf-8") as source:
    test_records = [json.loads(line) for line in source if line.strip()]

if EVAL_LIMIT is not None:
    test_records = test_records[:EVAL_LIMIT]

completed_ids = read_completed_ids(PREDICTIONS_PATH)
pending_records = [r for r in test_records if r["id"] not in completed_ids]

# Toi uu 1: Sap xep theo do dai prompt de gom cac cau ngan vao cung batch, giam 80% padding
pending_records.sort(key=lambda r: len(r.get("query", "")) + len(str(r.get("tools", []))))

print(f"Total: {len(test_records)} | Completed: {len(completed_ids)} | Pending: {len(pending_records)}")

# Toi uu 2: Bat dung token <|im_end|> de dung som
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
stop_ids = [tokenizer.eos_token_id]
if im_end_id is not None and im_end_id not in stop_ids:
    stop_ids.append(im_end_id)

with PREDICTIONS_PATH.open("a", encoding="utf-8") as output_file:
    for start in range(0, len(pending_records), BATCH_SIZE):
        batch = pending_records[start : start + BATCH_SIZE]
        prompts = [prompt_text(record, "vi") for record in batch]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_ids,
            )
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        generated_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        for record, raw_output in zip(batch, texts, strict=True):
            output_file.write(json.dumps({
                "id": record["id"],
                "query": record["query"],
                "gold": record.get("function_calls", []),
                "raw_output": raw_output,
                "batch_latency_ms": round(batch_latency_ms, 2),
                "batch_size": len(batch),
            }, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())

        done = len(completed_ids) + start + len(batch)
        batch_idx = start // BATCH_SIZE + 1
        total_batches = (len(pending_records) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Batch {batch_idx}/{total_batches} | Da xong: {done}/{len(test_records)} ({done / len(test_records) * 100:.1f}%) | Latency: {batch_latency_ms:.0f}ms")

print("Raw predictions saved to:", PREDICTIONS_PATH)
```

### Cell 13.4: Parse Tag va Tinh Metrics

```python
import re

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
            value = value.strip()
            try:
                arguments[parameter.strip()] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                arguments[parameter.strip()] = value
        calls.append({"name": name.strip(), "arguments": arguments})
    if not calls and open_tags > 0:
        errors.append("malformed_tool_call")
    return calls, errors

with PREDICTIONS_PATH.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

positive = negative = tool_correct = negative_correct = exact_match = syntax_errors = 0
total_latency_ms = 0.0

for row in rows:
    predicted, errors = parse_native_output(row["raw_output"])
    gold = row["gold"]
    is_positive = bool(gold)
    tool_match = [c["name"] for c in predicted] == [c["name"] for c in gold]
    exact = (predicted == gold)

    if errors:
        syntax_errors += 1
    if is_positive:
        positive += 1
        if tool_match:
            tool_correct += 1
        if exact and not errors:
            exact_match += 1
    else:
        negative += 1
        if not predicted and not errors:
            negative_correct += 1
            exact_match += 1

    total_latency_ms += row.get("batch_latency_ms", 0.0) / max(row.get("batch_size", 1), 1)

total = len(rows)
metrics = {
    "total_evaluated": total,
    "tool_selection_accuracy": round(tool_correct / positive, 4) if positive else 0.0,
    "negative_correctness": round(negative_correct / negative, 4) if negative else 0.0,
    "exact_match_accuracy": round(exact_match / total, 4) if total else 0.0,
    "syntax_error_rate": round(syntax_errors / total, 4) if total else 0.0,
    "avg_latency_ms": round(total_latency_ms / total, 2) if total else 0.0,
}

print("\n=== EVALUATION RESULTS ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

metrics_path = LOCAL_RUN_DIR / f"eval_metrics_{EXPERIMENT}_vi_test.json"
metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved metrics to {metrics_path}")
```

### Cell 13.5: Dong Bo Ket Qua Danh Gia Len Drive

```python
import shutil

for fname in [f"eval_predictions_{EXPERIMENT}_vi_test.jsonl", f"eval_metrics_{EXPERIMENT}_vi_test.json"]:
    src = LOCAL_RUN_DIR / fname
    if src.exists():
        shutil.copy(src, DRIVE_RUN_DIR / fname)
        print(f"Synced {fname} to Drive!")
```
