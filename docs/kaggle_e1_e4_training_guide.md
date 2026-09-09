# Kaggle E1-E4 Training Guide

Guide nay dung cho E1, E2, E3 va E4 voi Qwen3.5-4B tren 2 T4. Muc tieu la tang
throughput nhung giu nguyen
effective batch size 16 cua controlled track:

```text
per-device batch 2 x gradient accumulation 4 x 2 GPUs = 16
```

Khong goi `FastLanguageModel.for_inference()` trong guide nay. Ham do chi dung
sau training de generation/evaluation.

## Cell 1: Cau Hinh Va Preflight

Thay `DATA_ROOT` bang dung duong dan Dataset tren Kaggle cua ban. Sau khi chay
stage 1, attach output cua stage truoc vao notebook tiep theo truoc khi chay
Cell nay.

```python
from collections import Counter
import hashlib
import json
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/datasets/phcthnho/tool-calling-vi-experiments")
REVISION = "2026-09-02-full-dedup-seed42"
REVISION_DIR = DATA_ROOT / "benchmark_core" / REVISION
EXPERIMENT = "e1"  # e1, e2, e3, hoac e4
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

## Cell 2: Cai Dependency Va Kiem Tra GPU

Chay cell cai dat, restart kernel neu Kaggle yeu cau, roi chay lai tu Cell 1.

```python
%pip install -q "unsloth" "transformers>=5.2.0" "trl>=0.15" \
    "accelerate>=1.0" "peft>=0.14" "bitsandbytes>=0.43"
```

```python
import torch

assert torch.cuda.device_count() == 2, "Enable 2 x T4 before training"
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
```

## Nguyen Nhan Timeout Cu

Cau hinh cu chay 8 micro-batch moi optimizer step, in tqdm sau moi step, va
chay validation loss tren toan bo 7,701 mau sau epoch. Log E1 cho thay ca stdout
bi nghen (`IOStream.flush timed out`) va throughput khoang 22-25 giay/step.

Trainer khong materialize 60,000 mau thanh Python list trong moi DDP process.
JSONL duoc doc tung dong va tokenize qua generator de giam peak host RAM.
Cach nay cung tranh loi PyArrow khi schema JSON long co kieu khong dong nhat;
xu ly local, khong can HF API key.

Guide nay:

- dung batch 2, accumulation 4;
- tat tqdm va chi log moi 50 steps;
- tat full validation trong training mot epoch, vi khong co checkpoint trung gian
  nao duoc chon bang full validation;
- tat `find_unused_parameters`, vi DDP log xac nhan khong co parameter nao unused;
- luu checkpoint moi 250 steps va dung gracefully o moc step chi dinh de resume
  qua nhieu Kaggle session.

## Cell 3: Native Template Smoke Test

`import unsloth` phai xay ra truoc khi import `transformers`. Cell nay kiem tra
template cua dung checkpoint tren cac native training rows truoc khi ton GPU
cho DDP.

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
print("native tokenizer smoke test: PASS")
```

## Cell 4: Cau Hinh Stage

Cell nay dat cau hinh toi uu an toan de chay ngay. Khong tao full validation
trong training: voi mot epoch, full validation sau final checkpoint khong dung
de chon checkpoint va lam session vuot gio.

```python
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
STOP_AFTER_STEP = 1000
```

`STOP_AFTER_STEP` la optimizer step tuyet doi, khong phai so step chay them.
Tong step du kien voi effective batch 16 la `3750` cho E1/E2/E3 va `4100` cho
E4. Chay cac session lan luot:

| Experiment | Cac gia tri `STOP_AFTER_STEP` |
|---|---|
| E1, E2, E3 | `1000`, `2000`, `3000`, `3750` |
| E4 | `1000`, `2000`, `3000`, `4100` |

Moi session ket thuc sach truoc 12 gio. Publish `RUN_DIR` thanh Kaggle Notebook
Output/Dataset sau moi session, attach output do vao session ke tiep, roi copy
lai vao `RUN_DIR` truoc khi chay `torchrun`.

```python
# Chi dung o session 2 tro di. Doi path thanh Kaggle input cua output truoc do.
from pathlib import Path
import shutil

PREVIOUS_RUN_DIR = Path("/kaggle/input/e1-qwen35-4b-stage-1000/e1_qwen3.5-4b")
if PREVIOUS_RUN_DIR.exists() and not (RUN_DIR / "checkpoint").exists():
    shutil.copytree(PREVIOUS_RUN_DIR, RUN_DIR, dirs_exist_ok=True)
```

## Cell 5: Script Training

Tao script sau. `import unsloth` phai dung truoc
`transformers` va `peft`; dieu nay sua dung warning trong log.

```python
%%writefile train_kaggle.py
import unsloth

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastLanguageModel


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


class StopAtStep(TrainerCallback):
    def __init__(self, stop_after_step: int | None):
        self.stop_after_step = stop_after_step

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_after_step is not None and state.global_step >= self.stop_after_step:
            control.should_training_stop = True
        return control


def make_training_arguments(kwargs: dict) -> TrainingArguments:
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        # Transformers truoc v5 dung ten cu cho length grouping.
        kwargs["group_by_length"] = kwargs.pop("train_sampling_strategy") == "group_by_length"
        return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-id", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.run_dir) / "checkpoint"
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

    train_path = Path(args.data_root) / args.experiment / "instruction/train_chat.jsonl"
    def tokenized_rows():
        count = 0
        with train_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                if args.train_limit is not None and count >= args.train_limit:
                    break
                count += 1
                yield tokenize_assistant_only(
                    tokenizer, json.loads(line), args.max_seq_length
                )

    train_dataset = Dataset.from_generator(tokenized_rows)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        args.per_device_batch_size * args.gradient_accumulation_steps * world_size
    )
    if effective_batch_size != 16:
        raise ValueError(f"Expected effective batch size 16, got {effective_batch_size}")

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
        "disable_tqdm": True,
        "save_strategy": "steps",
        "save_steps": 250,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "seed": args.seed,
        "fp16": bool(torch.cuda.is_available()),
        "bf16": False,
        "ddp_backend": "nccl",
        "ddp_find_unused_parameters": False,
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
    print(
        f"world_size={world_size} effective_batch_size={effective_batch_size} "
        f"resume_from={checkpoint} stop_after_step={args.stop_after_step}"
    )
    trainer.train(resume_from_checkpoint=checkpoint)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("training stage complete:", output_dir)


if __name__ == "__main__":
    main()
```

## Cell 6: Chay Mot Stage

Moi stage xong thi publish `RUN_DIR`, attach no vao stage ke tiep va copy
checkpoint nhu phan cau hinh o tren.

```python
train_limit_arg = f"--train-limit {TRAIN_LIMIT}" if TRAIN_LIMIT is not None else ""
n_gpus = torch.cuda.device_count()
assert n_gpus == 2, f"Expected 2 T4 GPUs, got {n_gpus}"

!torchrun --nproc_per_node=2 train_kaggle.py \
    --experiment {EXPERIMENT} \
    --model-id {MODEL_ID} \
    --run-dir {RUN_DIR} \
    --data-root {DATA_ROOT} \
    --per-device-batch-size {PER_DEVICE_BATCH_SIZE} \
    --gradient-accumulation-steps {GRADIENT_ACCUMULATION_STEPS} \
    --stop-after-step {STOP_AFTER_STEP} \
    {train_limit_arg}
```

## Cell 7: Luu Run Manifest

Chay sau moi stage, truoc khi publish Notebook Output/Dataset.

```python
run_manifest = {
    "experiment": EXPERIMENT,
    "model": MODEL_ID,
    "revision": REVISION,
    "train_limit": TRAIN_LIMIT,
    "train_rows_used": len(train_rows) if TRAIN_LIMIT is None else TRAIN_LIMIT,
    "max_seq_length": 4096,
    "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": 16,
    "n_gpus": 2,
    "stop_after_step": STOP_AFTER_STEP,
    "seed": 42,
    "assistant_only_loss": True,
    "native_chat_template": True,
}
(RUN_DIR / "run_manifest.json").write_text(
    json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
```

## Cell 8: Nap Checkpoint Cho Evaluation

Chay Cell 8-10 trong notebook/session rieng sau stage cuoi. Attach Kaggle Output
cua stage cuoi va dat `CHECKPOINT_SOURCE` den thu muc `checkpoint` cua output
do. Khong danh gia full 7,712 mau trong cung session voi training.

```python
import json
import os
import re
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

CHECKPOINT_SOURCE = Path("/kaggle/input/e1-qwen35-4b-final/e1_qwen3.5-4b/checkpoint")
EVAL_LANGUAGE = "vi"
EVALUATION_NAME = f"{RUN_NAME}_{EVAL_LANGUAGE}_test"
PREDICTIONS_PATH = RUN_DIR / f"eval_predictions_{EVALUATION_NAME}.jsonl"
assert CHECKPOINT_SOURCE.is_dir(), CHECKPOINT_SOURCE

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
        "function": {"name": call["name"], "arguments": call.get("arguments", {})},
    }

def native_row(record: dict, language: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[language]},
        {"role": "user", "content": record["query"]},
    ]
    calls = [format_call(call) for call in record["function_calls"]]
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    else:
        fallback = "I cannot complete that request right now."
        if language == "vi":
            fallback = "Hiện tại tôi chưa thể thực hiện yêu cầu này."
        messages.append({"role": "assistant", "content": record.get("assistant_content") or fallback})
    return {
        "id": record["id"],
        "messages": messages,
        "tools": [format_tool(tool) for tool in record["tools"]],
    }

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(CHECKPOINT_SOURCE),
    max_seq_length=4096,
    load_in_4bit=True,
)
tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
FastLanguageModel.for_inference(model)
model.eval()
```

## Cell 9: Batched, Resumable Generation

Chay truoc voi 20 mau de kiem tra output. Sau do dat `EVAL_LIMIT = None` va
chay lai cell, cac ID da ghi trong JSONL se duoc bo qua.

```python
BATCH_SIZE = 4
MAX_NEW_TOKENS = 128
EVAL_LIMIT = 20

def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as source:
        return {json.loads(line)["id"] for line in source if line.strip()}

def prompt_text(record: dict) -> str:
    row = native_row(record, EVAL_LANGUAGE)
    return tokenizer.apply_chat_template(
        row["messages"][:-1], tools=row["tools"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )

test_records = read_jsonl(REVISION_DIR / EVAL_LANGUAGE / "test.jsonl")
if EVAL_LIMIT is not None:
    test_records = test_records[:EVAL_LIMIT]
completed_ids = read_completed_ids(PREDICTIONS_PATH)
pending_records = [record for record in test_records if record["id"] not in completed_ids]
print(f"completed={len(completed_ids)}, pending={len(pending_records)}")

with PREDICTIONS_PATH.open("a", encoding="utf-8") as output_file:
    for start in range(0, len(pending_records), BATCH_SIZE):
        batch = pending_records[start : start + BATCH_SIZE]
        encoded = tokenizer(
            [prompt_text(record) for record in batch],
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
        if done % 100 == 0 or done == len(test_records):
            peak = [round(torch.cuda.max_memory_allocated(i) / 1024**3, 2)
                    for i in range(torch.cuda.device_count())]
            print(f"processed={done}/{len(test_records)} batch_ms={batch_latency_ms:.0f} peak_vram_gib={peak}")
```

## Cell 10: Tinh Metric

Cell nay co the cham preview 20 mau sau smoke generation. Chi dung ket qua trong
bao cao khi `is_complete` la `true`, nghia la generation du `7,712` mau.

```python
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

rows = read_jsonl(PREDICTIONS_PATH)
assert len({row['id'] for row in rows}) == len(rows), "Duplicate prediction IDs"
expected_ids = {
    record["id"] for record in read_jsonl(REVISION_DIR / EVAL_LANGUAGE / "test.jsonl")
}
assert {row["id"] for row in rows} <= expected_ids, "Prediction contains an unknown test ID"
is_complete = len(rows) == len(expected_ids)
if not is_complete:
    print(f"PREVIEW ONLY: {len(rows)}/{len(expected_ids)} predictions. Set EVAL_LIMIT = None and resume generation for final metrics.")

positive = negative = tool_correct = negative_correct = exact_match = syntax_errors = 0
latency_ms = 0.0
scored_rows = []
for row in rows:
    predicted, errors = parse_native_output(row["raw_output"])
    gold = row["gold"]
    is_positive = bool(gold)
    tool_match = [call["name"] for call in predicted] == [call["name"] for call in gold]
    exact = predicted == gold
    if is_positive:
        positive += 1
        tool_correct += tool_match
    else:
        negative += 1
        negative_correct += not predicted and not errors
    exact_match += exact
    syntax_errors += bool(errors)
    latency_ms += row["batch_latency_ms"] / row["batch_size"]
    scored_rows.append({
        **row,
        "predicted": predicted,
        "errors": errors,
        "tool_match": tool_match if is_positive else (not predicted and not errors),
        "exact_match": exact,
    })

metrics = {
    "split": EVALUATION_NAME,
    "total_samples": len(rows),
    "expected_total_samples": len(expected_ids),
    "is_complete": is_complete,
    "positive_samples": positive,
    "negative_samples": negative,
    "tool_accuracy_pos_pct": round(100 * tool_correct / positive, 2),
    "non_fc_recall_pct": round(100 * negative_correct / negative, 2),
    "arga_exact_match_pct": round(100 * exact_match / len(rows), 2),
    "syntax_error_rate_pct": round(100 * syntax_errors / len(rows), 2),
    "avg_latency_ms": round(latency_ms / len(rows), 2),
}
(RUN_DIR / f"eval_predictions_{EVALUATION_NAME}_scored.json").write_text(
    json.dumps(scored_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(RUN_DIR / f"eval_metrics_{EVALUATION_NAME}.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
```

## Neu Gap OOM Hoac Con Nhieu VRAM

- VRAM cua E0 khong duoc dung de chon training batch: E0 la inference va model
  co the shard tren hai GPU, trong khi DDP training dat mot model replica, gradient
  va activation tren moi GPU.
- Chay trial `TRAIN_LIMIT = 256` voi cac cau hinh sau, theo dung thu tu. Moi cau
  hinh deu giu `effective_batch_size = 16`, nen khong thay doi training budget:

| Thu tu | Per-device batch | Gradient accumulation | Global effective batch |
|---|---:|---:|---:|
| Safe fallback | 1 | 8 | 16 |
| Default | 2 | 4 | 16 |
| Nen thu tiep | 4 | 2 | 16 |
| Muc toi da de thu | 8 | 1 | 16 |

- Chi dung batch 4 hoac 8 cho full run neu trial khong OOM va thoi gian moi
  optimizer step giam ro rang. Batch 8 co the OOM khi DDP gap batch gom nhieu
  prompt dai gan 4096 tokens.
- Khong dung batch 16 cho main controlled track: voi 2 GPU, accumulation toi
  thieu la 1 nen effective batch thanh 32 va khong con cung training budget.
- Khong doi epoch, learning rate, seed, sample IDs hoac `max_seq_length` 4096
  khi doi batch/accumulation.
- Khong giam `max_seq_length` 4096 trong main controlled track. Cach nay nhanh
  hon nhung thay doi training budget va co the cat tool schema/assistant target.
- Luon luu `RUN_DIR` sau moi stage, bao gom `checkpoint/checkpoint-*`; chi adapter
  o thu muc goc khong du de resume optimizer va scheduler.
