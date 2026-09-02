"""Train native Qwen3.5 tool-calling rows with Unsloth and assistant-only loss."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.models.slm.native_qwen import AssistantOnlyCollator, tokenize_assistant_only


@dataclass(frozen=True, slots=True)
class TrainConfig:
    train_file: Path
    output_dir: Path
    model_name: str = "unsloth/Qwen3.5-4B"
    eval_file: Path | None = None
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: float = 1.0
    learning_rate: float = 5e-7
    warmup_ratio: float = 0.05
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    seed: int = 42
    load_in_4bit: bool = True
    logging_steps: int = 10
    save_total_limit: int = 2
    resume_from_checkpoint: str | None = None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            rows.append(row)
    return rows


def _load_training_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch, datasets, transformers, and unsloth. "
            "Install the environment described by the Unsloth guide."
        ) from exc
    return torch, Dataset, Trainer, TrainingArguments, FastLanguageModel


def _training_arguments(
    training_arguments: Any,
    config: TrainConfig,
    has_eval: bool,
) -> Any:
    import torch

    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "seed": config.seed,
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
    }
    if has_eval:
        kwargs["eval_strategy"] = "epoch"
    try:
        return training_arguments(**kwargs)
    except TypeError:
        if has_eval:
            kwargs.pop("eval_strategy", None)
            kwargs["evaluation_strategy"] = "epoch"
        return training_arguments(**kwargs)


def train(config: TrainConfig) -> dict[str, Any]:
    """Run one SFT job and save its run configuration beside the adapter."""
    if not config.train_file.exists():
        raise FileNotFoundError(config.train_file)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _torch, dataset_type, trainer_type, arguments_type, model_loader = _load_training_dependencies()
    model, tokenizer = model_loader.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_16bit=not config.load_in_4bit,
        full_finetuning=False,
    )
    model = model_loader.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    train_rows = _read_rows(config.train_file)
    eval_rows = _read_rows(config.eval_file) if config.eval_file else []
    train_features = [
        tokenize_assistant_only(tokenizer, row, config.max_seq_length) for row in train_rows
    ]
    eval_features = [
        tokenize_assistant_only(tokenizer, row, config.max_seq_length) for row in eval_rows
    ]
    train_dataset = dataset_type.from_list(train_features)
    eval_dataset = dataset_type.from_list(eval_features) if eval_features else None
    arguments = _training_arguments(arguments_type, config, eval_dataset is not None)
    trainer = trainer_type(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    run_metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "assistant_only_loss": True,
        "chat_template": "loaded from model checkpoint at runtime",
    }
    (config.output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_metadata


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(_parse_args())
