"""Training loop cho Cross-Encoder (§Phase 3 method2_plan).

Viết tay thay vì dùng `transformers.Trainer` vì batch mang theo `schema_type`
dạng list[str] để route loss — Trainer sẽ cố collate nó thành tensor.

Ba điểm bám sát ràng buộc T4 (§4.3):

- `torch.amp` fp16 + `GradScaler` (T4 không có bf16).
- Dynamic padding, `max_length=256` — riêng thay đổi này đã ~4× nhanh hơn.
- Checkpoint mỗi `save_steps` vào `output_dir`, luôn hỗ trợ `resume_from`,
  vì Kaggle có thể ngắt session bất ngờ.

Curriculum 2 giai đoạn: warm-up trên glaive+xLAM (học kỹ năng span tổng quát)
rồi fine-tune trên custom_vi với lr thấp hơn.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from src.models.crossencoder.data_collator import CollatorConfig, CrossEncoderCollator
from src.models.crossencoder.dataset import CrossEncoderDataset
from src.models.crossencoder.heads import HeadConfig
from src.models.crossencoder.losses import HierarchicalLoss, LossConfig
from src.models.crossencoder.model import CrossEncoderForExtraction

logger = logging.getLogger(__name__)


@dataclass
class Stage:
    name: str
    sources: list[str]
    epochs: int
    lr: float


@dataclass
class CrossEncoderTrainConfig:
    model_name: str = "xlm-roberta-base"
    max_enum_size: int = 20
    dropout: float = 0.1

    train_path: Path = Path("data/method2/crossencoder/train.jsonl")
    val_path: Path = Path("data/method2/crossencoder/val.jsonl")
    max_length: int = 256
    padding: str = "longest"
    num_workers: int = 2

    output_dir: Path = Path("artifacts/method2/crossencoder/run01")
    batch_size: int = 32
    grad_accum: int = 2
    eval_batch_size: int = 16
    epochs: int = 3
    lr: float = 3e-5
    head_lr: float = 1e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = False
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    resume_from: str | None = None

    loss: LossConfig = field(default_factory=LossConfig)
    curriculum_enabled: bool = True
    stages: list[Stage] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CrossEncoderTrainConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        model_raw, data_raw, train_raw = raw.get("model", {}), raw.get("data", {}), raw.get("train", {})
        loss_raw, curriculum_raw = raw.get("loss", {}), raw.get("curriculum", {})
        defaults = cls()
        return cls(
            model_name=str(model_raw.get("name", defaults.model_name)),
            max_enum_size=int(model_raw.get("max_enum_size", defaults.max_enum_size)),
            dropout=float(model_raw.get("dropout", defaults.dropout)),
            train_path=Path(data_raw.get("train_path", defaults.train_path)),
            val_path=Path(data_raw.get("val_path", defaults.val_path)),
            max_length=int(data_raw.get("max_length", defaults.max_length)),
            padding=str(data_raw.get("padding", defaults.padding)),
            num_workers=int(data_raw.get("num_workers", defaults.num_workers)),
            output_dir=Path(train_raw.get("output_dir", defaults.output_dir)),
            batch_size=int(train_raw.get("batch_size", defaults.batch_size)),
            grad_accum=int(train_raw.get("grad_accum", defaults.grad_accum)),
            eval_batch_size=int(train_raw.get("eval_batch_size", defaults.eval_batch_size)),
            epochs=int(train_raw.get("epochs", defaults.epochs)),
            lr=float(train_raw.get("lr", defaults.lr)),
            head_lr=float(train_raw.get("head_lr", defaults.head_lr)),
            warmup_ratio=float(train_raw.get("warmup_ratio", defaults.warmup_ratio)),
            weight_decay=float(train_raw.get("weight_decay", defaults.weight_decay)),
            max_grad_norm=float(train_raw.get("max_grad_norm", defaults.max_grad_norm)),
            fp16=bool(train_raw.get("fp16", defaults.fp16)),
            gradient_checkpointing=bool(train_raw.get("gradient_checkpointing", False)),
            logging_steps=int(train_raw.get("logging_steps", defaults.logging_steps)),
            eval_steps=int(train_raw.get("eval_steps", defaults.eval_steps)),
            save_steps=int(train_raw.get("save_steps", defaults.save_steps)),
            save_total_limit=int(train_raw.get("save_total_limit", defaults.save_total_limit)),
            seed=int(train_raw.get("seed", defaults.seed)),
            resume_from=train_raw.get("resume_from"),
            loss=LossConfig(
                has_value_weight=float(loss_raw.get("has_value_weight", 1.0)),
                span_weight=float(loss_raw.get("span_weight", 1.0)),
                enum_weight=float(loss_raw.get("enum_weight", 1.0)),
                boolean_weight=float(loss_raw.get("boolean_weight", 1.0)),
                span_loss_combiner=str(loss_raw.get("span_loss_combiner", "sum")),
            ),
            curriculum_enabled=bool(curriculum_raw.get("enabled", True)),
            stages=[
                Stage(
                    name=str(s["name"]),
                    sources=list(s.get("sources", [])),
                    epochs=int(s.get("epochs", 1)),
                    lr=float(s.get("lr", defaults.lr)),
                )
                for s in curriculum_raw.get("stages", [])
            ],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_by_sources(dataset: CrossEncoderDataset, sources: Sequence[str]) -> Subset:
    wanted = set(sources)
    indices = [i for i, row in enumerate(dataset.rows) if row.get("source") in wanted]
    return Subset(dataset, indices)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    if "labels" in batch:
        moved["labels"] = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch["labels"].items()
        }
    moved["schema_type"] = batch.get("schema_type")
    return moved


def forward_batch(model: CrossEncoderForExtraction, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
        query_token_mask=batch["query_token_mask"],
    )


# ------------------------------------------------------------------ checkpoint


def save_checkpoint(
    model: CrossEncoderForExtraction,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    step: int,
    config: CrossEncoderTrainConfig,
    tag: str | None = None,
) -> Path:
    path = config.output_dir / (tag or f"checkpoint-{step}")
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
        },
        path / "trainer_state.pt",
    )
    _prune_checkpoints(config)
    return path


def _prune_checkpoints(config: CrossEncoderTrainConfig) -> None:
    checkpoints = sorted(
        (p for p in config.output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for stale in checkpoints[: -config.save_total_limit]:
        for item in sorted(stale.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
        stale.rmdir()


def find_last_checkpoint(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if (p / "trainer_state.pt").exists()]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda p: int(p.name.split("-")[-1])))


# ----------------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(
    model: CrossEncoderForExtraction,
    loader: DataLoader,
    loss_fn: HierarchicalLoss,
    device: torch.device,
    fp16: bool,
) -> dict[str, float]:
    """Metric theo head — dùng cho gate §Phase 3."""
    from src.models.crossencoder.evaluate import ComponentMetrics

    model.eval()
    metrics = ComponentMetrics()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            outputs = forward_batch(model, batch)
            losses = loss_fn(outputs, batch["labels"])
        total_loss += float(losses["loss"].item())
        n_batches += 1
        metrics.update(outputs, batch["labels"])
    model.train()
    result = metrics.compute()
    result["loss"] = round(total_loss / max(n_batches, 1), 4)
    return result


# ---------------------------------------------------------------------- train


def train(config: CrossEncoderTrainConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    collator = CrossEncoderCollator(
        CollatorConfig(
            tokenizer_name=config.model_name,
            max_length=config.max_length,
            padding=config.padding,
        ),
        tokenizer=tokenizer,
    )
    train_dataset = CrossEncoderDataset(config.train_path, collator, config.max_length)
    val_dataset = (
        CrossEncoderDataset(config.val_path, collator, config.max_length)
        if config.val_path.exists()
        else None
    )
    logger.info(
        "train=%d val=%d (bỏ %d cặp không căn được span trong cửa sổ %d token)",
        len(train_dataset),
        len(val_dataset) if val_dataset else 0,
        train_dataset.n_dropped_unalignable,
        config.max_length,
    )

    model = CrossEncoderForExtraction(
        model_name=config.model_name,
        head_config=HeadConfig(max_enum_size=config.max_enum_size, dropout=config.dropout),
    ).to(device)
    if config.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    loss_fn = HierarchicalLoss(config.loss)
    stages = (
        config.stages
        if config.curriculum_enabled and config.stages
        else [Stage(name="all", sources=[], epochs=config.epochs, lr=config.lr)]
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        if val_dataset
        else None
    )

    history: list[dict[str, Any]] = []
    global_step = 0
    resume_from = config.resume_from or find_last_checkpoint(config.output_dir)

    for stage in stages:
        data = subset_by_sources(train_dataset, stage.sources) if stage.sources else train_dataset
        if len(data) == 0:
            logger.warning("giai đoạn %s không có sample nào — bỏ qua", stage.name)
            continue
        loader = DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=config.num_workers,
            drop_last=False,
        )
        optimizer = _build_optimizer(model, stage.lr, config)
        total_steps = math.ceil(len(loader) / config.grad_accum) * stage.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(total_steps * config.warmup_ratio), total_steps
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.fp16 and device.type == "cuda")

        if resume_from:
            global_step = _load_trainer_state(resume_from, optimizer, scheduler, scaler)
            logger.info("resume từ %s tại step %d", resume_from, global_step)
            resume_from = None

        logger.info("giai đoạn %s: %d sample, %d step", stage.name, len(data), total_steps)
        model.train()
        started = time.perf_counter()

        for epoch in range(stage.epochs):
            for micro_step, batch in enumerate(loader):
                batch = move_batch(batch, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    outputs = forward_batch(model, batch)
                    losses = loss_fn(outputs, batch["labels"])
                    loss = losses["loss"] / config.grad_accum
                scaler.scale(loss).backward()

                if (micro_step + 1) % config.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if global_step % config.logging_steps == 0:
                        logger.info(
                            "stage=%s epoch=%d step=%d loss=%.4f has_value=%.4f sub=%.4f",
                            stage.name, epoch, global_step,
                            float(losses["loss"]), float(losses["loss_has_value"]),
                            float(losses["loss_sub"]),
                        )
                    if val_loader and global_step % config.eval_steps == 0:
                        metrics = evaluate(model, val_loader, loss_fn, device, config.fp16)
                        metrics.update({"step": global_step, "stage": stage.name})
                        history.append(metrics)
                        logger.info("eval @%d: %s", global_step, json.dumps(metrics))
                    if global_step % config.save_steps == 0:
                        save_checkpoint(
                            model, tokenizer, optimizer, scheduler, scaler, global_step, config
                        )

        logger.info("giai đoạn %s xong sau %.1f phút", stage.name, (time.perf_counter() - started) / 60)
        save_checkpoint(
            model, tokenizer, optimizer, scheduler, scaler, global_step, config,
            tag=f"stage-{stage.name}",
        )

    final_dir = config.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    final_metrics = evaluate(model, val_loader, loss_fn, device, config.fp16) if val_loader else {}
    report = {
        "final_metrics": final_metrics,
        "history": history,
        "n_train_pairs": len(train_dataset),
        "n_dropped_unalignable": train_dataset.n_dropped_unalignable,
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available()
            else None
        ),
    }
    (config.output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _build_optimizer(model: CrossEncoderForExtraction, lr: float, config: CrossEncoderTrainConfig):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")
    encoder_params = list(model.encoder.named_parameters())
    groups = [
        {
            "params": [p for n, p in encoder_params if not any(k in n for k in no_decay)],
            "lr": lr,
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in encoder_params if any(k in n for k in no_decay)],
            "lr": lr,
            "weight_decay": 0.0,
        },
        {
            "params": list(model.heads.parameters()),
            "lr": config.head_lr,
            "weight_decay": config.weight_decay,
        },
    ]
    return torch.optim.AdamW(groups)


def _load_trainer_state(path: str, optimizer, scheduler, scaler) -> int:
    state = torch.load(Path(path) / "trainer_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return int(state.get("step", 0))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[crossencoder] %(message)s")
    parser = argparse.ArgumentParser(description="Train Cross-Encoder for parameter extraction")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/crossencoder.yaml"))
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--show-config", action="store_true")
    args = parser.parse_args()

    config = CrossEncoderTrainConfig.from_yaml(args.config)
    # Notebook nội suy `None` thành chuỗi "None"; coi như không resume.
    if args.resume_from and args.resume_from.strip().lower() not in ("", "none", "null"):
        config.resume_from = args.resume_from
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_curriculum:
        config.curriculum_enabled = False

    if args.show_config:
        print(json.dumps({k: str(v) for k, v in config.__dict__.items()}, indent=2, ensure_ascii=False))
        return

    report = train(config)
    print(json.dumps(report["final_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
