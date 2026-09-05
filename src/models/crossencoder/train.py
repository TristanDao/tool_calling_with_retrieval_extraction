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

from src.models.crossencoder.data_collator import (
    DEFAULT_MAX_QUESTION_TOKENS,
    CollatorConfig,
    CrossEncoderCollator,
)
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
    #: Trần token của schema question. PHẢI khớp `extraction.max_question_tokens`
    #: bên pipeline.yaml — lệch là train/serve skew.
    max_question_tokens: int = DEFAULT_MAX_QUESTION_TOKENS
    dropout: float = 0.1
    #: Ablation §6.1 — bật head `should_call`. PHẢI bật cùng lúc với
    #: `pairs.should_call.enabled`, nếu không head được tạo mà không có
    #: hàng nào dạy nó (hoặc ngược lại, có nhãn mà không có head để học).
    enable_should_call: bool = False

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
    #: >0 để dừng sớm (smoke run) — đo throughput trước khi tiêu nhiều giờ GPU.
    max_steps: int | None = None

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
            max_question_tokens=int(
                model_raw.get("max_question_tokens", defaults.max_question_tokens)
            ),
            dropout=float(model_raw.get("dropout", defaults.dropout)),
            enable_should_call=bool(
                model_raw.get("enable_should_call", defaults.enable_should_call)
            ),
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
                should_call_weight=float(loss_raw.get("should_call_weight", 1.0)),
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
    stage_index: int = 0,
    epoch: int = 0,
) -> Path:
    path = config.output_dir / (tag or f"checkpoint-{step}")
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    # progress.json tách khỏi trainer_state.pt: đọc vị trí không cần torch.load
    # ~3 GB optimizer state, và thấy được cả checkpoint gắn tag `stage-*`.
    (path / "progress.json").write_text(
        json.dumps({"step": step, "stage_index": stage_index, "epoch": epoch}),
        encoding="utf-8",
    )
    torch.save(
        {
            "step": step,
            # Vị trí trong curriculum: thiếu hai trường này thì resume luôn quay
            # về giai đoạn đầu và train lại từ epoch 0.
            "stage_index": stage_index,
            "epoch": epoch,
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
    """Checkpoint mới nhất theo `step`, tính cả bản gắn tag `stage-*`.

    Sắp theo tên là sai với `stage-warmup` (không parse được số) và cũng sai với
    'checkpoint-1000' < 'checkpoint-500' khi so chuỗi — nên đọc `step` thật từ
    progress.json.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in list(output_dir.glob("checkpoint-*")) + list(output_dir.glob("stage-*")):
        if not (path / "trainer_state.pt").exists():
            continue
        progress = path / "progress.json"
        if progress.exists():
            step = int(json.loads(progress.read_text(encoding="utf-8")).get("step", 0))
        else:
            # Checkpoint sinh trước khi có progress.json.
            tail = path.name.split("-")[-1]
            step = int(tail) if tail.isdigit() else 0
        candidates.append((step, path))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


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

    dataset_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    collator = CrossEncoderCollator(
        CollatorConfig(
            tokenizer_name=config.model_name,
            max_length=config.max_length,
            padding=config.padding,
            max_question_tokens=config.max_question_tokens,
        ),
        tokenizer=tokenizer,
    )
    train_dataset = CrossEncoderDataset(config.train_path, collator, config.max_length)
    val_dataset = (
        CrossEncoderDataset(config.val_path, collator, config.max_length)
        if config.val_path.exists()
        else None
    )
    dataset_seconds = time.perf_counter() - dataset_started
    logger.info(
        "dựng dataset mất %.1f s", dataset_seconds
    )
    logger.info(
        "train=%d val=%d (bỏ %d cặp không căn được span trong cửa sổ %d token)",
        len(train_dataset),
        len(val_dataset) if val_dataset else 0,
        train_dataset.n_dropped_unalignable,
        config.max_length,
    )

    # PHẢI xác định checkpoint TRƯỚC khi dựng model: `from_pretrained` nạp cả
    # encoder lẫn 4 head đã train. Dựng model mới rồi chỉ nạp optimizer state là
    # mất sạch trọng số — resume khi đó còn tệ hơn train lại từ đầu.
    resume_path = config.resume_from or find_last_checkpoint(config.output_dir)
    if resume_path:
        model = CrossEncoderForExtraction.from_pretrained(resume_path).to(device)
        logger.info("nạp lại trọng số từ %s", resume_path)
    else:
        model = CrossEncoderForExtraction(
            model_name=config.model_name,
            head_config=HeadConfig(
                max_enum_size=config.max_enum_size,
                dropout=config.dropout,
                enable_should_call=config.enable_should_call,
            ),
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
    resume_stage, resume_epoch = 0, 0
    if resume_path:
        position = _read_trainer_state(resume_path)
        global_step = position["step"]
        resume_stage, resume_epoch = position["stage_index"], position["epoch"]
        logger.info(
            "resume: step=%d, giai đoạn #%d, epoch %d", global_step, resume_stage, resume_epoch
        )

    started_all = time.perf_counter()
    n_optimizer_steps = 0
    stop_early = False

    for stage_index, stage in enumerate(stages):
        if stage_index < resume_stage:
            logger.info("bỏ qua giai đoạn %s — đã xong ở lần chạy trước", stage.name)
            continue
        if stop_early:
            break
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

        if resume_path and stage_index == resume_stage:
            _load_trainer_state(resume_path, optimizer, scheduler, scaler)
            resume_path = None

        first_epoch = resume_epoch if stage_index == resume_stage else 0
        logger.info(
            "giai đoạn %s: %d sample, %d step, epoch %d..%d",
            stage.name, len(data), total_steps, first_epoch, stage.epochs - 1,
        )
        model.train()
        started = time.perf_counter()

        for epoch in range(first_epoch, stage.epochs):
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
                    n_optimizer_steps += 1

                    if global_step % config.logging_steps == 0:
                        logger.info(
                            "stage=%s epoch=%d step=%d loss=%.4f has_value=%.4f sub=%.4f",
                            stage.name, epoch, global_step,
                            losses["loss"].detach().item(),
                            losses["loss_has_value"].detach().item(),
                            losses["loss_sub"].detach().item(),
                        )
                    if val_loader and global_step % config.eval_steps == 0:
                        metrics = evaluate(model, val_loader, loss_fn, device, config.fp16)
                        metrics.update({"step": global_step, "stage": stage.name})
                        history.append(metrics)
                        logger.info("eval @%d: %s", global_step, json.dumps(metrics))
                    if global_step % config.save_steps == 0:
                        save_checkpoint(
                            model, tokenizer, optimizer, scheduler, scaler, global_step, config,
                            stage_index=stage_index, epoch=epoch,
                        )
                    if config.max_steps and n_optimizer_steps >= config.max_steps:
                        logger.info("SMOKE: dừng ở %d optimizer step", n_optimizer_steps)
                        stop_early = True
                        break
            if stop_early:
                break

        # `eval_steps=500` không bao giờ chạm trong giai đoạn finetune (chỉ 384
        # step) — cả giai đoạn quan trọng nhất sẽ không có số đo nào. Eval một
        # lần ở cuối mỗi giai đoạn để history luôn có mốc so sánh.
        # `not config.max_steps`: smoke chỉ đo throughput, mà eval 17,769 cặp nằm
        # trong cửa sổ tính `sec_per_step` sẽ thổi số đo lên (0.87 -> 1.44 s/step,
        # ước tính 1.1 -> 1.78 h) và làm hỏng chính cái guard rail này.
        if val_loader and not config.max_steps:
            metrics = evaluate(model, val_loader, loss_fn, device, config.fp16)
            metrics.update({"step": global_step, "stage": stage.name, "at": "end_of_stage"})
            history.append(metrics)
            logger.info("eval cuối giai đoạn %s: %s", stage.name, json.dumps(metrics))

        logger.info("giai đoạn %s xong sau %.1f phút", stage.name, (time.perf_counter() - started) / 60)
        save_checkpoint(
            model, tokenizer, optimizer, scheduler, scaler, global_step, config,
            tag=f"stage-{stage.name}",
            # epoch = stage.epochs → lần resume sau biết giai đoạn này đã xong.
            stage_index=stage_index, epoch=stage.epochs,
        )
        if stop_early:
            break

    final_dir = config.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    elapsed = time.perf_counter() - started_all
    smoke = bool(config.max_steps)
    # Smoke chỉ đo throughput; eval đầy đủ ở đây tốn vài phút mà không dùng vào việc gì.
    final_metrics = (
        evaluate(model, val_loader, loss_fn, device, config.fp16)
        if val_loader and not smoke
        else {}
    )
    sec_per_step = elapsed / n_optimizer_steps if n_optimizer_steps else None
    planned_steps = sum(
        math.ceil(
            math.ceil(
                len(subset_by_sources(train_dataset, st.sources) if st.sources else train_dataset)
                / config.batch_size
            )
            / config.grad_accum
        )
        * st.epochs
        for st in stages
    )
    # Chọn checkpoint: chỉ BÁO CÁO, không tự nạp lại — cùng quy ước với
    # Bi-Encoder để hai stage đọc được bằng một mắt.
    scored = [h for h in history if "has_value_f1" in h]
    best = max(scored, key=lambda h: h["has_value_f1"]) if scored else None
    report = {
        "final_metrics": final_metrics,
        "history": history,
        "final_checkpoint": str(final_dir),
        "training_duration_hours": round(elapsed / 3600, 3),
        "sec_per_step": round(sec_per_step, 2) if sec_per_step else None,
        "n_optimizer_steps": n_optimizer_steps,
        "planned_optimizer_steps": planned_steps,
        "estimated_hours_full_run": (
            round(sec_per_step * planned_steps / 3600, 2) if sec_per_step else None
        ),
        "stopped_early": smoke and n_optimizer_steps >= config.max_steps,
        "dataset_build_seconds": round(dataset_seconds, 1),
        "checkpoint_selection": {
            "metric": "has_value_f1",
            "best_value": best.get("has_value_f1") if best else None,
            "best_step": best.get("step") if best else None,
            "best_stage": best.get("stage") if best else None,
            "n_evaluations": len(scored),
            "note": "Chỉ báo cáo — model cuối là checkpoint cuối, không nạp lại best.",
        },
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


def _read_trainer_state(path: str) -> dict[str, int]:
    """Vị trí đã train tới, đọc trước khi dựng optimizer của giai đoạn."""
    progress = Path(path) / "progress.json"
    if progress.exists():
        data = json.loads(progress.read_text(encoding="utf-8"))
        return {
            "step": int(data.get("step", 0)),
            "stage_index": int(data.get("stage_index", 0)),
            "epoch": int(data.get("epoch", 0)),
        }
    state = torch.load(Path(path) / "trainer_state.pt", map_location="cpu")
    return {
        "step": int(state.get("step", 0)),
        # Checkpoint cũ (trước khi vá) không có hai khoá này → coi như giai đoạn
        # đầu, đúng hành vi cũ, không vỡ.
        "stage_index": int(state.get("stage_index", 0)),
        "epoch": int(state.get("epoch", 0)),
    }


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
    parser.add_argument(
        "--smoke",
        nargs="?",
        type=int,
        const=50,
        default=None,
        metavar="STEPS",
        help="Dừng sau N optimizer step (mặc định 50) để đo s/step và VRAM. "
             "Ghi vào output_dir riêng để không lẫn với run thật.",
    )
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
    if args.smoke is not None:
        config.max_steps = args.smoke
        config.output_dir = config.output_dir.parent / "smoke_run01"
        # Eval trên 17.7k cặp val mất vài phút — với 50 step thì nó lấn át hoàn
        # toàn số đo throughput. Save một lần ở cuối là đủ để kiểm tra resume.
        config.eval_steps = 10**9
        config.save_steps = max(args.smoke // 2, 1)
        config.logging_steps = max(args.smoke // 5, 1)
        logging.getLogger(__name__).info(
            "SMOKE: %d step, output %s", config.max_steps, config.output_dir
        )

    if args.show_config:
        print(json.dumps({k: str(v) for k, v in config.__dict__.items()}, indent=2, ensure_ascii=False))
        return

    report = train(config)
    print(json.dumps(report["final_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
