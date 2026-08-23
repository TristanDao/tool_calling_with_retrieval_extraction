"""Train Bi-Encoder bằng sentence-transformers + CachedMNRL + LoRA (§Phase 2).

Vì sao **CachedMultipleNegativesRankingLoss** chứ không phải gradient
accumulation: chất lượng MNRL tỉ lệ thuận với số in-batch negative. Gradient
accumulation chỉ chia nhỏ update, **không** làm tăng số negative trong một batch.
GradCache chia mini-batch và cache gradient của embedding nên đạt effective batch
256 với bộ nhớ gần như không đổi — vừa T4 16GB với BGE-M3 (§4.1).

Quy trình 2 vòng:

1. `train` — in-batch negative + hard negative round-0 (candidate/feature_group/BM25).
2. `mine` — dùng chính checkpoint round 1 retrieve top-k trên train, lấy tool sai
   xếp hạng cao (bỏ top-1 để tránh false negative) làm negative mới.
3. `train` lại **từ base**, không train tiếp từ round 1.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.sources import load_jsonl, write_jsonl

logger = logging.getLogger(__name__)


@dataclass
class LoraSettings:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )


@dataclass
class BiEncoderTrainConfig:
    model_name: str = "BAAI/bge-m3"
    max_seq_length: int = 192
    lora: LoraSettings = field(default_factory=LoraSettings)

    train_path: Path = Path("data/method2/biencoder/train.jsonl")
    val_path: Path = Path("data/method2/biencoder/val.jsonl")
    output_dir: Path = Path("artifacts/method2/biencoder/run01")
    tool_pool_path: Path = Path("data/method2/tool_pool.json")

    batch_size: int = 256
    mini_batch_size: int = 8
    scale: float = 20.0
    epochs: int = 3
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True
    eval_steps: int = 500
    save_steps: int = 500
    seed: int = 42
    n_negatives: int = 4
    resume_from: str | None = None
    #: Tên metric chọn checkpoint. Để trống thì tự dò từ log history.
    metric_for_best_model: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BiEncoderTrainConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        model_raw = raw.get("model", {})
        train_raw = raw.get("train", {})
        lora_raw = model_raw.get("lora", {})
        defaults = cls()
        return cls(
            model_name=str(model_raw.get("name", defaults.model_name)),
            max_seq_length=int(model_raw.get("max_seq_length", defaults.max_seq_length)),
            lora=LoraSettings(
                enabled=bool(lora_raw.get("enabled", True)),
                r=int(lora_raw.get("r", 16)),
                alpha=int(lora_raw.get("alpha", 32)),
                dropout=float(lora_raw.get("dropout", 0.05)),
                target_modules=list(lora_raw.get("target_modules", LoraSettings().target_modules)),
            ),
            train_path=Path(train_raw.get("train_path", defaults.train_path)),
            val_path=Path(train_raw.get("val_path", defaults.val_path)),
            output_dir=Path(train_raw.get("output_dir", defaults.output_dir)),
            tool_pool_path=Path(
                raw.get("pairs", {}).get("tool_pool_path", defaults.tool_pool_path)
            ),
            batch_size=int(train_raw.get("batch_size", defaults.batch_size)),
            mini_batch_size=int(train_raw.get("mini_batch_size", defaults.mini_batch_size)),
            scale=float(train_raw.get("scale", defaults.scale)),
            epochs=int(train_raw.get("epochs", defaults.epochs)),
            lr=float(train_raw.get("lr", defaults.lr)),
            warmup_ratio=float(train_raw.get("warmup_ratio", defaults.warmup_ratio)),
            scheduler=str(train_raw.get("scheduler", defaults.scheduler)),
            fp16=bool(train_raw.get("fp16", defaults.fp16)),
            gradient_checkpointing=bool(
                train_raw.get("gradient_checkpointing", defaults.gradient_checkpointing)
            ),
            eval_steps=int(train_raw.get("eval_steps", defaults.eval_steps)),
            save_steps=int(train_raw.get("save_steps", defaults.save_steps)),
            seed=int(train_raw.get("seed", defaults.seed)),
            n_negatives=int(raw.get("pairs", {}).get("n_hard_negatives", defaults.n_negatives)),
            resume_from=train_raw.get("resume_from"),
            metric_for_best_model=train_raw.get("metric_for_best_model"),
        )


# --------------------------------------------------------------------- data


def load_training_dataset(
    path: str | Path,
    tool_pool: dict[str, dict[str, Any]],
    n_negatives: int,
):
    """JSONL pairs → `datasets.Dataset` với cột anchor/positive/negative_i.

    MNRL coi mọi cột sau `positive` là negative tường minh, cộng thêm toàn bộ
    in-batch negative.
    """
    from datasets import Dataset

    columns: dict[str, list[str]] = {"anchor": [], "positive": []}
    for i in range(n_negatives):
        columns[f"negative_{i + 1}"] = []

    for row in load_jsonl(path):
        if not row.get("positive"):
            continue  # sample no-call chỉ dùng để hiệu chỉnh τ
        negatives = [n for n in row.get("negatives", []) if n in tool_pool]
        if len(negatives) < n_negatives:
            continue
        columns["anchor"].append(row["query"])
        columns["positive"].append(tool_pool[row["positive"]]["doc_text"])
        for i in range(n_negatives):
            columns[f"negative_{i + 1}"].append(tool_pool[negatives[i]]["doc_text"])

    return Dataset.from_dict(columns)


def build_ir_evaluator(val_path: str | Path, tool_pool: dict[str, dict[str, Any]]):
    """InformationRetrievalEvaluator trên val — Recall@k trong lúc train."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries: dict[str, str] = {}
    relevant: dict[str, set[str]] = {}
    for i, row in enumerate(load_jsonl(val_path)):
        if not row.get("positive"):
            continue
        qid = f"q{i}"
        queries[qid] = row["query"]
        relevant[qid] = {row["positive"]}

    corpus = {name: tool["doc_text"] for name, tool in tool_pool.items()}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name="custom_val",
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        show_progress_bar=True,
    )


# -------------------------------------------------------------------- model


def build_model(config: BiEncoderTrainConfig):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.model_name)
    model.max_seq_length = config.max_seq_length

    if config.lora.enabled:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
        )
        if hasattr(model, "add_adapter"):
            model.add_adapter(peft_config)
        else:  # sentence-transformers cũ hơn
            from peft import get_peft_model

            model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
    return model


# ------------------------------------------------------------------- train


def train(config: BiEncoderTrainConfig) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

    from src.models.biencoder.tool_pool import load_tool_pool

    tool_pool = load_tool_pool(config.tool_pool_path)
    train_dataset = load_training_dataset(config.train_path, tool_pool, config.n_negatives)
    logger.info("train pairs: %d", len(train_dataset))

    model = build_model(config)
    loss = CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=config.mini_batch_size, scale=config.scale
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.scheduler,
        fp16=config.fp16,
        bf16=False,  # T4 không hỗ trợ bf16
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="steps" if config.val_path.exists() else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        logging_steps=50,
        seed=config.seed,
        report_to=[],
    )

    evaluator = build_ir_evaluator(config.val_path, tool_pool) if config.val_path.exists() else None
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    import time

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=config.resume_from)
    duration_sec = time.perf_counter() - started

    final_dir = config.output_dir / "final"
    save_model(model, final_dir)
    metrics = {k: float(v) for k, v in (evaluator(model) if evaluator else {}).items()}

    log_history = list(getattr(trainer.state, "log_history", []))
    report = {
        "final_metrics": metrics,
        "n_train_pairs": len(train_dataset),
        "training_duration_sec": round(duration_sec, 1),
        "training_duration_hours": round(duration_sec / 3600, 3),
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available()
            else None
        ),
        "log_history": log_history,
        "checkpoint_selection": select_best_checkpoint(log_history, config.metric_for_best_model),
        "final_checkpoint": str(final_dir),
    }
    (config.output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Giữ tên cũ cho tương thích với script đã viết trước đó.
    (config.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return report


def select_best_checkpoint(
    log_history: list[dict[str, Any]],
    metric_name: str | None = None,
) -> dict[str, Any]:
    """Chọn checkpoint tốt nhất từ log history của trainer.

    Tên metric của `InformationRetrievalEvaluator` phụ thuộc phiên bản
    sentence-transformers, nên không hardcode: nếu `metric_name` không có trong
    log thì rơi về khoá đầu tiên khớp hậu tố quen thuộc. Toàn bộ `log_history`
    vẫn được ghi lại để lần chạy sau khai báo đúng tên trong config.
    """
    evals = [entry for entry in log_history if any("eval" in k for k in entry)]
    if not evals:
        return {"metric": metric_name, "resolved_metric": None, "reason": "không có eval nào"}

    candidates = sorted({k for entry in evals for k in entry if k.startswith("eval_")})
    resolved = metric_name if metric_name and any(metric_name in c for c in candidates) else None
    if resolved is None:
        for suffix in ("ndcg@10", "recall@5", "accuracy@1", "mrr@10"):
            match = next((c for c in candidates if c.endswith(suffix)), None)
            if match:
                resolved = match
                break
    if resolved is None:
        return {
            "metric": metric_name,
            "resolved_metric": None,
            "available_metrics": candidates,
            "reason": "không khớp metric nào",
        }

    scored = [(entry.get(resolved), entry.get("step")) for entry in evals if resolved in entry]
    best_value, best_step = max(scored, key=lambda pair: (pair[0] is not None, pair[0]))
    return {
        "metric": metric_name,
        "resolved_metric": resolved,
        "best_value": best_value,
        "best_step": best_step,
        "n_evaluations": len(scored),
        "available_metrics": candidates,
        "note": (
            "Chỉ báo cáo, không tự nạp lại. Đặt load_best_model_at_end sau khi đã "
            "biết tên metric chính xác từ available_metrics của lần chạy này."
        ),
    }


def save_model(model, output_dir: str | Path) -> None:
    """Merge LoRA rồi lưu, để `index.py` nạp được như model thường."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    auto_model = model[0].auto_model
    if hasattr(auto_model, "merge_and_unload"):
        model[0].auto_model = auto_model.merge_and_unload()
    model.save(str(output_dir))


# -------------------------------------------------------- hard negative mining


def mine_hard_negatives(
    model_path: str,
    pairs_path: str | Path,
    output_path: str | Path,
    tool_pool_path: str | Path = "data/method2/tool_pool.json",
    top_k: int = 20,
    skip_top: int = 1,
    n_negatives: int = 4,
    batch_size: int = 64,
    max_seq_length: int = 192,
) -> dict[str, Any]:
    """Round 2: lấy tool sai nhưng xếp hạng cao làm negative tường minh."""
    import numpy as np

    from src.models.biencoder.index import load_encoder
    from src.models.biencoder.tool_pool import load_tool_pool

    tool_pool = load_tool_pool(tool_pool_path)
    names = sorted(tool_pool)
    model = load_encoder(model_path, max_seq_length)
    doc_embeddings = model.encode(
        [tool_pool[n]["doc_text"] for n in names],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    rows = [r for r in load_jsonl(pairs_path)]
    positive_rows = [r for r in rows if r.get("positive")]
    query_embeddings = model.encode(
        [r["query"] for r in positive_rows],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    mined = 0
    updates: dict[int, list[str]] = {}
    for i, row in enumerate(positive_rows):
        scores = doc_embeddings @ query_embeddings[i]
        order = np.argpartition(-scores, min(top_k, len(names) - 1))[:top_k]
        order = sorted(order, key=lambda j: -scores[j])
        gold = set(row.get("all_gold") or [row["positive"]])
        negatives = [names[j] for j in order[skip_top:] if names[j] not in gold]
        if negatives:
            updates[id(row)] = negatives[:n_negatives]
            mined += 1

    for row in positive_rows:
        if id(row) in updates:
            row["negatives"] = updates[id(row)]

    write_jsonl(output_path, rows)
    return {"n_rows": len(rows), "n_mined": mined, "output": str(output_path)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[biencoder] %(message)s")
    parser = argparse.ArgumentParser(description="Train Bi-Encoder for tool retrieval")
    parser.add_argument("command", choices=["train", "mine", "show-config"])
    parser.add_argument("--config", type=Path, default=Path("configs/method2/biencoder.yaml"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = BiEncoderTrainConfig.from_yaml(args.config)
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.output_dir:
        config.output_dir = args.output_dir

    if args.command == "show-config":
        print(json.dumps({k: str(v) for k, v in config.__dict__.items()}, indent=2, ensure_ascii=False))
        return

    if args.command == "train":
        report = train(config)
        print(json.dumps(report["final_metrics"], indent=2))
        print(json.dumps(report["checkpoint_selection"], ensure_ascii=False, indent=2))
        return

    import yaml

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    mining = raw.get("mining", {})
    result = mine_hard_negatives(
        model_path=args.model or str(config.output_dir / "final"),
        pairs_path=config.train_path,
        output_path=mining.get("output_path", "data/method2/biencoder/train_mined.jsonl"),
        tool_pool_path=config.tool_pool_path,
        top_k=int(mining.get("top_k", 20)),
        skip_top=int(mining.get("skip_top", 1)),
        n_negatives=int(mining.get("n_negatives", 4)),
        max_seq_length=config.max_seq_length,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
