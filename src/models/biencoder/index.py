"""Pre-compute embedding cho toàn bộ tool pool (§Phase 2.4 method2_plan).

`E(t)` không phụ thuộc query nên chỉ cần tính **một lần**; tại inference chỉ còn
1 forward pass cho query + 1 phép nhân ma trận `1×d · d×N`. Đây là nguồn gốc
lợi thế latency/scalability của Method 2 khi N tăng từ 3 lên 1000.

Thời gian đo theo plan: ~2 phút cho 4,4k tool trên T4. `t_index_build` được ghi
riêng, **không** cộng vào latency/query (§Phase 5).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.biencoder.tool_pool import build_document_text, load_tool_pool


@dataclass
class IndexConfig:
    model_path: str = "BAAI/bge-m3"
    tool_pool_path: Path = Path("data/method2/tool_pool.json")
    embeddings_path: Path = Path("data/method2/index/tool_embeddings.npy")
    tool_ids_path: Path = Path("data/method2/index/tool_ids.json")
    batch_size: int = 64
    max_seq_length: int = 192
    normalize: bool = True
    include_param_names_in_doc: bool = True
    device: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IndexConfig":
        defaults = cls()
        index_raw = raw.get("index", {})
        model_raw = raw.get("model", {})
        return cls(
            model_path=str(model_raw.get("name", defaults.model_path)),
            tool_pool_path=Path(
                raw.get("pairs", {}).get("tool_pool_path", defaults.tool_pool_path)
            ),
            embeddings_path=Path(index_raw.get("embeddings_path", defaults.embeddings_path)),
            tool_ids_path=Path(index_raw.get("tool_ids_path", defaults.tool_ids_path)),
            batch_size=int(index_raw.get("batch_size", defaults.batch_size)),
            max_seq_length=int(model_raw.get("max_seq_length", defaults.max_seq_length)),
            normalize=bool(index_raw.get("normalize", defaults.normalize)),
        )


def load_encoder(model_path: str, max_seq_length: int = 192, device: str | None = None):
    """Nạp SentenceTransformer, chấp nhận cả checkpoint LoRA đã merge."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=device)
    model.max_seq_length = max_seq_length
    return model


def build_index(config: IndexConfig) -> dict[str, Any]:
    import numpy as np

    pool = load_tool_pool(config.tool_pool_path)
    names = sorted(pool)
    documents = [
        pool[name].get("doc_text")
        or build_document_text(pool[name], config.include_param_names_in_doc)
        for name in names
    ]

    model = load_encoder(config.model_path, config.max_seq_length, config.device)
    started = time.perf_counter()
    embeddings = model.encode(
        documents,
        batch_size=config.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - started

    config.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.embeddings_path, embeddings.astype("float32"))
    config.tool_ids_path.parent.mkdir(parents=True, exist_ok=True)
    config.tool_ids_path.write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "model": config.model_path,
        "n_tools": len(names),
        "dim": int(embeddings.shape[1]),
        "normalized": config.normalize,
        "max_seq_length": config.max_seq_length,
        "t_index_build_sec": round(elapsed, 3),
    }
    (config.embeddings_path.parent / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute tool embeddings")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/biencoder.yaml"))
    parser.add_argument("--model", type=str, default=None, help="Checkpoint đã fine-tune")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import yaml

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    config = IndexConfig.from_dict(raw)
    if args.model:
        config.model_path = args.model
    if args.device:
        config.device = args.device

    meta = build_index(config)
    print(
        f"[index] {meta['n_tools']} tool × {meta['dim']}d trong "
        f"{meta['t_index_build_sec']}s → {config.embeddings_path}"
    )


if __name__ == "__main__":
    main()
