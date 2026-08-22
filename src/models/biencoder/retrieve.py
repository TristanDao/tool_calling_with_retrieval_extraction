"""Truy hồi tool từ index đã pre-compute, kèm abstention và chọn số call.

Hai cơ chế §5 method2_plan mà plan gốc còn thiếu:

- **Abstention** (`τ`): `max_i sim(q, t_i) < τ` → `<no_tool_call>`. Không có nó
  thì Negative Recall = 0% vì Bi-Encoder luôn trả về top-k.
- **Chọn số call** (`τ_call`, `k_max`): lấy mọi tool có `sim >= τ_call`, tối đa
  `k_max`. Chiến lược thay thế `gap`: lấy top-1, thêm top-2 nếu
  `sim_1 − sim_2 < δ`.

Cả hai ngưỡng phải được hiệu chỉnh trên validation và **freeze** trước khi chạy
test (xem `evaluate.calibrate_thresholds`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

NO_TOOL_CALL = "<no_tool_call>"


@dataclass
class RetrievalThresholds:
    """Ngưỡng đã hiệu chỉnh trên val. Ghi ra `thresholds.json` rồi freeze."""

    tau: float = 0.0
    tau_call: float = 1.0
    k_max: int = 3
    gap_delta: float = 0.05
    strategy: str = "absolute"  # absolute | gap
    calibrated_on: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalThresholds":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class RetrievalResult:
    query: str
    ranked: list[tuple[str, float]]
    selected: list[str]
    abstained: bool

    def to_ranked_tools(self, top_k: int = 10) -> list[dict[str, Any]]:
        """Định dạng `ranked_tools` theo prediction contract của evaluator."""
        return [{"name": name, "score": float(score)} for name, score in self.ranked[:top_k]]


class ToolRetriever:
    """Query → top-k tool, có abstention. Index nạp một lần, tái dùng."""

    def __init__(
        self,
        model,
        tool_names: Sequence[str],
        embeddings,
        thresholds: RetrievalThresholds | None = None,
        normalize_query: bool = True,
    ) -> None:
        import numpy as np

        self.model = model
        self.tool_names = list(tool_names)
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.name_to_row = {name: i for i, name in enumerate(self.tool_names)}
        self.thresholds = thresholds or RetrievalThresholds()
        self.normalize_query = normalize_query

    @classmethod
    def from_paths(
        cls,
        model_path: str,
        embeddings_path: str | Path,
        tool_ids_path: str | Path,
        thresholds_path: str | Path | None = None,
        max_seq_length: int = 192,
        device: str | None = None,
    ) -> "ToolRetriever":
        import numpy as np

        from src.models.biencoder.index import load_encoder

        model = load_encoder(model_path, max_seq_length, device)
        embeddings = np.load(Path(embeddings_path))
        names = json.loads(Path(tool_ids_path).read_text(encoding="utf-8"))
        thresholds = (
            RetrievalThresholds.load(thresholds_path) if thresholds_path else None
        )
        return cls(model, names, embeddings, thresholds)

    def encode_queries(self, queries: Sequence[str], batch_size: int = 64):
        return self.model.encode(
            list(queries),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_query,
        )

    def score(
        self,
        query_embedding,
        candidate_names: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Cosine similarity, sắp giảm dần. `candidate_names=None` → toàn pool."""
        import numpy as np

        if candidate_names is None:
            rows = np.arange(len(self.tool_names))
        else:
            rows = np.array(
                [self.name_to_row[n] for n in candidate_names if n in self.name_to_row],
                dtype=int,
            )
            if rows.size == 0:
                return []
        scores = self.embeddings[rows] @ np.asarray(query_embedding, dtype="float32")
        order = np.argsort(-scores)
        return [(self.tool_names[rows[i]], float(scores[i])) for i in order]

    def select(
        self,
        ranked: list[tuple[str, float]],
        thresholds: RetrievalThresholds | None = None,
    ) -> tuple[list[str], bool]:
        """Áp abstention rồi chọn số call. Trả `(tools, abstained)`."""
        thresholds = thresholds or self.thresholds
        if not ranked:
            return [], True
        if ranked[0][1] < thresholds.tau:
            return [], True
        return select_tools(ranked, thresholds), False

    def retrieve(
        self,
        query: str,
        candidate_names: Sequence[str] | None = None,
        thresholds: RetrievalThresholds | None = None,
    ) -> RetrievalResult:
        embedding = self.encode_queries([query])[0]
        ranked = self.score(embedding, candidate_names)
        selected, abstained = self.select(ranked, thresholds)
        return RetrievalResult(query=query, ranked=ranked, selected=selected, abstained=abstained)

    def retrieve_batch(
        self,
        queries: Sequence[str],
        candidate_names_per_query: Sequence[Sequence[str] | None] | None = None,
        thresholds: RetrievalThresholds | None = None,
        batch_size: int = 64,
    ) -> list[RetrievalResult]:
        embeddings = self.encode_queries(queries, batch_size=batch_size)
        results: list[RetrievalResult] = []
        for i, query in enumerate(queries):
            candidates = (
                candidate_names_per_query[i] if candidate_names_per_query else None
            )
            ranked = self.score(embeddings[i], candidates)
            selected, abstained = self.select(ranked, thresholds)
            results.append(
                RetrievalResult(query=query, ranked=ranked, selected=selected, abstained=abstained)
            )
        return results


def select_tools(
    ranked: list[tuple[str, float]],
    thresholds: RetrievalThresholds,
) -> list[str]:
    """Chọn danh sách tool sẽ gọi từ ranking (giả định đã qua abstention)."""
    if not ranked:
        return []
    if thresholds.strategy == "gap":
        selected = [ranked[0][0]]
        for name, score in ranked[1 : thresholds.k_max]:
            if ranked[0][1] - score < thresholds.gap_delta:
                selected.append(name)
            else:
                break
        return selected
    selected = [name for name, score in ranked if score >= thresholds.tau_call]
    if not selected:
        selected = [ranked[0][0]]
    return selected[: thresholds.k_max]
