"""Pipeline Method 2: Bi-Encoder → Cross-Encoder → Validator → function_calls.

Xuất đúng **prediction contract** của `src/evaluation` (xem `docs/evaluation.md`
§3) để so sánh với ba method còn lại trên cùng evaluator, cùng normalization,
cùng rule so khớp. Không tự tính N-FCEM/ArgEM ở đây.

Hai chế độ bắt buộc theo §6.3 experimental_plan:

| Chế độ | Input Cross-Encoder | Trả lời |
|---|---|---|
| `oracle` | Tool gold | Extraction tốt đến đâu, độc lập retrieval |
| `pipeline` | Bi-Encoder top-k + abstention | Hiệu năng hệ thống thật |

`ArgA_oracle − ArgA_pipeline` chính là phần lỗi do retrieval, con số này phải
xuất hiện tường minh trong báo cáo.

Latency đo **tách 4 giai đoạn** (§10.1): `t_query_embed`, `t_retrieve`,
`t_cross_encode`, `t_validate`. `t_index_build` ghi riêng, không cộng vào
latency/query.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.evaluation.error_analysis import classify_errors
from src.models.biencoder.retrieve import RetrievalThresholds, ToolRetriever, select_tools
from src.models.crossencoder.data_collator import DEFAULT_MAX_QUESTION_TOKENS
from src.models.crossencoder.inference import CrossEncoderExtractor, ExtractionConfig
from src.models.pipeline.validator import ArgumentValidator
from src.models.sources import load_jsonl, write_jsonl

MODE_PIPELINE = "pipeline"
MODE_ORACLE = "oracle"

#: Cơ chế abstention. `tau` = ngưỡng cosine của Bi-Encoder (§5.1, mặc định);
#: `should_call` = head nhị phân của Cross-Encoder (ablation §6.1).
ABSTENTION_TAU = "tau"
ABSTENTION_SHOULD_CALL = "should_call"

# Bốn nhóm lỗi của §9 experimental_plan (Ersoy et al.).
ERROR_WRONG_VALUE = "W"
ERROR_TRANSLATION = "T"
ERROR_PARAPHRASE = "P"
ERROR_INCOMPLETE = "I"


@dataclass
class StageTimings:
    t_query_embed: float = 0.0
    t_retrieve: float = 0.0
    t_cross_encode: float = 0.0
    t_validate: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.t_query_embed + self.t_retrieve + self.t_cross_encode + self.t_validate
        ) * 1000

    def as_ms(self) -> dict[str, float]:
        return {k: round(v * 1000, 3) for k, v in asdict(self).items()}


@dataclass
class Method2Config:
    biencoder_path: str = "artifacts/method2/biencoder/run01/final"
    crossencoder_path: str = "artifacts/method2/crossencoder/run01/final"
    embeddings_path: Path = Path("data/method2/index/tool_embeddings.npy")
    tool_ids_path: Path = Path("data/method2/index/tool_ids.json")
    tool_pool_path: Path = Path("data/method2/tool_pool.json")
    thresholds_path: Path | None = Path("artifacts/method2/biencoder/run01/thresholds.json")
    max_seq_length: int = 192
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    #: `candidates` = xếp hạng trong candidate pool của sample; `pool` = toàn bộ.
    retrieval_scope: str = "candidates"
    #: `tau` | `should_call` — xem ABSTENTION_*.
    abstention: str = ABSTENTION_TAU
    #: Ngưỡng cho `should_call`, hiệu chỉnh trên val rồi FREEZE như τ.
    should_call_threshold: float = 0.5
    device: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Method2Config":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        defaults = cls()
        models = raw.get("models", {})
        index = raw.get("index", {})
        extraction_raw = raw.get("extraction", {})
        normalizer_raw = extraction_raw.get("normalizer", {}) or {}

        from src.models.crossencoder.normalize import NormalizerConfig

        reference_date = normalizer_raw.get("reference_date")
        if isinstance(reference_date, str):
            from datetime import date

            reference_date = date.fromisoformat(reference_date)

        return cls(
            biencoder_path=str(models.get("biencoder", defaults.biencoder_path)),
            crossencoder_path=str(models.get("crossencoder", defaults.crossencoder_path)),
            embeddings_path=Path(index.get("embeddings_path", defaults.embeddings_path)),
            tool_ids_path=Path(index.get("tool_ids_path", defaults.tool_ids_path)),
            tool_pool_path=Path(index.get("tool_pool_path", defaults.tool_pool_path)),
            thresholds_path=(
                Path(models["thresholds"]) if models.get("thresholds") else defaults.thresholds_path
            ),
            max_seq_length=int(models.get("max_seq_length", defaults.max_seq_length)),
            retrieval_scope=str(raw.get("retrieval_scope", defaults.retrieval_scope)),
            abstention=str(raw.get("abstention", defaults.abstention)),
            should_call_threshold=float(
                raw.get("should_call_threshold", defaults.should_call_threshold)
            ),
            extraction=ExtractionConfig(
                max_length=int(extraction_raw.get("max_length", 256)),
                has_value_threshold=float(extraction_raw.get("has_value_threshold", 0.5)),
                fallback_has_value_threshold=float(
                    extraction_raw.get("fallback_has_value_threshold", 0.3)
                ),
                max_answer_len=int(extraction_raw.get("max_answer_len", 30)),
                batch_size=int(extraction_raw.get("batch_size", 64)),
                max_question_tokens=int(
                    extraction_raw.get("max_question_tokens", DEFAULT_MAX_QUESTION_TOKENS)
                ),
                use_normalizer=bool(extraction_raw.get("use_normalizer", True)),
                normalizer=NormalizerConfig(
                    enabled=bool(extraction_raw.get("use_normalizer", True)),
                    reference_date=reference_date,
                    century=int(normalizer_raw.get("century", 2000)),
                ),
            ),
        )


class Method2Pipeline:
    def __init__(
        self,
        retriever: ToolRetriever,
        extractor: CrossEncoderExtractor,
        validator: ArgumentValidator,
        tool_pool: dict[str, dict[str, Any]],
        thresholds: RetrievalThresholds | None = None,
        retrieval_scope: str = "candidates",
        abstention: str = ABSTENTION_TAU,
        should_call_threshold: float = 0.5,
    ) -> None:
        self.retriever = retriever
        self.extractor = extractor
        self.validator = validator
        self.tool_pool = tool_pool
        self.thresholds = thresholds or retriever.thresholds
        self.retrieval_scope = retrieval_scope
        self.abstention = abstention
        self.should_call_threshold = should_call_threshold
        if abstention == ABSTENTION_SHOULD_CALL and not extractor.has_should_call:
            # Quay lặng lẽ về τ là hỏng cả ablation: báo cáo sẽ ghi
            # "should_call" trong khi con số đo được là của baseline. Đúng lớp
            # lỗi đã xảy ra với `strategy` ở §7.10b — lần này chặn ngay.
            raise ValueError(
                "abstention='should_call' nhưng checkpoint Cross-Encoder không có "
                "head đó. Train lại với model.enable_should_call=true, hoặc để "
                "abstention='tau'."
            )

    @classmethod
    def from_config(cls, config: Method2Config) -> "Method2Pipeline":
        from transformers import AutoTokenizer

        from src.models.biencoder.tool_pool import load_tool_pool
        from src.models.crossencoder.model import CrossEncoderForExtraction

        retriever = ToolRetriever.from_paths(
            model_path=config.biencoder_path,
            embeddings_path=config.embeddings_path,
            tool_ids_path=config.tool_ids_path,
            thresholds_path=config.thresholds_path
            if config.thresholds_path and Path(config.thresholds_path).exists()
            else None,
            max_seq_length=config.max_seq_length,
            device=config.device,
        )
        model = CrossEncoderForExtraction.from_pretrained(config.crossencoder_path)
        tokenizer = AutoTokenizer.from_pretrained(config.crossencoder_path, use_fast=True)
        extractor = CrossEncoderExtractor(
            model,
            tokenizer,
            config.extraction,
            device=config.device or ("cuda" if _cuda_available() else "cpu"),
        )
        validator = ArgumentValidator(
            fallback_threshold=config.extraction.fallback_has_value_threshold,
            normalizer=extractor.normalizer,
        )
        return cls(
            retriever,
            extractor,
            validator,
            load_tool_pool(config.tool_pool_path),
            retrieval_scope=config.retrieval_scope,
            abstention=config.abstention,
            should_call_threshold=config.should_call_threshold,
        )

    # ------------------------------------------------------------- inference

    def run_sample(self, sample: dict[str, Any], mode: str = MODE_PIPELINE) -> dict[str, Any]:
        query = str(sample.get("query", ""))
        timings = StageTimings()
        schemas = self._sample_schemas(sample)
        should_call_prob: float | None = None

        if mode == MODE_ORACLE:
            selected = list(
                dict.fromkeys(
                    call["name"] for call in (sample.get("function_calls") or []) if call.get("name")
                )
            )
            ranked: list[tuple[str, float]] = []
            abstained = not selected
        else:
            started = time.perf_counter()
            embedding = self.retriever.encode_queries([query])[0]
            timings.t_query_embed = time.perf_counter() - started

            started = time.perf_counter()
            candidates = list(schemas) if self.retrieval_scope == "candidates" and schemas else None
            ranked = self.retriever.score(embedding, candidates)
            timings.t_retrieve = time.perf_counter() - started

            if self.abstention == ABSTENTION_SHOULD_CALL:
                # Một forward Cross-Encoder cấp tool trên top-1. Tính vào
                # `t_cross_encode` vì đó đúng là compute của Cross-Encoder; dồn
                # vào `t_retrieve` sẽ đổ chi phí này lên Bi-Encoder.
                started = time.perf_counter()
                should_call_prob = self._should_call_prob(query, ranked, schemas)
                timings.t_cross_encode += time.perf_counter() - started
                abstained = (
                    should_call_prob is None
                    or should_call_prob < self.should_call_threshold
                )
                selected = [] if abstained else select_tools(ranked, self.thresholds)
            else:
                selected, abstained = self.retriever.select(ranked, self.thresholds)

        function_calls: list[dict[str, Any]] = []
        raw_calls: list[dict[str, Any]] = []
        statuses: list[str] = []

        for tool_name in selected:
            schema = schemas.get(tool_name) or self.tool_pool.get(tool_name)
            if schema is None:
                continue
            started = time.perf_counter()
            predictions = self.extractor.predict(query, schema)
            arguments = self.extractor.to_arguments(predictions)
            timings.t_cross_encode += time.perf_counter() - started

            started = time.perf_counter()
            result = self.validator.validate(arguments, schema, predictions)
            timings.t_validate += time.perf_counter() - started

            function_calls.append({"name": tool_name, "arguments": result.arguments})
            statuses.append(result.status)
            raw_calls.append(
                {
                    "name": tool_name,
                    "status": result.status,
                    "missing_required": result.missing_required,
                    "dropped_keys": result.dropped_keys,
                    # Gộp cả param optional kiểu array/object: validator chỉ thấy
                    # được các param required, nhưng bảng coverage §3.2 cần đủ.
                    "unsupported_params": sorted(
                        set(result.unsupported_params)
                        | {p.name for p in predictions if p.unsupported}
                    ),
                    "schema_errors": result.schema_errors,
                    "params": [asdict(p) for p in predictions],
                }
            )

        prediction = {
            "id": sample.get("id"),
            "function_calls": function_calls,
            "ranked_tools": [{"name": n, "score": round(s, 6)} for n, s in ranked[:10]],
            "telemetry": {
                "latency_ms": round(timings.total_ms, 3),
                **timings.as_ms(),
            },
            "metadata": {
                "model": "method_2",
                "cost_basis": "unavailable_gpu_cost",
                "mode": mode,
                "abstained": abstained,
                "validation_status": statuses,
                # Ghi lại xác suất thô để quét lại ngưỡng OFFLINE, không cần
                # GPU — đúng cách `replay_selection.py` đã dò ra bug §7.10b.
                **(
                    {"should_call_prob": round(should_call_prob, 6)}
                    if should_call_prob is not None
                    else {}
                ),
            },
        }
        raw = {
            "id": sample.get("id"),
            "query": query,
            "mode": mode,
            "ranked_tools": [{"name": n, "score": round(s, 6)} for n, s in ranked[:20]],
            "selected": selected,
            "calls": raw_calls,
        }
        return {"prediction": prediction, "raw": raw, "timings": timings}

    def run_dataset(
        self,
        samples: Iterable[dict[str, Any]],
        mode: str = MODE_PIPELINE,
        output_dir: str | Path = "results/method2/predictions",
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        predictions: list[dict[str, Any]] = []
        raws: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        latencies: list[StageTimings] = []

        for sample in samples:
            outcome = self.run_sample(sample, mode)
            predictions.append(outcome["prediction"])
            raws.append(outcome["raw"])
            latencies.append(outcome["timings"])
            errors.extend(classify_errors(sample, outcome["prediction"], outcome["raw"]))

        name = f"{mode}_predictions.jsonl" if mode == MODE_ORACLE else "predictions.jsonl"
        write_jsonl(output_dir / name, predictions)
        write_jsonl(output_dir / f"raw_predictions_{mode}.jsonl", raws)
        write_jsonl(output_dir / f"errors_{mode}.jsonl", errors)

        latency = summarize_latency(latencies)
        (output_dir / f"latency_{mode}.json").write_text(
            json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "n_samples": len(predictions),
            "n_errors": len(errors),
            "latency": latency,
            "output_dir": str(output_dir),
        }

    def _should_call_prob(
        self,
        query: str,
        ranked: list[tuple[str, float]],
        schemas: dict[str, dict[str, Any]],
    ) -> float | None:
        """P(cần gọi) đo trên tool đứng đầu ranking — đúng như plan §5.1 mô tả.

        Không có ranking thì không có tool nào để hỏi, và câu trả lời đúng là
        abstain: trả `None` để nơi gọi xử lý, chứ không trả 0.0 (0.0 là một câu
        trả lời của model, `None` là "model chưa được hỏi").
        """
        if not ranked:
            return None
        top_name = ranked[0][0]
        schema = schemas.get(top_name) or self.tool_pool.get(top_name)
        if schema is None:
            return None
        return self.extractor.should_call_prob(query, schema)

    def _sample_schemas(self, sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {t["name"]: t for t in (sample.get("tools") or []) if t.get("name")}


# ------------------------------------------------------------------ latency


def summarize_latency(timings: Sequence[StageTimings]) -> dict[str, Any]:
    if not timings:
        return {}

    def percentile(values: list[float], q: float) -> float:
        from src.evaluation.statistics import percentile

        return round(percentile(values, q), 3)

    report: dict[str, Any] = {"n": len(timings)}
    stages = ["t_query_embed", "t_retrieve", "t_cross_encode", "t_validate"]
    for stage in stages + ["total"]:
        values = [
            t.total_ms if stage == "total" else getattr(t, stage) * 1000 for t in timings
        ]
        report[stage] = {
            "mean_ms": round(sum(values) / len(values), 3),
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99),
        }
    if _cuda_available():
        import torch

        report["gpu_peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return report


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ------------------------------------------------------------ error analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Method 2 pipeline over a dataset")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/pipeline.yaml"))
    parser.add_argument("--gold", type=Path, required=True, help="File gold theo unified structure")
    parser.add_argument("--mode", choices=[MODE_PIPELINE, MODE_ORACLE], default=MODE_PIPELINE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/method2/predictions"))
    parser.add_argument("--limit", type=int, default=None)
    # Hai cờ dưới đây để chạy nhánh ablation §6.1 mà KHÔNG phải sửa file config:
    # config trong dataset upload đã bị khoá SHA, sửa nó là phải đóng gói lại.
    parser.add_argument(
        "--abstention",
        choices=[ABSTENTION_TAU, ABSTENTION_SHOULD_CALL],
        default=None,
        help="Ghi đè `abstention` của config.",
    )
    parser.add_argument(
        "--should-call-threshold",
        type=float,
        default=None,
        help="Ngưỡng đã FREEZE từ val (scripts/method2/calibrate_should_call.py).",
    )
    parser.add_argument(
        "--crossencoder",
        type=str,
        default=None,
        help="Checkpoint Cross-Encoder khác (nhánh ablation ghi ra thư mục riêng).",
    )
    parser.add_argument(
        "--biencoder",
        type=str,
        default=None,
        help="Checkpoint Bi-Encoder khác.",
    )
    args = parser.parse_args()

    config = Method2Config.from_yaml(args.config)
    if args.abstention:
        config.abstention = args.abstention
    if args.should_call_threshold is not None:
        config.should_call_threshold = args.should_call_threshold
    if args.crossencoder:
        config.crossencoder_path = args.crossencoder
    if args.biencoder:
        config.biencoder_path = args.biencoder
    pipeline = Method2Pipeline.from_config(config)

    samples = list(load_jsonl(args.gold))
    if args.limit:
        samples = samples[: args.limit]

    report = pipeline.run_dataset(samples, mode=args.mode, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
