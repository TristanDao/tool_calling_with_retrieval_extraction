"""Inference pipeline for Cross-Encoder.

Convert (query, tool_schema) → dict of arguments.

Ba khác biệt so với bản đầu (§2.3d method2_plan):

1. **Batch**: toàn bộ parameter của một tool đi trong **một** forward pass.
2. **Span decode chuẩn SQuAD**: chọn (i, j) maximize ``start[i] + end[j]`` với
   ràng buộc ``i <= j <= i + max_answer_len``, thay vì argmax độc lập rồi swap.
3. **Normalizer**: span text đi qua `SpanNormalizer` trước khi ghi vào argument.

Text của span lấy theo `offset_mapping` trên query gốc, không ghép lại từ token,
nên không dính artefact `▁` của SentencePiece.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.models.crossencoder.data_collator import (
    SCHEMA_TYPE_BOOLEAN,
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_NUMBER,
    SCHEMA_TYPE_SHOULD_CALL,
    SCHEMA_TYPE_STRING,
    DEFAULT_MAX_QUESTION_TOKENS,
    UNSUPPORTED_TYPES,
    build_schema_question,
    cap_question,
    iter_parameters,
)
from src.models.crossencoder.label_generator import BOOLEAN_LABEL_TRUE
from src.models.crossencoder.model import CrossEncoderForExtraction
from src.models.crossencoder.normalize import NormalizerConfig, SpanNormalizer


@dataclass
class ExtractionConfig:
    max_length: int = 256
    has_value_threshold: float = 0.5
    #: Ngưỡng thấp hơn, chỉ dùng khi validator thấy thiếu required param.
    fallback_has_value_threshold: float = 0.3
    max_answer_len: int = 30
    batch_size: int = 64
    #: PHẢI khớp giá trị lúc train, nếu không model thấy question dài hơn hẳn
    #: những gì nó từng học (và tokenizer ném "Sequence to truncate too short"
    #: với param có description ~900 ký tự trong glaive/xLAM).
    max_question_tokens: int = DEFAULT_MAX_QUESTION_TOKENS
    use_normalizer: bool = True
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)


@dataclass
class ParamPrediction:
    """Dự đoán cho một parameter, giữ đủ logit để ghi `raw_predictions.jsonl`."""

    name: str
    routing_type: str
    required: bool
    has_value_prob: float
    value: Any = None
    span_text: str | None = None
    span_char: tuple[int, int] | None = None
    span_score: float | None = None
    enum_index: int | None = None
    normalizer_applied: list[str] = field(default_factory=list)
    unsupported: bool = False
    normalization_failed: bool = False
    has_value_logit: float | None = None
    enum_logits: list[float] | None = None
    boolean_logits: list[float] | None = None


def decode_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    max_answer_len: int,
) -> tuple[int, int, float]:
    """(start, end, score) maximize start[i] + end[j] với i <= j <= i+max_answer_len."""
    # Không dùng `finfo.min`: `scores` là tổng hai logit nên min + min sẽ tràn
    # thành -inf. Một hằng số đủ âm mà cộng đôi vẫn hữu hạn là an toàn hơn.
    neg = -1e30 if start_logits.dtype == torch.float32 else -1e4
    start = start_logits.masked_fill(~valid_mask, neg)
    end = end_logits.masked_fill(~valid_mask, neg)
    length = start.size(0)

    scores = start.unsqueeze(1) + end.unsqueeze(0)  # scores[i][j] = start[i] + end[j]
    idx = torch.arange(length, device=start.device)
    start_idx = idx.unsqueeze(1)  # i
    end_idx = idx.unsqueeze(0)  # j
    allowed = (end_idx >= start_idx) & (end_idx - start_idx < max_answer_len)
    scores = scores.masked_fill(~allowed, neg)

    flat = int(scores.argmax().item())
    best_start, best_end = divmod(flat, length)
    return best_start, best_end, float(scores[best_start, best_end].item())


class CrossEncoderExtractor:
    """Trích xuất argument cho một (query, tool) — batch theo parameter."""

    def __init__(
        self,
        model: CrossEncoderForExtraction,
        tokenizer: PreTrainedTokenizerBase,
        config: ExtractionConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()
        self.device = torch.device(device)
        # Một schema question lặp lại ở rất nhiều tool/param; cache để không
        # tokenize lại mỗi chunk.
        self._question_cache: dict[str, tuple[str, int]] = {}
        self.model.to(self.device)
        self.model.eval()
        self.normalizer = SpanNormalizer(self.config.normalizer)

    @property
    def has_should_call(self) -> bool:
        """Checkpoint này có head `should_call` không (ablation §6.1).

        Kiểm bằng head thật chứ không bằng config: một checkpoint cũ nạp qua
        `from_pretrained` sẽ dựng `HeadConfig` từ dict đã lưu, ở đó khoá này
        không tồn tại nên head không được tạo — pipeline phải quay về ngưỡng τ,
        và phải biết điều đó mà không cần ai truyền cờ.
        """
        return getattr(getattr(self.model, "heads", None), "should_call", None) is not None

    @torch.no_grad()
    def should_call_prob(self, query: str, tool_schema: dict[str, Any]) -> float | None:
        """P(query này cần gọi tool này). `None` nếu checkpoint không có head.

        Một forward pass cấp tool, đúng khuôn question mà `dataset` đã dạy —
        `Tool=... Desc=... Params=...`. Lệch khuôn ở đây là train/serve skew,
        loại lỗi không báo mà chỉ làm điểm số tệ đi.
        """
        if not self.has_should_call:
            return None
        pseudo_param = {
            "name": tool_schema.get("name", ""),
            "description": tool_schema.get("description", ""),
            "param_names": [p["name"] for p in iter_parameters(tool_schema)],
            "routing_type": SCHEMA_TYPE_SHOULD_CALL,
        }
        question = cap_question(
            self.tokenizer,
            build_schema_question(pseudo_param),
            self.config.max_question_tokens,
            self._question_cache,
        )[0]
        # Gọi dạng batch (list 1 phần tử) chứ không dạng chuỗi đơn: đi đúng
        # nhánh mà `_predict_chunk` đi, nên pad và `sequence_ids` hành xử giống
        # hệt. Cũng để ngỏ đường hỏi nhiều tool trong một lượt về sau.
        encoded = self.tokenizer(
            [query],
            [question],
            max_length=self.config.max_length,
            padding=True,
            truncation="only_first",
            return_tensors="pt",
        )
        query_mask = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
        for col, sid in enumerate(encoded.sequence_ids(0)):
            if sid == 0:
                query_mask[0, col] = True
        model_inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "query_token_mask": query_mask.to(self.device),
        }
        if "token_type_ids" in encoded:
            model_inputs["token_type_ids"] = encoded["token_type_ids"].to(self.device)
        outputs = self.model(**model_inputs)
        return float(torch.sigmoid(outputs["should_call"])[0].item())

    @torch.no_grad()
    def predict(self, query: str, tool_schema: dict[str, Any]) -> list[ParamPrediction]:
        params = list(iter_parameters(tool_schema))
        if not params:
            return []

        predictions: list[ParamPrediction] = []
        supported = [p for p in params if p["routing_type"] not in UNSUPPORTED_TYPES]
        for param in params:
            if param["routing_type"] in UNSUPPORTED_TYPES:
                predictions.append(
                    ParamPrediction(
                        name=param["name"],
                        routing_type=param["routing_type"],
                        required=bool(param.get("required")),
                        has_value_prob=0.0,
                        unsupported=True,
                    )
                )
        if not supported:
            return predictions

        for chunk_start in range(0, len(supported), self.config.batch_size):
            chunk = supported[chunk_start : chunk_start + self.config.batch_size]
            predictions.extend(self._predict_chunk(query, chunk))
        return predictions

    def _predict_chunk(
        self, query: str, params: list[dict[str, Any]]
    ) -> list[ParamPrediction]:
        questions = [
            cap_question(
                self.tokenizer,
                build_schema_question(p),
                self.config.max_question_tokens,
                self._question_cache,
            )[0]
            for p in params
        ]
        encoded = self.tokenizer(
            [query] * len(params),
            questions,
            max_length=self.config.max_length,
            padding=True,
            truncation="only_first",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping")
        query_mask = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
        for row in range(len(params)):
            seq_ids = encoded.sequence_ids(row)
            for col, sid in enumerate(seq_ids):
                if sid == 0:
                    query_mask[row, col] = True

        model_inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "query_token_mask": query_mask.to(self.device),
        }
        if "token_type_ids" in encoded:
            model_inputs["token_type_ids"] = encoded["token_type_ids"].to(self.device)

        outputs = self.model(**model_inputs)
        has_value_probs = torch.sigmoid(outputs["has_value"]).float().cpu()

        results: list[ParamPrediction] = []
        for row, param in enumerate(params):
            pred = ParamPrediction(
                name=param["name"],
                routing_type=param["routing_type"],
                required=bool(param.get("required")),
                has_value_prob=float(has_value_probs[row].item()),
                has_value_logit=float(outputs["has_value"][row].item()),
            )
            routing_type = param["routing_type"]
            if routing_type in (SCHEMA_TYPE_STRING, SCHEMA_TYPE_NUMBER):
                start, end, score = decode_span(
                    outputs["span_start"][row].float().cpu(),
                    outputs["span_end"][row].float().cpu(),
                    query_mask[row],
                    self.config.max_answer_len,
                )
                offsets = offset_mapping[row].tolist()
                char_start = int(offsets[start][0])
                char_end = int(offsets[end][1])
                span_text = query[char_start:char_end]
                pred.span_text = span_text
                pred.span_char = (char_start, char_end)
                pred.span_score = score
                if self.config.use_normalizer:
                    norm = self.normalizer.normalize(span_text, param)
                    pred.value = norm.value
                    pred.normalizer_applied = norm.applied
                    pred.normalization_failed = not norm.ok
                else:
                    pred.value = span_text.strip()
            elif routing_type == SCHEMA_TYPE_ENUM:
                enum_values = param.get("enum") or []
                logits = outputs["enum_logits"][row][: len(enum_values)]
                pred.enum_logits = logits.detach().float().cpu().tolist()
                idx = int(logits.argmax().item()) if len(enum_values) else 0
                pred.enum_index = idx
                if 0 <= idx < len(enum_values):
                    pred.value = enum_values[idx]
            elif routing_type == SCHEMA_TYPE_BOOLEAN:
                idx = int(outputs["boolean_logits"][row].argmax().item())
                pred.boolean_logits = outputs["boolean_logits"][row].detach().float().cpu().tolist()
                pred.value = idx == BOOLEAN_LABEL_TRUE
            results.append(pred)
        return results

    def to_arguments(
        self,
        predictions: list[ParamPrediction],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Lọc theo `has_value` rồi lắp ráp dict argument."""
        thr = self.config.has_value_threshold if threshold is None else threshold
        args: dict[str, Any] = {}
        for pred in predictions:
            if pred.unsupported or pred.value is None:
                continue
            if pred.has_value_prob < thr:
                continue
            args[pred.name] = pred.value
        return args

    def extract(self, query: str, tool_schema: dict[str, Any]) -> dict[str, Any]:
        return self.to_arguments(self.predict(query, tool_schema))


@torch.no_grad()
def extract_arguments(
    model: CrossEncoderForExtraction,
    tokenizer: PreTrainedTokenizerBase,
    query: str,
    tool_schema: dict[str, Any],
    device: str | torch.device = "cpu",
    has_value_threshold: float = 0.5,
    max_length: int = 256,
) -> dict[str, Any]:
    """API tương thích ngược — bọc `CrossEncoderExtractor`."""
    extractor = CrossEncoderExtractor(
        model,
        tokenizer,
        ExtractionConfig(
            max_length=max_length,
            has_value_threshold=has_value_threshold,
        ),
        device=device,
    )
    return extractor.extract(query, tool_schema)


def load_cross_encoder(
    model_path: str,
    device: str | torch.device = "cpu",
) -> tuple[CrossEncoderForExtraction, PreTrainedTokenizerBase]:
    model = CrossEncoderForExtraction.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model.to(device)
    return model, tokenizer
