"""Data collator for Cross-Encoder.

Each sample = 1 (query, parameter) pair. Tokenize theo BERT-QA format:
[CLS] <query> [SEP] Param=... Type=...[. Enum=...] [SEP]

Query luôn là segment đầu tiên nên token index của span label (tính từ 1, ngay
sau [CLS]) không đổi khi batch được pad bên phải.
"""

import re
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


SCHEMA_TYPE_STRING = "string"
SCHEMA_TYPE_NUMBER = "number"
SCHEMA_TYPE_BOOLEAN = "boolean"
SCHEMA_TYPE_ENUM = "enum"
SCHEMA_TYPE_ARRAY = "array"
SCHEMA_TYPE_OBJECT = "object"

#: Hàng cấp TOOL, không phải cấp parameter (ablation §6.1: `should_call` head).
#: Đi chung batch với hàng parameter và được `HierarchicalLoss` tách ra — hai
#: loại hàng không được huấn luyện chéo head của nhau.
SCHEMA_TYPE_SHOULD_CALL = "should_call"

_VALID_TYPES = (
    SCHEMA_TYPE_STRING,
    SCHEMA_TYPE_NUMBER,
    SCHEMA_TYPE_BOOLEAN,
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_ARRAY,
    SCHEMA_TYPE_OBJECT,
)

#: Type không được model hỗ trợ — §3.2 method2_plan, báo cáo coverage riêng.
UNSUPPORTED_TYPES = (SCHEMA_TYPE_ARRAY, SCHEMA_TYPE_OBJECT)


_TYPE_MAP: dict[str, str] = {
    "str": SCHEMA_TYPE_STRING,
    "string": SCHEMA_TYPE_STRING,
    "int": SCHEMA_TYPE_NUMBER,
    "integer": SCHEMA_TYPE_NUMBER,
    "float": SCHEMA_TYPE_NUMBER,
    "double": SCHEMA_TYPE_NUMBER,
    "number": SCHEMA_TYPE_NUMBER,
    "bool": SCHEMA_TYPE_BOOLEAN,
    "boolean": SCHEMA_TYPE_BOOLEAN,
    "enum": SCHEMA_TYPE_ENUM,
    "list": SCHEMA_TYPE_ARRAY,
    "array": SCHEMA_TYPE_ARRAY,
    "object": SCHEMA_TYPE_OBJECT,
    "dict": SCHEMA_TYPE_OBJECT,
}

#: Type gốc theo JSON Schema — normalizer cần phân biệt integer với number.
_RAW_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "object": "object",
    "dict": "object",
}

_OPTIONAL_SUFFIX_RE = re.compile(r",\s*optional\s*$", re.IGNORECASE)
_LIST_GENERIC_RE = re.compile(r"^list(\[.*\])?$", re.IGNORECASE)


def _clean_type(raw: Any) -> str:
    if isinstance(raw, (list, tuple)):
        # JSON Schema cho phép type là list, vd ["string", "null"].
        non_null = [t for t in raw if str(t).lower() != "null"]
        raw = non_null[0] if non_null else ""
    return _OPTIONAL_SUFFIX_RE.sub("", str(raw or "").strip()).lower()


def normalize_schema_type(raw: Any) -> str:
    """Type dùng để route head. `integer` gộp vào `number`."""
    cleaned = _clean_type(raw)
    if not cleaned:
        return SCHEMA_TYPE_STRING
    if cleaned in _TYPE_MAP:
        return _TYPE_MAP[cleaned]
    if _LIST_GENERIC_RE.match(cleaned):
        return SCHEMA_TYPE_ARRAY
    return SCHEMA_TYPE_STRING


def raw_schema_type(raw: Any) -> str:
    """Type gốc, giữ nguyên `integer` để normalizer coerce đúng."""
    cleaned = _clean_type(raw)
    if not cleaned:
        return "string"
    if cleaned in _RAW_TYPE_MAP:
        return _RAW_TYPE_MAP[cleaned]
    if _LIST_GENERIC_RE.match(cleaned):
        return "array"
    return "string"


def resolve_param_type(param: dict[str, Any]) -> str:
    """Routing type của một parameter.

    JSON Schema biểu diễn enum bằng ``{"type": "string", "enum": [...]}`` chứ
    không phải ``"type": "enum"``, nên phải xét sự có mặt của khoá ``enum`` trước.
    """
    if param.get("enum"):
        return SCHEMA_TYPE_ENUM
    return normalize_schema_type(param.get("type", SCHEMA_TYPE_STRING))


def iter_parameters(tool_schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Duyệt parameter của một tool theo unified master structure.

    Nhận cả ``{"parameters": {"type": "object", "properties": {...},
    "required": [...]}}`` lẫn dạng phẳng ``{"parameters": {"name": {...}}}``.
    """
    params = tool_schema.get("parameters") or {}
    if not isinstance(params, dict):
        return
    properties = params.get("properties")
    if not isinstance(properties, dict):
        properties = {k: v for k, v in params.items() if isinstance(v, dict)}
        required: Any = []
    else:
        required = params.get("required")
    # Glaive có sample ghi `required: true` ở cấp object và gắn cờ `required`
    # vào từng property. Chỉ dạng list mới là JSON Schema hợp lệ.
    required_set = set(required) if isinstance(required, list) else None
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        param = dict(spec)
        param["name"] = name
        param["required"] = (
            name in required_set
            if required_set is not None
            else bool(spec.get("required", False))
        )
        param["routing_type"] = resolve_param_type(param)
        param["value_type"] = raw_schema_type(param.get("type", "string"))
        yield param


def build_should_call_question(tool: dict[str, Any]) -> str:
    """Question cấp tool cho head `should_call`.

    Cùng khuôn `Khoá=giá trị. ` với question cấp parameter để encoder không phải
    học hai văn phong. Liệt kê tên parameter chứ không liệt kê mô tả của chúng:
    tên đủ để nhận ra tool làm gì, còn mô tả đầy đủ thì tràn `max_question_tokens`
    và đẩy query ra khỏi cửa sổ.
    """
    name = tool.get("name", "")
    desc = str(tool.get("description", "") or "").strip()
    params = "|".join(str(p) for p in (tool.get("param_names") or []))
    return f"Tool={name}. Desc={desc}. Params={params}"


def build_schema_question(param: dict[str, Any]) -> str:
    if param.get("routing_type") == SCHEMA_TYPE_SHOULD_CALL:
        return build_should_call_question(param)
    name = param["name"]
    desc = str(param.get("description", "") or "").strip()
    routing_type = param.get("routing_type") or resolve_param_type(param)
    if routing_type == SCHEMA_TYPE_ENUM:
        enum_values = param.get("enum", []) or []
        enum_str = "|".join(str(v) for v in enum_values)
        return f"Param={name}. Desc={desc}. Type=enum. Enum={enum_str}"
    declared = param.get("value_type") or raw_schema_type(param.get("type", "string"))
    return f"Param={name}. Desc={desc}. Type={declared}"


#: Trần token mặc định cho schema question. Dùng chung giữa training
#: (`CrossEncoderCollator`) và inference (`inference.py`) — hai bên phải cắt
#: GIỐNG HỆT nhau, nếu không model được train trên question đã cắt mà lúc chạy
#: thật lại thấy question đầy đủ (train/serve skew).
DEFAULT_MAX_QUESTION_TOKENS = 96


def cap_question(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_question_tokens: int = DEFAULT_MAX_QUESTION_TOKENS,
    cache: dict[str, tuple[str, int]] | None = None,
) -> tuple[str, int]:
    """(question đã cắt xuống trần, số token thật của nó).

    `truncation="only_first"` chỉ cắt query, nên question dài hơn `max_length`
    làm tokenizer ném "Sequence to truncate too short". Đo trên dữ liệu: p99 =
    198 ký tự (~66 token) nhưng dài nhất 883 ký tự (461 token) — chỉ 23/142k
    dòng, đủ để giết cả job giữa chừng.
    """
    if cache is not None and question in cache:
        return cache[question]
    ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    if len(ids) > max_question_tokens:
        text = tokenizer.decode(ids[:max_question_tokens])
        # Decode rồi encode lại có thể lệch vài token, nên đếm lại cho đúng.
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        text = question
    entry = (text, len(ids))
    if cache is not None:
        cache[question] = entry
    return entry


@dataclass
class CollatorConfig:
    tokenizer_name: str = "xlm-roberta-base"
    max_length: int = 256
    padding: str = "longest"
    truncation: str = "only_first"
    #: Trần token cho phần schema question. `only_first` chỉ cắt query, nên
    #: question dài hơn `max_length` làm tokenizer ném "Sequence to truncate too
    #: short". Đo thực tế: p99 = 198 ký tự (~66 token), dài nhất 883 ký tự.
    max_question_tokens: int = DEFAULT_MAX_QUESTION_TOKENS


class CrossEncoderCollator:
    def __init__(
        self,
        config: CollatorConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = tokenizer or AutoTokenizer.from_pretrained(
            config.tokenizer_name, use_fast=True
        )
        # Một schema question lặp lại ở rất nhiều dòng (cùng tool, cùng param),
        # nên cache theo chuỗi gốc để không tokenize 142k lần.
        self._question_cache: dict[str, tuple[str, int]] = {}

    def _n_special_pair(self) -> int:
        """Số special token khi encode cặp — XLM-R: `<s> A </s> </s> B </s>` = 4."""
        try:
            return int(self.tokenizer.num_special_tokens_to_add(pair=True))
        except (AttributeError, TypeError):
            return 4

    def _question_entry(self, param: dict[str, Any]) -> tuple[str, int]:
        return cap_question(
            self.tokenizer,
            build_schema_question(param),
            self.config.max_question_tokens,
            self._question_cache,
        )

    def query_token_budget(self, param: dict[str, Any]) -> int:
        """Số token query CÒN LẠI sau khi trừ question và special token.

        Đây là con số `dataset` phải dùng để căn span. Căn theo `max_length`
        rồi để collator cắt query ngắn hơn thì nhãn span trỏ ra ngoài chuỗi và
        bị `_collate_labels._clip` kẹp về vị trí sai — hỏng nhãn mà không báo.
        """
        _, n_question = self._question_entry(param)
        return self.config.max_length - n_question - self._n_special_pair()

    def encode_one(
        self,
        query: str,
        param: dict[str, Any],
        labels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Tokenize 1 cặp (query, param). Không pad — pad ở `__call__`."""
        question, _ = self._question_entry(param)
        encoded = self.tokenizer(
            query,
            question,
            max_length=self.config.max_length,
            padding=False,
            truncation=self.config.truncation,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        token_type_ids = list(encoded.get("token_type_ids") or [0] * len(input_ids))
        query_token_mask = self._query_token_mask(encoded, len(input_ids))

        item: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "query_token_mask": query_token_mask,
            "schema_type": param.get("routing_type") or resolve_param_type(param),
            # Số giá trị enum hợp lệ. `inference.py` cắt logits về đúng khoảng
            # này; metric phải làm y hệt, nếu không nó chấm một thứ mà pipeline
            # thật không bao giờ sinh ra.
            "enum_size": len(param.get("enum") or []),
        }
        if labels is not None:
            item["labels"] = labels
        return item

    def _query_token_mask(self, encoded: Any, length: int) -> list[bool]:
        """True tại token thuộc segment query (không tính special token)."""
        seq_ids = None
        if hasattr(encoded, "sequence_ids"):
            try:
                seq_ids = encoded.sequence_ids(0)
            except (ValueError, TypeError):
                seq_ids = None
        if seq_ids is not None:
            return [sid == 0 for sid in seq_ids]
        # Fallback cho tokenizer chậm: cắt tại SEP đầu tiên.
        ids = list(encoded["input_ids"])
        sep_id = self.tokenizer.sep_token_id
        first_sep = ids.index(sep_id) if sep_id in ids else length
        mask = [False] * length
        for i in range(1, min(first_sep, length)):
            mask[i] = True
        return mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id or 0
        max_len = max(len(b["input_ids"]) for b in batch)
        if self.config.padding == "max_length":
            max_len = self.config.max_length

        def _pad(seq: list[Any], fill: Any) -> list[Any]:
            return list(seq) + [fill] * (max_len - len(seq))

        out: dict[str, Any] = {
            "input_ids": torch.tensor(
                [_pad(b["input_ids"], pad_id) for b in batch], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [_pad(b["attention_mask"], 0) for b in batch], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                [_pad(b["token_type_ids"], 0) for b in batch], dtype=torch.long
            ),
            "query_token_mask": torch.tensor(
                [_pad(b["query_token_mask"], False) for b in batch], dtype=torch.bool
            ),
            "schema_type": [b["schema_type"] for b in batch],
            "enum_size": self._enum_sizes(batch),
        }
        if "labels" in batch[0]:
            out["labels"] = self._collate_labels([b["labels"] for b in batch], max_len)
            out["labels"]["enum_size"] = out["enum_size"]
        return out

    def _collate_labels(
        self, labels_list: list[dict[str, Any]], max_len: int
    ) -> dict[str, Any]:
        def _clip(idx: Any) -> int:
            return min(max(int(idx), 0), max_len - 1)

        # `.get(..., 0)` chứ không `[...]`: hàng cấp tool (`should_call`) không
        # có nhãn span/enum/boolean. Số 0 ở đây là chỗ giữ tensor cho đúng hình,
        # `HierarchicalLoss` đã loại hẳn những hàng đó khỏi các head tương ứng
        # nên giá trị không bao giờ được học.
        return {
            "has_value": torch.tensor(
                [lbl.get("has_value", 0) for lbl in labels_list], dtype=torch.long
            ),
            "span_start": torch.tensor(
                [_clip(lbl.get("span_start", 0)) for lbl in labels_list], dtype=torch.long
            ),
            "span_end": torch.tensor(
                [_clip(lbl.get("span_end", 0)) for lbl in labels_list], dtype=torch.long
            ),
            "enum_label": torch.tensor(
                [lbl.get("enum_label", 0) for lbl in labels_list], dtype=torch.long
            ),
            "boolean_label": torch.tensor(
                [lbl.get("boolean_label", 0) for lbl in labels_list], dtype=torch.long
            ),
            "should_call": torch.tensor(
                [lbl.get("should_call", 0) for lbl in labels_list], dtype=torch.long
            ),
            "schema_type": [lbl["schema_type"] for lbl in labels_list],
        }

    @staticmethod
    def _enum_sizes(batch: list[dict[str, Any]]) -> list[int]:
        return [int(b.get("enum_size", 0)) for b in batch]
