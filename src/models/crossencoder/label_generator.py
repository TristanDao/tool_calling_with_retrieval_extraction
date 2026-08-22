"""Rule-based label generator for Cross-Encoder.

Với mỗi (query, parameter, gold_value) sinh nhãn train:

- string/number/integer: căn `gold_value` vào query theo **chuỗi nguyên văn** →
  (start, end) token. Không căn được → `SKIP` (§3.3 method2_plan: non-verbatim
  nằm ngoài khả năng của span head, không được bịa nhãn).
- enum: khớp gold với danh sách enum → class id.
- boolean: lấy thẳng từ gold `true`/`false`. **Không** đoán theo cue từ — cue ở
  cấp toàn query không gắn được với parameter cụ thể (bug §2.3c). Cue chỉ dùng
  tùy chọn để *lọc* sample mà query không hề có manh mối.
- param không xuất hiện trong arguments và là optional → `has_value=0`.
- array/object → `SKIP` với lý do `unsupported_type`.

`generate()` trả về dict nhãn, hoặc `SkipLabel` mang lý do để thống kê §1.5.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.models.crossencoder.data_collator import (
    SCHEMA_TYPE_BOOLEAN,
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_NUMBER,
    SCHEMA_TYPE_STRING,
    UNSUPPORTED_TYPES,
    resolve_param_type,
)


SKIP_LABEL = "skip"


@dataclass(frozen=True)
class SkipLabel:
    """Sample bị loại khỏi train, kèm lý do để thống kê `label_stats.json`."""

    reason: str

    def __str__(self) -> str:  # giữ tương thích với so sánh `== SKIP_LABEL`
        return SKIP_LABEL

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SkipLabel):
            return self.reason == other.reason
        return other == SKIP_LABEL

    def __hash__(self) -> int:
        return hash(SKIP_LABEL)


BOOLEAN_TRUE_CUES = [
    r"\bcó\b", r"\bmuốn\b", r"\bcần\b", r"\bnên\b", r"\bđược\b",
    r"\bđồng ý\b", r"\bchấp nhận\b", r"\byes\b", r"\btrue\b", r"\bđúng\b",
]

BOOLEAN_FALSE_CUES = [
    r"\bkhông\b", r"\bchưa\b", r"\bđừng\b", r"\bcấm\b", r"\bmiễn\b",
    r"\bno\b", r"\bfalse\b", r"\bsai\b", r"\btừ chối\b",
]

BOOLEAN_LABEL_TRUE = 0
BOOLEAN_LABEL_FALSE = 1


@dataclass
class LabelGeneratorConfig:
    tokenizer_name: str = "xlm-roberta-base"
    max_length: int = 256
    #: Bỏ sample boolean mà query không chứa cue nào (tránh train model đoán mò).
    require_boolean_cue: bool = False
    #: Cửa sổ ký tự quanh keyword của param khi dò cue boolean.
    boolean_cue_window: int = 60
    #: Cho phép căn span bỏ qua hoa/thường và khác biệt khoảng trắng.
    relaxed_span_match: bool = True


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class LabelGenerator:
    def __init__(
        self,
        config: LabelGeneratorConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = tokenizer or AutoTokenizer.from_pretrained(
            config.tokenizer_name, use_fast=True
        )
        self.true_patterns = [re.compile(p, re.IGNORECASE) for p in BOOLEAN_TRUE_CUES]
        self.false_patterns = [re.compile(p, re.IGNORECASE) for p in BOOLEAN_FALSE_CUES]

    # ------------------------------------------------------------------ spans

    def find_char_span(self, query: str, value: str) -> tuple[int, int] | None:
        """Vị trí ký tự [start, end) của `value` trong `query`, hoặc None."""
        query_n = _nfc(query)
        value_n = _nfc(str(value)).strip()
        if not value_n:
            return None
        idx = query_n.find(value_n)
        if idx >= 0:
            return idx, idx + len(value_n)
        if not self.config.relaxed_span_match:
            return None
        idx = query_n.lower().find(value_n.lower())
        if idx >= 0:
            return idx, idx + len(value_n)
        # Khoảng trắng trong gold có thể khác query (vd "Hà  Nội").
        pattern = r"\s+".join(re.escape(tok) for tok in value_n.split())
        match = re.search(pattern, query_n, re.IGNORECASE)
        if match:
            return match.start(), match.end()
        return None

    def _align_span(
        self,
        query: str,
        value: str,
        max_length: int,
    ) -> tuple[int, int, int, int] | None:
        """(start_tok, end_tok, char_start, char_end) — token index tính cả [CLS]."""
        char_span = self.find_char_span(query, value)
        if char_span is None:
            return None
        char_start, char_end_excl = char_span
        char_end = char_end_excl - 1
        offsets = self.tokenizer(
            _nfc(query),
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length - 2,
        )["offset_mapping"]
        start_tok: int | None = None
        end_tok: int | None = None
        for tok_idx, (s, e) in enumerate(offsets):
            if e <= s:
                continue
            if start_tok is None and s <= char_start < e:
                start_tok = tok_idx + 1
            if s <= char_end < e:
                end_tok = tok_idx + 1
        if start_tok is None or end_tok is None or end_tok < start_tok:
            return None
        if end_tok >= max_length - 1:
            return None
        return start_tok, end_tok, char_start, char_end_excl

    # ---------------------------------------------------------------- boolean

    def _detect_boolean_cue(self, query: str, param: dict[str, Any]) -> bool:
        """Query có manh mối boolean nào **gần** keyword của param không.

        Keyword lấy từ tên param (tách `_`) và các từ trong `description`.
        Không tìm được keyword thì xét toàn query.
        """
        text = _nfc(query).lower()
        window = self.config.boolean_cue_window
        keywords = [w for w in re.split(r"[_\s]+", str(param.get("name", "")).lower()) if len(w) > 2]
        desc = str(param.get("description", "") or "").lower()
        keywords += [w for w in re.findall(r"\w{4,}", desc)]

        regions: list[str] = []
        for kw in keywords:
            pos = text.find(kw)
            if pos >= 0:
                regions.append(text[max(0, pos - window) : pos + len(kw) + window])
        if not regions:
            regions = [text]

        patterns = self.true_patterns + self.false_patterns
        return any(pat.search(region) for region in regions for pat in patterns)

    # --------------------------------------------------------------- generate

    def generate(
        self,
        query: str,
        param: dict[str, Any],
        gold_value: Any,
        is_required: bool | None = None,
    ) -> dict[str, Any] | SkipLabel:
        routing_type = param.get("routing_type") or resolve_param_type(param)
        if is_required is None:
            is_required = bool(param.get("required", False))

        result: dict[str, Any] = {
            "has_value": 0,
            "span_start": 0,
            "span_end": 0,
            "enum_label": 0,
            "boolean_label": BOOLEAN_LABEL_TRUE,
            "schema_type": routing_type,
        }

        if routing_type in UNSUPPORTED_TYPES:
            return SkipLabel("unsupported_type")

        if gold_value is None:
            if is_required:
                # Gold mâu thuẫn: required nhưng không có giá trị → không train.
                return SkipLabel("required_without_value")
            return result

        if routing_type == SCHEMA_TYPE_ENUM:
            enum_values = [str(v) for v in (param.get("enum") or [])]
            value_str = str(gold_value)
            if value_str not in enum_values:
                lowered = [v.lower() for v in enum_values]
                if value_str.lower() in lowered:
                    result["has_value"] = 1
                    result["enum_label"] = lowered.index(value_str.lower())
                    return result
                return SkipLabel("enum_value_not_in_schema")
            result["has_value"] = 1
            result["enum_label"] = enum_values.index(value_str)
            return result

        if routing_type == SCHEMA_TYPE_BOOLEAN:
            if isinstance(gold_value, str):
                lowered = gold_value.strip().lower()
                if lowered in ("true", "có", "yes", "1"):
                    bool_value = True
                elif lowered in ("false", "không", "no", "0"):
                    bool_value = False
                else:
                    return SkipLabel("boolean_value_unparseable")
            else:
                bool_value = bool(gold_value)
            if self.config.require_boolean_cue and not self._detect_boolean_cue(query, param):
                return SkipLabel("boolean_no_cue")
            result["has_value"] = 1
            result["boolean_label"] = BOOLEAN_LABEL_TRUE if bool_value else BOOLEAN_LABEL_FALSE
            return result

        if routing_type in (SCHEMA_TYPE_STRING, SCHEMA_TYPE_NUMBER):
            if isinstance(gold_value, bool):
                return SkipLabel("type_mismatch")
            span = self._align_span(query, str(gold_value), self.config.max_length)
            if span is None:
                return SkipLabel("non_verbatim")
            start_tok, end_tok, char_start, char_end = span
            result["has_value"] = 1
            result["span_start"] = start_tok
            result["span_end"] = end_tok
            result["char_start"] = char_start
            result["char_end"] = char_end
            return result

        return SkipLabel("unknown_type")
