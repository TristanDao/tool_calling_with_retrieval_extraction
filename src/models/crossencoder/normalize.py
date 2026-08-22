"""Normalizer hậu xử lý cho span đã dự đoán (§4 Phase 4, method2_plan).

Quy tắc bất di bất dịch:

1. Chỉ chạy trên **span model đã dự đoán**, không chạy trên query gốc.
2. Chỉ áp dụng khi **JSON Schema cho phép** — số chỉ coerce khi type là
   integer/number, ngày chỉ parse khi `format: date`.
3. Không synonym mapping, không LLM (§8.4 experimental_plan cấm cho metric chính).

Thứ tự xử lý: phân cách nghìn → thập phân VN → đơn vị tiền → đơn vị thời gian →
ngày tháng → whitespace/Unicode.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

# ---------------------------------------------------------------- patterns

#: Hệ số nhân theo đơn vị tiếng Việt. Sắp theo độ dài giảm dần khi match.
_MULTIPLIERS: list[tuple[str, float]] = [
    (r"tỷ|tỉ", 1e9),
    (r"triệu|tr\b", 1e6),
    (r"nghìn|ngàn|ngh\b|k\b", 1e3),
]

_MONEY_PARAM_RE = re.compile(
    r"price|vnd|amount|fee|cost|budget|salary|gia_|_gia|money|payment", re.IGNORECASE
)
_MINUTE_PARAM_RE = re.compile(r"minute|_min\b|duration|phut", re.IGNORECASE)
_HOUR_PARAM_RE = re.compile(r"hour|_hrs?\b|gio\b", re.IGNORECASE)
_DATE_PARAM_RE = re.compile(r"date|ngay|day\b|deadline|checkin|checkout", re.IGNORECASE)

_TIME_UNIT_TO_MINUTES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(ngày|ngay)\b", re.IGNORECASE), 1440.0),
    (re.compile(r"\b(tiếng|giờ|gio|h)\b", re.IGNORECASE), 60.0),
    (re.compile(r"\b(phút|phut|min|minutes?)\b", re.IGNORECASE), 1.0),
    (re.compile(r"\b(giây|giay|s|sec|seconds?)\b", re.IGNORECASE), 1.0 / 60.0),
]

_NUMBER_CORE_RE = re.compile(r"[-+]?\d[\d.,\s]*")
_THOUSAND_DOT_RE = re.compile(r"^[-+]?\d{1,3}(\.\d{3})+$")
_THOUSAND_COMMA_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3})+$")

_RELATIVE_DAYS: dict[str, int] = {
    "hôm nay": 0,
    "hom nay": 0,
    "nay": 0,
    "mai": 1,
    "ngày mai": 1,
    "ngay mai": 1,
    "mốt": 2,
    "ngày mốt": 2,
    "hôm qua": -1,
    "hom qua": -1,
}

_DMY_RE = re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})(?:\s*[/\-.]\s*(\d{2,4}))?\b")
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_VN_WORD_DATE_RE = re.compile(
    r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})(?:\s*năm\s*(\d{4}))?", re.IGNORECASE
)


@dataclass
class NormalizationResult:
    """Giá trị sau chuẩn hoá kèm vết các rule đã áp dụng (cho ablation §6.2)."""

    value: Any
    applied: list[str] = field(default_factory=list)
    ok: bool = True

    @property
    def changed(self) -> bool:
        return bool(self.applied)


@dataclass
class NormalizerConfig:
    enabled: bool = True
    #: Ngày tham chiếu cho biểu thức tương đối ("mai"). None → bỏ qua nhóm này.
    reference_date: date | None = None
    #: Thế kỷ mặc định khi năm viết 2 chữ số.
    century: int = 2000


# ------------------------------------------------------------------ helpers


def normalize_text(text: str) -> str:
    """NFC + trim + collapse whitespace + bỏ dấu câu thừa ở hai đầu."""
    cleaned = unicodedata.normalize("NFC", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" \t\n\r.,;:!?\"'“”‘’()[]")


def _strip_multiplier(text: str) -> tuple[str, float, str | None]:
    """Tách hệ số nhân đơn vị tiền ra khỏi chuỗi."""
    for pattern, factor in _MULTIPLIERS:
        match = re.search(rf"(?<=[\d\s])({pattern})", text, re.IGNORECASE)
        if match:
            return text[: match.start()] + text[match.end() :], factor, match.group(0)
    return text, 1.0, None


def parse_number(text: str) -> float | None:
    """Parse số theo quy ước VN lẫn EN.

    - ``50.000`` / ``50,000`` → 50000 (phân cách nghìn)
    - ``1,5`` → 1.5 (thập phân VN)
    - ``1.234,5`` → 1234.5 (cả hai, cái sau là thập phân)
    """
    match = _NUMBER_CORE_RE.search(text)
    if not match:
        return None
    core = re.sub(r"\s+", "", match.group(0)).rstrip(".,")
    if not core or not re.search(r"\d", core):
        return None

    has_dot = "." in core
    has_comma = "," in core
    if has_dot and has_comma:
        decimal_sep = "." if core.rfind(".") > core.rfind(",") else ","
        thousand_sep = "," if decimal_sep == "." else "."
        core = core.replace(thousand_sep, "").replace(decimal_sep, ".")
    elif has_dot:
        core = core.replace(".", "") if _THOUSAND_DOT_RE.match(core) else core
    elif has_comma:
        core = core.replace(",", "") if _THOUSAND_COMMA_RE.match(core) else core.replace(",", ".")

    try:
        return float(core)
    except ValueError:
        return None


def parse_date(text: str, reference_date: date | None, century: int = 2000) -> str | None:
    """Trả về ISO ``YYYY-MM-DD``, hoặc None nếu không nhận dạng được."""
    cleaned = unicodedata.normalize("NFC", text).strip().lower()

    iso = _ISO_RE.search(cleaned)
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"

    word = _VN_WORD_DATE_RE.search(cleaned)
    if word:
        day, month = int(word.group(1)), int(word.group(2))
        year = int(word.group(3)) if word.group(3) else (reference_date.year if reference_date else None)
        return _format_date(year, month, day)

    dmy = _DMY_RE.search(cleaned)
    if dmy:
        day, month = int(dmy.group(1)), int(dmy.group(2))
        raw_year = dmy.group(3)
        if raw_year:
            year = int(raw_year)
            if year < 100:
                year += century
        else:
            year = reference_date.year if reference_date else None
        return _format_date(year, month, day)

    if reference_date is not None:
        for phrase, delta in sorted(_RELATIVE_DAYS.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
                return (reference_date + timedelta(days=delta)).isoformat()
    return None


def _format_date(year: int | None, month: int, day: int) -> str | None:
    if year is None or not (1 <= month <= 12) or not (1 <= day <= 31):
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


# ----------------------------------------------------------------- main API


class SpanNormalizer:
    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()

    def normalize(self, span_text: str, param: dict[str, Any]) -> NormalizationResult:
        """Chuẩn hoá span theo schema của `param`."""
        text = normalize_text(span_text)
        if not self.config.enabled:
            return NormalizationResult(value=text, applied=[])
        if not text:
            return NormalizationResult(value=text, applied=[], ok=False)

        value_type = param.get("value_type") or param.get("type") or "string"
        if isinstance(value_type, (list, tuple)):
            value_type = next((t for t in value_type if t != "null"), "string")
        value_type = str(value_type).lower()
        param_name = str(param.get("name", ""))
        fmt = str(param.get("format", "") or "").lower()

        if value_type in ("integer", "number"):
            return self._normalize_number(text, param_name, value_type)
        if fmt in ("date", "date-time") or (fmt == "" and _DATE_PARAM_RE.search(param_name) and value_type == "string"):
            iso = parse_date(text, self.config.reference_date, self.config.century)
            if iso is not None and fmt in ("date", "date-time"):
                return NormalizationResult(value=iso, applied=["date"])
            # Không có `format` trong schema → không được ép kiểu, chỉ trim.
            return NormalizationResult(value=text, applied=[])
        return NormalizationResult(value=text, applied=[])

    def _normalize_number(
        self, text: str, param_name: str, value_type: str
    ) -> NormalizationResult:
        applied: list[str] = []
        stripped, multiplier, unit = _strip_multiplier(text)
        if multiplier != 1.0:
            applied.append(f"money_unit:{unit}")

        number = parse_number(stripped)
        if number is None:
            return NormalizationResult(value=text, applied=[], ok=False)
        if re.search(r"[.,]", stripped):
            applied.append("separator")

        number *= multiplier

        time_factor = self._time_factor(text, param_name)
        if time_factor is not None and time_factor != 1.0:
            number *= time_factor
            applied.append("time_unit")

        if value_type == "integer" or float(number).is_integer():
            if abs(number - round(number)) < 1e-9:
                return NormalizationResult(value=int(round(number)), applied=applied)
        return NormalizationResult(value=float(number), applied=applied)

    def _time_factor(self, text: str, param_name: str) -> float | None:
        """Hệ số quy đổi đơn vị thời gian trong span về đơn vị của param."""
        target: float
        # Xét "hour" trước: `duration_hours` khớp cả hai pattern.
        if _HOUR_PARAM_RE.search(param_name):
            target = 60.0
        elif _MINUTE_PARAM_RE.search(param_name):
            target = 1.0
        else:
            return None
        for pattern, minutes in _TIME_UNIT_TO_MINUTES:
            if pattern.search(text):
                return minutes / target
        return None


def is_money_param(param_name: str) -> bool:
    return bool(_MONEY_PARAM_RE.search(param_name))
