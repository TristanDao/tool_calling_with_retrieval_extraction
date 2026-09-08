"""Structured configuration for the evaluation pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(kw_only=True)
class NormalizationConfig:
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    collapse_whitespace: bool = True
    trim_whitespace: bool = True
    casefold_strings: bool = True
    coerce_numbers: bool = True
    coerce_booleans: bool = True
    normalize_dates: bool = True
    casefold_identifiers: bool = False
    global_aliases: dict[str, Any] = field(default_factory=dict)
    parameter_aliases: dict[str, dict[str, Any]] = field(default_factory=dict)
    unordered_array_paths: tuple[str, ...] = ()

    @classmethod
    def strict(cls) -> NormalizationConfig:
        """Biến thể strict: tắt mọi nới lỏng ở mức GIÁ TRỊ.

        Giữ Unicode NFC; tắt nới lỏng whitespace, case, type và alias.
        Strict và normalized chấm cùng output sau postprocessing, nên chênh
        lệch không phải ablation của normalizer trong inference.
        """
        return cls(
            unicode_form="NFC",
            collapse_whitespace=False,
            trim_whitespace=False,
            casefold_strings=False,
            coerce_numbers=False,
            coerce_booleans=False,
            normalize_dates=False,
            casefold_identifiers=False,
        )

    @classmethod
    def from_alias_file(cls, path: Path, base: NormalizationConfig | None = None) -> NormalizationConfig:
        current = asdict(base or cls())
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Normalization config must be a JSON object")
        unicode_form = raw.get("unicode_form", current["unicode_form"])
        if unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError(f"Unsupported Unicode normalization form: {unicode_form}")
        current["global_aliases"] = raw.get("global", raw.get("global_aliases", {}))
        current["parameter_aliases"] = raw.get("by_parameter", raw.get("parameter_aliases", {}))
        current["unordered_array_paths"] = tuple(raw.get("unordered_array_paths", ()))
        for key in (
            "unicode_form",
            "collapse_whitespace",
            "trim_whitespace",
            "casefold_strings",
            "coerce_numbers",
            "coerce_booleans",
            "normalize_dates",
            "casefold_identifiers",
        ):
            if key in raw:
                current[key] = raw[key]
        return cls(**current)


@dataclass(kw_only=True)
class EvaluationConfig:
    retrieval_ks: tuple[int, ...] = (1, 3, 5, 10)
    slice_fields: tuple[str, ...] = (
        "source",
        "call_count",
        "schema_parameter_count",
        "candidate_tool_count",
        "feature_group",
    )
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42
    strict_coverage: bool = True
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
