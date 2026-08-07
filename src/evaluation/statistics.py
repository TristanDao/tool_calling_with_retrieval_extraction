"""Bootstrap confidence intervals and paired exact significance tests."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from src.evaluation.efficiency_metrics import percentile


def bootstrap_mean_interval(
    values: Sequence[bool | float],
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if not values:
        return {"estimate": None, "lower": None, "upper": None, "samples": samples}
    numeric = [float(value) for value in values]
    estimate = sum(numeric) / len(numeric)
    if samples <= 0:
        return {"estimate": estimate, "lower": None, "upper": None, "samples": 0}
    rng = random.Random(seed)
    bootstrap_values = [
        sum(rng.choice(numeric) for _ in numeric) / len(numeric) for _ in range(samples)
    ]
    alpha = 1.0 - confidence_level
    return {
        "estimate": estimate,
        "lower": percentile(bootstrap_values, alpha / 2),
        "upper": percentile(bootstrap_values, 1 - alpha / 2),
        "samples": samples,
        "confidence_level": confidence_level,
    }


def paired_bootstrap_difference(
    first: Sequence[bool | float],
    second: Sequence[bool | float],
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("Paired bootstrap inputs must have equal length")
    differences = [float(a) - float(b) for a, b in zip(first, second, strict=True)]
    return bootstrap_mean_interval(differences, samples, confidence_level, seed)


def exact_mcnemar(first: Sequence[bool], second: Sequence[bool]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("McNemar inputs must have equal length")
    first_only = sum(a and not b for a, b in zip(first, second, strict=True))
    second_only = sum(b and not a for a, b in zip(first, second, strict=True))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant_pairs": discordant,
        "p_value_two_sided_exact": p_value,
    }
