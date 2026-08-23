"""Tests for paired statistical comparisons."""

from __future__ import annotations

from src.evaluation.statistics import (
    bootstrap_mean_interval,
    exact_mcnemar,
    paired_bootstrap_difference,
)


def test_bootstrap_is_reproducible() -> None:
    first = bootstrap_mean_interval([True, True, False, True], 100, 0.95, 42)
    second = bootstrap_mean_interval([True, True, False, True], 100, 0.95, 42)
    assert first == second
    assert first["estimate"] == 0.75


def test_paired_tests() -> None:
    first = [True, True, False, True]
    second = [False, True, False, False]
    mcnemar = exact_mcnemar(first, second)
    difference = paired_bootstrap_difference(first, second, 100, 0.95, 3)
    assert mcnemar["first_only_correct"] == 2
    assert mcnemar["second_only_correct"] == 0
    assert mcnemar["p_value_two_sided_exact"] == 0.5
    assert difference["estimate"] == 0.5
