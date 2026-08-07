"""Latency, throughput, token, cost, and memory metrics."""

from __future__ import annotations

import math
from typing import Any

from src.evaluation.types import PredictionRecord


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float], total_count: int) -> dict[str, Any]:
    return {
        "coverage": len(values) / total_count if total_count else None,
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def compute_efficiency_metrics(
    predictions: list[PredictionRecord],
    overall_success: list[bool],
) -> dict[str, Any]:
    total_count = len(predictions)
    latencies = [item.telemetry["latency_ms"] for item in predictions if "latency_ms" in item.telemetry]
    input_tokens = [item.telemetry["input_tokens"] for item in predictions if "input_tokens" in item.telemetry]
    output_tokens = [item.telemetry["output_tokens"] for item in predictions if "output_tokens" in item.telemetry]
    costs = [item.telemetry["cost_usd"] for item in predictions if "cost_usd" in item.telemetry]
    memory = [
        item.telemetry["gpu_peak_memory_mb"]
        for item in predictions
        if "gpu_peak_memory_mb" in item.telemetry
    ]
    latency = _summary(latencies, total_count)
    mean_latency = latency["mean"]
    cost_coverage_complete = len(costs) == total_count and total_count > 0
    total_cost = sum(costs) if costs else None
    complete_total_cost = sum(costs) if cost_coverage_complete else None
    correct_count = sum(overall_success)
    token_pairs = [
        item.telemetry.get("input_tokens", 0.0) + item.telemetry.get("output_tokens", 0.0)
        for item in predictions
        if "input_tokens" in item.telemetry and "output_tokens" in item.telemetry
    ]
    return {
        "latency_ms": latency,
        "estimated_sequential_throughput_qps": (
            1000.0 / mean_latency if mean_latency and mean_latency > 0 else None
        ),
        "input_tokens": _summary(input_tokens, total_count),
        "output_tokens": _summary(output_tokens, total_count),
        "total_tokens": _summary(token_pairs, total_count),
        "cost": {
            "coverage": len(costs) / total_count if total_count else None,
            "observed_total_usd": total_cost,
            "total_usd": complete_total_cost,
            "usd_per_query": (
                complete_total_cost / total_count if complete_total_cost is not None else None
            ),
            "usd_per_1k_queries": (
                complete_total_cost * 1000 / total_count
                if complete_total_cost is not None
                else None
            ),
            "usd_per_correct_call": (
                complete_total_cost / correct_count
                if complete_total_cost is not None and correct_count
                else None
            ),
        },
        "gpu_peak_memory_mb": {
            "coverage": len(memory) / total_count if total_count else None,
            "max": max(memory) if memory else None,
            "mean": sum(memory) / len(memory) if memory else None,
        },
    }
