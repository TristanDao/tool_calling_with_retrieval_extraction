"""Order-invariant matching for single-call and multi-call predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from functools import cache

from src.evaluation.normalization import canonical_json, flatten_arguments
from src.evaluation.types import CallAlignment, FunctionCall


def tool_name_counter(calls: tuple[FunctionCall, ...]) -> Counter[str]:
    return Counter(call.name for call in calls)


def tool_names_exact(gold: tuple[FunctionCall, ...], prediction: tuple[FunctionCall, ...]) -> bool:
    return tool_name_counter(gold) == tool_name_counter(prediction)


def call_signature(call: FunctionCall) -> str:
    return canonical_json({"name": call.name, "arguments": call.arguments})


def calls_exact(gold: tuple[FunctionCall, ...], prediction: tuple[FunctionCall, ...]) -> bool:
    return Counter(call_signature(call) for call in gold) == Counter(
        call_signature(call) for call in prediction
    )


def _pair_score(gold: FunctionCall, prediction: FunctionCall) -> int:
    gold_flat = flatten_arguments(gold.arguments)
    pred_flat = flatten_arguments(prediction.arguments)
    exact = int(gold.arguments == prediction.arguments)
    pair_overlap = sum(
        canonical_json(value) == canonical_json(pred_flat.get(key))
        for key, value in gold_flat.items()
        if key in pred_flat
    )
    key_overlap = len(set(gold_flat) & set(pred_flat))
    return exact * 1_000_000 + pair_overlap * 1_000 + key_overlap


def _best_assignment(
    left: list[FunctionCall],
    right: list[FunctionCall],
    score: Callable[[FunctionCall, FunctionCall], int],
) -> list[tuple[int, int]]:
    if not left or not right:
        return []
    if max(len(left), len(right)) > 12:
        available = set(range(len(right)))
        pairs: list[tuple[int, int]] = []
        for left_index, left_call in enumerate(left):
            if not available:
                break
            right_index = max(available, key=lambda idx: score(left_call, right[idx]))
            available.remove(right_index)
            pairs.append((left_index, right_index))
        return pairs
    if len(left) > len(right):
        swapped = _best_assignment(right, left, lambda a, b: score(b, a))
        return [(right_index, left_index) for left_index, right_index in swapped]

    @cache
    def solve(left_index: int, used_mask: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if left_index == len(left):
            return 0, ()
        best_score = -1
        best_pairs: tuple[tuple[int, int], ...] = ()
        for right_index in range(len(right)):
            if used_mask & (1 << right_index):
                continue
            tail_score, tail_pairs = solve(left_index + 1, used_mask | (1 << right_index))
            current_score = score(left[left_index], right[right_index]) + tail_score
            if current_score > best_score:
                best_score = current_score
                best_pairs = ((left_index, right_index),) + tail_pairs
        return best_score, best_pairs

    return list(solve(0, 0)[1])


def align_calls(
    gold: tuple[FunctionCall, ...],
    prediction: tuple[FunctionCall, ...],
) -> list[CallAlignment]:
    gold_groups: dict[str, list[FunctionCall]] = defaultdict(list)
    pred_groups: dict[str, list[FunctionCall]] = defaultdict(list)
    for call in gold:
        gold_groups[call.name].append(call)
    for call in prediction:
        pred_groups[call.name].append(call)
    alignments: list[CallAlignment] = []
    for name in sorted(set(gold_groups) | set(pred_groups)):
        gold_calls = gold_groups[name]
        pred_calls = pred_groups[name]
        pairs = _best_assignment(gold_calls, pred_calls, _pair_score)
        used_gold = {gold_index for gold_index, _ in pairs}
        used_pred = {pred_index for _, pred_index in pairs}
        alignments.extend(
            CallAlignment(gold=gold_calls[gold_index], prediction=pred_calls[pred_index])
            for gold_index, pred_index in pairs
        )
        alignments.extend(
            CallAlignment(gold=call, prediction=None)
            for index, call in enumerate(gold_calls)
            if index not in used_gold
        )
        alignments.extend(
            CallAlignment(gold=None, prediction=call)
            for index, call in enumerate(pred_calls)
            if index not in used_pred
        )
    return alignments
