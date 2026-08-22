"""Test cho bug §2.3d — decode span kiểu argmax độc lập."""

import pytest

torch = pytest.importorskip("torch")

from src.models.crossencoder.inference import decode_span

LENGTH = 10


def _logits(peaks: dict[int, float]) -> "torch.Tensor":
    out = torch.zeros(LENGTH)
    for idx, value in peaks.items():
        out[idx] = value
    return out


def _all_valid() -> "torch.Tensor":
    return torch.ones(LENGTH, dtype=torch.bool)


def test_joint_decode_beats_independent_argmax() -> None:
    # argmax(start)=5, argmax(end)=2 → bản cũ swap thành (2, 5), sai cả hai đầu.
    start = _logits({5: 10.0, 1: 9.0})
    end = _logits({2: 10.0, 7: 1.0})

    best_start, best_end, score = decode_span(start, end, _all_valid(), max_answer_len=30)

    assert (best_start, best_end) == (1, 2)  # 9 + 10 = 19, cao nhất trong các cặp hợp lệ
    assert score == pytest.approx(19.0)


def test_decode_never_returns_end_before_start() -> None:
    start = _logits({8: 5.0})
    end = _logits({1: 5.0})

    best_start, best_end, _ = decode_span(start, end, _all_valid(), max_answer_len=30)

    assert best_start <= best_end


def test_max_answer_len_rejects_overlong_span() -> None:
    start = _logits({0: 10.0})
    end = _logits({9: 10.0, 2: 4.0})

    best_start, best_end, _ = decode_span(start, end, _all_valid(), max_answer_len=3)

    assert best_end - best_start < 3
    assert (best_start, best_end) == (0, 2)


def test_decode_stays_inside_query_mask() -> None:
    mask = torch.zeros(LENGTH, dtype=torch.bool)
    mask[1:4] = True
    start = _logits({7: 20.0, 2: 1.0})
    end = _logits({8: 20.0, 3: 1.0})

    best_start, best_end, _ = decode_span(start, end, mask, max_answer_len=30)

    assert mask[best_start] and mask[best_end]
    assert (best_start, best_end) == (2, 3)
