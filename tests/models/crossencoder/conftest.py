"""Tokenizer giả cho unit test — không cần tải model từ HuggingFace.

Tách token theo khoảng trắng, giữ offset ký tự thật, mô phỏng đủ API mà
collator và label generator dùng: `sequence_ids`, `offset_mapping`,
`pad_token_id`, `sep_token_id`.
"""

import re

import pytest

CLS_ID = 0
PAD_ID = 1
SEP_ID = 2
_TOKEN_RE = re.compile(r"\S+")


class FakeEncoding(dict):
    def __init__(self, data: dict, seq_ids: list[int | None]) -> None:
        super().__init__(data)
        self._seq_ids = seq_ids

    def sequence_ids(self, index: int = 0) -> list[int | None]:
        return self._seq_ids


class FakeTokenizer:
    """Word-level tokenizer đủ dùng cho test."""

    pad_token_id = PAD_ID
    sep_token_id = SEP_ID
    cls_token_id = CLS_ID

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            ids.append(10 + (hash(token) % 1000))
            offsets.append((match.start(), match.end()))
        return ids, offsets

    def __call__(
        self,
        text,
        text_pair=None,
        max_length: int = 256,
        padding=False,
        truncation=False,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        return_tensors=None,
        **kwargs,
    ) -> FakeEncoding:
        if isinstance(text, list):
            raise NotImplementedError("FakeTokenizer chỉ hỗ trợ 1 cặp mỗi lần gọi")

        query_ids, query_offsets = self._tokenize(text)
        if not add_special_tokens and text_pair is None:
            limit = max_length if truncation else len(query_ids)
            data = {
                "input_ids": query_ids[:limit],
                "attention_mask": [1] * len(query_ids[:limit]),
            }
            if return_offsets_mapping:
                data["offset_mapping"] = query_offsets[:limit]
            return FakeEncoding(data, [0] * len(data["input_ids"]))

        pair_ids, pair_offsets = self._tokenize(text_pair or "")
        if truncation:
            budget = max_length - 3
            if len(query_ids) + len(pair_ids) > budget:
                query_ids = query_ids[: max(budget - len(pair_ids), 1)]
                query_offsets = query_offsets[: len(query_ids)]

        input_ids = [CLS_ID] + query_ids + [SEP_ID] + pair_ids + [SEP_ID]
        seq_ids = (
            [None] + [0] * len(query_ids) + [None] + [1] * len(pair_ids) + [None]
        )
        data = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if return_offsets_mapping:
            data["offset_mapping"] = (
                [(0, 0)] + query_offsets + [(0, 0)] + pair_offsets + [(0, 0)]
            )
        return FakeEncoding(data, seq_ids)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
