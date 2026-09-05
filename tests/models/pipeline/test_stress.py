"""Stress test Phase 7 — kiểm tra cách dựng haystack và wiring của runner.

Chất lượng model không thuộc phạm vi test này (model giả): điều cần chốt là
haystack đúng N, lồng nhau giữa các N, luôn chứa gold, và `same_domain` báo
trung thực mức độ bị lấp bằng distractor ngẫu nhiên.
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crossencoder"))
from conftest import FakeTokenizer  # noqa: E402

from src.models.biencoder.retrieve import RetrievalThresholds, ToolRetriever  # noqa: E402
from src.models.crossencoder.inference import (  # noqa: E402
    CrossEncoderExtractor,
    ExtractionConfig,
)
from src.models.pipeline.method2 import Method2Pipeline  # noqa: E402
from src.models.pipeline.stress import (  # noqa: E402
    DISTRACTOR_RANDOM,
    DISTRACTOR_SAME_DOMAIN,
    StressConfig,
    build_haystack,
    distractor_order,
    reference_group,
    run_stress,
    sample_queries,
)
from models.pipeline.test_method2_end_to_end import FakeCrossEncoder  # noqa: E402
from src.models.pipeline.validator import ArgumentValidator  # noqa: E402

GROUPS = ["Ẩm thực", "Y tế", "Khác"]


def _tool(index: int, group: str) -> dict:
    return {
        "name": f"tool_{index:02d}",
        "description": f"Công cụ số {index}",
        "feature_group": group,
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "Địa điểm"}},
            "required": ["location"],
        },
    }


@pytest.fixture
def pool() -> dict[str, dict]:
    # 4 tool mỗi nhóm có nhãn thật, phần còn lại là "Khác" — đúng hình dạng pool
    # thật, nơi 4.424/4.464 tool chưa được phân loại.
    tools = [_tool(i, GROUPS[0]) for i in range(4)]
    tools += [_tool(i + 4, GROUPS[1]) for i in range(4)]
    tools += [_tool(i + 8, GROUPS[2]) for i in range(12)]
    return {t["name"]: t for t in tools}


@pytest.fixture
def groups(pool) -> dict[str, str]:
    return {name: tool["feature_group"] for name, tool in pool.items()}


def _sample(sample_id: str, gold: str | None, pool: dict) -> dict:
    calls = [{"name": gold, "arguments": {"location": "Hà Nội"}}] if gold else []
    tools = [pool[gold]] if gold else [pool["tool_00"]]
    return {
        "id": sample_id,
        "source": "custom_vi",
        "query": "Tìm quán phở bò ở Hà Nội",
        "function_calls": calls,
        "tools": tools,
        "metadata": {},
    }


# ------------------------------------------------------------- dựng haystack


def test_haystack_dung_kich_thuoc_va_luon_chua_gold(pool, groups):
    sample = _sample("s1", "tool_00", pool)
    order = distractor_order(sample, ["tool_00"], sorted(pool), groups, DISTRACTOR_RANDOM, 42)

    for n in (2, 5, 12):
        haystack = build_haystack(sample, n, order, pool, groups, DISTRACTOR_RANDOM)
        names = [t["name"] for t in haystack.tools]
        assert len(names) == n
        assert len(set(names)) == n
        assert "tool_00" in names


def test_haystack_long_nhau_giua_cac_n(pool, groups):
    sample = _sample("s1", "tool_00", pool)
    order = distractor_order(sample, ["tool_00"], sorted(pool), groups, DISTRACTOR_RANDOM, 42)

    sets = [
        {t["name"] for t in build_haystack(sample, n, order, pool, groups, DISTRACTOR_RANDOM).tools}
        for n in (2, 5, 12)
    ]
    assert sets[0] < sets[1] < sets[2]


def test_gold_khong_bao_gio_bi_distractor_lan_at(pool, groups):
    sample = _sample("s1", "tool_00", pool)
    order = distractor_order(sample, ["tool_00"], sorted(pool), groups, DISTRACTOR_RANDOM, 42)

    assert "tool_00" not in order


def test_same_domain_uu_tien_cung_nhom_roi_lap_ngau_nhien(pool, groups):
    sample = _sample("s1", "tool_00", pool)
    order = distractor_order(
        sample, ["tool_00"], sorted(pool), groups, DISTRACTOR_SAME_DOMAIN, 42
    )

    # Nhóm có 4 tool, 1 là gold → chỉ còn 3 distractor cùng nhóm.
    assert {groups[n] for n in order[:3]} == {GROUPS[0]}
    pure = build_haystack(sample, 4, order, pool, groups, DISTRACTOR_SAME_DOMAIN)
    diluted = build_haystack(sample, 10, order, pool, groups, DISTRACTOR_SAME_DOMAIN)
    assert pure.purity == 1.0
    assert diluted.purity == pytest.approx(3 / 9)


def test_random_khong_bao_purity(pool, groups):
    sample = _sample("s1", "tool_00", pool)
    order = distractor_order(sample, ["tool_00"], sorted(pool), groups, DISTRACTOR_RANDOM, 42)

    assert build_haystack(sample, 10, order, pool, groups, DISTRACTOR_RANDOM).purity is None


def test_negative_lay_nhom_tham_chieu_tu_candidate_pool(pool, groups):
    negative = _sample("neg", None, pool)

    assert reference_group(negative, groups) == GROUPS[0]


def test_nhom_khac_khong_duoc_coi_la_mot_nhom(pool, groups):
    sample = _sample("s1", "tool_08", pool)  # tool_08 thuộc "Khác"

    assert reference_group(sample, groups) is None


def test_gold_nhieu_hon_n_thi_bi_cat_va_duoc_ghi_nhan(pool, groups):
    sample = _sample("s1", "tool_00", pool)
    sample["function_calls"].append({"name": "tool_01", "arguments": {"location": "Huế"}})
    sample["tools"].append(pool["tool_01"])
    order = distractor_order(
        sample, ["tool_00", "tool_01"], sorted(pool), groups, DISTRACTOR_RANDOM, 42
    )

    haystack = build_haystack(sample, 1, order, pool, groups, DISTRACTOR_RANDOM)

    assert haystack.truncated is True
    assert [t["name"] for t in haystack.tools] == ["tool_00"]


# --------------------------------------------------------------- lấy mẫu query


def test_sample_queries_can_bang_positive_va_negative(pool):
    samples = [_sample(f"p{i}", "tool_00", pool) for i in range(50)]
    samples += [_sample(f"n{i}", None, pool) for i in range(50)]

    picked = sample_queries(samples, 20, seed=42)

    assert len(picked) == 20
    assert sum(1 for s in picked if s["function_calls"]) == 10
    assert [s["id"] for s in picked] == [s["id"] for s in sample_queries(samples, 20, 42)]


# ------------------------------------------------------------------- runner


@pytest.fixture
def pipeline(pool):
    import numpy as np

    names = sorted(pool)
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(names), 8)).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_vector = embeddings[names.index("tool_00")]

    class FixedEncoder:
        def encode(self, texts, **kwargs):
            return np.array([query_vector for _ in texts], dtype="float32")

    retriever = ToolRetriever(
        FixedEncoder(),
        names,
        embeddings,
        RetrievalThresholds(tau=0.1, tau_call=0.9, k_max=3, strategy="absolute"),
        normalize_query=False,
    )
    extractor = CrossEncoderExtractor(
        FakeCrossEncoder(), FakeTokenizer(), ExtractionConfig(max_length=64, batch_size=8)
    )
    return Method2Pipeline(
        retriever, extractor, ArgumentValidator(), pool, retrieval_scope="candidates"
    )


def test_run_stress_ghi_mot_dong_va_mot_file_cho_moi_n(pipeline, pool, tmp_path):
    samples = [_sample(f"p{i}", "tool_00", pool) for i in range(4)]
    samples += [_sample(f"n{i}", None, pool) for i in range(4)]
    config = StressConfig(
        n_values=(2, 5, 12), distractor_modes=(DISTRACTOR_RANDOM,), output_dir=tmp_path
    )

    report = run_stress(pipeline, config, samples=samples, pool=pool)

    assert [row["n"] for row in report["rows"]] == [2, 5, 12]
    for n in (2, 5, 12):
        rows = (tmp_path / f"predictions_{DISTRACTOR_RANDOM}_n{n}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(rows) == len(samples)
        assert json.loads(rows[0])["id"] == "p0"
    assert (tmp_path / "stress_report.json").exists()
    assert "Tool Set Acc" in (tmp_path / "stress_summary.md").read_text(encoding="utf-8")


def test_moi_dong_co_du_metric_va_latency_bon_giai_doan(pipeline, pool, tmp_path):
    samples = [_sample("p0", "tool_00", pool), _sample("n0", None, pool)]
    config = StressConfig(n_values=(2,), output_dir=tmp_path)

    row = run_stress(pipeline, config, samples=samples, pool=pool)["rows"][0]

    assert set(row) >= {
        "tool_set_accuracy", "recall_at_1", "mrr", "negative_recall",
        "arg_em_given_correct_tool", "latency",
    }
    assert set(row["latency"]) >= {
        "t_query_embed", "t_retrieve", "t_cross_encode", "t_validate", "total"
    }


def test_runner_ep_scope_candidates_roi_tra_lai_nguyen_trang(pipeline, pool, tmp_path):
    pipeline.retrieval_scope = "pool"
    samples = [_sample("p0", "tool_00", pool)]

    run_stress(pipeline, StressConfig(n_values=(2,), output_dir=tmp_path), samples, pool)

    assert pipeline.retrieval_scope == "pool"


def test_config_doc_dung_stress_yaml_that():
    config = StressConfig.from_yaml("configs/method2/stress.yaml")

    assert config.n_values == (3, 10, 50, 100, 500, 1000)
    assert config.n_queries == 200
    # `same_domain` đang tắt vì 4.424 tool chưa có nhãn feature_group.
    assert config.distractor_modes == (DISTRACTOR_RANDOM,)


def test_bao_cao_ve_duoc_bang_plot_stress(pipeline, pool, tmp_path):
    """Chống lệch schema: script vẽ đọc đúng file mà `run_stress` vừa sinh ra.

    Hai file ở hai thư mục khác nhau (`src/` và `scripts/`) nên không có gì bắt
    chúng đi cùng nhau; đổi tên một key trong `rows` là hình lặng lẽ trống.
    """
    import importlib.util

    import matplotlib

    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location(
        "plot_stress", "scripts/method2/plot_stress.py"
    )
    plot_stress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_stress)

    samples = [_sample("p0", "tool_00", pool), _sample("n0", None, pool)]
    report = run_stress(
        pipeline, StressConfig(n_values=(2, 5), output_dir=tmp_path), samples, pool
    )

    for figure in plot_stress.plot_report(report, tmp_path):
        assert figure.stat().st_size > 0
    # Mỗi series phải có số thật ở mọi N — None hết nghĩa là key đã đổi tên.
    rows = plot_stress._rows_for_mode(report, DISTRACTOR_RANDOM)
    for key, _ in plot_stress.ACCURACY_SERIES:
        assert any(row.get(key) is not None for row in rows), key
    for stage, _ in plot_stress.LATENCY_STAGES:
        assert all(row["latency"][stage]["p50_ms"] is not None for row in rows), stage


def test_sample_queries_nhan_duoc_generator(pool):
    """`sources.load_jsonl` là generator; hàm duyệt 3 lượt nên phải tự vật chất hoá.

    Trước khi sửa: lượt hai trở đi rỗng → stress test chạy trên 0 query, không
    ném lỗi, chỉ ra một bảng toàn `None`.
    """
    samples = [_sample(f"p{i}", "tool_00", pool) for i in range(10)]
    samples += [_sample(f"n{i}", None, pool) for i in range(10)]

    picked = sample_queries(iter(samples), 8, seed=42)

    assert len(picked) == 8
    assert sum(1 for s in picked if s["function_calls"]) == 4
