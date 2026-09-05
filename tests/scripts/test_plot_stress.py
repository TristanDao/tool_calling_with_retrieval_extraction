"""Biểu đồ Phase 7 — kiểm cái mà một hình vẽ có thể sai âm thầm.

Không so pixel: điều cần chắc là script chạy được trên đúng `stress_report.json`
mà `run_stress` sinh ra, ra đủ 2 hình mỗi mode, và không chết khi một metric
toàn `None` (negative recall vắng mặt nếu tập query không có sample no-call).
"""

import importlib.util
import json

import matplotlib

matplotlib.use("Agg")


def _load():
    spec = importlib.util.spec_from_file_location(
        "plot_stress", "scripts/method2/plot_stress.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(mode: str, n: int, **overrides):
    row = {
        "mode": mode,
        "n": n,
        "tool_set_accuracy": 0.9,
        "recall_at_1": 0.95,
        "arg_em_given_correct_tool": 0.6,
        "negative_recall": 0.74,
        "latency": {
            stage: {"p50_ms": value, "p95_ms": value * 1.3}
            for stage, value in (
                ("t_query_embed", 6.0), ("t_retrieve", 0.2), ("t_cross_encode", 40.0),
                ("t_validate", 0.4), ("total", 46.6),
            )
        },
    }
    row.update(overrides)
    return row


def test_moi_mode_ra_hai_hinh(tmp_path):
    module = _load()
    report = {"rows": [_row(m, n) for m in ("random", "same_domain") for n in (3, 10, 50)]}

    written = module.plot_report(report, tmp_path)

    assert sorted(p.name for p in written) == [
        "stress_accuracy_random.png", "stress_accuracy_same_domain.png",
        "stress_latency_random.png", "stress_latency_same_domain.png",
    ]
    assert all(p.stat().st_size > 0 for p in written)


def test_metric_toan_none_khong_lam_chet_hinh(tmp_path):
    module = _load()
    report = {"rows": [_row("random", n, negative_recall=None) for n in (3, 10, 50)]}

    written = module.plot_report(report, tmp_path)

    assert len(written) == 2


def test_cli_ghi_hinh_canh_file_bao_cao(tmp_path, monkeypatch):
    """`--output-dir` mặc định = thư mục chứa report, đúng chỗ Kaggle đọc lại."""
    module = _load()
    report = {"config": {"pool_size": 20, "n_queries": 2, "seed": 42},
              "rows": [_row("random", n) for n in (2, 5)]}
    path = tmp_path / "stress_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["plot_stress.py", "--report", str(path)])
    module.main()

    assert (tmp_path / "stress_accuracy_random.png").exists()
