"""Run manifest cho stage `evaluation` (§11 method2_plan).

Trước đây `_audit_complete` chỉ có bộ check của run TRAIN. Gọi nguyên vẹn cho
Phase 5 sẽ báo thiếu `checkpoint`/`peak_vram`/`duration` — những thứ một run
đánh giá vốn không có — khiến `missing` mất hết ý nghĩa cảnh báo.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.run_manifest import STAGE_EVALUATION, build_run_manifest

TRAIN_ONLY = {"checkpoint", "checkpoint_selection", "duration", "retrieval_metrics"}
EVAL_ONLY = {
    "predictions",
    "raw_predictions",
    "errors_classified",
    "latency_stages",
    "seed",
    "metrics_normalized",
    "metrics_strict",
}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _eval_run(tmp_path: Path, *, with_strict: bool = True) -> tuple[Path, Path]:
    run_dir = tmp_path / "predictions" / "custom_seen"
    run_dir.mkdir(parents=True)
    for name in ("predictions.jsonl", "oracle_predictions.jsonl",
                 "raw_predictions_pipeline.jsonl", "errors_pipeline.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    _write(run_dir / "latency_pipeline.json",
           {"total": {"p50_ms": 58.68}, "gpu_peak_memory_mb": 3280.1})

    metrics = {"config": {"seed": 42},
               "metrics": {"extraction": {"normalized_arg_em_given_correct_tool": 0.62}}}
    if with_strict:
        metrics["metrics"]["strict_extraction"] = {
            "normalized_arg_em_given_correct_tool": 0.41
        }
    report = tmp_path / "report.json"
    _write(report, metrics)
    return run_dir, report


def test_stage_evaluation_khong_doi_artifact_cua_train(tmp_path: Path) -> None:
    run_dir, report = _eval_run(tmp_path)
    manifest = build_run_manifest(
        run_dir, Path("configs/method2/pipeline.yaml"),
        STAGE_EVALUATION, {"metrics": report},
    )
    checks = manifest["audit_complete"]["checks"]
    assert TRAIN_ONLY.isdisjoint(checks), "run đánh giá không được đòi artifact train"
    assert EVAL_ONLY <= set(checks)
    assert not (EVAL_ONLY & set(manifest["audit_complete"]["missing"]))


def test_ghi_nhan_seed_latency_vram(tmp_path: Path) -> None:
    run_dir, report = _eval_run(tmp_path)
    manifest = build_run_manifest(
        run_dir, Path("configs/method2/pipeline.yaml"),
        STAGE_EVALUATION, {"metrics": report},
    )
    evaluation = manifest["evaluation"]
    assert evaluation["seed"] == 42          # §11: seed phải ghi lại được
    assert evaluation["latency_p50_ms"] == 58.68
    assert evaluation["peak_vram_mb"] == 3280.1
    assert evaluation["missing_artifacts"] == []


def test_thieu_ban_strict_thi_bao_thieu(tmp_path: Path) -> None:
    """§11 đòi CẢ strict lẫn normalized — thiếu một bản là audit phải kêu."""
    run_dir, report = _eval_run(tmp_path, with_strict=False)
    manifest = build_run_manifest(
        run_dir, Path("configs/method2/pipeline.yaml"),
        STAGE_EVALUATION, {"metrics": report},
    )
    assert "metrics_strict" in manifest["audit_complete"]["missing"]
    assert "metrics_normalized" not in manifest["audit_complete"]["missing"]


def test_stage_train_van_giu_bo_check_cu(tmp_path: Path) -> None:
    manifest = build_run_manifest(
        tmp_path, Path("configs/method2/biencoder.yaml"), "biencoder", {}
    )
    checks = manifest["audit_complete"]["checks"]
    assert TRAIN_ONLY <= set(checks)
    assert EVAL_ONLY.isdisjoint(checks)
