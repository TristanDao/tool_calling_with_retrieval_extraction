"""Hiệu chỉnh ngưỡng `should_call` — cùng hàm mục tiêu với τ.

Ablation §6.1 so hai cơ chế abstention; chọn ngưỡng theo hai tiêu chí khác nhau
là so hai thứ khác nhau. Test này ghim mục tiêu đó lại (Macro-F1) và ghim luôn
quy tắc "thiếu dữ liệu thì dừng, không đoán".
"""

import importlib.util
import json

import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        "calibrate_should_call", "scripts/method2/calibrate_should_call.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )


def _corpus(tmp_path, probs_by_id, negatives):
    gold = [
        {"id": key, "query": "q", "function_calls": [] if key in negatives else [{"name": "t"}]}
        for key in probs_by_id
    ]
    predictions = [
        {"id": key, "function_calls": [], "metadata": {"should_call_prob": prob}}
        for key, prob in probs_by_id.items()
    ]
    gold_path, pred_path = tmp_path / "gold.jsonl", tmp_path / "pred.jsonl"
    _write(gold_path, gold)
    _write(pred_path, predictions)
    return gold_path, pred_path


def test_chon_nguong_tach_duoc_hai_lop(tmp_path):
    module = _load()
    # positive dồn ở 0.8-0.9, negative ở 0.1-0.2 → ngưỡng tối ưu nằm giữa.
    probs = {"p1": 0.90, "p2": 0.85, "p3": 0.80, "n1": 0.20, "n2": 0.15, "n3": 0.10}
    gold_path, pred_path = _corpus(tmp_path, probs, negatives={"n1", "n2", "n3"})
    pairs, report = module.collect_probs(gold_path, pred_path)

    threshold, detail, curve = module.sweep(pairs, module._grid(0.0, 1.0, 0.05))

    assert report == {"n_scored": 6, "n_missing_prob": 0, "n_negative": 3, "n_positive": 3}
    assert 0.20 < threshold <= 0.80
    assert detail["macro_f1"] == pytest.approx(1.0)
    assert len(curve) == 21


def test_prediction_thieu_prob_duoc_dem_chu_khong_lam_lech_ket_qua(tmp_path):
    module = _load()
    gold_path, pred_path = _corpus(
        tmp_path, {"p1": 0.9, "n1": 0.1}, negatives={"n1"}
    )
    rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines()]
    rows.append({"id": "p2", "function_calls": [], "metadata": {}})
    _write(pred_path, rows)

    _, report = module.collect_probs(gold_path, pred_path)

    assert report["n_missing_prob"] == 1
    assert report["n_scored"] == 2


def test_cli_ghi_file_va_bao_da_freeze(tmp_path, monkeypatch, capsys):
    module = _load()
    probs = {"p1": 0.9, "p2": 0.8, "n1": 0.2, "n2": 0.1}
    gold_path, pred_path = _corpus(tmp_path, probs, negatives={"n1", "n2"})
    out = tmp_path / "should_call_threshold.json"
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate_should_call.py", "--gold", str(gold_path),
         "--predictions", str(pred_path), "--output", str(out)],
    )

    module.main()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["objective"] == "macro_f1"
    assert payload["calibrated_on"] == str(gold_path)
    assert 0.2 < payload["should_call_threshold"] <= 0.8
    assert "FREEZE" in capsys.readouterr().out


def test_khong_co_prob_nao_thi_dung_chu_khong_doan(tmp_path, monkeypatch):
    module = _load()
    gold_path, pred_path = _corpus(tmp_path, {"p1": 0.9}, negatives=set())
    _write(pred_path, [{"id": "p1", "function_calls": [], "metadata": {}}])
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate_should_call.py", "--gold", str(gold_path),
         "--predictions", str(pred_path), "--output", str(tmp_path / "o.json")],
    )

    with pytest.raises(SystemExit, match="should_call_prob"):
        module.main()


def test_val_khong_co_no_call_thi_dung(tmp_path, monkeypatch):
    """Không có negative thì mọi ngưỡng đều "hoàn hảo" — con số vô nghĩa."""
    module = _load()
    gold_path, pred_path = _corpus(tmp_path, {"p1": 0.9, "p2": 0.8}, negatives=set())
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate_should_call.py", "--gold", str(gold_path),
         "--predictions", str(pred_path), "--output", str(tmp_path / "o.json")],
    )

    with pytest.raises(SystemExit, match="no-call"):
        module.main()
