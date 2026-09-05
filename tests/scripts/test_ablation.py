"""Phase 6 — sinh config nhánh ablation và luật "khác đúng một knob".

Ablation hỏng theo một kiểu duy nhất mà không ai nhận ra: nhánh khác baseline ở
hai chỗ thay vì một. Test quan trọng nhất ở đây là
`test_bat_duoc_nhanh_khac_control_o_knob_khong_khai` — nó chứng minh cái chốt
chặn thật sự chặn, chứ không phải một hàm luôn trả về "ổn".
"""

import importlib.util
import json
import sys

import pytest
import yaml


def _load():
    spec = importlib.util.spec_from_file_location(
        "ablation", "scripts/method2/ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Phải đăng ký vào sys.modules TRƯỚC khi exec: `from __future__ import
    # annotations` khiến dataclass phân giải type hint qua `sys.modules[__module__]`
    # lúc tạo lớp, mà module nạp bằng spec thì chưa có ở đó.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ resolve


def test_giao_thuc_rut_gon_ap_cho_ca_control(tmp_path):
    module = _load()

    module.write_configs(tmp_path)

    control = yaml.safe_load((tmp_path / "control.biencoder.yaml").read_text(encoding="utf-8"))
    assert control["train"]["epochs"] == 1
    assert control["mining"]["enabled"] is False
    # Control KHÔNG được mượn run02: nó phải ghi vào thư mục ablation của nó.
    assert control["train"]["output_dir"] == "artifacts/method2/ablation/control"


def test_moi_nhanh_ghi_vao_thu_muc_rieng(tmp_path):
    module = _load()

    module.write_configs(tmp_path)

    dirs = {
        name: yaml.safe_load(
            (tmp_path / f"{name}.biencoder.yaml").read_text(encoding="utf-8")
        )["train"]["output_dir"]
        for name in ("control", "nneg0", "nneg8", "doc_name_desc")
    }
    assert len(set(dirs.values())) == len(dirs)


def test_nhanh_doi_doc_text_phai_doi_ca_pool_lan_index(tmp_path):
    """Không dựng lại index thì train doc_text mới mà đánh giá bằng embedding cũ."""
    module = _load()

    module.write_configs(tmp_path)

    arm = yaml.safe_load(
        (tmp_path / "doc_name_desc.biencoder.yaml").read_text(encoding="utf-8")
    )
    pool = yaml.safe_load(
        (tmp_path / "doc_name_desc.tool_pool.yaml").read_text(encoding="utf-8")
    )
    assert pool["include_param_names_in_doc"] is False
    assert "ablation/doc_name_desc" in arm["pairs"]["tool_pool_path"]
    assert "ablation/doc_name_desc" in arm["index"]["embeddings_path"]
    assert "ablation/doc_name_desc" in arm["index"]["tool_ids_path"]


def test_nhanh_khong_doi_doc_text_thi_khong_sinh_config_pool(tmp_path):
    module = _load()

    module.write_configs(tmp_path)

    assert not (tmp_path / "nneg0.tool_pool.yaml").exists()
    assert (tmp_path / "doc_name_desc.tool_pool.yaml").exists()


# ------------------------------------------------------------------- verify


def test_diff_bo_qua_knob_duong_dan():
    module = _load()
    control = {"biencoder": {"train": {"output_dir": "a", "epochs": 1}}}
    arm = {"biencoder": {"train": {"output_dir": "b", "epochs": 1}}}

    assert module.diff_arm(control, arm) == set()


def test_diff_bat_knob_that_su_doi():
    module = _load()
    control = {"biencoder": {"pairs": {"n_hard_negatives": 4}}}
    arm = {"biencoder": {"pairs": {"n_hard_negatives": 8}}}

    assert module.diff_arm(control, arm) == {"pairs.n_hard_negatives"}


def test_bat_duoc_nhanh_khac_control_o_knob_khong_khai(monkeypatch, tmp_path):
    """Chốt chặn phải thật sự chặn — dựng đúng lỗi mà nó sinh ra để phòng."""
    module = _load()
    sloppy = module.Arm(
        "sloppy",
        {module.BIENCODER: {"pairs.n_hard_negatives": 8}},
        "khai 1 knob nhưng config lệch 2",
    )
    monkeypatch.setattr(module, "ARMS", (module.ARMS[0], sloppy))
    original = module.resolve_arm

    def leaky(arm, base):
        resolved = original(arm, base)
        if arm.name == "sloppy":
            # Lần sửa tay trước còn sót lại — đúng kiểu tai nạn thật.
            module.set_path(resolved[module.BIENCODER], "train.lr", 9.9e-5)
        return resolved

    monkeypatch.setattr(module, "resolve_arm", leaky)

    with pytest.raises(SystemExit, match="train.lr"):
        module.write_configs(tmp_path)


def test_nhanh_dung_khai_bao_thi_qua(tmp_path):
    module = _load()

    resolved = module.write_configs(tmp_path)

    assert module.verify_arms(resolved) == []


# ------------------------------------------------------------------- report


def _metrics(ndcg: float) -> dict:
    return {
        "scope": "pool",
        "overall": {
            "ndcg@10": ndcg, "mrr": 0.9,
            "full_recall@1": 0.8, "full_recall@5": 0.95,
        },
        "by_tool_split": {
            "seen": {"full_recall@1": 0.87},
            "unseen": {"full_recall@1": 0.71},
        },
    }


def test_report_gom_du_nhanh_va_danh_dau_nhanh_chua_chay(tmp_path):
    module = _load()
    for name, ndcg in (("control", 0.7146), ("nneg0", 0.6802)):
        path = tmp_path / name
        path.mkdir()
        (path / "metrics.json").write_text(
            json.dumps(_metrics(ndcg)), encoding="utf-8"
        )

    report = module.collect_report(tmp_path)
    markdown = module.report_markdown(report)

    by_arm = {row["arm"]: row for row in report["rows"]}
    assert by_arm["control"]["ndcg@10"] == 0.7146
    assert by_arm["nneg0"]["unseen_full_recall@1"] == 0.71
    assert "missing" in by_arm["nneg8"]
    assert "*chưa chạy*" in markdown
    # Cảnh báo không so với Phase 5 phải nằm ngay trong bảng, không chỉ ở docs.
    assert "Phase 5" in markdown


def test_report_ghi_lai_giao_thuc_da_dung(tmp_path):
    """Người đọc bảng phải thấy ngay là số này đo ở 1 epoch."""
    module = _load()

    report = module.collect_report(tmp_path)

    assert report["protocol"][module.FAMILY_BI][module.BIENCODER]["train.epochs"] == 1


def test_control_dung_lai_dung_bo_pair_cua_baseline(tmp_path):
    """Sinh lại pair cho control là mở cửa cho "giống nhưng không y hệt"."""
    module = _load()

    module.write_configs(tmp_path)

    control = yaml.safe_load((tmp_path / "control.biencoder.yaml").read_text(encoding="utf-8"))
    assert control["train"]["train_path"] == "data/method2/biencoder/train.jsonl"
    assert control["pairs"]["output_dir"] == "data/method2/biencoder"


def test_nhanh_doi_pairs_thi_ghi_ra_thu_muc_rieng(tmp_path):
    """Không được ghi đè `data/method2/biencoder/` — preflight khoá SHA file đó."""
    module = _load()

    module.write_configs(tmp_path)

    for name in ("nneg0", "nneg8", "doc_name_desc"):
        config = yaml.safe_load((tmp_path / f"{name}.biencoder.yaml").read_text(encoding="utf-8"))
        assert config["train"]["train_path"] == f"data/method2/ablation/{name}/train.jsonl"
        assert config["pairs"]["output_dir"] == f"data/method2/ablation/{name}"


def test_nhanh_chi_doi_knob_training_khong_can_sinh_lai_pair():
    module = _load()
    training_only = module.Arm("lr", {module.BIENCODER: {"train.lr": 1e-5}}, "")

    assert training_only.rebuilds_pairs is False
    assert module.ARMS[1].rebuilds_pairs is True   # nneg0 đổi pairs.*


# ------------------------------------------------------- họ Cross-Encoder


def test_nhanh_ce_chi_sinh_config_crossencoder(tmp_path):
    """Kèm biencoder.yaml cho một nhánh CE là mời người chạy dùng nhầm file."""
    module = _load()

    module.write_configs(tmp_path)

    assert (tmp_path / "should_call.crossencoder.yaml").exists()
    assert not (tmp_path / "should_call.biencoder.yaml").exists()
    assert not (tmp_path / "control.crossencoder.yaml").exists()


def test_nhanh_ce_doi_ngoai_control_cua_ho_no(tmp_path):
    """`ce_control` giữ nguyên curriculum; so nó với `control` (1 epoch) là so nhầm."""
    module = _load()

    resolved = module.write_configs(tmp_path)

    assert module.verify_arms(resolved) == []
    assert resolved["ce_control"][module.CROSSENCODER]["train"]["epochs"] == 3
    assert resolved["control"][module.BIENCODER]["train"]["epochs"] == 1


def test_should_call_ghi_pair_ra_thu_muc_rieng(tmp_path):
    """`data/method2/crossencoder/train.jsonl` bị preflight khoá SHA."""
    module = _load()

    module.write_configs(tmp_path)

    config = yaml.safe_load(
        (tmp_path / "should_call.crossencoder.yaml").read_text(encoding="utf-8")
    )
    assert config["pairs"]["should_call"]["enabled"] is True
    assert config["model"]["enable_should_call"] is True
    assert config["data"]["train_path"] == "data/method2/ablation/should_call/train.jsonl"
    assert config["pairs"]["output_dir"] == "data/method2/ablation/should_call"


def test_doi_backbone_khong_can_sinh_lai_pair():
    """Nhãn ghi ra đĩa là char span, tokenizer-free (§1.4) — đúng để làm việc này."""
    module = _load()
    arms = {arm.name: arm for arm in module.ARMS}

    assert arms["backbone_e5"].rebuilds_ce_pairs is False
    assert arms["backbone_bge"].rebuilds_ce_pairs is False
    assert arms["should_call"].rebuilds_ce_pairs is True


def test_nhanh_doi_nhieu_knob_van_qua_neu_khai_du_va_co_caveat(tmp_path):
    module = _load()

    resolved = module.write_configs(tmp_path)

    assert module.verify_arms(resolved) == []
    bge = {arm.name: arm for arm in module.ARMS}["backbone_bge"]
    # Ba knob, khai đủ ba — chốt chặn không cản, nhưng caveat bắt phải công bố.
    assert len(bge.overrides[module.CROSSENCODER]) == 3
    assert "effective batch" in bge.caveat


def test_caveat_hien_trong_bao_cao(tmp_path):
    module = _load()

    markdown = module.report_markdown(module.collect_report(tmp_path))

    assert "Nhánh không sạch một biến" in markdown
    assert "backbone_bge" in markdown


def test_bao_cao_ce_doc_lat_custom_vi_chu_khong_phai_overall(tmp_path):
    """Gate §Phase 3 đo trên custom_vi; bảng ablation phải đọc cùng lát đó."""
    module = _load()
    path = tmp_path / "backbone_e5"
    path.mkdir()
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "overall": {"has_value_f1": 0.11, "span_em": 0.11,
                            "enum_accuracy": 0.11, "boolean_accuracy": 0.11},
                "by_source": {
                    "custom_vi": {"has_value_f1": 0.93, "span_em": 0.71,
                                  "enum_accuracy": 0.88, "boolean_accuracy": 0.95}
                },
            }
        ),
        encoding="utf-8",
    )

    row = {r["arm"]: r for r in module.collect_report(tmp_path)["rows"]}["backbone_e5"]

    assert row["measured_on"] == "custom_vi"
    assert row["has_value_f1"] == 0.93


def test_bao_cao_ce_lui_ve_overall_khi_thieu_lat_custom_vi(tmp_path):
    module = _load()
    path = tmp_path / "backbone_e5"
    path.mkdir()
    (path / "metrics.json").write_text(
        json.dumps({"overall": {"has_value_f1": 0.5, "span_em": 0.4,
                                "enum_accuracy": 0.3, "boolean_accuracy": 0.2}}),
        encoding="utf-8",
    )

    row = {r["arm"]: r for r in module.collect_report(tmp_path)["rows"]}["backbone_e5"]

    assert row["measured_on"] == "overall"
    assert row["has_value_f1"] == 0.5
