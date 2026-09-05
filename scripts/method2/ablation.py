"""Phase 6 §6.3–6.4 — dựng config cho từng nhánh ablation Bi-Encoder, rồi gộp kết quả.

Hai việc, và việc thứ nhất mới là lý do file này tồn tại.

**Sinh config thay vì sửa tay.** Ablation hỏng theo đúng một kiểu: nhánh khác
baseline ở hai chỗ thay vì một, vì lần sửa YAML trước còn sót lại. Ở đây mỗi
nhánh khai báo đúng những knob nó đổi, config được sinh ra từ base, và
`verify_arms()` **đối chiếu ngược**: config đã resolve của nhánh phải khác
`control` đúng bằng tập knob đã khai, không hơn. Sai lệch là dừng, trước khi
tiêu giờ GPU chứ không phải sau.

**Giao thức rút gọn dùng chung** (§7.16 mục C): 1 epoch, một vòng, không mining
— áp cho MỌI nhánh, `control` bao gồm. `control` phải được chạy lại ở đúng giao
thức này chứ không mượn `run02` (3 epoch + Round 2): so nhánh 1-epoch với run02
là đo số epoch chứ không đo ablation.

    python scripts/method2/ablation.py configs      # sinh + kiểm
    python scripts/method2/ablation.py report       # gộp kết quả

Số Phase 6 đo ở giao thức 1-epoch nên **không so trực tiếp** với bảng Phase 5.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BIENCODER = "biencoder"
TOOL_POOL = "tool_pool"
CROSSENCODER = "crossencoder"

#: Nhánh thuộc họ nào. Hai họ có control riêng và giao thức riêng: Bi-Encoder
#: rút gọn được xuống 1 epoch, còn §6.1 cần head học đến nơi nên giữ nguyên
#: curriculum. Trộn chung một control là so nhầm.
FAMILY_BI = "bi"
FAMILY_CE = "ce"

ABLATION_DIR = Path("configs/method2/ablation")
RESULTS_DIR = Path("results/method2/ablation")

#: Giao thức dùng chung, theo từng họ. Áp cho mọi nhánh của họ đó, control
#: bao gồm — control phải chịu đúng giao thức mà các nhánh chịu.
SHARED_PROTOCOL: dict[str, dict[str, dict[str, Any]]] = {
    FAMILY_BI: {
        BIENCODER: {
            "train.epochs": 1,
            "mining.enabled": False,
            "train.metric_for_best_model": None,
        }
    },
    # §6.1 không rút gọn: head `should_call` phải được học đến nơi thì con số
    # abstention mới nói lên điều gì. §6.5 (backbone) thì rút gọn được, nhưng
    # hai nhánh cùng họ phải cùng giao thức nên giữ nguyên cả hai.
    FAMILY_CE: {},
}

#: Knob thuộc về "chạy ở đâu", không thuộc về "đo cái gì" — bỏ qua khi đối chiếu.
PATH_KNOBS: tuple[str, ...] = (
    "train.output_dir",
    "train.train_path",
    "train.val_path",
    "pairs.output_dir",
    "pairs.stats_path",
    "mining.output_path",
    "thresholds.output_path",
    "evaluate.output_dir",
    "index.embeddings_path",
    "index.tool_ids_path",
    "pairs.tool_pool_path",
    "output_path",
    "stats_path",
    "data.train_path",
    "data.val_path",
)


@dataclass(frozen=True)
class Arm:
    name: str
    #: Knob đổi so với control, theo đường dẫn có chấm trong file config tương ứng.
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    question: str = ""
    family: str = FAMILY_BI
    #: Knob buộc phải đổi kèm nhưng KHÔNG thuộc câu hỏi của nhánh — ví dụ hạ
    #: batch_size cho vừa VRAM. Vẫn phải khai (chốt chặn bắt), và được in kèm
    #: trong báo cáo để người đọc biết phép so này không sạch một biến.
    caveat: str = ""

    @property
    def rebuilds_pool(self) -> bool:
        """Đổi `tool_pool` là đổi `doc_text` → phải dựng lại pool VÀ index.

        Không dựng lại index thì nhánh này train trên doc_text mới nhưng đánh giá
        trên embedding cũ — con số ra vẫn "hợp lý" nên không ai nhận ra.
        """
        return TOOL_POOL in self.overrides

    @property
    def rebuilds_ce_pairs(self) -> bool:
        """Nhánh Cross-Encoder có cần sinh lại pair không.

        `model.name` (§6.5) KHÔNG cần: nhãn ghi ra đĩa là char span,
        tokenizer-free — đúng lý do §1.4 chọn char span thay vì token index.
        Chỉ `pairs.*` mới làm đổi nội dung file.
        """
        return any(
            knob.startswith("pairs.") for knob in self.overrides.get(CROSSENCODER, {})
        )

    @property
    def rebuilds_pairs(self) -> bool:
        """Nhánh có cần bộ pair riêng không.

        `control` (và mọi nhánh chỉ đổi knob training) sinh ra **đúng** bộ pair
        của baseline: cùng `n_hard_negatives`, cùng seed, cùng nguồn. Cho nó
        dùng thẳng `data/method2/biencoder/` vừa tiết kiệm 60 MB và ~15 phút mỗi
        nhánh, vừa loại hẳn khả năng control train trên một bộ dữ liệu "giống
        nhưng không y hệt" baseline.
        """
        return self.rebuilds_pool or any(
            knob.startswith("pairs.") for knob in self.overrides.get(BIENCODER, {})
        )


#: §6.4 — số hard negative; §6.3 — document text; §6.1 — head; §6.5 — backbone.
ARMS: tuple[Arm, ...] = (
    Arm("control", {}, "Mốc so sánh, chạy lại ở đúng giao thức rút gọn"),
    Arm(
        "nneg0",
        {BIENCODER: {"pairs.n_hard_negatives": 0}},
        "§6.4 — hard negative có đáng số giờ nó tốn không?",
    ),
    Arm(
        "nneg8",
        {BIENCODER: {"pairs.n_hard_negatives": 8}},
        "§6.4 — gấp đôi hard negative có còn cải thiện không?",
    ),
    Arm(
        "doc_name_desc",
        {TOOL_POOL: {"include_param_names_in_doc": False}},
        "§6.3 — tên parameter trong doc_text có giúp truy hồi không?",
    ),
    Arm("ce_control", {}, "Mốc so sánh của họ Cross-Encoder", family=FAMILY_CE),
    Arm(
        "should_call",
        {
            CROSSENCODER: {
                "pairs.should_call.enabled": True,
                "model.enable_should_call": True,
            }
        },
        "§6.1 — head học được có hơn ngưỡng cosine τ không?",
        family=FAMILY_CE,
    ),
    Arm(
        "backbone_e5",
        {CROSSENCODER: {"model.name": "intfloat/multilingual-e5-base"}},
        "§6.5 — backbone khác cùng cỡ có hơn xlm-roberta-base không?",
        family=FAMILY_CE,
    ),
    Arm(
        "backbone_bge",
        {
            CROSSENCODER: {
                "model.name": "BAAI/bge-m3",
                "train.batch_size": 16,
                "train.grad_accum": 4,
            }
        },
        "§6.5 — backbone 568M có hơn 278M không?",
        family=FAMILY_CE,
        caveat=(
            "bge-m3 ở batch 32 × len 256 là ~17,5 GB → OOM trên T4. batch_size "
            "16 + grad_accum 4 giữ effective batch 64 y như control, nhưng số "
            "micro-batch đổi nên bước tối ưu không còn giống hệt."
        ),
    ),
)

BASE_CONFIGS: dict[str, Path] = {
    BIENCODER: Path("configs/method2/biencoder.yaml"),
    TOOL_POOL: Path("configs/method2/tool_pool.yaml"),
    CROSSENCODER: Path("configs/method2/crossencoder.yaml"),
}

#: Control của từng họ.
CONTROL_OF: dict[str, str] = {FAMILY_BI: "control", FAMILY_CE: "ce_control"}


# ------------------------------------------------------------------ resolve


def set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def get_path(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def config_kinds(arm: Arm) -> tuple[str, ...]:
    """File config mà nhánh này thật sự dùng."""
    if arm.family == FAMILY_CE:
        return (CROSSENCODER,)
    return (BIENCODER, TOOL_POOL) if arm.rebuilds_pool else (BIENCODER,)


def resolve_arm(arm: Arm, base: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Config của một nhánh = base + giao thức của họ + override riêng + đường dẫn."""
    resolved = {key: copy.deepcopy(value) for key, value in base.items()}
    for kind, knobs in SHARED_PROTOCOL[arm.family].items():
        for dotted, value in knobs.items():
            set_path(resolved[kind], dotted, value)
    for kind, knobs in arm.overrides.items():
        for dotted, value in knobs.items():
            set_path(resolved[kind], dotted, value)

    run = f"artifacts/method2/ablation/{arm.name}"
    data = f"data/method2/ablation/{arm.name}"
    bi = resolved[BIENCODER]
    if arm.rebuilds_pairs:
        set_path(bi, "pairs.output_dir", data)
        set_path(bi, "pairs.stats_path", f"{data}/pairs_stats.json")
        set_path(bi, "train.train_path", f"{data}/train.jsonl")
        set_path(bi, "train.val_path", f"{data}/val.jsonl")
    set_path(bi, "train.output_dir", run)
    set_path(bi, "thresholds.output_path", f"{run}/thresholds.json")
    set_path(bi, "evaluate.output_dir", f"{RESULTS_DIR.as_posix()}/{arm.name}")
    if arm.rebuilds_pool:
        # Pool riêng → index riêng → cả `pairs` lẫn `retrieve` phải trỏ vào đó.
        pool_path = f"{data}/tool_pool.json"
        set_path(resolved[TOOL_POOL], "output_path", pool_path)
        set_path(resolved[TOOL_POOL], "stats_path", f"{data}/tool_pool_stats.json")
        set_path(bi, "pairs.tool_pool_path", pool_path)
        set_path(bi, "index.embeddings_path", f"{data}/index/tool_embeddings.npy")
        set_path(bi, "index.tool_ids_path", f"{data}/index/tool_ids.json")

    ce = resolved[CROSSENCODER]
    set_path(ce, "train.output_dir", run)
    if arm.rebuilds_ce_pairs:
        # KHÔNG ghi đè `data/method2/crossencoder/` — preflight khoá SHA file đó.
        set_path(ce, "pairs.output_dir", data)
        set_path(ce, "pairs.stats_path", f"{data}/label_stats.json")
        set_path(ce, "data.train_path", f"{data}/train.jsonl")
        set_path(ce, "data.val_path", f"{data}/val.jsonl")
    return resolved


# ------------------------------------------------------------------- verify


def _flatten(config: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(config, dict):
        return {prefix: config}
    flat: dict[str, Any] = {}
    for key, value in config.items():
        flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def diff_arm(control: dict[str, dict[str, Any]], arm: dict[str, dict[str, Any]]) -> set[str]:
    """Knob khác nhau giữa hai nhánh, bỏ qua knob đường dẫn."""
    changed: set[str] = set()
    for kind in sorted(set(control) | set(arm)):
        left = _flatten(control.get(kind, {}))
        right = _flatten(arm.get(kind, {}))
        for knob in sorted(set(left) | set(right)):
            if knob in PATH_KNOBS or left.get(knob) == right.get(knob):
                continue
            changed.add(knob)
    return changed


def verify_arms(resolved: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Mỗi nhánh phải khác control CỦA HỌ NÓ đúng bằng những knob nó khai báo."""
    problems: list[str] = []
    for arm in ARMS:
        if arm.name in CONTROL_OF.values():
            continue
        control = resolved[CONTROL_OF[arm.family]]
        declared = {knob for knobs in arm.overrides.values() for knob in knobs}
        actual = diff_arm(control, resolved[arm.name])
        if actual != declared:
            problems.append(
                f"{arm.name}: khai {sorted(declared) or 'không gì'} nhưng thực tế khác "
                f"{sorted(actual) or 'không gì'}"
            )
    return problems


# ------------------------------------------------------------------- report


def collect_report(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        path = results_dir / arm.name / "metrics.json"
        base = {
            "arm": arm.name,
            "family": arm.family,
            "question": arm.question,
            "caveat": arm.caveat,
        }
        if not path.exists():
            rows.append({**base, "missing": str(path)})
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        overall = report.get("overall") or {}
        if arm.family == FAMILY_CE:
            # Gate §Phase 3 đo trên custom_vi chứ không phải `overall` (xLAM
            # chiếm 14.452/17.769 cặp) — bảng ablation phải đọc cùng lát đó,
            # nếu không hai nơi nói hai con số khác nhau về cùng một run.
            slice_ = (report.get("by_source") or {}).get("custom_vi") or overall
            rows.append(
                {
                    **base,
                    "measured_on": "custom_vi" if slice_ is not overall else "overall",
                    "has_value_f1": slice_.get("has_value_f1"),
                    "span_em": slice_.get("span_em"),
                    "enum_accuracy": slice_.get("enum_accuracy"),
                    "boolean_accuracy": slice_.get("boolean_accuracy"),
                }
            )
            continue
        by_split = report.get("by_tool_split") or {}
        rows.append(
            {
                **base,
                "scope": report.get("scope"),
                "ndcg@10": overall.get("ndcg@10"),
                "mrr": overall.get("mrr"),
                "full_recall@1": overall.get("full_recall@1"),
                "full_recall@5": overall.get("full_recall@5"),
                "seen_full_recall@1": (by_split.get("seen") or {}).get("full_recall@1"),
                "unseen_full_recall@1": (by_split.get("unseen") or {}).get("full_recall@1"),
            }
        )
    return {"protocol": SHARED_PROTOCOL, "rows": rows}


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def report_markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    lines = ["# Phase 6 — ablation", ""]

    bi = [r for r in rows if r.get("family", FAMILY_BI) == FAMILY_BI]
    if bi:
        lines += [
            "## Bi-Encoder (§6.3 doc text, §6.4 hard negative)",
            "",
            "Giao thức rút gọn dùng chung: 1 epoch, một vòng, không mining. "
            "**Không so trực tiếp với bảng Phase 5** (3 epoch + Round 2).",
            "",
            "| Nhánh | nDCG@10 | MRR | full R@1 | full R@5 | R@1 seen | R@1 unseen | Câu hỏi |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in bi:
            if row.get("missing"):
                lines.append(
                    f"| {row['arm']} | — | — | — | — | — | — | *chưa chạy* — {row['question']} |"
                )
                continue
            lines.append(
                "| {arm} | {ndcg} | {mrr} | {r1} | {r5} | {seen} | {unseen} | {q} |".format(
                    arm=row["arm"],
                    ndcg=_cell(row["ndcg@10"]),
                    mrr=_cell(row["mrr"]),
                    r1=_cell(row["full_recall@1"]),
                    r5=_cell(row["full_recall@5"]),
                    seen=_cell(row["seen_full_recall@1"]),
                    unseen=_cell(row["unseen_full_recall@1"]),
                    q=row["question"],
                )
            )
        lines.append("")

    ce = [r for r in rows if r.get("family") == FAMILY_CE]
    if ce:
        lines += [
            "## Cross-Encoder (§6.1 should_call head, §6.5 backbone)",
            "",
            "Giữ nguyên curriculum của Phase 3. Metric đọc trên lát **custom_vi**, "
            "cùng lát mà gate §Phase 3 dùng — `overall` bị xLAM chi phối.",
            "",
            "| Nhánh | đo trên | has_value F1 | span EM | enum acc | bool acc | Câu hỏi |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in ce:
            if row.get("missing"):
                lines.append(
                    f"| {row['arm']} | — | — | — | — | — | *chưa chạy* — {row['question']} |"
                )
                continue
            lines.append(
                "| {arm} | {on} | {hv} | {span} | {enum} | {boolean} | {q} |".format(
                    arm=row["arm"],
                    on=row.get("measured_on", "—"),
                    hv=_cell(row["has_value_f1"]),
                    span=_cell(row["span_em"]),
                    enum=_cell(row["enum_accuracy"]),
                    boolean=_cell(row["boolean_accuracy"]),
                    q=row["question"],
                )
            )
        lines.append("")
        lines.append(
            "*§6.1 so bằng cách chạy CÙNG checkpoint dưới `--abstention tau` và "
            "`--abstention should_call` (§7.17), không phải so với bảng Phase 5.*"
        )
        lines.append("")

    caveats = [r for r in rows if r.get("caveat")]
    if caveats:
        lines += ["## Nhánh không sạch một biến", ""]
        lines += [f"- **{r['arm']}** — {r['caveat']}" for r in caveats]
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- cli


def write_configs(output_dir: Path = ABLATION_DIR) -> dict[str, dict[str, dict[str, Any]]]:
    import yaml

    base = {
        kind: (yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        for kind, path in BASE_CONFIGS.items()
    }
    resolved = {arm.name: resolve_arm(arm, base) for arm in ARMS}

    problems = verify_arms(resolved)
    if problems:
        raise SystemExit(
            "[ablation] nhánh khác control ở knob KHÔNG khai báo:\n  "
            + "\n  ".join(problems)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        # Chỉ ghi file config mà nhánh thật sự dùng: một nhánh Bi-Encoder mà
        # kèm theo `crossencoder.yaml` là mời người chạy dùng nhầm file.
        for kind in config_kinds(arm):
            path = output_dir / f"{arm.name}.{kind}.yaml"
            path.write_text(
                yaml.safe_dump(
                    resolved[arm.name][kind], allow_unicode=True, sort_keys=False
                ),
                encoding="utf-8",
            )
            print(f"[ablation] {arm.name:14s} {kind:12s} → {path}")
        needs = [
            label
            for label, flag in (
                ("tool_pool + index", arm.rebuilds_pool),
                ("pair Bi-Encoder", arm.rebuilds_pairs),
                ("pair Cross-Encoder", arm.rebuilds_ce_pairs),
            )
            if flag
        ]
        print(
            f"[ablation] {arm.name:14s} "
            + (f"CẦN dựng lại: {', '.join(needs)}" if needs else "dùng lại pair của baseline")
        )
        if arm.caveat:
            print(f"[ablation] {arm.name:14s} LƯU Ý: {arm.caveat}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 — ablation Bi-Encoder")
    parser.add_argument("command", choices=["configs", "report"])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "configs":
        write_configs(args.output_dir or ABLATION_DIR)
        return

    results_dir = args.output_dir or RESULTS_DIR
    report = collect_report(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "ablation_summary.md").write_text(
        report_markdown(report), encoding="utf-8"
    )
    print(report_markdown(report))


if __name__ == "__main__":
    main()
