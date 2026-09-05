"""Cell của notebook phải chạy được theo đúng thứ tự từ trên xuống.

Cell `run_manifest` từng bị đặt TRƯỚC cell định nghĩa `GOLD`: cú pháp đúng, đủ
khối, qua hết mọi kiểm tra hiện có — và chết bằng `NameError` ngay phút đầu của
"Save & Run All" trên Kaggle. Kiểm cú pháp từng cell không thấy được lớp lỗi này
vì nó nằm ở QUAN HỆ giữa các cell.

Luật kiểm: một tên được gán ở đâu đó trong notebook thì lần ĐỌC đầu tiên không
được nằm ở cell trước cell gán nó lần đầu.
"""

import ast
import json
from pathlib import Path

import pytest

from scripts.test_notebooks import _strip_shell

NOTEBOOKS = sorted(Path("notebooks").glob("method2_*.ipynb"))


def _stores_and_loads(tree: ast.AST) -> tuple[set[str], set[str]]:
    stores: set[str] = set()
    loads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (stores if isinstance(node.ctx, ast.Store) else loads).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            stores.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.add(node.name)
        elif isinstance(node, ast.arg):
            stores.add(node.arg)
    return stores, loads


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_khong_doc_bien_truoc_khi_no_duoc_gan(path):
    cells = [
        cell for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    ]
    first_store: dict[str, int] = {}
    first_load: dict[str, int] = {}
    for index, cell in enumerate(cells):
        stores, loads = _stores_and_loads(ast.parse(_strip_shell("".join(cell["source"]))))
        for name in stores:
            first_store.setdefault(name, index)
        for name in loads:
            first_load.setdefault(name, index)

    early = {
        name: (first_load[name], store)
        for name, store in first_store.items()
        if name in first_load and first_load[name] < store
    }
    assert not early, (
        f"{path.name}: đọc trước khi gán — "
        + ", ".join(f"{n} (đọc ở cell {l}, gán ở cell {s})" for n, (l, s) in sorted(early.items()))
    )


def test_luat_kiem_that_su_bat_duoc_loi(tmp_path):
    """Chứng minh luật trên không phải test rỗng: dựng đúng lỗi đã xảy ra."""
    notebook = {"cells": [
        {"cell_type": "code", "source": ["for _, tag in GOLD:\n", "    print(tag)\n"]},
        {"cell_type": "code", "source": ["GOLD = [('a', 'b')]\n"]},
    ]}
    path = tmp_path / "bad.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")

    with pytest.raises(AssertionError, match="GOLD"):
        test_khong_doc_bien_truoc_khi_no_duoc_gan(path)
