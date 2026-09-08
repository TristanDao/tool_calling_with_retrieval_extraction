"""Generate ordered Kaggle notebooks for Method 2 completion without editing old runs."""

from __future__ import annotations

import json
from pathlib import Path

STAGES = ["validation", "round1", "round2", "evaluation", "stress", "normalizer_ablation"]

INPUT_HELPERS = '''from pathlib import Path
import hashlib, json

def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def choose_bundle(input_root: Path) -> Path:
    candidates = []
    for manifest in sorted(input_root.rglob("completion_manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("files") and all((manifest.parent / name).is_file() for name in data["files"]):
            candidates.append(manifest)
    if not candidates or len({file_digest(p) for p in candidates}) != 1:
        raise RuntimeError(f"Attach one complete bundle identity. Complete candidates: {[str(p) for p in candidates]}")
    print("Bundle selected:", candidates[0].parent)
    return candidates[0].parent

def choose_stage(input_root: Path, stage: str, bundle_digest: str) -> Path:
    candidates = []
    identities = set()
    for marker in sorted(input_root.rglob(f"{stage}_completed.json")):
        if marker.parts[-4:-1] != ("results", "method2", "completion"):
            continue
        root = marker.parents[3]
        data = json.loads(marker.read_text(encoding="utf-8"))
        hashes = data.get("checkpoint_hashes", {})
        if data.get("stage") != stage or data.get("completed") is not True or not hashes:
            continue
        if data.get("bundle_manifest_sha256") != bundle_digest:
            print("Skipping marker from another bundle:", marker)
            continue
        invalid = [name for name, digest in hashes.items()
                   if not (root / name).is_file() or file_digest(root / name) != digest]
        if invalid:
            print("Skipping incomplete or mismatched stage output:", marker, invalid[:3])
            continue
        candidates.append(marker)
        identities.add(json.dumps(hashes, sort_keys=True))
    if not candidates or len(identities) != 1:
        raise RuntimeError(f"Attach one complete {stage} checkpoint identity. Verified candidates: {[str(p) for p in candidates]}")
    print(f"{stage} input selected:", candidates[0])
    return candidates[0]

'''

SETUP = '''from pathlib import Path
import os, shutil, subprocess, sys, json
''' + INPUT_HELPERS + '''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
bundle = choose_bundle(Path("/kaggle/input"))
work = Path("/kaggle/working")
for name in ["src", "scripts", "configs", "data", "docs", "notebooks"]:
    shutil.copytree(bundle / name, work / name, dirs_exist_ok=True)
shutil.copy2(bundle / "same_domain_feasibility.json", work / "same_domain_feasibility.json")
shutil.copy2(bundle / "completion_manifest.json", work / "completion_manifest.json")
os.chdir(work)
caches = list(Path("/kaggle/input").rglob("models--BAAI--bge-m3"))
if caches:
    cache_hub = caches[0].parent
    os.environ["HF_HUB_CACHE"] = str(cache_hub)
    os.environ["HF_HOME"] = str(cache_hub.parent)
pins = json.loads(Path("configs/method2/pinned_versions.json").read_text())["pinned"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *[f"{k}=={v}" for k,v in pins.items()], "jsonschema", "pyyaml", "matplotlib", "rank_bm25", "datasets", "accelerate"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
lora_smoke = """import torch
from transformers import XLMRobertaConfig, XLMRobertaModel
from peft import LoraConfig
config = XLMRobertaConfig(vocab_size=32, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, max_position_embeddings=32)
model = XLMRobertaModel(config)
model.add_adapter(LoraConfig(r=2, lora_alpha=4, target_modules=["query", "value"]))
model(torch.tensor([[0, 5, 2]])).last_hidden_state.square().mean().backward()
assert any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
print("LoRA environment smoke passed")
"""
subprocess.run([sys.executable, "-c", lora_smoke], check=True)
'''

RESTORE = '''prior_stage = {prior!r}
if prior_stage:
    prior_marker = choose_stage(Path("/kaggle/input"), prior_stage, file_digest(work / "completion_manifest.json"))
    previous = prior_marker.parents[3]
    for name in ["artifacts/method2", "results/method2/completion", "data/method2/index"]:
        if (previous / name).exists():
            shutil.copytree(previous / name, work / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoint-*"))
if {needs_validation!r}:
    validation_marker = choose_stage(Path("/kaggle/input"), "validation", file_digest(work / "completion_manifest.json"))
    validation_root = validation_marker.parents[3]
    shutil.copytree(validation_root / "results/method2/completion/validation", work / "results/method2/completion/validation", dirs_exist_ok=True)
    shutil.copy2(validation_marker, work / "results/method2/completion/validation_completed.json")
    shutil.copytree(validation_root / "artifacts/method2/crossencoder", work / "artifacts/method2/crossencoder", dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoint-*"))
ce = Path("artifacts/method2/crossencoder/run01/final")
if {needs_ce!r} and not (ce / "crossencoder_heads.pt").exists():
    heads = [p for p in Path("/kaggle/input").rglob("crossencoder_heads.pt") if p.parent.name == "final"]
    if len(heads) != 1:
        raise RuntimeError(f"Attach one CE final checkpoint (model + heads + tokenizer), found {{len(heads)}}")
    shutil.copytree(heads[0].parent, ce, dirs_exist_ok=True)
'''


def make_notebooks(output: Path = Path("notebooks")) -> list[Path]:
    written = []
    for index, stage in enumerate(STAGES, 1):
        prior = "round1" if stage == "round2" else ("evaluation" if stage in STAGES[4:] else ("round2" if stage == "evaluation" else None))
        restore = RESTORE.replace("{prior!r}", repr(prior)).replace("{needs_ce!r}", repr(stage not in ("round1", "round2"))).replace("{needs_validation!r}", repr(stage == "evaluation")).replace("{{len(heads)}}", "{len(heads)}")
        run = f'''subprocess.run([sys.executable, "scripts/method2/run_completion.py", "{stage}"], check=True)
print(Path("results/method2/completion/{stage}_completed.json").read_text(encoding="utf-8")[:2000])
'''
        save = f'''archive = work / "method2_completion_{stage}_reports.tar.gz"
items = [name for name in ["results/method2/completion", "data/method2/biencoder/train_mined.jsonl"] if Path(name).exists()]
subprocess.run(["tar", "czf", str(archive), *items], check=True)
print(archive, archive.stat().st_size)
print("Save the whole notebook output for the next stage; this small archive does not contain model weights or the index.")
'''
        cells = [{"cell_type": "markdown", "metadata": {}, "source": [f"# Method 2 completion {index}: {stage}\n", "Attach the strict completion bundle. See docs/method2_completion.md for the required previous outputs. GPU T4, fresh session. Test results must not select hyperparameters.\n"]}]
        for source in (SETUP, restore, run, save):
            compile(source, f"{stage}_cell", "exec")
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": source.splitlines(keepends=True)})
        path = output / f"method2_completion_{index:02d}_{stage}.ipynb"
        path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding="utf-8")
        written.append(path)
    return written


def make_ce_repair_notebook(output: Path = Path("notebooks")) -> Path:
    restore = '''baseline_marker = choose_stage(Path("/kaggle/input"), "validation", file_digest(work / "completion_manifest.json"))
baseline_root = baseline_marker.parents[3]
baseline_meta = json.loads(baseline_marker.read_text())
baseline_path = Path(baseline_meta["results"]["validation_gate"]["checkpoint"])
baseline = work / "artifacts/method2/crossencoder/baseline_for_repair/final"
shutil.copytree(baseline_root / baseline_path, baseline)
patches = list(Path("/kaggle/input").rglob("ce_repair_manifest.json"))
if len(patches) != 1:
    raise RuntimeError(f"Attach exactly one CE repair addon; found {len(patches)}")
patch_path = patches[0]
patch_data = json.loads(patch_path.read_text())
parent_data = json.loads((work / "completion_manifest.json").read_text())
if patch_data["parent_completion_sha256"] != file_digest(work / "completion_manifest.json"):
    raise RuntimeError("CE addon belongs to another bundle")
if set(patch_data["files"]) & set(parent_data["files"]):
    raise RuntimeError("CE addon must not overwrite frozen bundle files")
for name, digest in patch_data["files"].items():
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Invalid addon path")
    source = patch_path.parent / relative
    if file_digest(source) != digest:
        raise RuntimeError(f"Addon hash mismatch: {name}")
    target = work / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
shutil.copy2(patch_path, work / "ce_repair_manifest.json")
print((work / "ce_repair_audit.json").read_text())
'''
    run = '''subprocess.run([sys.executable, "scripts/method2/run_ce_repair.py"], check=True)
print(Path("results/method2/ce_repair/comparison.json").read_text())
'''
    save = '''archive = work / "method2_ce_repair_reports.tar.gz"
subprocess.run(["tar", "czf", str(archive), "results/method2/ce_repair", "results/method2/completion", "ce_repair_audit.json", "ce_repair_manifest.json"], check=True)
print("Save the whole notebook output. The reports archive omits model weights.")
print(archive)
'''
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# Method 2 — 01b CE supervision repair\n",
        "Inputs: complete validation notebook 01 output (includes baseline CE and parent bundle), plus ce_repair_v1 addon. GPU T4 with Internet. No round2 input needed here.\n",
        "Train fresh XLM-R with corrected mention labels, same 2+2 curriculum. Compare both checkpoints on validation only; do not use test to select. A failed quality gate remains failed even when the run completes.\n"]}]
    for source in (SETUP, restore, run, save):
        compile(source, "ce_repair_cell", "exec")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": source.splitlines(keepends=True)})
    path = output / "method2_ce_repair_01b.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                               "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    for path in make_notebooks():
        print(path)
    print(make_ce_repair_notebook())
