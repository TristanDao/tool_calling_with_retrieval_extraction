"""Execute isolated Method 2 GPU stages with validation-only tuning and provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.sources import load_jsonl, sha256_file

ROOT = Path("results/method2/completion")


def command(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def verify_bundle() -> None:
    manifest = json.loads(Path("completion_manifest.json").read_text(encoding="utf-8"))
    bad = [name for name, expected in manifest["files"].items()
           if not Path(name).exists() or sha256_file(Path(name)) != expected]
    if bad:
        raise ValueError(f"Bundle changed or incomplete: {bad[:10]}")


def verify_previous(stage: str) -> dict[str, Any]:
    previous = json.loads((ROOT / f"{stage}_completed.json").read_text(encoding="utf-8"))
    if previous["bundle_manifest_sha256"] != sha256_file(Path("completion_manifest.json")):
        raise ValueError("Previous stage must use the identical strict completion bundle")
    for name, expected in previous["checkpoint_hashes"].items():
        if not Path(name).exists() or sha256_file(Path(name)) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    return previous


def run(stage: str, crossencoder: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU stage requires CUDA; use review_outputs.py for offline CPU reporting")
    verify_bundle()
    marker = ROOT / f"{stage}_completed.json"
    if marker.exists():
        raise FileExistsError(f"Stage already completed; preserve {marker} and use a fresh workspace for a new run")
    ROOT.mkdir(parents=True, exist_ok=True)
    config1 = "configs/method2/strict_round1.yaml"
    config2 = "configs/method2/strict_round2.yaml"
    bi1 = "artifacts/method2/biencoder/strict_round1/final"
    bi2 = "artifacts/method2/biencoder/strict_round2/final"
    generated: dict[str, Any] = {}
    if stage == "validation":
        command("scripts/method2/validation_gate.py", "--model", str(crossencoder),
                "--output", str(ROOT / "validation"))
        generated["validation_gate"] = json.loads((ROOT / "validation/validation_gate.json").read_text(encoding="utf-8"))
    elif stage == "round1":
        command("-m", "src.models.biencoder.train", "train", "--config", config1, "--single-gpu")
        command("-m", "src.models.run_manifest", "--run-dir", str(Path(bi1).parent), "--config", config1)
    elif stage == "round2":
        if not (ROOT / "round1_completed.json").exists():
            raise ValueError("Attach the new strict round1 output including its completion marker first")
        verify_previous("round1")
        command("-m", "src.models.biencoder.train", "mine", "--config", config1, "--model", bi1)
        command("-m", "src.models.biencoder.train", "train", "--config", config2, "--single-gpu")
        command("-m", "src.models.biencoder.index", "--config", config2, "--model", bi2)
        command("-m", "src.models.biencoder.evaluate", "calibrate", "--config", config2,
                "--model", bi2, "--pairs", "data/method2/biencoder/val.jsonl")
        for scope in ("candidates", "pool"):
            command("-m", "src.models.biencoder.evaluate", "evaluate", "--config", config2,
                    "--model", bi2, "--pairs", "data/method2/biencoder/val.jsonl", "--scope", scope,
                    "--output", str(ROOT / f"retrieval_val_{scope}.json"))
        command("-m", "src.models.run_manifest", "--run-dir", str(Path(bi2).parent), "--config", config2,
                "--report", f"retrieval={ROOT / 'retrieval_val_candidates.json'}")
    else:
        if not (ROOT / "round2_completed.json").exists():
            raise ValueError("Attach strict round2 output and its completion marker before evaluation")
        verify_previous("round2")
        if stage == "evaluation":
            validation = verify_previous("validation")
            gate = validation["results"]["validation_gate"]
            if gate["checkpoint"] != str(crossencoder):
                raise ValueError("Evaluate the checkpoint that was measured on validation")
            generated["quality_gate_passed"] = gate["passed"]
        if stage in ("stress", "normalizer_ablation"):
            baseline = verify_previous("evaluation")
        from src.models.pipeline.method2 import Method2Config, Method2Pipeline

        cfg = Method2Config.from_yaml("configs/method2/completion_pipeline.yaml")
        cfg.crossencoder_path = str(crossencoder)
        if not cfg.thresholds_path.exists():
            raise FileNotFoundError("Frozen validation thresholds are required")
        if stage in ("stress", "normalizer_ablation"):
            resolved = baseline["results"]["resolved_pipeline"]
            if str(crossencoder) != resolved["crossencoder"] or sha256_file(cfg.thresholds_path) != resolved["thresholds_sha256"]:
                raise ValueError("Ablation/stress checkpoint selection and thresholds must match evaluation")
        if stage == "normalizer_ablation":
            cfg.extraction.use_normalizer = False
            cfg.extraction.normalizer.enabled = False
        pipeline = Method2Pipeline.from_config(cfg)
        if stage == "stress":
            from src.models.pipeline.stress import StressConfig, run_stress

            stress = StressConfig.from_yaml("configs/method2/stress.yaml")
            stress.output_dir = ROOT / "stress"
            warm = list(load_jsonl(stress.gold_path))[:3]
            for sample in warm:
                pipeline.run_sample(sample)
            torch.cuda.reset_peak_memory_stats()
            generated = run_stress(pipeline, stress)
            command("scripts/method2/plot_stress.py", "--report", str(stress.output_dir / "stress_report.json"))
        else:
            sets = {"benchmark": "data/benchmark_vi/test.jsonl",
                    "custom_seen": "data/custom_vi/v1/test_seen.jsonl",
                    "custom_unseen": "data/custom_vi/v1/test_unseen.jsonl"}
            for tag, path in sets.items():
                samples = list(load_jsonl(Path(path)))
                output = ROOT / stage / tag
                for mode in ("pipeline", "oracle"):
                    for sample in samples[:3]:
                        pipeline.run_sample(sample, mode)
                    torch.cuda.reset_peak_memory_stats()
                    pipeline.run_dataset(samples, mode, output)
            report_root = ROOT / stage / "report"
            command("scripts/method2/review_outputs.py", "--source", ".",
                    "--predictions-root", str(ROOT / stage), "--output", str(report_root),
                    "--protocol", "strict_unseen_without_normalizer" if stage == "normalizer_ablation" else "strict_unseen")
            command("scripts/method2/plot_completion.py", "--tables", str(report_root / "tables.json"),
                    "--output", str(report_root / "figures"))
        generated["resolved_pipeline"] = {"biencoder": cfg.biencoder_path, "crossencoder": str(crossencoder),
                                          "thresholds_sha256": sha256_file(cfg.thresholds_path),
                                          "normalizer_enabled": cfg.extraction.use_normalizer}
    checkpoint_hashes = {str(p): sha256_file(p) for root in (Path(bi1), Path(bi2), crossencoder, Path("data/method2/index"))
                         if root.exists() for p in root.rglob("*") if p.is_file()}
    thresholds = Path(bi2).parent / "thresholds.json"
    if thresholds.exists():
        checkpoint_hashes[str(thresholds)] = sha256_file(thresholds)
    marker.write_text(json.dumps({"stage": stage, "completed": True,
                      "bundle_manifest_sha256": sha256_file(Path("completion_manifest.json")),
                      "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                      "warmup": "3 per evaluation mode; excluded from measurements",
                      "checkpoint_hashes": checkpoint_hashes, "results": generated},
                     ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validation", "round1", "round2", "evaluation", "stress", "normalizer_ablation"])
    parser.add_argument("--crossencoder", type=Path, default=Path("artifacts/method2/crossencoder/run01/final"))
    args = parser.parse_args()
    run(args.stage, args.crossencoder)


if __name__ == "__main__":
    main()
