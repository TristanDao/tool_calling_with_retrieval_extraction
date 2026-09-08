"""Run CE-only oracle validation and require all declared component and call gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.config import EvaluationConfig, NormalizationConfig
from src.evaluation.evaluator import evaluate
from src.evaluation.extraction_metrics import normalize_calls
from src.evaluation.io import load_gold, load_predictions
from src.evaluation.matching import align_calls
from src.evaluation.normalization import ArgumentNormalizer, canonical_json
from src.models.sources import load_jsonl, write_jsonl


def argument_diagnostics(gold_path: Path, predictions_path: Path) -> dict[str, Any]:
    from src.evaluation.io import align_predictions

    gold = load_gold(gold_path)
    predictions = align_predictions(gold, load_predictions(predictions_path))
    normalizer = ArgumentNormalizer(NormalizationConfig())
    correct_calls = total_calls = 0
    counts: dict[str, list[int]] = {}
    for sample, pred in zip(gold, predictions, strict=True):
        for pair in align_calls(normalize_calls(sample, sample.function_calls, normalizer),
                                normalize_calls(sample, pred.function_calls, normalizer)):
            if pair.gold is None:
                continue
            total_calls += 1
            expected = pair.gold.arguments
            actual = pair.prediction.arguments if pair.prediction else {}
            correct_calls += int(pair.prediction is not None and pred.parse_valid and
                                 canonical_json(expected) == canonical_json(actual))
            schema = sample.tool_schemas.get(pair.gold.name, {}).get("parameters", {})
            required = set(schema.get("required", []))
            for key, value in expected.items():
                param = schema.get("properties", {}).get(key, {})
                kind = "enum" if "enum" in param else str(param.get("type", "unknown"))
                for label in [kind] + (["required"] if key in required else []):
                    tally = counts.setdefault(label, [0, 0])
                    tally[1] += 1
                    tally[0] += int(key in actual and canonical_json(value) == canonical_json(actual[key]))
    return {"argument_em": correct_calls / total_calls if total_calls else None,
            "gold_call_count": total_calls, "correct_call_count": correct_calls,
            "values_on_gold": {k: {"correct": v[0], "support": v[1], "accuracy": v[0] / v[1]}
                               for k, v in counts.items()}}


def run(config_path: Path, model_path: Path, output: Path) -> dict[str, Any]:
    import yaml
    from src.models.biencoder.retrieve import RetrievalThresholds
    from src.models.crossencoder.evaluate import check_gates, evaluate_file
    from src.models.crossencoder.inference import CrossEncoderExtractor, ExtractionConfig, load_cross_encoder
    from src.models.pipeline.method2 import Method2Pipeline
    from src.models.pipeline.validator import ArgumentValidator

    if output.exists():
        raise FileExistsError(f"Use a new validation output: {output}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    components = evaluate_file(str(model_path), cfg["data"]["val_path"],
                               max_length=cfg["data"]["max_length"],
                               batch_size=cfg["train"]["eval_batch_size"])
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_cross_encoder(str(model_path), device)
    extractor = CrossEncoderExtractor(model, tokenizer, ExtractionConfig(
        max_length=cfg["data"]["max_length"],
        max_question_tokens=cfg["model"].get("max_question_tokens", 96)), device=device)
    pipeline = Method2Pipeline(None, extractor, ArgumentValidator(normalizer=extractor.normalizer),
                               {}, thresholds=RetrievalThresholds())
    samples = [s for split in ("val_seen", "val_unseen")
               for s in load_jsonl(Path(f"data/custom_vi/v1/{split}.jsonl"))]
    for sample in samples:
        pipeline.run_sample(sample, "oracle")
        break
    pipeline.run_dataset(samples, "oracle", output)
    write_jsonl(output / "gold.jsonl", samples)
    evaluate(output / "gold.jsonl", output / "oracle_predictions.jsonl", output / "evaluation",
             EvaluationConfig(), output / "oracle_predictions.jsonl")
    diagnostics = argument_diagnostics(output / "gold.jsonl", output / "oracle_predictions.jsonl")
    values = {**components["by_source"].get("custom_vi", {}), "argument_em": diagnostics["argument_em"]}
    gate = check_gates(values, cfg["gates"])
    gate.update({"measured_on": "custom_val_seen+custom_val_unseen", "components": components,
                 "inference_diagnostics": diagnostics, "checkpoint": str(model_path),
                 "next_action": "freeze_checkpoint" if gate["passed"] else "inspect_validation_errors_before_retraining"})
    (output / "validation_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in gate.items() if k != "components"}, ensure_ascii=False, indent=2))
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/method2/crossencoder.yaml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.model, args.output)


if __name__ == "__main__":
    main()
