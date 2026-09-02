"""Model package with lazy imports for optional training dependencies."""

from typing import Any

__all__ = [
    "CrossEncoderForExtraction",
    "CrossEncoderHeads",
    "CrossEncoderCollator",
    "HierarchicalLoss",
    "LabelGenerator",
    "HeadConfig",
    "LossConfig",
    "extract_arguments",
]


def __getattr__(name: str) -> Any:
    if name == "CrossEncoderForExtraction":
        from src.models.crossencoder.model import CrossEncoderForExtraction

        return CrossEncoderForExtraction
    if name in {"CrossEncoderHeads", "HeadConfig"}:
        from src.models.crossencoder.heads import CrossEncoderHeads, HeadConfig

        return {"CrossEncoderHeads": CrossEncoderHeads, "HeadConfig": HeadConfig}[name]
    if name == "CrossEncoderCollator":
        from src.models.crossencoder.data_collator import CrossEncoderCollator

        return CrossEncoderCollator
    if name == "HierarchicalLoss":
        from src.models.crossencoder.losses import HierarchicalLoss

        return HierarchicalLoss
    if name == "LabelGenerator":
        from src.models.crossencoder.label_generator import LabelGenerator

        return LabelGenerator
    if name == "LossConfig":
        from src.models.crossencoder.losses import LossConfig

        return LossConfig
    if name == "extract_arguments":
        from src.models.crossencoder.inference import extract_arguments

        return extract_arguments
    raise AttributeError(name)
