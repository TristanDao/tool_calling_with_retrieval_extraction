"""Cross-Encoder for schema-aware parameter extraction.

Hierarchical heads on top of BGE-M3 encoder:
- has_value (binary)
- span start/end (string/number params)
- enum logits (enum params)
- boolean logits (2-way)
"""

from src.models.crossencoder.data_collator import CrossEncoderCollator
from src.models.crossencoder.heads import CrossEncoderHeads, HeadConfig
from src.models.crossencoder.inference import extract_arguments
from src.models.crossencoder.label_generator import LabelGenerator
from src.models.crossencoder.losses import HierarchicalLoss, LossConfig
from src.models.crossencoder.model import CrossEncoderForExtraction

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
