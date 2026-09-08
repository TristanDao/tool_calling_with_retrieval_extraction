"""Cross-Encoder model wrapper.

BAAI/bge-m3 encoder + CrossEncoderHeads.
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

from src.models.crossencoder.heads import CrossEncoderHeads, HeadConfig


class CrossEncoderForExtraction(nn.Module):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        head_config: HeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.head_config = head_config or HeadConfig()
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name)
        encoder_hidden = self._get_encoder_hidden_size()
        self.head_config = HeadConfig(
            hidden_size=encoder_hidden,
            max_enum_size=self.head_config.max_enum_size,
            dropout=self.head_config.dropout,
            enable_should_call=self.head_config.enable_should_call,
        )
        self.heads = CrossEncoderHeads(self.head_config)

    def _get_encoder_hidden_size(self) -> int:
        if hasattr(self.encoder, "config") and hasattr(self.encoder.config, "hidden_size"):
            return int(self.encoder.config.hidden_size)
        sample = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            out = self.encoder(sample)
        if hasattr(out, "last_hidden_state"):
            return int(out.last_hidden_state.size(-1))
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return int(out.hidden_states[-1].size(-1))
        raise ValueError("Cannot determine encoder hidden size")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        query_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_kwargs: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            encoder_kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None and self._encoder_supports_token_type():
            encoder_kwargs["token_type_ids"] = token_type_ids
        encoder_out = self.encoder(**encoder_kwargs)
        hidden_states = encoder_out.last_hidden_state
        return self.heads(hidden_states, query_token_mask=query_token_mask)

    def _encoder_supports_token_type(self) -> bool:
        return hasattr(self.encoder, "embeddings") and hasattr(
            self.encoder.embeddings, "token_type_embeddings"
        )

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(save_path)
        torch.save(
            {
                "head_config": self.head_config.__dict__,
                "state_dict": self.heads.state_dict(),
            },
            save_path / "crossencoder_heads.pt",
        )

    @classmethod
    def from_pretrained(cls, load_dir: str | Path) -> "CrossEncoderForExtraction":
        load_path = Path(load_dir)
        head_ckpt = torch.load(load_path / "crossencoder_heads.pt", map_location="cpu")
        head_cfg_dict = head_ckpt["head_config"]
        head_config = HeadConfig(**head_cfg_dict)
        instance = cls(model_name=str(load_path), head_config=head_config)
        instance.heads.load_state_dict(head_ckpt["state_dict"])
        return instance
