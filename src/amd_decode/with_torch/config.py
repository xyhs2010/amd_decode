"""Qwen3.5-9B text configuration.

Only the language-model half of the multimodal checkpoint is used.  This
branch targets one model, so the sizes the kernels hardcode are asserted here
instead of being plumbed through as options.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


@dataclass(frozen=True)
class Qwen35Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_dim: int
    max_position_embeddings: int
    layer_types: tuple[str, ...]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    def __post_init__(self) -> None:
        assert self.hidden_size == 4096
        assert self.intermediate_size == 12288
        assert self.num_hidden_layers == 32
        assert self.head_dim == 256
        assert self.rope_dim == 64
        assert self.num_attention_heads == 16
        assert self.num_key_value_heads == 4
        assert self.linear_num_key_heads == 16
        assert self.linear_num_value_heads == 32
        assert self.linear_key_head_dim == self.linear_value_head_dim == 128
        assert self.linear_conv_kernel_dim == 4
        assert len(self.layer_types) == self.num_hidden_layers
        assert sum(t == FULL_ATTENTION for t in self.layer_types) == 8

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_full_attention(self) -> tuple[bool, ...]:
        return tuple(t == FULL_ATTENTION for t in self.layer_types)

    @property
    def kv_layers(self) -> tuple[int, ...]:
        """Layer indices that own a K/V cache (8 of them)."""
        return tuple(i for i, t in enumerate(self.layer_types) if t == FULL_ATTENTION)

    @property
    def linear_layers(self) -> tuple[int, ...]:
        """Layer indices that own a DeltaNet recurrent state (24 of them)."""
        return tuple(i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen35Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        text = raw.get("text_config", raw)
        rope = text.get("rope_parameters", {})
        head_dim = text["head_dim"]
        return cls(
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=head_dim,
            vocab_size=text["vocab_size"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_theta=float(rope.get("rope_theta", 10000.0)),
            rope_dim=int(head_dim * float(rope.get("partial_rotary_factor", 1.0))),
            max_position_embeddings=text.get("max_position_embeddings", 262144),
            layer_types=tuple(text["layer_types"]),
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
        )
