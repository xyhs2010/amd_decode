"""Qwen3 model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_scaling: dict | None

    @property
    def num_key_value_groups(self) -> int:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 10000.0),
            max_position_embeddings=raw.get("max_position_embeddings", 32768),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            rope_scaling=raw.get("rope_scaling"),
        )
