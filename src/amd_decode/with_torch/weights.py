"""Sharded Qwen3.5 weight access.

The checkpoint is the multimodal ``Qwen3_5ForConditionalGeneration``; only the
``model.language_model`` tensors are read.  Vision and MTP tensors stay on disk
forever.  Weights are memory-mapped and read on demand, so the full checkpoint
is never materialized in RAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..utils.safetensor_io import SafeTensorsFile

EMBED = "model.language_model.embed_tokens.weight"
FINAL_NORM = "model.language_model.norm.weight"
LM_HEAD = "lm_head.weight"

_COMMON = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

_FULL_ATTENTION = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)

_LINEAR_ATTENTION = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
)


class Qwen35Weights:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        shards = sorted(set(self.weight_map.values()))
        self._files = {name: SafeTensorsFile(self.model_dir / name) for name in shards}

    def get(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"tensor {name!r} not in index") from exc
        return self._files[shard].get(name, dtype=dtype)

    def layer(self, layer_idx: int, full_attention: bool, dtype: torch.dtype | None = None
              ) -> dict[str, torch.Tensor]:
        prefix = f"model.language_model.layers.{layer_idx}."
        names = _COMMON + (_FULL_ATTENTION if full_attention else _LINEAR_ATTENTION)
        return {name: self.get(prefix + name, dtype=dtype) for name in names}
