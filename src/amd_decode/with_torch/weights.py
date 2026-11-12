"""Sharded Qwen3 weight access.

Weights stay in shard files and are read on demand, so the full 16 GB model is
never duplicated in RAM. ``Qwen3Torch`` (see ``model.py``) streams one layer at
a time to the GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .safetensor_io import SafeTensorsFile

LAYER_TENSORS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


class Qwen3Weights:
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

    def layer(self, layer_idx: int, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        return {name: self.get(prefix + name, dtype=dtype) for name in LAYER_TENSORS}
