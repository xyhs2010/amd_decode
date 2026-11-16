"""``Qwen3Torch`` with the int8 KV cache in host memory.

Everything else - weight streaming, ``_layer``, ``forward`` - is the reference
implementation; only the cache's storage changes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..with_torch.kv_cache import KVCache
from ..with_torch.model import Qwen3Torch
from .host_memory import Hip
from .kv_cache import HostKVCache


class HostQwen3(Qwen3Torch):
    """Qwen3 decode with int8 K/V in pinned, GPU-addressable host memory."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        weight_bits: int = 4,
    ) -> None:
        if dtype is not torch.float16:
            raise ValueError("HostQwen3 supports float16 only")
        super().__init__(model_dir, device=device, dtype=dtype, weight_bits=weight_bits)
        self._hip = Hip()

    def make_kv_cache(self, max_len: int, batch_size: int = 1) -> KVCache:
        return HostKVCache(
            self.num_layers,
            max_len=max_len,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
            batch_size=batch_size,
            device=self.device,
            hip=self._hip,
        )
