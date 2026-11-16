"""``Qwen3Torch`` with the KV cache in host memory.

Only the cache changes: ``_layer`` is inherited unchanged, so the maths is
byte-identical to the reference implementation.  ``forward`` still streams one
layer's weights at a time; the KV cache is never resident in VRAM.

``make_kv_cache`` is the hook used by ``amd_decode.with_torch.align``: when a
model defines it, ``align`` gives that model a cache of the required capacity
(the base class keeps using ``KVCache``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..with_torch.model import Qwen3Torch
from .host_memory import HOST_MALLOC_FLAGS, Hip
from .kv_cache import HostKVCache


class HostQwen3(Qwen3Torch):
    """Qwen3 decode with K/V in pinned, GPU-addressable host memory."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        weight_bits: int = 4,
        *,
        kv_dtype: torch.dtype | None = None,
        kv_flags: int = HOST_MALLOC_FLAGS,
    ) -> None:
        super().__init__(model_dir, device=device, dtype=dtype, weight_bits=weight_bits)
        # 16-bit by default: same width as the activations.
        self.kv_dtype = dtype if kv_dtype is None else kv_dtype
        self.kv_flags = kv_flags
        self._hip = Hip()

    def make_kv_cache(self, max_len: int, batch_size: int = 1) -> HostKVCache:
        return HostKVCache(
            self.num_layers,
            max_len=max_len,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
            batch_size=batch_size,
            dtype=self.kv_dtype,
            device=self.device,
            flags=self.kv_flags,
            hip=self._hip,
        )

    @property
    def kv_nbytes_per_token(self) -> int:
        """Host bytes each decode step must read: all layers, K and V."""
        return (
            2
            * self.num_layers
            * self.cfg.num_key_value_heads
            * self.cfg.head_dim
            * torch.empty((), dtype=self.kv_dtype).element_size()
        )
