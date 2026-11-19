"""The fp16 KV cache for Qwen3.5's full-attention layers.

Only the 8 layers whose ``layer_types`` entry is ``full_attention`` own K/V;
the 24 delta-net layers keep their conv/recurrent state on the model
(``Qwen35Torch._delta``).  Both are seeded from the same offline prefill.

Buffers are fp16 ``[1, 4, max_len, 256]`` in VRAM, allocated per layer on first
use.  The offline prefill is bf16 dense, so seeding converts once; decode
appends fp16 tokens into the same buffer.
"""

from __future__ import annotations

import torch

KV_HEADS = 4
HEAD_DIM = 256


class KVCache:
    def __init__(self, max_len: int) -> None:
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.max_len = max_len
        self.length = 0
        self._k: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}
        self._len: dict[int, int] = {}

    def _buffers(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self._k:
            shape = (1, KV_HEADS, self.max_len, HEAD_DIM)
            self._k[layer_idx] = torch.empty(shape, dtype=torch.float16, device="cuda")
            self._v[layer_idx] = torch.empty(shape, dtype=torch.float16, device="cuda")
        return self._k[layer_idx], self._v[layer_idx]

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``[1, 4, T, 256]`` K/V and return the prefix ``[0, pos + T)``."""
        k_buf, v_buf = self._buffers(layer_idx)
        pos = self._len.get(layer_idx, 0)
        n = k.shape[-2]
        if pos + n > self.max_len:
            raise RuntimeError(
                f"layer {layer_idx}: writing {n} tokens at {pos} exceeds "
                f"capacity {self.max_len}"
            )
        k_buf[:, :, pos : pos + n].copy_(k)
        v_buf[:, :, pos : pos + n].copy_(v)
        self._len[layer_idx] = pos + n
        self.length = pos + n
        return self.prefix(layer_idx)

    @property
    def layers(self) -> list[int]:
        """Layer indices that own K/V (the 8 full-attention layers once seeded)."""
        return sorted(self._len)

    def prefix(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer's cached K/V ``[1, H, n, D]`` as fp16 views."""
        n = self._len.get(layer_idx, 0)
        return self._k[layer_idx][:, :, :n], self._v[layer_idx][:, :, :n]
