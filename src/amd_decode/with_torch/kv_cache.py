"""The int8 KV cache used by ``Qwen3Torch``.

K/V are stored as int8 plus scales and dequantised to fp16 before attention,
so ``_layer`` is unchanged.  This implementation keeps the buffers in VRAM;
``with_host.HostKVCache`` overrides ``_allocate``/``free`` to put the same
layout in pinned host memory.

Scale layouts:

* the seeded prefix keeps the prefill machine's native q8 layout (K: per
  (head, block, channel), V: per token), read straight from the saved cache;
* tokens appended during decode get a per-(head, token) symmetric scale from
  :func:`quantize_kv`.
"""

from __future__ import annotations

import math

import torch

from ..utils.prefill_cache import SavedKVCache, expand_block_scale


def quantize_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(head, token) symmetric int8 for newly generated K/V.

    ``x`` is [B, H, T, D] on any device; returns int8 ``q`` with the same shape
    and a float32 scale of shape [B, H, T, 1].
    """
    scale = x.float().abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


class KVCache:
    """Per-layer int8 K/V, shaped [B, H_kv, max_len, D], with fp16 dequant."""

    def __init__(
        self,
        num_layers: int,
        max_len: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        batch_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        self.num_layers = num_layers
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = torch.float16
        device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device

        shape = (num_layers, batch_size, num_kv_heads, max_len, head_dim)
        scale_shape = (num_layers, batch_size, num_kv_heads, max_len, 1)
        self.k_q = self._allocate(shape, torch.int8)
        self.v_q = self._allocate(shape, torch.int8)
        self.k_tail_scale = self._allocate(scale_shape, torch.float32)
        self.v_scale = self._allocate(scale_shape, torch.float32)

        self._prefix_len = [0] * num_layers
        self._k_block_scale: list[torch.Tensor | None] = [None] * num_layers
        self._block_size: list[int | None] = [None] * num_layers
        self._len = [0] * num_layers
        self.length = 0
        self.seeded_len = 0

    def _allocate(self, shape: tuple[int, ...], dtype: torch.dtype):
        """Allocate one cache buffer; ``HostKVCache`` overrides the storage."""
        tensor = torch.empty(math.prod(shape), dtype=dtype, device=self.device)
        return tensor.view(shape)

    def free(self) -> None:
        """Release non-tensor storage; device buffers are freed by the GC."""

    # ------------------------------------------------------------------ helpers

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.k_q, self.v_q, self.k_tail_scale, self.v_scale)
        )

    def bytes_read(self, length: int) -> int:
        """Bytes a decode step pulls for ``length`` cached tokens."""
        heads = self.num_layers * self.batch_size * self.num_kv_heads
        n = heads * (2 * self.head_dim + 4) * length  # int8 K/V + fp32 V scale
        n += heads * 4 * max(0, length - self.seeded_len)  # K tail scales
        return n

    # ------------------------------------------------------------------ seeding

    def seed_layer(
        self,
        layer_idx: int,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v_q: torch.Tensor,
        v_scale: torch.Tensor,
        block_size: int,
        position: int = 0,
    ) -> None:
        """Copy a saved q8 prefill layer as stored (int8 + native scales)."""
        n = k_q.shape[-2]
        if k_q.shape[:-2] != (self.batch_size, self.num_kv_heads):
            raise ValueError(f"unexpected k_q shape {tuple(k_q.shape)}")
        if position + n > self.max_len:
            raise RuntimeError(
                f"layer {layer_idx}: prefix of {n} at {position} exceeds "
                f"capacity {self.max_len}"
            )
        sl = slice(position, position + n)
        self.k_q[layer_idx, :, :, sl].copy_(k_q)
        self.v_q[layer_idx, :, :, sl].copy_(v_q)
        self.v_scale[layer_idx, :, :, sl].copy_(v_scale.unsqueeze(0))
        self._k_block_scale[layer_idx] = k_scale.contiguous()
        self._block_size[layer_idx] = block_size
        self._prefix_len[layer_idx] = position + n
        self.seeded_len = max(self.seeded_len, position + n)

    def seed_from_prefill(self, saved: SavedKVCache, position: int = 0) -> None:
        """Seed the whole cache from a hydra_llm prefill directory."""
        if saved.num_layers != self.num_layers:
            raise ValueError(
                f"cache has {self.num_layers} layers, prefill has {saved.num_layers}"
            )
        seq_len = saved.seq_len
        for layer_idx in range(saved.num_layers):
            k_q, k_scale, v_q, v_scale = saved.layer_quantized(layer_idx)
            block = saved.block_size or -(-seq_len // k_scale.shape[1])
            self.seed_layer(layer_idx, k_q, k_scale, v_q, v_scale, block, position)
        self.set_length(position + seq_len)

    def set_length(self, length: int) -> None:
        if not 0 <= length <= self.max_len:
            raise ValueError(f"length {length} outside [0, {self.max_len}]")
        self._len = [length] * self.num_layers
        self.length = length

    # ------------------------------------------------------------------ decode

    def dequantize(self, layer_idx: int, length: int | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fp16 [B, H, length, D] K/V for one layer."""
        total = self._len[layer_idx] if length is None else length
        if total <= 0:
            raise ValueError("nothing to dequantise")
        prefix = min(self._prefix_len[layer_idx], total)

        k_f = self.k_q[layer_idx, :, :, :total].float()
        parts = []
        if prefix:
            k_scale = self._k_block_scale[layer_idx]
            if k_scale is None:
                raise RuntimeError(f"layer {layer_idx}: prefix has no block scale")
            expanded = expand_block_scale(
                k_scale.to(self.device), prefix, self._block_size[layer_idx]
            )
            parts.append(k_f[:, :, :prefix] * expanded.unsqueeze(0))
        if total > prefix:
            tail_scale = self.k_tail_scale[layer_idx, :, :, prefix:total].float()
            parts.append(k_f[:, :, prefix:] * tail_scale)

        k = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
        v = (self.v_q[layer_idx, :, :, :total].float()
             * self.v_scale[layer_idx, :, :, :total].float())
        return k.to(self.dtype), v.to(self.dtype)

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantise new fp16 K/V into the cache, return the dequantised prefix."""
        position = self._len[layer_idx]
        n = k.shape[-2]
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {tuple(k.shape)} != {tuple(v.shape)}")
        if k.shape[:-2] != (self.batch_size, self.num_kv_heads):
            raise ValueError(
                f"expected [B={self.batch_size}, H={self.num_kv_heads}, T, "
                f"D={self.head_dim}], got {tuple(k.shape)}"
            )
        if position + n > self.max_len:
            raise RuntimeError(
                f"layer {layer_idx}: writing {n} tokens at {position} exceeds "
                f"capacity {self.max_len}"
            )

        k_q, k_s = quantize_kv(k)
        v_q, v_s = quantize_kv(v)
        sl = slice(position, position + n)
        self.k_q[layer_idx, :, :, sl].copy_(k_q)
        self.v_q[layer_idx, :, :, sl].copy_(v_q)
        self.k_tail_scale[layer_idx, :, :, sl].copy_(k_s)
        self.v_scale[layer_idx, :, :, sl].copy_(v_s)

        self._len[layer_idx] = position + n
        self.length = self._len[layer_idx]
        return self.dequantize(layer_idx)
