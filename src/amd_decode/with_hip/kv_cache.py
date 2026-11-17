"""q8 KV cache for the HIP decode backend: pinned host memory, one sequence.

K/V live in host RAM as int8 ``[L, H, T, D]`` with fp32 scales.  The seeded
prefix keeps the prefill machine's per-(head, block, channel) K scales; tokens
appended during prefill/decode use per-token K and V scales.

The decode kernel reads the int8 buffers directly.  ``dequantize`` exists only
for the short torch prefill pass of ``answer_prefix``, which runs once per
sample.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ..utils.prefill_cache import SavedKVCache, expand_block_scale
from ..with_host.host_memory import Hip, HostBuffer
from ..with_torch.kv_cache import quantize_kv


class KVView(NamedTuple):
    """Pointers and lengths the decode kernel needs for one layer."""

    k: torch.Tensor        # int8 [H, max_len, D]
    v: torch.Tensor        # int8 [H, max_len, D]
    k_tail: torch.Tensor   # fp32 [H, max_len], appended tokens
    v_scale: torch.Tensor  # fp32 [H, max_len]
    k_block: torch.Tensor | None  # fp32 [H, n_blocks, D], seeded prefix
    n: int
    prefix: int
    block_size: int


class HipKVCache:
    """int8 K/V plus scales in pinned host memory, shaped ``[L, H, T, D]``."""

    def __init__(self, hip: Hip, num_layers: int, num_kv_heads: int, head_dim: int,
                 max_len: int, device: str | torch.device = "cuda") -> None:
        self.hip = hip
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.dtype = torch.float16
        device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device

        kv_shape = (num_layers, num_kv_heads, max_len, head_dim)
        scale_shape = (num_layers, num_kv_heads, max_len)
        self.k, self.k_dev, self.k_cpu = self._alloc(kv_shape, torch.int8)
        self.v, self.v_dev, self.v_cpu = self._alloc(kv_shape, torch.int8)
        self.k_tail, self.k_tail_dev, self.k_tail_cpu = self._alloc(
            scale_shape, torch.float32
        )
        self.v_scale, self.v_scale_dev, self.v_scale_cpu = self._alloc(
            scale_shape, torch.float32
        )

        # Prefix K scales, uploaded once per layer at seed time.
        self.k_block: list[torch.Tensor | None] = [None] * num_layers
        self.block_size = 0
        self.prefix_len = 0
        self._len = [0] * num_layers
        self.length = 0

    def _alloc(self, shape: tuple[int, ...], dtype: torch.dtype):
        buf = HostBuffer(self.hip, math.prod(shape), dtype, self.device)
        return buf, buf.cuda.view(shape), buf.cpu.view(shape)

    def free(self) -> None:
        for buf in (self.k, self.v, self.k_tail, self.v_scale):
            buf.free()

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass

    # ------------------------------------------------------------------ seeding

    def seed_from_prefill(self, saved: SavedKVCache) -> None:
        seq = saved.seq_len
        if saved.num_layers != self.num_layers:
            raise ValueError(
                f"cache has {self.num_layers} layers, prefill has {saved.num_layers}"
            )
        if seq > self.max_len:
            raise RuntimeError(
                f"prefix of {seq} exceeds cache capacity {self.max_len}"
            )
        for layer_idx in range(self.num_layers):
            k_q, k_scale, v_q, v_scale = saved.layer_quantized(layer_idx)
            self.k_cpu[layer_idx, :, :seq].copy_(k_q[0])
            self.v_cpu[layer_idx, :, :seq].copy_(v_q[0])
            self.v_scale_cpu[layer_idx, :, :seq].copy_(v_scale.squeeze(-1))
            self.k_block[layer_idx] = (
                k_scale.contiguous().to(self.device)
            )
            self._len[layer_idx] = seq
        blocks = self.k_block[0].shape[1]
        self.block_size = saved.block_size or -(-seq // blocks)
        self.prefix_len = seq
        self.length = seq

    # ------------------------------------------------------------------- decode

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Quantise new fp16 K/V ``[1, H, T, D]`` into the cache."""
        pos = self._len[layer_idx]
        n = k.shape[-2]
        if k.shape != v.shape or k.shape[:2] != (1, self.num_kv_heads):
            raise ValueError(
                f"expected [1, {self.num_kv_heads}, T, {self.head_dim}], "
                f"got {tuple(k.shape)} and {tuple(v.shape)}"
            )
        if pos + n > self.max_len:
            raise RuntimeError(
                f"layer {layer_idx}: writing {n} tokens at {pos} exceeds "
                f"capacity {self.max_len}"
            )
        k_q, k_scale = quantize_kv(k)
        v_q, v_scale = quantize_kv(v)
        sl = slice(pos, pos + n)
        self.k_dev[layer_idx, :, sl].copy_(k_q[0])
        self.v_dev[layer_idx, :, sl].copy_(v_q[0])
        self.k_tail_dev[layer_idx, :, sl].copy_(k_scale[0, :, :, 0])
        self.v_scale_dev[layer_idx, :, sl].copy_(v_scale[0, :, :, 0])
        self._len[layer_idx] = pos + n
        self.length = pos + n

    def view(self, layer_idx: int) -> KVView:
        """The layer's buffers and metadata, for the decode kernel."""
        return KVView(
            self.k_dev[layer_idx], self.v_dev[layer_idx],
            self.k_tail_dev[layer_idx], self.v_scale_dev[layer_idx],
            self.k_block[layer_idx], self._len[layer_idx],
            self.prefix_len, self.block_size,
        )

    def dequantize(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fp16 ``[1, H, T, D]`` K/V for the torch prefill pass."""
        n = self._len[layer_idx]
        if n <= 0:
            raise ValueError("nothing to dequantise")
        k = self.k_dev[layer_idx, :, :n].float()
        prefix = min(self.prefix_len, n)
        if prefix:
            k[:, :prefix] *= expand_block_scale(
                self.k_block[layer_idx], prefix, self.block_size
            )
        if n > prefix:
            k[:, prefix:n] *= self.k_tail_dev[
                layer_idx, :, prefix:n
            ].float().unsqueeze(-1)
        v = (self.v_dev[layer_idx, :, :n].float()
             * self.v_scale_dev[layer_idx, :, :n].float().unsqueeze(-1))
        return k.unsqueeze(0).to(self.dtype), v.unsqueeze(0).to(self.dtype)
