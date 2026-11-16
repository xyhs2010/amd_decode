"""KV cache backed by host memory instead of VRAM.

``HostKVCache`` implements the same ``append`` protocol as
``amd_decode.with_torch.KVCache``, but every layer's K/V lives in one pinned
host allocation that the GPU addresses directly (see ``host_memory``).  A 16K
fp16 context costs 2.4 GB of host RAM for all 36 layers and *no* VRAM, which is
what makes longer contexts possible on a 16 GB card.

The returned K/V are views over host memory.  The reference ``_layer`` calls
``repeat_interleave`` on them, which materialises a device copy before the
matmuls, so byte-level results match a device-resident cache exactly; the
saving is that the full cache never has to be resident in VRAM.
"""

from __future__ import annotations

import torch

from .host_memory import HOST_MALLOC_FLAGS, HostBuffer, Hip


class HostKVCache:
    """Per-layer K/V in host memory, shaped [B, H_kv, max_len, D] per layer.

    All layers share one ``hipHostMalloc`` allocation (2 * num_layers blobs) so
    there is a single owner and a single free.

    Capacity is fixed at construction: attention reads the prefix
    ``[..., :length, :]`` and ``append`` writes at ``length``.  Use
    :meth:`set_layer` + :meth:`set_length` to seed the cache from a prefill
    (e.g. a saved cache on disk) instead of appending token by token.
    """

    def __init__(
        self,
        num_layers: int,
        max_len: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        flags: int = HOST_MALLOC_FLAGS,
        hip: Hip | None = None,
    ) -> None:
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.num_layers = num_layers
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device
        self.hip = hip or Hip()
        self.flags = flags

        numel = (
            2 * num_layers * batch_size * num_kv_heads * max_len * head_dim
        )
        self._buf = HostBuffer(self.hip, numel, dtype, self.device, flags=flags)
        self._all = self._buf.cuda.view(
            2, num_layers, batch_size, num_kv_heads, max_len, head_dim
        )
        self.k = self._all[0]
        self.v = self._all[1]
        self._len = [0] * num_layers
        self.length = 0

    # ------------------------------------------------------------------ helpers

    @property
    def nbytes(self) -> int:
        return self._buf.nbytes

    def layer_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The full-capacity [B, H_kv, max_len, D] K/V for one layer."""
        return self.k[layer_idx], self.v[layer_idx]

    def _write(self, layer_idx: int, position: int, k: torch.Tensor,
               v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = k.shape[-2]
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {tuple(k.shape)} != {tuple(v.shape)}")
        if k.shape[:-2] != (self.batch_size, self.num_kv_heads):
            raise ValueError(
                f"expected [B={self.batch_size}, H={self.num_kv_heads}, T, "
                f"D={self.head_dim}], got {tuple(k.shape)}"
            )
        if k.shape[-1] != self.head_dim:
            raise ValueError(f"head_dim {k.shape[-1]} != {self.head_dim}")
        if position < 0 or position + n > self.max_len:
            raise RuntimeError(
                f"layer {layer_idx}: writing {n} tokens at {position} exceeds "
                f"capacity {self.max_len}"
            )
        # Element-wise copy into host memory; cheap on non-coherent mappings.
        self.k[layer_idx, :, :, position:position + n].copy_(k)
        self.v[layer_idx, :, :, position:position + n].copy_(v)
        return (
            self.k[layer_idx, :, :, :position + n],
            self.v[layer_idx, :, :, :position + n],
        )

    # ------------------------------------------------------------------ protocol

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new [B, H_kv, T_new, D] K/V and return the whole prefix."""
        position = self._len[layer_idx]
        k_all, v_all = self._write(layer_idx, position, k, v)
        self._len[layer_idx] = position + k.shape[-2]
        self.length = self._len[layer_idx]
        return k_all, v_all

    def set_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
                  position: int = 0) -> None:
        """Overwrite a layer's K/V slice without touching ``length``.

        Used to seed the cache from a prefill; call :meth:`set_length`
        afterwards so ``forward`` can validate ``position_offset``.
        """
        self._write(layer_idx, position, k, v)
        self._len[layer_idx] = position + k.shape[-2]

    def set_length(self, length: int) -> None:
        if not 0 <= length <= self.max_len:
            raise ValueError(f"length {length} outside [0, {self.max_len}]")
        self._len = [length] * self.num_layers
        self.length = length

    # ------------------------------------------------------------------ cleanup

    def free(self) -> None:
        self._buf.free()

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass
