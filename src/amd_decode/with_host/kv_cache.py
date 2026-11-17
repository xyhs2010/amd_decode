"""Host-memory extension of the with_torch q8 KV cache.

``KVCache`` owns all append/dequantise logic; this subclass only changes where
the buffers live.  The GPU addresses the pinned host buffers directly, so the
cache costs no VRAM and is read/written over PCIe.
"""

from __future__ import annotations

import math

from ..with_torch.kv_cache import KVCache
from .host_memory import HOST_MALLOC_FLAGS, HostBuffer, Hip


class HostKVCache(KVCache):
    """``KVCache`` whose buffers live in pinned, GPU-addressable host memory."""

    def __init__(self, *args, hip: Hip | None = None,
                 flags: int = HOST_MALLOC_FLAGS, **kwargs) -> None:
        self._hip = hip or Hip()
        self._flags = flags
        self._owners: list[HostBuffer] = []
        super().__init__(*args, **kwargs)

    def _allocate(self, shape: tuple[int, ...], dtype):
        buf = HostBuffer(self._hip, math.prod(shape), dtype, self.device,
                         flags=self._flags)
        self._owners.append(buf)
        return buf.cuda.view(shape)

    def free(self) -> None:
        for buf in self._owners:
            buf.free()
        self._owners.clear()

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass
