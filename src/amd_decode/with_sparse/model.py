"""``HipQwen35`` whose decode attention reads only the policy's selected tokens.

Weight streaming, the FFN kernels, DeltaNet and the cache are inherited; only
``_decode_attn`` changes, so dense decode stays the reference.
"""

from __future__ import annotations

from ..with_hip import HipQwen35
from .ops import sparse_attention
from .policy import WindowPolicy


class SparseQwen35(HipQwen35):
    def __init__(self, model_dir, policy=None, **kwargs) -> None:
        super().__init__(model_dir, **kwargs)
        self.policy = policy if policy is not None else WindowPolicy()

    def load_prefill(self, saved=None, max_len: int | None = None) -> None:
        super().load_prefill(saved, max_len)
        if saved is None:
            self.policy.reset()
            return
        keys = {li: self._kv.prefix(li)[0][0] for li in self._kv.layers}
        self.policy.prepare(keys, self._kv.length)

    def _decode_attn(self, q, kv, layer_idx):
        ids = self.policy.select(q, kv.length, layer_idx)
        return sparse_attention(q, kv, layer_idx, ids)
