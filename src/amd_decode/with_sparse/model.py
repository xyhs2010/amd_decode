"""``HipQwen3`` whose decode attention reads only the policy's selected tokens.

Prefill, the FFN kernels, weight streaming and the q8 cache are all inherited;
only ``_decode_attn`` changes, so the dense backend stays the reference.
"""

from __future__ import annotations

import torch.nn.functional as F

from ..with_hip import HipQwen3
from .ops import sparse_attention
from .policy import WindowPolicy


class SparseQwen3(HipQwen3):
    def __init__(self, model_dir, policy=None, **kwargs) -> None:
        super().__init__(model_dir, **kwargs)
        self.policy = policy if policy is not None else WindowPolicy()

    def set_kv_cache(self, kv) -> None:
        super().set_kv_cache(kv)
        if kv is not None:
            self.policy.prepare(kv)

    def _decode_attn(self, q, kv, layer_idx, p):
        ids = self.policy.select(q, kv, layer_idx)
        out = sparse_attention(q, kv, layer_idx, ids)
        return F.linear(out, p["self_attn.o_proj.weight"])
