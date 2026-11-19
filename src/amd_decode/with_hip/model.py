"""Qwen3.5 decode with the int4 GEMV on a HIP kernel.

Every quantized projection - attention, delta-net, MLP and ``lm_head`` - stays
packed and goes through :func:`q4_matmul` instead of being dequantised to fp16
every step.  Attention and the cache are still the reference ``with_torch``
implementation; the fused device-attention kernel is not ported yet.
"""

from __future__ import annotations

import torch.nn.functional as F

from ..with_torch.model import Qwen35Torch, _quantizable
from .ops import q4_matmul


class HipQwen35(Qwen35Torch):
    def _prepare(self, name, tensor):
        if _quantizable(tensor):
            return self._packed(name, tensor)
        return super()._prepare(name, tensor)

    def linear(self, x, w):
        if not isinstance(w, tuple):
            return F.linear(x, w)
        shape = x.shape
        out = q4_matmul(x.reshape(-1, shape[-1]), w)
        return out.view(shape[:-1] + (out.shape[-1],))
