"""Qwen3.5 decode with the hot operations on HIP kernels.

Every quantized projection - attention, delta-net, MLP and ``lm_head`` - stays
packed and goes through :func:`q4_matmul` instead of being dequantised to fp16
every step, and single-token attention reads the fp16 K/V cache directly in
:func:`decode_attention`.  Prefill and multi-token chunks still use the
reference ``with_torch`` attention.
"""

from __future__ import annotations

import torch.nn.functional as F

from ..with_torch.model import Qwen35Torch, _quantizable
from .ops import decode_attention, q4_matmul


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

    def _decode_attn(self, q, kv, layer_idx):
        return decode_attention(q, kv, layer_idx)
