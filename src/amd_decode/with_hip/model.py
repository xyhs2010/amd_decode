"""Qwen3.5 decode with the hot operations on HIP kernels.

Every quantized projection stays packed and goes through :func:`q4_matmul`, and
single-token attention reads the fp16 K/V cache directly in
:func:`decode_attention`.  Two weights need more than int4:

* the ``linear_attn.*`` projections are kept **fp16 in VRAM** (~3 GiB): their
  int4 error is what accumulates in the 24 delta-net recurrent states, and it
  costs about 3 of 46 LongBench v2 samples;
* ``lm_head`` stays int4, but its top-k tokens are re-ranked with the fp16 rows
  read from host memory, which recovers the int4 logits error with no VRAM.

Prefill and multi-token chunks still use the reference ``with_torch`` attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..with_torch.model import Qwen35Torch, _quantizable
from ..with_torch.weights import LM_HEAD
from .ops import decode_attention, q4_matmul

RERANK_TOP = 64


class _RerankLmHead:
    """int4 lm_head + the fp16 rows still in host memory for re-ranking."""

    __slots__ = ("packed", "host")

    def __init__(self, packed, host) -> None:
        self.packed = packed
        self.host = host


class HipQwen35(Qwen35Torch):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # name -> fp16 weight on the GPU, built once per attention projection.
        self._fp16: dict[str, torch.Tensor] = {}

    def _prepare(self, name, tensor):
        if name == LM_HEAD:
            return _RerankLmHead(self._packed(name, tensor), tensor)
        if (tensor.dim() == 2 and tensor.shape[-1] % 128 == 0
                and ".linear_attn." in name):
            if name not in self._fp16:
                self._fp16[name] = tensor.to(self.device, self.dtype)
            return self._fp16[name]
        if _quantizable(tensor):
            return self._packed(name, tensor)
        return super()._prepare(name, tensor)

    def linear(self, x, w):
        if isinstance(w, _RerankLmHead):
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            logits = q4_matmul(x2, w.packed)
            top = torch.topk(logits[-1], RERANK_TOP).indices
            rows = w.host[top.cpu()].to(x.device, x.dtype)
            logits[-1].index_copy_(0, top, F.linear(x2[-1], rows))
            return logits.view(shape[:-1] + (logits.shape[-1],))
        if not isinstance(w, tuple):
            return F.linear(x, w)
        shape = x.shape
        out = q4_matmul(x.reshape(-1, shape[-1]), w)
        return out.view(shape[:-1] + (out.shape[-1],))

    def _decode_attn(self, q, kv, layer_idx):
        return decode_attention(q, kv, layer_idx)
