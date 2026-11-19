"""Python wrappers over the compiled HIP kernels (``_ops``)."""

from __future__ import annotations

import torch

from . import _ops

WARP_TOKENS = 512  # tokens per partial, mirrored in attn.hip
HEADS = 16         # query heads
KV_HEADS = 4
HEAD_DIM = 256


def q4_matmul(x: torch.Tensor, weight) -> torch.Tensor:
    """``x @ W.T`` with ``W`` stored as int4 ``(packed q, scale, zero)``.

    ``x`` is fp16 ``[M, K]``, ``q`` is uint8 ``[N, K/2]`` and ``scale``/``zero``
    are fp16 ``[N, K/128]``; returns fp16 ``[M, N]``.
    """
    q, scale, zero = weight
    x = x.contiguous()
    m, k = x.shape
    n = q.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _ops.q4_matmul(
        x.data_ptr(), q.data_ptr(), scale.data_ptr(), zero.data_ptr(),
        out.data_ptr(), m, n, k,
    )
    return out


def decode_attention(q: torch.Tensor, kv, layer_idx: int) -> torch.Tensor:
    """Decode attention over the fp16 K/V cache, before the output gate.

    ``q`` is fp16 ``[1, 16, 1, 256]``; returns fp16 ``[1, 1, 4096]``.
    """
    k, v = kv.prefix(layer_idx)
    n = k.shape[-2]
    if (n <= 0 or q.dtype is not torch.float16
            or tuple(q.shape) != (1, HEADS, 1, HEAD_DIM)):
        raise ValueError(
            f"decode_attention expects q [1, 16, 1, 256] fp16, got "
            f"{tuple(q.shape)} {q.dtype} (n={n})"
        )
    q = q.contiguous()
    parts = -(-n // WARP_TOKENS)
    po = torch.empty((HEADS, parts, HEAD_DIM), dtype=torch.float32,
                     device=q.device)
    ps = torch.empty((HEADS, parts, 2), dtype=torch.float32, device=q.device)
    out = torch.empty((HEADS, HEAD_DIM), dtype=torch.float16, device=q.device)
    _ops.attn_split(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), po.data_ptr(), ps.data_ptr(),
        n, k.stride(1), k.stride(2), v.stride(1), v.stride(2), parts,
    )
    _ops.attn_merge(po.data_ptr(), ps.data_ptr(), out.data_ptr(), parts)
    return out.view(1, 1, HEADS * HEAD_DIM)
