"""Python wrappers over the compiled sparse attention kernels (``_sparse_ops``)."""

from __future__ import annotations

import torch

from . import _sparse_ops

WARP_TOKENS = 256  # id slots per partial, mirrored in attn_sparse.hip
HEADS = 16         # query heads, one id row each
HEAD_DIM = 256


def sparse_attention(q: torch.Tensor, kv, layer_idx: int,
                     ids: torch.Tensor) -> torch.Tensor:
    """Decode attention over ``ids`` only, before the output gate.

    ``q`` is fp16 ``[1, 16, 1, 256]``; ``ids`` is int32 ``[16, m]`` with ``-1``
    padding, one row per query head; returns fp16 ``[1, 1, 4096]``.
    """
    m = ids.shape[1]
    if (m <= 0 or q.dtype is not torch.float16
            or tuple(q.shape) != (1, HEADS, 1, HEAD_DIM)):
        raise ValueError(
            f"sparse_attention expects q [1, 16, 1, 256] fp16, got "
            f"{tuple(q.shape)} {q.dtype} (m={m})"
        )
    if ids.dtype is not torch.int32 or tuple(ids.shape) != (HEADS, m):
        raise ValueError(
            f"ids must be int32 [{HEADS}, m], got {ids.dtype} {tuple(ids.shape)}"
        )
    k, v = kv.prefix(layer_idx)  # [1, 4, n, 256] fp16, strided over max_len
    q = q.contiguous()
    ids = ids.contiguous()
    parts = -(-m // WARP_TOKENS)
    po = torch.empty((HEADS, parts, HEAD_DIM), dtype=torch.float32, device=q.device)
    ps = torch.empty((HEADS, parts, 2), dtype=torch.float32, device=q.device)
    out = torch.empty((HEADS, HEAD_DIM), dtype=torch.float16, device=q.device)
    _sparse_ops.attn_split(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        k.stride(1), v.stride(1), ids.data_ptr(),
        po.data_ptr(), ps.data_ptr(), parts, m,
    )
    _sparse_ops.attn_merge(po.data_ptr(), ps.data_ptr(), out.data_ptr(), parts)
    return out.view(1, 1, HEADS * HEAD_DIM)
