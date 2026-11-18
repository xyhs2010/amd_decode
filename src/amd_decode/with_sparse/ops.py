"""Python wrapper over the compiled sparse attention kernels (``_sparse_ops``)."""

from __future__ import annotations

import torch

from . import _sparse_ops

WARP_TOKENS = 256  # id slots per partial, mirrored in attn_sparse.hip


def sparse_attention(q: torch.Tensor, kv, layer_idx: int,
                     ids: torch.Tensor) -> torch.Tensor:
    """Decode attention over ``ids`` only, before ``o_proj``.

    ``q`` is fp16 ``[1, 32, 1, 128]``; ``ids`` is int32 ``[32, m]`` with ``-1``
    padding, one row per query head; returns fp16 ``[1, 1, 4096]``.
    """
    view = kv.view(layer_idx)
    rows = 32
    m = ids.shape[1]
    if (m <= 0 or q.dtype is not torch.float16 or tuple(q.shape) != (1, rows, 1, 128)
            or view.k.shape[2] != 128):
        raise ValueError(
            f"sparse_attention expects q [1, 32, 1, 128] fp16, got "
            f"{tuple(q.shape)} {q.dtype} (m={m})"
        )
    if ids.dtype is not torch.int32 or tuple(ids.shape) != (rows, m):
        raise ValueError(
            f"ids must be int32 [{rows}, m], got {ids.dtype} {tuple(ids.shape)}"
        )
    q = q.contiguous()
    ids = ids.contiguous()
    parts = -(-m // WARP_TOKENS)
    po = torch.empty((rows, parts, 128), dtype=torch.float32, device=q.device)
    ps = torch.empty((rows, parts, 2), dtype=torch.float32, device=q.device)
    out = torch.empty((rows, 128), dtype=torch.float16, device=q.device)
    kb = view.k_block
    _sparse_ops.attn_split(
        q.data_ptr(), view.k.data_ptr(), view.v.data_ptr(),
        view.k_tail.data_ptr(), view.v_scale.data_ptr(),
        0 if kb is None else kb.data_ptr(), ids.data_ptr(),
        po.data_ptr(), ps.data_ptr(),
        m, view.prefix, view.block_size, view.k.shape[1],
        0 if kb is None else kb.shape[1], parts, m,
    )
    _sparse_ops.attn_merge(po.data_ptr(), ps.data_ptr(), out.data_ptr(), parts)
    return out.view(1, 1, rows * 128)
