"""Python wrappers over the compiled HIP kernels (``_ops``)."""

from __future__ import annotations

import torch

from . import _ops

WARP_TOKENS = 256  # tokens per partial, mirrored in attn.hip


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


def q8_attention(q: torch.Tensor, kv, layer_idx: int) -> torch.Tensor:
    """Decode attention over the cached int8 K/V, before ``o_proj``.

    ``q`` is fp16 ``[1, 32, 1, 128]``; returns fp16 ``[1, 1, 4096]``.
    """
    view = kv.view(layer_idx)
    n = view.n
    heads = view.k.shape[0]
    if (n <= 0 or q.dtype is not torch.float16 or q.shape[0] != 1
            or q.shape[2] != 1 or heads != 8 or q.shape[1] != 4 * heads
            or view.k.shape[2] != 128):
        raise ValueError(
            f"q8_attention expects q [1, 32, 1, 128] fp16 with 8 KV heads, got "
            f"{tuple(q.shape)} {q.dtype} and {heads} KV heads (n={n})"
        )
    q = q.contiguous()
    parts = -(-n // WARP_TOKENS)
    rows = heads * 4
    po = torch.empty((rows, parts, 128), dtype=torch.float32, device=q.device)
    ps = torch.empty((rows, parts, 2), dtype=torch.float32, device=q.device)
    out = torch.empty((rows, 128), dtype=torch.float16, device=q.device)
    kb = view.k_block
    _ops.attn_split(
        q.data_ptr(), view.k.data_ptr(), view.v.data_ptr(),
        view.k_tail.data_ptr(), view.v_scale.data_ptr(),
        0 if kb is None else kb.data_ptr(), po.data_ptr(), ps.data_ptr(),
        n, view.prefix, view.block_size, view.k.shape[1],
        0 if kb is None else kb.shape[1], parts,
    )
    _ops.attn_merge(po.data_ptr(), ps.data_ptr(), out.data_ptr(), parts)
    return out.view(1, 1, rows * 128)
