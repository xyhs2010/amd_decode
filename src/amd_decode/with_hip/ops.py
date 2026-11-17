"""Python wrappers over the compiled HIP kernels (``_ops``)."""

from __future__ import annotations

import torch

from . import _ops


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
