"""Int4 weight-only quantization for the W4A16 decode path.

Weights are stored as asymmetric 4-bit groups of 128; activations stay fp16 and
the GEMV accumulates in fp32.  There is no torchao compatibility requirement on
this branch, so the qparams are computed in fp32 and stored as fp16 directly
(no bf16 round trip).

The q values are 0..15, so :func:`pack_int4` / :func:`unpack_int4` are a
lossless storage codec: two values per byte.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 128
QMAX = 15
MID_POINT = 8.0


def quantize_int4(
    weight: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight into ``(q, scale, zero)``.

    ``q`` is int8 with values 0..15, shaped ``[out, in]``; ``scale`` and
    ``zero`` are fp16 ``[out, in / 128]``.  Dequantization is
    ``(q - 8) * scale + zero``.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(weight.shape)}")
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"in_features {weight.shape[-1]} is not divisible by group_size {group_size}"
        )

    out_features, in_features = weight.shape
    groups = weight.float().view(out_features, in_features // group_size, group_size)

    min_val = groups.amin(dim=-1)
    max_val = groups.amax(dim=-1)
    scale = ((max_val - min_val) / float(QMAX)).clamp_min(torch.finfo(torch.float32).tiny)
    zero = min_val + scale * MID_POINT

    q = torch.round(
        (groups - (zero - scale * MID_POINT).unsqueeze(-1)) / scale.unsqueeze(-1)
    ).clamp(0, QMAX)

    return (
        q.to(torch.int8).view(out_features, in_features),
        scale.to(torch.float16),
        zero.to(torch.float16),
    )


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack int8 q (values 0..15) two per byte along the last dim."""
    if q.shape[-1] % 2:
        raise ValueError(f"last dim {q.shape[-1]} is odd; cannot pack int4")
    u = q.to(torch.uint8)
    return (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`; returns int8 ``[..., in_features]``."""
    u = packed.to(torch.uint8)
    out = torch.empty(u.shape[:-1] + (in_features,), dtype=torch.uint8,
                      device=u.device)
    out[..., 0::2] = u & 0x0F
    out[..., 1::2] = u >> 4
    # Values are 0..15, so the uint8 and int8 bit patterns agree.
    return out.view(torch.int8)


def dequantize_int4(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float16,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize ``q``/``scale``/``zero`` to a dense weight on ``device``."""
    out_features, in_features = q.shape
    groups = in_features // group_size
    q = q.to(device=device).to(dtype).view(out_features, groups, group_size)
    s = scale.to(device=device).to(dtype).unsqueeze(-1)
    z = zero.to(device=device).to(dtype).unsqueeze(-1)
    return ((q - MID_POINT) * s + z).view(out_features, in_features)
