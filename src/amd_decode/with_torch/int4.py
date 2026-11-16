"""Int4 weight-only quantization that matches torchao's TINYGEMM path.

The prefill machine quantized Qwen3-8B with::

    Int4WeightOnlyConfig(
        group_size=128,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
    )

For a 2-D bf16 weight [out, in] torchao 0.18 computes asymmetric group-wise
qparams in bf16 (``_choose_qparams_affine_tinygemm``)::

    scale = clamp((max - min) / 15, min=finfo(bf16).smallest_normal)
    zero  = min + scale * 8
    q     = clamp(round((w - (zero - 8 * scale)) / scale), 0, 15)

and dequantizes as ``(q - 8) * scale + zero``. Qwen3-8B's linear shapes are all
multiples of 1024 in and 8 out, so torchao's padding step is a no-op and we can
skip it.

We only reproduce the qparams and the dequantized weights; the CUDA-only
``_convert_weight_to_int4pack`` bit layout is irrelevant for a torch GEMV.

:func:`pack_int4` / :func:`unpack_int4` are a storage codec for the cache: the
q values are 0..15, so two fit per byte with no change to the dequant maths.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 128
QMAX = 15
MID_POINT = 8.0


def quantize_int4(
    weight: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight the same way torchao's tinygemm path does.

    Returns ``(q, scale, zero)`` with shapes ``[out, in]``, ``[out, in/128]``,
    ``[out, in/128]``. ``q`` holds values 0..15.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(weight.shape)}")
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"in_features {weight.shape[-1]} is not divisible by group_size {group_size}"
        )
    if weight.dtype != torch.bfloat16:
        weight = weight.to(torch.bfloat16)

    out_features, in_features = weight.shape
    groups = weight.view(out_features, in_features // group_size, group_size)

    # Mirror torchao: all qparam arithmetic happens in bf16.
    min_val = groups.amin(dim=-1)
    max_val = groups.amax(dim=-1)
    eps = torch.finfo(torch.bfloat16).smallest_normal
    scale = ((max_val - min_val) / float(QMAX)).clamp_min(eps)
    zero = min_val + scale * MID_POINT

    # zero_point_domain == FLOAT: zero = min + scale * mid_point
    min_val_q = zero - scale * MID_POINT
    q = torch.round(
        (groups - min_val_q.unsqueeze(-1)) / scale.unsqueeze(-1)
    ).clamp(0, QMAX)

    return q.to(torch.int8).view(out_features, in_features), scale, zero


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack int8 q (values 0..15) two per byte along the last dim.

    Returns uint8 ``[..., in // 2]``; the even index is the low nibble.
    """
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
    s = scale.to(device=device).to(dtype).repeat_interleave(group_size, dim=1)
    z = zero.to(device=device).to(dtype).repeat_interleave(group_size, dim=1)
    w = (q.to(device=device).to(dtype) - MID_POINT) * s + z
    # torchao's reference dequant returns bf16 (the scale dtype); round to bf16
    # before the final cast so the effective weights match as closely as
    # possible. bf16 -> fp16 is exact for weight magnitudes.
    if dtype != torch.bfloat16:
        w = w.to(torch.bfloat16)
    return w.to(dtype)
