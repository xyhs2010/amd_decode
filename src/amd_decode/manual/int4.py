from ..with_torch.int4 import GROUP_SIZE, QMAX, MID_POINT
import torch


class Q4Weight:
    nsteps = 10

    def __init__(self, weight: torch.Tensor, group_size: int = GROUP_SIZE):
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
        self.group_size = group_size
        self.odim, self.idim = weight.shape
        self.int4_w, self.zero_points, self.scales = self.quant(weight)

    def to_gpu(self):
        self.int4_w = self.int4_w.to("cuda")
        self.zero_points = self.zero_points.to("cuda")
        self.scales = self.scales.to("cuda")

    def dequant(self):
        qweight = self.unpack_int4(self.int4_w, self.idim)
        qweight = qweight.to(torch.float16).view(
            self.odim, self.idim // self.group_size, self.group_size
        )
        return ((qweight - MID_POINT) * self.scales + self.zero_points).view(
            self.odim, self.idim
        )

    def quant(self, weight: torch.Tensor):
        from math import log10

        odim, idim, group_size = self.odim, self.idim, self.group_size
        groups = weight.cuda().float().view(odim, idim // group_size, group_size)
        zero_points = groups.mean(dim=-1, keepdim=True).half()

        min_scales = groups.std(dim=-1, keepdim=True).half() * 3 / QMAX
        min_scales = min_scales.clamp_min(torch.finfo(torch.float16).tiny)
        acc = torch.logspace(0, log10(3), self.nsteps).half().cuda()
        scale_pool = min_scales[..., None] * acc  # 前 3 轴为 groups 形状

        group_ext = groups[..., None]
        zero_ext = zero_points[..., None]
        scale_w = (group_ext - zero_ext) / scale_pool + MID_POINT
        quant_w = torch.round(scale_w).clamp(0, QMAX)
        dequant_w = (quant_w - MID_POINT) * scale_pool + zero_ext
        diff = torch.mean((group_ext - dequant_w) ** 2, dim=2)
        arg_scale = torch.argmin(diff, dim=-1, keepdim=True)

        scales = torch.gather(scale_pool[:, :, 0, :], dim=2, index=arg_scale)

        scale_w = (groups - zero_points) / scales + MID_POINT
        quant_w = torch.round(scale_w).clamp(0, QMAX)
        int4_w = self.pack_int4(quant_w.to(torch.int8).view(odim, idim))
        return int4_w, zero_points, scales

    @staticmethod
    def pack_int4(q: torch.Tensor) -> torch.Tensor:
        """Pack int8 q (values 0..15) two per byte along the last dim."""
        if q.shape[-1] % 2:
            raise ValueError(f"last dim {q.shape[-1]} is odd; cannot pack int4")
        u = q.to(torch.uint8)
        return (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()

    @staticmethod
    def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
        """Inverse of :func:`pack_int4`; returns int8 ``[..., in_features]``."""
        u = packed.to(torch.uint8)
        out = torch.empty(
            u.shape[:-1] + (in_features,), dtype=torch.uint8, device=u.device
        )
        out[..., 0::2] = u & 0x0F
        out[..., 1::2] = u >> 4
        # Values are 0..15, so the uint8 and int8 bit patterns agree.
        return out.view(torch.int8)
