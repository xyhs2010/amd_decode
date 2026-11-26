import torch
from .int4 import Q4Weight


def rms(x: torch.Tensor, weight: torch.Tensor, rms_norm_eps=1e-6) -> torch.Tensor:
    """Zero-centred RMSNorm (input/post/final/q_norm/k_norm)."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + rms_norm_eps)
    return (xf * (1.0 + weight.float())).to(x.dtype)


def rms_gated(
    x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, rms_norm_eps=1e-6
) -> torch.Tensor:
    """DeltaNet's gated RMSNorm: norm, fp16 weight, then silu(z)."""
    import torch.nn.functional as F

    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + rms_norm_eps)
    xf = weight * xf.to(x.dtype)
    return (xf * F.silu(gate.float())).to(x.dtype)


def linear_int4(x: torch.Tensor, w: Q4Weight) -> torch.Tensor:
    import torch.nn.functional as F

    w = w.dequant()
    return F.linear(x, w)
