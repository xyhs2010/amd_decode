"""Rotary position embedding, including a transformers-compatible YaRN path.

The YaRN formula here mirrors ``transformers.modeling_rope_utils``:

  * ``inv_freq`` is interpolated for low frequencies and extrapolated for high
    frequencies, with a linear ramp in between;
  * ``attention_factor`` (a.k.a. mscale) defaults to ``0.1 * ln(factor) + 1``
    and is applied to cos/sin.

Keeping this identical matters: the KV cache was prefilled with YaRN (factor 8,
original_max_position_embeddings 32768), so decode must use the same map.
"""

from __future__ import annotations

import math

import torch

from .config import Qwen3Config


def _find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_mask(min_: float, max_: float, dim: int) -> torch.Tensor:
    if min_ == max_:
        max_ += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
    return linear.clamp(0, 1)


def rope_parameters(cfg: Qwen3Config) -> tuple[torch.Tensor, float]:
    """Return ``(inv_freq, attention_scaling)`` on CPU/float32."""
    dim = cfg.head_dim
    base = cfg.rope_theta
    scaling = cfg.rope_scaling or {}
    rope_type = scaling.get("type", scaling.get("rope_type", "default"))

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs

    if rope_type in (None, "default"):
        return inv_freq_extrapolation, 1.0
    if rope_type != "yarn":
        raise NotImplementedError(f"unsupported rope_scaling type: {rope_type!r}")

    factor = float(scaling["factor"])
    attention_factor = scaling.get("attention_factor")
    attention_factor = (
        0.1 * math.log(factor) + 1.0
        if attention_factor is None
        else float(attention_factor)
    )
    beta_fast = float(scaling.get("beta_fast", 32))
    beta_slow = float(scaling.get("beta_slow", 1))
    max_pos = int(scaling.get("original_max_position_embeddings", cfg.max_position_embeddings))

    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = _find_correction_range(beta_fast, beta_slow, dim, base, max_pos)
    extrapolation_factor = 1.0 - _linear_ramp_mask(low, high, dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    return inv_freq, attention_factor


def build_cos_sin(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``positions``: 1-D absolute positions. Returns cos/sin of shape [T, D]."""
    inv_freq = inv_freq.to(positions.device)
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling).to(dtype)
    sin = (emb.sin() * attention_scaling).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape [B, H, T, D] with cos/sin of shape [T, D]."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos[None, None, :, :] + rotated * sin[None, None, :, :]
