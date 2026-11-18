"""Partial rotary position embedding for Qwen3.5.

Only the first 64 of the 256 head dimensions are rotated
(``partial_rotary_factor = 0.25``), with ``rope_theta = 1e7`` and no YaRN:
the model is natively trained for 262144 tokens, and this branch caps the
context there.  The text-only path uses identical T/H/W positions, in which
case the interleaved mRoPE layout degenerates to this standard 1-D map.
"""

from __future__ import annotations

import torch

from .config import Qwen35Config


def build_cos_sin(
    positions: torch.Tensor,
    cfg: Qwen35Config,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``positions``: 1-D absolute positions. Returns cos/sin of shape [T, 64]."""
    dim = cfg.rope_dim
    inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    freqs = positions.to(torch.float32)[:, None] * inv_freq.to(positions.device)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Rotate the first ``rope_dim`` channels of ``x`` [B, H, T, D]."""
    half = rope_dim // 2
    x_rot = x[..., :rope_dim]
    rotated = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
    out = x.clone()
    out[..., :rope_dim] = x_rot * cos[None, None] + rotated * sin[None, None]
    return out
