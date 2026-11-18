"""HIP kernel backend for Qwen3.5 decode.

Placeholder: the kernels (int4 GEMV, fused q8 decode attention, DeltaNet
state step) are ported after the reference path and the cache are validated.
Until then this is the same W4A16 torch implementation.
"""

from __future__ import annotations

from ..with_torch.model import Qwen35Torch


class HipQwen35(Qwen35Torch):
    """Qwen3.5 decode with the hot operations on HIP kernels."""
