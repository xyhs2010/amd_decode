"""``Qwen35Torch`` with the K/V cache in pinned host memory.

Placeholder until the Qwen3.5 cache is ported: the host variant only changes
where the 8 attention layers' int8 K/V live, so it will inherit everything from
the reference implementation once the cache interface exists.
"""

from __future__ import annotations

from ..with_torch.model import Qwen35Torch


class HostQwen35(Qwen35Torch):
    """Qwen3.5 decode with int8 K/V in pinned, GPU-addressable host memory."""
