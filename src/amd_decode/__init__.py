"""The decode-backend protocol shared by every implementation.

A backend is any object implementing :class:`DecodeModel`; ``with_torch``,
``with_host`` and future kernel backends all speak it, so ``align`` and the
evaluation harness only depend on these four operations.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch

LayerHook = Callable[[int, torch.Tensor], Any]


@runtime_checkable
class DecodeModel(Protocol):
    """Minimal interface for a decode backend."""

    def set_kv_cache(self, kv: Any) -> None:
        """Attach the cache that ``forward`` should use (or ``None``)."""

    def set_hook(self, hook: LayerHook | None) -> None:
        """Install ``hook(layer_idx, output)``; its return value is the feature."""

    def forward(self, input_ids, position_offset: int | None = None) -> torch.Tensor:
        """Run one pass.

        ``position_offset=None`` means "start at the attached cache's current
        ``length``", or 0 when no cache is attached.
        """

    def get_features(self) -> dict[int, Any]:
        """Features captured by the most recent ``forward``."""


def main() -> None:
    print("Hello from amd-decode!")
