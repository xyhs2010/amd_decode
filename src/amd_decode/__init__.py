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

    def load_prefill(self, saved: Any = None, max_len: int | None = None) -> None:
        """Build the cache and seed it from an offline prefill; ``None`` clears it."""

    def set_hook(self, hook: LayerHook | None) -> None:
        """Install ``hook(layer_idx, output)``; its return value is the feature."""

    def forward(self, input_ids, position_offset: int | None = None) -> torch.Tensor:
        """Run one pass.

        ``position_offset=None`` means "start at the length of the cache loaded
        by :meth:`load_prefill`", or 0 when no cache is loaded.
        """

    def get_features(self) -> dict[int, Any]:
        """Features captured by the most recent ``forward``."""


def main() -> None:
    print("Hello from amd-decode!")
