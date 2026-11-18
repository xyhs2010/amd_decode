"""Sparse decode on top of the HIP backend: a policy picks the KV to attend."""

from .model import SparseQwen3
from .policy import WindowPolicy

__all__ = ["SparseQwen3", "WindowPolicy"]
