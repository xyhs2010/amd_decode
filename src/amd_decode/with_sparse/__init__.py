"""Sparse decode on top of the HIP backend: a policy picks the KV to attend."""

from .model import SparseQwen3
from .policy import NaiveIvfPolicy, WindowPolicy

__all__ = ["NaiveIvfPolicy", "SparseQwen3", "WindowPolicy"]
