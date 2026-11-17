"""Qwen3 decode with HIP kernels for the hot operations.

Only the FFN matmuls are on the GPU kernel so far; attention, weight streaming
and the q8 KV cache are the reference implementation.
"""

from .model import HipQwen3

__all__ = ["HipQwen3"]
