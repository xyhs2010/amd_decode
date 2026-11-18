"""KV selection policies for sparse decode.

``SparseQwen3`` calls ``prepare(kv)`` once after the cache is seeded, then
``select(q, kv, layer_idx)`` on every decode layer.  Selection never runs during
prefill.
"""

from __future__ import annotations

import torch
from ..with_hip.kv_cache import HipKVCache


class Policy:
    """Decode-time KV selection.

    ``select`` returns int32 token ids ``[num_kv_heads, m]`` on the query's
    device; ``-1`` marks padding, which the kernel skips.  Any subset of
    ``[0, n)`` is legal - blocks, top-k tokens and IVF posting lists all fit -
    but the ids must be unique (a repeat is counted twice by the softmax) and
    must include the current token, or decode silently loses its input.
    """

    def prepare(self, kv: HipKVCache) -> None:
        """Build any index the policy needs; called once per attached cache."""

    def select(self, q, kv: HipKVCache, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError


class WindowPolicy(Policy):
    """The first ``sink`` tokens plus the most recent ``window``, nothing else.

    ``window >= length`` selects every token in order, which makes sparse decode
    bitexact with dense - use that to validate the gather path.
    """

    def __init__(self, window: int = 512, sink: int = 4) -> None:
        self.window = window
        self.sink = sink

    def select(self, q, kv: HipKVCache, layer_idx: int) -> torch.Tensor:
        n = kv.view(layer_idx).n
        start = max(0, n - self.window)
        # ``min`` keeps the sink from overlapping the window when the cache is
        # still short; a duplicate id would be counted twice by the softmax.
        head = min(self.sink, start)
        ids = torch.cat(
            [
                torch.arange(head, device=q.device),
                torch.arange(start, n, device=q.device),
            ]
        ).to(torch.int32)
        return ids.expand(kv.num_kv_heads, -1).contiguous()


class NaiveIvfPolicy(Policy):
    def __init__(self, nlist: int, nprobe: int) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.num_layers = self.num_kv_heads = self.head_dim = -1

    def prepare(self, kv: HipKVCache) -> None:
        from .kmeans import kmeans_gpu

        self.num_layers = kv.num_layers
        self.num_kv_heads = kv.num_kv_heads
        self.head_dim = kv.head_dim
        self.centroids = [
            torch.Tensor(self.num_kv_heads, self.nlist, self.head_dim, device="cuda")
            for _ in range(self.num_layers)
        ]
        for li in range(self.num_layers):
            k, _ = kv.dequantize(li)
            for hi in range(self.num_kv_heads):
                self.centroids[li][hi] = kmeans_gpu(k[0, hi], self.nprobe)

    def select(self, q, kv: HipKVCache, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError
