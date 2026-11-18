"""KV selection policies for sparse decode.

``SparseQwen3`` calls ``prepare(kv)`` once after the cache is seeded, then
``select(q, kv, layer_idx)`` on every decode layer.  Selection never runs during
prefill.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ..with_hip.kv_cache import HipKVCache


class Policy:
    """Decode-time KV selection.

    ``select`` returns int32 token ids ``[num_attention_heads, m]`` on the
    query's device, one row per query head; ``-1`` marks padding, which the
    kernel skips.  Any subset of ``[0, n)`` is legal - blocks, top-k tokens and
    IVF posting lists all fit - but a row's ids must be unique (a repeat is
    counted twice by the softmax) and must include the current token, or decode
    silently loses its input.
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
        return ids.expand(q.shape[1], -1).contiguous()


class NaiveIvfPolicy(Policy):
    """Cluster each (layer, K/V head)'s keys and attend the nearest clusters.

    ``prepare`` k-means the keys into ``nlist`` centroids and keeps the token
    ids grouped by cluster.  ``select`` keeps the ``nprobe`` clusters whose
    centroid scores highest with the query.

    The last ``window`` seeded tokens stay out of the index, and so does every
    token decoded afterwards; everything after ``index_len`` is always attended,
    however long decode runs.  Index and tail are disjoint by construction, so
    no id is ever counted twice - and no token falls out of reach either.
    """

    def __init__(self, nlist: int | None = None, nprobe: int | None = None,
                 window: int = 512,
                 index_file: str | Path | None = None) -> None:
        # ``nlist=None`` means 32 indexed tokens per cluster; ``nprobe=None``
        # means ``2 * sqrt(nlist)`` clusters per query head.
        self.nlist = nlist
        self.nprobe = nprobe
        self.window = window
        self.index_file = None if index_file is None else Path(index_file)
        self.centroids = []   # [L] -> [H, nlist, D]
        self.order = []       # [L] -> [H, n_index]: token ids grouped by cluster
        self.start = []       # [L] -> [H, nlist + 1]: cluster offsets
        self.index_len = 0    # tokens [0, index_len) are in the index

    def save(self) -> None:
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "nlist": self.nlist, "window": self.window,
            "index_len": self.index_len,
            "centroids": self.centroids, "order": self.order, "start": self.start,
        }, self.index_file)

    def load(self) -> None:
        data = torch.load(self.index_file)
        if self.nlist is not None and self.nlist != data["nlist"]:
            raise ValueError(
                f"{self.index_file} holds nlist={data['nlist']}, "
                f"policy wants nlist={self.nlist}"
            )
        if self.window != data["window"]:
            raise ValueError(
                f"{self.index_file} holds window={data['window']}, "
                f"policy wants window={self.window}"
            )
        self.nlist = data["nlist"]
        self.centroids = data["centroids"]
        self.order = data["order"]
        self.start = data["start"]
        self.index_len = data["index_len"]

    def prepare(self, kv: HipKVCache) -> None:
        from .kmeans import kmeans_gpu

        index_len = max(0, kv.view(0).n - self.window)
        if self.index_file is not None and self.index_file.exists():
            self.load()
            if self.index_len != index_len:
                raise ValueError(
                    f"{self.index_file} was built for index_len {self.index_len}, "
                    f"this cache gives {index_len}"
                )
            return

        if self.nlist is None:
            self.nlist = max(1, index_len // 32)
        self.centroids = [
            torch.empty(kv.num_kv_heads, self.nlist, kv.head_dim,
                        dtype=torch.float16, device="cuda")
            for _ in range(kv.num_layers)
        ]
        self.order, self.start, self.index_len = [], [], 0
        if kv.view(0).n <= self.window:  # only the always-attended tail
            return
        self.index_len = kv.view(0).n - self.window
        for li in range(kv.num_layers):
            k, _ = kv.dequantize(li)
            order, start = [], []
            for hi in range(kv.num_kv_heads):
                # fp32: fp16 sums overflow here; 10 Lloyd rounds are within
                # 0.2% of the converged inertia
                labels, centroids = kmeans_gpu(
                    k[0, hi, :self.index_len].float(), self.nlist, max_iter=10
                )
                self.centroids[li][hi] = centroids
                order.append(torch.argsort(labels).to(torch.int32))
                counts = torch.bincount(labels, minlength=self.nlist)
                start.append(torch.cat(
                    [counts.new_zeros(1), counts.cumsum(0)]
                ).to(torch.int32))
            self.order.append(torch.stack(order))
            self.start.append(torch.stack(start))
        if self.index_file is not None:
            self.save()

    def select(self, q: torch.Tensor, kv: HipKVCache, layer_idx: int) -> torch.Tensor:
        assert q.shape[0] == 1 and q.shape[2] == 1
        n, rows = kv.view(layer_idx).n, q.shape[1]
        # Everything the index does not cover: the seed tail plus every token
        # decoded so far.  Always attended, never a duplicate of an indexed id.
        tail = torch.arange(self.index_len, n, device=q.device, dtype=torch.int32)
        if not self.order:
            return tail.unsqueeze(0).expand(rows, -1).contiguous()

        nprobe = self.nprobe or int(2 * math.sqrt(self.nlist))
        nprobe = min(nprobe, self.nlist)
        group = rows // kv.num_kv_heads
        q = q.view(kv.num_kv_heads, group, kv.head_dim)
        scores = torch.einsum("hgd,hkd->hgk", q, self.centroids[layer_idx])
        top = scores.topk(nprobe, dim=-1).indices           # [H, group, nprobe]
        # unsqueeze + gather keeps the row per K/V head; ``start[:, top]``
        # would broadcast into [H, H, group, nprobe] instead.
        start = self.start[layer_idx].unsqueeze(1).expand(-1, group, -1)
        order = self.order[layer_idx].unsqueeze(1).expand(-1, group, -1)
        st = start.gather(2, top)
        length = start.gather(2, top + 1) - st              # [H, group, nprobe]
        off = torch.arange(int(length.max()), device=q.device)
        keep = off < length.unsqueeze(-1)
        width = keep.shape[-1]
        pos = (st.unsqueeze(-1) + off).clamp_max(order.shape[2] - 1)
        ids = order.gather(2, pos.reshape(kv.num_kv_heads, group, nprobe * width))
        keep = keep.reshape(kv.num_kv_heads, group, nprobe * width)
        ids = torch.where(keep, ids, -1).reshape(rows, -1)
        return torch.cat([ids, tail.unsqueeze(0).expand(rows, -1)], 1).contiguous()
