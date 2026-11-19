"""KV selection policies for sparse decode.

``SparseQwen35`` calls ``prepare(keys, length)`` once after the cache is seeded,
then ``select(q, n, layer_idx)`` on every decode layer.  Neither selection nor
index building runs during prefill.

``keys`` maps a global layer index to that layer's K ``[H, length, D]``; the
policies never see a cache object, so the offline index builder passes mmap'd
prefill tensors straight in.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


class Policy:
    """Decode-time KV selection.

    ``select`` returns int32 token ids ``[num_attention_heads, m]`` on the
    query's device, one row per query head; ``-1`` marks padding, which the
    kernel skips.  Any subset of ``[0, n)`` is legal - blocks, top-k tokens and
    IVF posting lists all fit - but a row's ids must be unique (a repeat is
    counted twice by the softmax) and must include the current token, or decode
    silently loses its input.
    """

    def prepare(self, keys: dict[int, torch.Tensor], length: int) -> None:
        """Build any index the policy needs from the seeded K of each layer."""

    def reset(self) -> None:
        """Forget state built for a previous cache."""

    def select(self, q: torch.Tensor, n: int, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError


class WindowPolicy(Policy):
    """The first ``sink`` tokens plus the most recent ``window``, nothing else.

    ``window >= n`` selects every token in order, which makes sparse decode
    match dense - use that to validate the gather path.
    """

    def __init__(self, window: int = 512, sink: int = 4) -> None:
        self.window = window
        self.sink = sink

    def select(self, q: torch.Tensor, n: int, layer_idx: int) -> torch.Tensor:
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
        self.centroids: dict[int, torch.Tensor] = {}  # [L] -> [H, nlist, D]
        self.order: dict[int, torch.Tensor] = {}      # [L] -> [H, n_index]
        self.start: dict[int, torch.Tensor] = {}      # [L] -> [H, nlist + 1]
        self.index_len = 0    # tokens [0, index_len) are in the index

    def reset(self) -> None:
        self.centroids, self.order, self.start = {}, {}, {}
        self.index_len = 0

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

    def prepare(self, keys: dict[int, torch.Tensor], length: int) -> None:
        from .kmeans import kmeans_gpu

        index_len = max(0, length - self.window)
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
        self.reset()
        if index_len <= 0:  # only the always-attended tail
            return
        self.index_len = index_len
        for layer_idx, k in keys.items():
            device = k.device if k.device.type == "cuda" else torch.device("cuda")
            heads, _, head_dim = k.shape
            centroids = torch.empty(heads, self.nlist, head_dim,
                                    dtype=torch.float16, device=device)
            order, start = [], []
            for hi in range(heads):
                # fp32: fp16 sums overflow here; 10 Lloyd rounds are within
                # 0.2% of the converged inertia
                x = k[hi, :index_len].to(device=device, dtype=torch.float16).float()
                labels, centroid = kmeans_gpu(x, self.nlist, max_iter=10)
                centroids[hi] = centroid
                order.append(torch.argsort(labels).to(torch.int32))
                counts = torch.bincount(labels, minlength=self.nlist)
                start.append(torch.cat(
                    [counts.new_zeros(1), counts.cumsum(0)]
                ).to(torch.int32))
                del x
            self.centroids[layer_idx] = centroids
            self.order[layer_idx] = torch.stack(order)
            self.start[layer_idx] = torch.stack(start)
        if self.index_file is not None:
            self.save()

    def select(self, q: torch.Tensor, n: int, layer_idx: int) -> torch.Tensor:
        assert q.shape[0] == 1 and q.shape[2] == 1
        rows = q.shape[1]
        # Everything the index does not cover: the seed tail plus every token
        # decoded so far.  Always attended, never a duplicate of an indexed id.
        tail = torch.arange(self.index_len, n, device=q.device, dtype=torch.int32)
        order = self.order.get(layer_idx)
        if order is None:
            return tail.unsqueeze(0).expand(rows, -1).contiguous()

        nprobe = min(self.nprobe or int(2 * math.sqrt(self.nlist)), self.nlist)
        centroids = self.centroids[layer_idx]
        heads = centroids.shape[0]
        group = rows // heads
        qg = q.reshape(heads, group, q.shape[-1]).float()
        scores = torch.einsum("hgd,hkd->hgk", qg, centroids.float())
        top = scores.topk(nprobe, dim=-1).indices           # [H, group, nprobe]
        # unsqueeze + gather keeps the row per K/V head; ``start[:, top]``
        # would broadcast into [H, H, group, nprobe] instead.
        start = self.start[layer_idx].unsqueeze(1).expand(-1, group, -1)
        order = order.unsqueeze(1).expand(-1, group, -1)
        st = start.gather(2, top)
        length = start.gather(2, top + 1) - st              # [H, group, nprobe]
        off = torch.arange(int(length.max()), device=q.device)
        keep = off < length.unsqueeze(-1)
        width = keep.shape[-1]
        pos = (st.unsqueeze(-1) + off).clamp_max(order.shape[2] - 1)
        ids = order.gather(2, pos.reshape(heads, group, nprobe * width))
        keep = keep.reshape(heads, group, nprobe * width)
        ids = torch.where(keep, ids, -1).reshape(rows, -1)
        return torch.cat([ids, tail.unsqueeze(0).expand(rows, -1)], 1).contiguous()
