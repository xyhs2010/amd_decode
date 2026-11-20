#!/usr/bin/env python
"""Build the sparse-attention IVF indices for saved prefill caches, offline.

Run from the repo root::

    PYTHONPATH=src .venv/bin/python scripts/build_ivf_index.py --lengths 32768 --limit 2

No model weights are loaded: the K of every full-attention layer is read
straight from the saved cache (a bf16 mmap view) and ``NaiveIvfPolicy`` writes
``<output>/<sample_id>.pt``.  ``nlist`` defaults to 32 indexed tokens per
cluster, so it follows the context length.  Existing files are skipped, so an
interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from amd_decode.utils.prefill_cache import SavedKVCache
from amd_decode.with_sparse.kmeans import _assign
from amd_decode.with_sparse.policy import NaiveIvfPolicy


def rebucket(saved: SavedKVCache, old_path: Path, out: Path) -> NaiveIvfPolicy:
    """Reuse the centroids of an existing index, re-assign this cache's keys."""
    old = torch.load(old_path)
    policy = NaiveIvfPolicy(old["nlist"], window=old["window"], index_file=out)
    policy.reset()
    policy.index_len = saved.seq_len - old["window"]
    for layer_idx in _kv_layers(saved):
        centroids = old["centroids"][layer_idx]
        k = saved.layer_tensors(layer_idx)["k"][0]
        order, start = [], []
        for head in range(k.shape[0]):
            x = k[head, : policy.index_len].to("cuda", torch.float16).float()
            labels = _assign(x, centroids[head].to("cuda", torch.float32))
            del x
            order.append(labels.argsort().to(torch.int32))
            counts = torch.bincount(labels, minlength=policy.nlist)
            start.append(torch.cat([counts.new_zeros(1), counts.cumsum(0)])
                         .to(torch.int32))
        policy.centroids[layer_idx] = centroids
        policy.order[layer_idx] = torch.stack(order)
        policy.start[layer_idx] = torch.stack(start)
    policy.save()
    return policy


def _parse_lengths(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    return [int(part) for part in value.replace(",", " ").split()]


def _kv_layers(saved: SavedKVCache) -> list[int]:
    return [
        i for i, layer_type in enumerate(saved.meta["layer_types"])
        if layer_type == "full_attention"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root",
                        default="data/kv_cache/Qwen3.5-9B/bfloat16/jsonl")
    parser.add_argument("--output", default="data/ivf")
    parser.add_argument("--lengths", type=_parse_lengths, default=None,
                        metavar="N[,N...]|all")
    parser.add_argument("--nlist", type=int, default=None,
                        help="clusters per (layer, KV head); default 32 tokens "
                             "per cluster")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebucket-from", default=None, metavar="DIR",
                        help="reuse the centroids in DIR/<sample_id>.pt and only "
                             "re-assign the tokens (no k-means)")
    args = parser.parse_args()

    rows = []
    for path in Path(args.cache_root).glob("L*/*/*/meta.json"):
        meta = json.loads(path.read_text())
        if args.lengths is not None and int(meta["length"]) not in args.lengths:
            continue
        rows.append((int(meta["length"]), path.parent.name, path.parent))
    rows.sort()
    if args.limit:
        rows = rows[: args.limit]

    for number, (_, name, path) in enumerate(rows, 1):
        out = Path(args.output) / f"{name}.pt"
        if out.exists() and not args.overwrite:
            print(f"[{number}/{len(rows)}] skip {name}")
            continue

        saved = SavedKVCache(path)
        started = time.time()
        if args.rebucket_from:
            old = Path(args.rebucket_from) / f"{name}.pt"
            policy = rebucket(saved, old, out)
        else:
            keys = {i: saved.layer_tensors(i)["k"][0] for i in _kv_layers(saved)}
            policy = NaiveIvfPolicy(args.nlist, window=args.window, index_file=out)
            policy.prepare(keys, saved.seq_len)
        print(f"[{number}/{len(rows)}] {name}: len {saved.seq_len}, "
              f"nlist {policy.nlist}, index_len {policy.index_len}, "
              f"{time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
