#!/usr/bin/env python
"""Build the sparse-attention IVF indices for saved prefill caches, offline.

Run from the repo root::

    .venv/bin/python scripts/build_ivf_index.py --lengths all \
        --per-task-limit 1 --window 512 --output data/ivf

No model weights are loaded: each sample's ``SavedKVCache`` is seeded straight
into a ``HipKVCache`` and ``NaiveIvfPolicy`` writes ``<output>/<sample_id>.pt``.
``nlist`` defaults to 32 indexed tokens per cluster, so it follows the context
length.  Samples are picked the way ``eval_ruler.py`` picks them - sorted by
(length, task, index), first ``--per-task-limit`` per pair - so the indices
match the evaluation set.  Existing files are skipped, so an interrupted run
resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from amd_decode.bench.ruler import _sample_index
from amd_decode.utils.prefill_cache import SavedKVCache
from amd_decode.with_host.host_memory import Hip
from amd_decode.with_hip.kv_cache import HipKVCache
from amd_decode.with_sparse.policy import NaiveIvfPolicy


def _parse_lengths(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    return [int(part) for part in value.replace(",", " ").split()]


def _select(paths: list[Path], lengths: list[int] | None,
            per_task_limit: int) -> list[tuple[int, str, int, Path]]:
    """``eval_ruler``'s rule: sort by (length, task, index), keep N per pair."""
    rows = []
    for path in paths:
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("benchmark") != "ruler":
            continue
        length, task = int(meta.get("length", 0)), str(meta.get("task", ""))
        if lengths is not None and length not in lengths:
            continue
        sample_id = str(meta.get("sample_id", path.name))
        rows.append((length, task, _sample_index(sample_id, meta), path))
    rows.sort()
    kept: list[tuple[int, str, int, Path]] = []
    seen: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (row[0], row[1])
        if seen.get(key, 0) < per_task_limit:
            kept.append(row)
        seen[key] = seen.get(key, 0) + 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-root", default="data/kv_cache")
    parser.add_argument("--output", default="data/ivf")
    parser.add_argument("--lengths", type=_parse_lengths, default=None,
                        metavar="N[,N...]|all")
    parser.add_argument("--per-task-limit", type=int, default=1)
    parser.add_argument("--nlist", type=int, default=None,
                        help="clusters per (layer, KV head); default 32 tokens "
                             "per cluster")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = sorted({p.parent for p in Path(args.kv_root).rglob("meta.json")})
    rows = _select(paths, args.lengths, args.per_task_limit)
    if args.limit:
        rows = rows[: args.limit]
    hip = Hip()
    for number, (_, _, _, path) in enumerate(rows, 1):
        saved = SavedKVCache(path)
        name = str(saved.meta.get("sample_id", path.name))
        out = Path(args.output) / f"{name}.pt"
        if out.exists() and not args.overwrite:
            print(f"[{number}/{len(rows)}] skip {name}")
            continue

        k_q = saved.layer_quantized(0)[0]  # [1, H, T, D]
        kv = HipKVCache(hip, saved.num_layers, k_q.shape[1], k_q.shape[3],
                        saved.seq_len)
        kv.seed_from_prefill(saved)
        policy = NaiveIvfPolicy(args.nlist, window=args.window, index_file=out)
        started = time.time()
        policy.prepare(kv)
        kv.free()
        print(f"[{number}/{len(rows)}] {name}: len {saved.seq_len}, "
              f"nlist {policy.nlist}, index_len {policy.index_len}, "
              f"{time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
