"""Sparse decode must match dense when the policy selects every token.

Run from the repo root (after ``pip install -e .``)::

    PYTHONPATH=src .venv/bin/python examples/sparse_align.py

With ``window >= seq_len`` and no sink, ``WindowPolicy`` returns ``arange(n)``,
so the gather kernel reads the cache in exactly the dense order.  Its fp32
online softmax still accumulates in a different order than torch's, so the
logits agree within ``align``'s tolerance rather than bit-for-bit.  A short
window then runs the genuinely sparse path, whose output is only checked to be
finite - its quality is what the policy experiments are for.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from amd_decode.align import align
from amd_decode.utils.prefill_cache import SavedKVCache
from amd_decode.with_hip import HipQwen35
from amd_decode.with_sparse import SparseQwen35, WindowPolicy


def find_sample(root: Path) -> Path:
    """The shortest saved prefill: decode replay is one weight stream per token."""
    paths = sorted(root.glob("L*/*/*/meta.json"))
    if not paths:
        raise SystemExit(f"no saved prefill under {root}; pass --sample")
    return min((p.parent for p in paths), key=lambda d: SavedKVCache(d).seq_len)


def decode_logits(model, saved, tokens) -> list[torch.Tensor]:
    model.load_prefill(saved, saved.seq_len + len(tokens) + 1)
    position, out = saved.seq_len, []
    for token in tokens:
        out.append(model.forward([token], position_offset=position).detach().cpu())
        position += 1
    model.load_prefill(None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="data/models/Qwen3.5-9B")
    parser.add_argument("--cache-root",
                        default="data/kv_cache/Qwen3.5-9B/bfloat16/jsonl")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[9707, 11, 1879, 13])
    args = parser.parse_args()

    sample = Path(args.sample) if args.sample else find_sample(Path(args.cache_root))
    saved = SavedKVCache(sample)
    print(f"sample: {sample} (seq_len {saved.seq_len})")

    settings = dict(device=args.device, dtype=torch.float16)
    dense = HipQwen35(args.model_dir, **settings)
    every = SparseQwen35(
        args.model_dir, policy=WindowPolicy(window=saved.seq_len, sink=0), **settings
    )
    # One packed-weight cache for all instances: three copies would not fit.
    every._int4_cache = dense._int4_cache

    print("\n== every token selected: dense vs sparse (decode) ==")
    if not align(dense, every, saved, args.tokens, decode=True):
        raise SystemExit("alignment failed")

    print("\n== window=512, sink=4 (quality not checked) ==")
    sparse = SparseQwen35(
        args.model_dir, policy=WindowPolicy(window=512, sink=4), **settings
    )
    sparse._int4_cache = dense._int4_cache
    reference = decode_logits(dense, saved, args.tokens)
    short = decode_logits(sparse, saved, args.tokens)
    finite = all(bool(torch.isfinite(t).all()) for t in short)
    drift = max((a - b).abs().max().item() for a, b in zip(reference, short))
    print(f"finite: {finite}, max logit drift vs dense: {drift:.3e}")


if __name__ == "__main__":
    main()
