"""Sparse decode must match dense when the policy selects every token.

Run from the repo root:

    .venv/bin/python examples/sparse_align.py

With ``window >= seq_len`` and no sink, ``WindowPolicy`` returns ``arange(n)``,
so the gather kernel reads the cache in exactly the dense order and the logits
are bit-identical.  Any difference is a bug in ``attn_sparse.hip``.  A short
window then runs the genuinely sparse path, whose output is only checked to be
finite - its quality is what the policy experiments are for.

There is no saved prefill in ``data/kv_cache`` here, so this exercises the
per-token tail scales only; the seeded-prefix block scales need a real sample.
"""

from __future__ import annotations

import argparse

import torch

from amd_decode.align import align
from amd_decode.with_hip import HipQwen3
from amd_decode.with_sparse import SparseQwen3, WindowPolicy


def logits(model, kv, tokens, decode: bool) -> list[torch.Tensor]:
    model.set_kv_cache(kv)
    chunks = [[tok] for tok in tokens] if decode else [tokens]
    position, out = kv.length, []
    for chunk in chunks:
        out.append(model.forward(chunk, position_offset=position).detach().cpu())
        position += len(chunk)
    model.set_kv_cache(None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="data/models/Qwen3-8B-Base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[9707, 11, 1879, 13, 358, 1079, 13, 576])
    args = parser.parse_args()

    settings = dict(device=args.device, dtype=torch.float16, weight_bits=4)
    n = len(args.tokens)

    reference = HipQwen3(args.model_dir, **settings)
    sparse = SparseQwen3(
        args.model_dir, policy=WindowPolicy(window=n, sink=0), **settings
    )

    print("== every token selected: dense vs sparse ==")
    dense = logits(reference, reference.make_kv_cache(n), args.tokens, True)
    every = logits(sparse, sparse.make_kv_cache(n), args.tokens, True)
    diff = max((a - b).abs().max().item() for a, b in zip(dense, every))
    print(f"max |dense - sparse| over decode steps: {diff:.3e}")
    if diff != 0.0:
        raise SystemExit("sparse gather does not reproduce dense decode")

    print("\n== align (all layers, decode) ==")
    if not align(
        reference, sparse,
        (reference.make_kv_cache(n), sparse.make_kv_cache(n)),
        args.tokens, decode=True,
    ):
        raise SystemExit("alignment failed")

    print("\n== window=2, sink=1 (quality not checked) ==")
    sparse.policy = WindowPolicy(window=2, sink=1)
    short = logits(sparse, sparse.make_kv_cache(n), args.tokens, True)
    finite = all(bool(torch.isfinite(t).all()) for t in short)
    drift = max((a - b).abs().max().item() for a, b in zip(dense, short))
    print(f"finite: {finite}, max logit drift vs dense: {drift:.3e}")


if __name__ == "__main__":
    main()
