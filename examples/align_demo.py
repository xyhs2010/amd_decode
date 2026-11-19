"""Alignment demo: Qwen35Torch against itself and a ``_layer`` override.

Run from the repo root:

    PYTHONPATH=src .venv/bin/python examples/align_demo.py

The second case is the template for a custom layer: subclass ``Qwen35Torch``,
keep the ``_layer`` signature, and let ``align`` prove the new maths matches the
reference before wiring it into the hot path.  Pass a ``SavedKVCache`` as
``align``'s third argument to compare both backends on a seeded cache.
"""

from __future__ import annotations

import argparse

import torch

from amd_decode.align import align
from amd_decode.with_torch import Qwen35Torch


class SameQwen35(Qwen35Torch):
    """A custom ``_layer`` copied verbatim from the base class."""

    def _layer(self, x, p, cos, sin, kv, layer_idx, past):
        h = x
        n = self._rms(x, p["input_layernorm.weight"])
        if self.cfg.is_full_attention[layer_idx]:
            mixer = self._full_attn(n, p, cos, sin, kv, layer_idx, past)
        else:
            mixer = self._delta_net(n, p, layer_idx)
        x = h + mixer

        h = x
        x = h + self._mlp(self._rms(x, p["post_attention_layernorm.weight"]), p)
        return x


def run(model_dir, reference_cls, candidate_cls, tokens, *, device, dtype) -> bool:
    reference = reference_cls(model_dir, device=device, dtype=dtype)
    candidate = candidate_cls(model_dir, device=device, dtype=dtype)
    return align(reference, candidate, None, tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="data/models/Qwen3.5-9B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[9707, 11, 1879, 13])
    args = parser.parse_args()

    print("== Qwen35Torch vs Qwen35Torch (sanity check) ==")
    ok_self = run(
        args.model_dir, Qwen35Torch, Qwen35Torch, args.tokens,
        device=args.device, dtype=torch.float16,
    )

    print("\n== Qwen35Torch vs SameQwen35 (overridden _layer) ==")
    ok_layer = run(
        args.model_dir, Qwen35Torch, SameQwen35, args.tokens,
        device=args.device, dtype=torch.float16,
    )

    if not (ok_self and ok_layer):
        raise SystemExit("alignment failed")


if __name__ == "__main__":
    main()
