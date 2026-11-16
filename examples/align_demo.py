"""Alignment demo: Qwen3Torch against itself and against a ``_layer`` override.

Run from the repo root:

    PYTHONPATH=src .venv/bin/python examples/align_demo.py
    PYTHONPATH=src .venv/bin/python examples/align_demo.py --weight-bits 4 --decode

The second case is the one to copy when writing a real custom layer: subclass
``Qwen3Torch``, keep the ``_layer`` signature, and let ``align`` prove the new
maths is byte-identical to the reference before wiring it into the hot path.
"""

from __future__ import annotations

import argparse

import torch

from amd_decode.with_torch import Qwen3Torch, align


class SameQwen3(Qwen3Torch):
    """A custom ``_layer`` that is copied verbatim from the base class.

    Replace the body with the new implementation; as long as it computes the
    same thing, ``align`` should report byte-identical outputs.
    """

    def _layer(self, x, p, cos, sin, kv, layer_idx, past):
        h = x
        q, k, v = self._attn_prep(
            self._rms(x, p["input_layernorm.weight"]), p, cos, sin
        )
        if kv is None:
            k_all, v_all = k, v
        else:
            k_all, v_all = kv.append(layer_idx, k, v)
        x = h + self._attn_out(q, k_all, v_all, p, past)

        h = x
        x = h + self._mlp(self._rms(x, p["post_attention_layernorm.weight"]), p)
        return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="data/models/Qwen3-8B-Base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-bits", type=int, default=16, choices=(4, 16))
    parser.add_argument("--decode", action="store_true",
                        help="feed one token at a time through the KV cache")
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[9707, 11, 1879, 13])
    args = parser.parse_args()

    common = dict(
        device=args.device,
        dtype=torch.float16,
        weight_bits=args.weight_bits,
        decode=args.decode,
    )

    print("== Qwen3Torch vs Qwen3Torch (sanity check) ==")
    ok_self = align(
        args.model_dir, Qwen3Torch, Qwen3Torch, args.tokens, **common
    )

    print("\n== Qwen3Torch vs SameQwen3 (overridden _layer) ==")
    ok_layer = align(
        args.model_dir, Qwen3Torch, SameQwen3, args.tokens, **common
    )

    if not (ok_self and ok_layer):
        raise SystemExit("alignment failed")


if __name__ == "__main__":
    main()
