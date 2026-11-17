"""``HostQwen3`` with the FFN matmuls running on an int4 HIP kernel.

Weight streaming, attention and the q8 KV cache are the reference
implementation.  Only two methods change: ``_load_layer`` keeps the FFN weights
packed (no fp16 dequantization) and ``_mlp`` feeds those packed weights to the
kernel, so a decode step reads ~2.7 GiB of int4 instead of expanding ~14 GiB to
fp16 in VRAM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..with_host import HostQwen3
from ..with_torch.int4 import pack_int4, quantize_int4
from .ops import q4_matmul


class HipQwen3(HostQwen3):
    _FFN = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")

    def __init__(self, *args, weight_bits: int = 4, **kwargs) -> None:
        if weight_bits != 4:
            raise ValueError("HipQwen3 requires 4-bit weights")
        super().__init__(*args, weight_bits=weight_bits, **kwargs)

    # ------------------------------------------------------------------ weights

    def _load_layer(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}."
        p = {}
        for name, tensor in self.weights.layer(layer_idx).items():
            p[name] = (
                self._q4(prefix + name, tensor)
                if name in self._FFN
                else self._linear_weight(prefix + name, tensor)
            )
        return p

    def _q4(self, name: str, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Packed int4 and fp16 qparams on the GPU, computed once per weight.

        The kernel takes fp16 scale/zero; the base ``_int4_cache`` (bf16) is
        only used for the q/k/v/o weights, so the two never mix.
        """
        if name not in self._int4_cache:
            q, scale, zero = quantize_int4(tensor)
            self._int4_cache[name] = (
                pack_int4(q).to(self.device),
                scale.to(self.device).to(self.dtype),
                zero.to(self.device).to(self.dtype),
            )
        return self._int4_cache[name]

    # --------------------------------------------------------------------- mlp

    def _mlp(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        gate = F.silu(q4_matmul(x, p["mlp.gate_proj.weight"]))
        up = q4_matmul(x, p["mlp.up_proj.weight"])
        out = q4_matmul(gate * up, p["mlp.down_proj.weight"])
        return out.view(shape[:-1] + (out.shape[-1],))
