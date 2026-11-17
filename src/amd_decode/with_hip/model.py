"""``Qwen3Torch`` with the FFN matmuls on an int4 HIP kernel and a host q8 cache.

Weight streaming and the reference helpers are inherited.  What changes:

* ``_load_layer`` / ``_mlp``: FFN weights stay packed (no fp16 dequantization),
  so a decode step reads ~2.7 GiB of int4 instead of ~14 GiB of fp16;
* ``make_kv_cache`` / ``_layer``: the q8 cache lives in host RAM and decode
  attention reads it without materialising fp16 K/V on the GPU.

Prefill (``answer_prefix``, T > 1) still goes through the reference
``_attn_out`` after a one-off ``dequantize``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..with_host.host_memory import Hip
from ..with_torch import Qwen3Torch
from ..with_torch.int4 import pack_int4, quantize_int4
from .kv_cache import HipKVCache
from .ops import q4_matmul, q8_attention


class HipQwen3(Qwen3Torch):
    _FFN = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")

    def __init__(self, model_dir, device="cuda", dtype=torch.float16,
                 weight_bits: int = 4) -> None:
        if weight_bits != 4 or dtype is not torch.float16:
            raise ValueError("HipQwen3 requires 4-bit weights and float16")
        super().__init__(model_dir, device=device, dtype=dtype, weight_bits=weight_bits)
        self._hip = Hip()

    def make_kv_cache(self, max_len: int, batch_size: int = 1) -> HipKVCache:
        if batch_size != 1:
            raise ValueError("HipQwen3 supports batch_size=1")
        return HipKVCache(
            self._hip, self.num_layers, self.cfg.num_key_value_heads,
            self.cfg.head_dim, max_len, device=self.device,
        )

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

    # --------------------------------------------------------------- attention

    def _layer(self, x, p, cos, sin, kv, layer_idx, past):
        h = x
        q, k, v = self._attn_prep(
            self._rms(x, p["input_layernorm.weight"]), p, cos, sin
        )
        if kv is None:
            attn = self._attn_out(q, k, v, p, past)
        else:
            kv.write(layer_idx, k, v)
            if q.shape[0] == 1 and q.shape[2] == 1:
                attn = self._decode_attn(q, kv, layer_idx, p)
            else:
                k_all, v_all = kv.dequantize(layer_idx)
                attn = self._attn_out(q, k_all, v_all, p, past)
        x = h + attn

        h = x
        x = h + self._mlp(self._rms(x, p["post_attention_layernorm.weight"]), p)
        return x

    def _decode_attn(self, q, kv, layer_idx, p):
        """Decode attention reads the int8 host buffers directly."""
        out = q8_attention(q, kv, layer_idx)
        return F.linear(out, p["self_attn.o_proj.weight"])
