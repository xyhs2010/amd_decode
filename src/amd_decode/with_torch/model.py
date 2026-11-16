"""A small, explicit Qwen3 forward pass for decode experiments.

Design goals:
  * no transformers dependency, so every step is easy to replace;
  * one code path for prefill (T > 1, no cache) and decode (T = 1, cache);
  * layer streaming: only one layer's weights are resident on the GPU, so the
    16 GB gfx906 card can run a bf16/fp16 8B model without quantizing it.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import DecodeModel
from .config import Qwen3Config
from .int4 import dequantize_int4, pack_int4, quantize_int4, unpack_int4
from .kv_cache import KVCache
from .rope import apply_rope, build_cos_sin, rope_parameters
from .weights import Qwen3Weights

# Sentinel for "use the cache attached with set_kv_cache"; an explicit
# ``kv=None`` still means the cache-less branch.
_ATTACHED = object()


class Qwen3Torch(DecodeModel):
    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        weight_bits: int = 4,
    ) -> None:
        if weight_bits not in (4, 16):
            raise ValueError(f"weight_bits must be 4 or 16, got {weight_bits}")
        self.cfg = Qwen3Config.from_model_dir(model_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.weight_bits = weight_bits
        self.weights = Qwen3Weights(model_dir)
        # name -> (packed q, scale, zero) on the GPU, computed once per weight.
        # Packed q is half the int8 size and is read every step, so keeping it
        # on the device avoids streaming ~3.5 GiB per step over PCIe.
        self._int4_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Kept on CPU: embedding lookup only touches a few rows, and lm_head is
        # moved to the GPU only after all layers have released their weights.
        self.embed = self.weights.get("model.embed_tokens.weight")
        self.lm_head = self.weights.get("lm_head.weight")
        self.final_norm = self.weights.get("model.norm.weight")

        self.inv_freq, self.attention_scaling = rope_parameters(self.cfg)
        self.num_layers = self.cfg.num_hidden_layers

        # DecodeModel protocol state (see amd_decode.align).
        self._kv = None
        self._hook = None
        self._features: dict[int, object] = {}

    # ------------------------------------------------------------- align protocol

    def set_kv_cache(self, kv) -> None:
        """Attach the cache ``forward`` uses when ``kv`` is not passed."""
        self._kv = kv

    def set_hook(self, hook) -> None:
        """Install ``hook(layer_idx, output)``, called after every layer.

        The hook's return value is what :meth:`get_features` reports for that
        layer, so a hook that copies to CPU keeps VRAM bounded.
        """
        self._hook = hook

    def get_features(self) -> dict[int, object]:
        """Layer features captured by the most recent :meth:`forward`."""
        return self._features

    def make_kv_cache(self, max_len: int, batch_size: int = 1) -> KVCache:
        """Allocate the int8 KV cache; ``with_host`` overrides the storage."""
        if self.dtype is not torch.float16:
            raise ValueError("the q8 KV cache dequantises to float16")
        return KVCache(
            self.num_layers,
            max_len=max_len,
            num_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
            batch_size=batch_size,
            device=self.device,
        )

    # ------------------------------------------------------------------ internals

    def _to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        # Cast on the GPU: bf16 -> fp16 conversion on the CPU is slow and would
        # need an extra full-size buffer.
        return tensor.to(self.device).to(self.dtype)

    @staticmethod
    def _is_int4_weight(name: str) -> bool:
        # torchao quantizes nn.Linear modules, not embeddings or norms.
        return name == "lm_head.weight" or name.endswith("_proj.weight")

    def _linear_weight(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Return a dense weight on the GPU, quantizing int4 if requested."""
        if self.weight_bits != 4 or not self._is_int4_weight(name):
            return self._to_gpu(tensor)
        try:
            if name not in self._int4_cache:
                q, scale, zero = quantize_int4(tensor)
                self._int4_cache[name] = (
                    pack_int4(q).to(self.device),
                    scale.to(self.device),
                    zero.to(self.device),
                )
        except ValueError:
            # Same behaviour as torchao: skip weights whose in_features is not
            # divisible by the group size.
            return self._to_gpu(tensor)
        q, scale, zero = self._int4_cache[name]
        q = unpack_int4(q, q.shape[-1] * 2)
        return dequantize_int4(q, scale, zero, device=self.device, dtype=self.dtype)

    def _load_layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        return {
            name: self._linear_weight(prefix + name, tensor)
            for name, tensor in self.weights.layer(layer_idx).items()
        }

    def _rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Matches transformers' Qwen3RMSNorm ordering.
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return weight * (xf * torch.rsqrt(var + self.cfg.rms_norm_eps)).to(x.dtype)

    def _attn_prep(
        self,
        x: torch.Tensor,
        p: dict[str, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        heads = self.cfg.num_attention_heads
        kv_heads = self.cfg.num_key_value_heads
        head_dim = self.cfg.head_dim

        q = F.linear(x, p["self_attn.q_proj.weight"]).view(batch, seq, heads, head_dim)
        k = F.linear(x, p["self_attn.k_proj.weight"]).view(batch, seq, kv_heads, head_dim)
        v = F.linear(x, p["self_attn.v_proj.weight"]).view(batch, seq, kv_heads, head_dim)

        q = self._rms(q.transpose(1, 2), p["self_attn.q_norm.weight"])
        k = self._rms(k.transpose(1, 2), p["self_attn.k_norm.weight"])
        v = v.transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        return q, k, v

    def _attn_out(
        self,
        q: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        p: dict[str, torch.Tensor],
        past: int,
    ) -> torch.Tensor:
        batch, heads, seq, head_dim = q.shape
        rep = self.cfg.num_key_value_groups
        k = k_all.repeat_interleave(rep, dim=1)
        v = v_all.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if seq > 1:
            qpos = torch.arange(past, past + seq, device=q.device)
            kpos = torch.arange(0, past + seq, device=q.device)
            mask = qpos[:, None] >= kpos[None, :]
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(batch, seq, heads * head_dim)
        return F.linear(out, p["self_attn.o_proj.weight"])

    def _mlp(self, x: torch.Tensor, p: dict[str, torch.Tensor]) -> torch.Tensor:
        gate = F.silu(F.linear(x, p["mlp.gate_proj.weight"]))
        up = F.linear(x, p["mlp.up_proj.weight"])
        return F.linear(gate * up, p["mlp.down_proj.weight"])

    def _layer(
        self,
        x: torch.Tensor,
        p: dict[str, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: KVCache | None,
        layer_idx: int,
        past: int,
    ) -> torch.Tensor:
        h = x
        q, k, v = self._attn_prep(self._rms(x, p["input_layernorm.weight"]), p, cos, sin)
        if kv is None:
            k_all, v_all = k, v
        else:
            k_all, v_all = kv.append(layer_idx, k, v)
        x = h + self._attn_out(q, k_all, v_all, p, past)

        h = x
        x = h + self._mlp(self._rms(x, p["post_attention_layernorm.weight"]), p)
        return x

    # ------------------------------------------------------------------ public

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | list[int],
        position_offset: int | None = None,
        kv=_ATTACHED,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """Run one forward pass.

        Args:
            input_ids: [T] or [B, T] token ids.
            position_offset: absolute position of the first input token; ``None``
                means "start at the attached cache's ``length``" (0 without one).
            kv: optional cache. When given, new K/V are appended and attention
                runs over the whole cache. Omit to use the cache attached with
                :meth:`set_kv_cache`; pass ``None`` for no cache.
            logits_to_keep: if set, only return the last N positions' logits.

        Returns:
            logits of shape [B, T, vocab] (or [B, N, vocab]).
        """
        if kv is _ATTACHED:
            kv = self._kv
        if position_offset is None:
            position_offset = 0 if kv is None else kv.length
        self._features = {}
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        if ids.dim() == 1:
            ids = ids[None, :]
        if ids.dim() != 2:
            raise ValueError(f"input_ids must be 1-D or 2-D, got shape {tuple(ids.shape)}")

        if kv is not None and kv.length != position_offset:
            raise ValueError(
                f"cache length {kv.length} != position_offset {position_offset}"
            )

        seq = ids.shape[1]
        past = 0 if kv is None else position_offset
        x = self.embed[ids.cpu()].to(self.device).to(self.dtype)

        positions = torch.arange(position_offset, position_offset + seq, device=self.device)
        cos, sin = build_cos_sin(positions, self.inv_freq, self.attention_scaling, self.dtype)

        for layer_idx in range(self.num_layers):
            p = self._load_layer(layer_idx)
            x = self._layer(x, p, cos, sin, kv, layer_idx, past)
            if self._hook is not None:
                self._features[layer_idx] = self._hook(layer_idx, x)
            del p

        x = self._rms(x, self.final_norm.to(self.device).to(self.dtype))
        lm_head = self._linear_weight("lm_head.weight", self.lm_head)
        logits = F.linear(x, lm_head)
        del lm_head

        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return logits
