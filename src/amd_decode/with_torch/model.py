"""Qwen3.5-9B decode pass, W4A16 reference implementation.

The text model is 32 layers: 24 Gated DeltaNet (linear attention) layers and 8
gated full-attention layers (every fourth layer).  One :class:`Qwen35Torch`
implements both; a kernel backend only replaces :meth:`linear` and the decode
attention.

W4A16: weights are int4 with fp16 scale/zero, activations and the dense
dequantized weights are fp16, and the GEMV accumulates in fp32.  DeltaNet's
recurrent state and gating are kept in fp32, matching the upstream model.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import DecodeModel
from .config import Qwen35Config
from .int4 import dequantize_int4, pack_int4, quantize_int4, unpack_int4
from .kv_cache import KVCache
from .rope import apply_rope, build_cos_sin
from .weights import EMBED, FINAL_NORM, LM_HEAD, Qwen35Weights


def _quantizable(tensor: torch.Tensor) -> bool:
    """Every 2-D weight whose in_features is a multiple of the group size."""
    return tensor.dim() == 2 and tensor.shape[-1] % 128 == 0


class Qwen35Torch(DecodeModel):
    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if dtype is not torch.float16:
            raise ValueError("the W4A16 decode path requires float16")
        self.cfg = Qwen35Config.from_model_dir(model_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.weights = Qwen35Weights(model_dir)

        # name -> (packed q, fp16 scale, fp16 zero) on the GPU, computed once.
        self._int4_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Embedding stays on CPU (a decode step gathers a few rows); lm_head is
        # dequantized only after all layers have released their weights.
        self.embed = self.weights.get(EMBED)
        self.lm_head = self.weights.get(LM_HEAD)
        self.final_norm = self.weights.get(FINAL_NORM)

        self.num_layers = self.cfg.num_hidden_layers

        # DecodeModel protocol state (see amd_decode.align).
        self._kv: KVCache | None = None
        self._delta: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
        self._hook = None
        self._features: dict[int, object] = {}

    # ------------------------------------------------------------- align protocol

    def load_prefill(self, saved=None, max_len: int | None = None) -> None:
        """Build and seed the cache from an offline prefill; ``None`` clears it."""
        if saved is None:
            self._kv = None
            self._delta = None
            return

        self._kv = self._new_kv_cache(max_len)
        self._delta = {}
        for layer_idx in range(saved.num_layers):
            tensors = saved.layer_tensors(layer_idx)
            if "k" in tensors:
                self._kv.append(
                    layer_idx,
                    tensors["k"].to(self.device, self.dtype),
                    tensors["v"].to(self.device, self.dtype),
                )
            else:
                self._delta[layer_idx] = (
                    tensors["conv_state"].to(self.device, self.dtype),
                    tensors["recurrent_state"].to(self.device),
                )

    def _new_kv_cache(self, max_len: int) -> KVCache:
        """K/V storage for the full-attention layers; ``with_host`` overrides."""
        return KVCache(max_len)

    def set_hook(self, hook) -> None:
        self._hook = hook

    def get_features(self) -> dict[int, object]:
        return self._features

    # ------------------------------------------------------------ weight loading

    def _packed(self, name: str, tensor: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if name not in self._int4_cache:
            q, scale, zero = quantize_int4(tensor)
            self._int4_cache[name] = (
                pack_int4(q).to(self.device),
                scale.to(self.device),
                zero.to(self.device),
            )
        return self._int4_cache[name]

    def _prepare(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Return the dense fp16 weight used by :meth:`linear`.

        A kernel backend overrides this to keep the int4 triple packed.
        """
        if not _quantizable(tensor):
            # Norms are zero-centred fp32 in the checkpoint; A_log must stay
            # fp32 as well.  Everything else 1-D/3-D (dt_bias, conv1d) is fp16.
            if tensor.dtype is torch.float32:
                return tensor.to(self.device)
            return tensor.to(self.device).to(self.dtype)
        q, scale, zero = self._packed(name, tensor)
        return dequantize_int4(
            unpack_int4(q, tensor.shape[-1]), scale, zero,
            device=self.device, dtype=self.dtype,
        )

    def _load_layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        prefix = f"model.language_model.layers.{layer_idx}."
        tensors = self.weights.layer(
            layer_idx, self.cfg.is_full_attention[layer_idx]
        )
        return {
            name: self._prepare(prefix + name, tensor)
            for name, tensor in tensors.items()
        }

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """The single matmul hook every projection goes through."""
        return F.linear(x, w)

    # --------------------------------------------------------------------- norms

    def _rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Zero-centred RMSNorm (input/post/final/q_norm/k_norm)."""
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.cfg.rms_norm_eps)
        return (xf * (1.0 + weight.float())).to(x.dtype)

    def _rms_gated(self, x: torch.Tensor, weight: torch.Tensor,
                   gate: torch.Tensor) -> torch.Tensor:
        """DeltaNet's gated RMSNorm: norm, fp16 weight, then silu(z)."""
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.cfg.rms_norm_eps)
        xf = weight * xf.to(x.dtype)
        return (xf * F.silu(gate.float())).to(x.dtype)

    # ----------------------------------------------------------- full attention

    def _attention(self, q: torch.Tensor, k_all: torch.Tensor,
                   v_all: torch.Tensor, past: int) -> torch.Tensor:
        """GQA softmax attention; returns [B, T, H * D] before o_proj."""
        batch, heads, seq, head_dim = q.shape
        rep = self.cfg.num_key_value_groups
        k = k_all.repeat_interleave(rep, dim=1)
        v = v_all.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if seq > 1:
            qpos = torch.arange(past, past + seq, device=q.device)
            kpos = torch.arange(0, past + seq, device=q.device)
            scores = scores.masked_fill(qpos[:, None] < kpos[None, :],
                                        torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v)
        return out.transpose(1, 2).reshape(batch, seq, heads * head_dim)

    def _full_attn(self, x, p, cos, sin, kv, layer_idx, past):
        batch, seq, _ = x.shape
        heads = self.cfg.num_attention_heads
        kv_heads = self.cfg.num_key_value_heads
        head_dim = self.cfg.head_dim

        qg = self.linear(x, p["self_attn.q_proj.weight"]).view(
            batch, seq, heads, 2 * head_dim
        )
        q, gate = qg.chunk(2, dim=-1)
        q = self._rms(q, p["self_attn.q_norm.weight"]).transpose(1, 2)

        k = self.linear(x, p["self_attn.k_proj.weight"]).view(
            batch, seq, kv_heads, head_dim
        )
        k = self._rms(k, p["self_attn.k_norm.weight"]).transpose(1, 2)
        v = self.linear(x, p["self_attn.v_proj.weight"]).view(
            batch, seq, kv_heads, head_dim
        ).transpose(1, 2)

        q = apply_rope(q, cos, sin, self.cfg.rope_dim)
        k = apply_rope(k, cos, sin, self.cfg.rope_dim)

        if kv is None:
            k_all, v_all = k, v
        else:
            k_all, v_all = kv.append(layer_idx, k, v)

        out = self._attention(q, k_all, v_all, past)
        out = out * torch.sigmoid(gate.reshape(batch, seq, heads * head_dim))
        return self.linear(out, p["self_attn.o_proj.weight"])

    # ----------------------------------------------------------- gated delta net

    def _causal_conv(self, x: torch.Tensor, weight: torch.Tensor,
                     state: torch.Tensor | None) -> torch.Tensor:
        """Depthwise causal conv + SiLU over [B, T, C], keeping 4 left tokens."""
        batch, seq, channels = x.shape
        kernel = weight.shape[-1]
        buf = x.transpose(1, 2)
        if state is None:
            state = torch.zeros(batch, channels, kernel,
                                dtype=x.dtype, device=x.device)
        buf = torch.cat((state, buf), dim=-1)
        state.copy_(buf[..., -kernel:])
        out = F.conv1d(buf, weight, groups=channels)
        # The first output only looks at the cached tokens; upstream drops it.
        return F.silu(out[..., 1:]).transpose(1, 2)

    def _delta_rule(self, q, k, v, g, beta, state):
        """One causal pass of the gated delta rule, token by token."""
        batch, seq, k_heads, k_dim = q.shape
        v_heads, v_dim = v.shape[-2:]
        repeat = v_heads // k_heads
        q = q.repeat_interleave(repeat, dim=2).float()
        k = k.repeat_interleave(repeat, dim=2).float()
        v = v.float()
        beta = beta.float()

        q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        q = q / math.sqrt(k_dim)

        if state is None:
            state = torch.zeros(batch, v_heads, k_dim, v_dim,
                                dtype=torch.float32, device=q.device)
        out = []
        for t in range(seq):
            state = state * g[:, t].exp()[..., None, None]
            kv_mem = (state * k[:, t].unsqueeze(-1)).sum(dim=-2)
            delta = (v[:, t] - kv_mem) * beta[:, t].unsqueeze(-1)
            state = state + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
            out.append((state * q[:, t].unsqueeze(-1)).sum(dim=-2))
        return torch.stack(out, dim=1), state

    def _delta_net(self, x, p, layer_idx):
        batch, seq, _ = x.shape
        cfg = self.cfg
        key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim

        conv_state = recurrent_state = None
        if self._delta is not None:
            conv_state, recurrent_state = self._delta.get(layer_idx, (None, None))
            if conv_state is None:
                conv_state = torch.zeros(
                    batch, key_dim * 2 + value_dim, cfg.linear_conv_kernel_dim,
                    dtype=x.dtype, device=x.device,
                )

        mixed = self.linear(x, p["linear_attn.in_proj_qkv.weight"])
        mixed = self._causal_conv(mixed, p["linear_attn.conv1d.weight"], conv_state)
        q, k, v = mixed.split((key_dim, key_dim, value_dim), dim=-1)
        q = q.view(batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = k.view(batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = v.view(batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim)

        z = self.linear(x, p["linear_attn.in_proj_z.weight"]).view(
            batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim
        )
        beta = self.linear(x, p["linear_attn.in_proj_b.weight"]).sigmoid()
        a = self.linear(x, p["linear_attn.in_proj_a.weight"])
        g = -p["linear_attn.A_log"].float().exp() * F.softplus(
            a.float() + p["linear_attn.dt_bias"].float()
        )

        out, recurrent_state = self._delta_rule(q, k, v, g, beta, recurrent_state)
        if self._delta is not None:
            self._delta[layer_idx] = (conv_state, recurrent_state)

        out = self._rms_gated(out.to(x.dtype), p["linear_attn.norm.weight"], z)
        return self.linear(out.reshape(batch, seq, value_dim),
                           p["linear_attn.out_proj.weight"])

    # ---------------------------------------------------------------- layer glue

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

    def _mlp(self, x, p):
        gate = F.silu(self.linear(x, p["mlp.gate_proj.weight"]))
        up = self.linear(x, p["mlp.up_proj.weight"])
        return self.linear(gate * up, p["mlp.down_proj.weight"])

    # ------------------------------------------------------------------ public

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | list[int],
        position_offset: int | None = None,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
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
        cos, sin = build_cos_sin(positions, self.cfg, self.dtype)

        for layer_idx in range(self.num_layers):
            p = self._load_layer(layer_idx)
            x = self._layer(x, p, cos, sin, kv, layer_idx, past)
            if self._hook is not None:
                self._features[layer_idx] = self._hook(layer_idx, x)
            del p

        x = self._rms(x, self.final_norm.to(self.device).to(self.dtype))
        lm_head = self._prepare(LM_HEAD, self.lm_head)
        logits = F.linear(x, lm_head)
        del lm_head

        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return logits
