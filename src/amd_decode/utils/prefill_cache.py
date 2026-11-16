"""Reader for the dense-prefill KV caches written by hydra_llm.

Expected layout (one directory per sample)::

    meta.json
    tokens.safetensors
    layer_00.safetensors ... layer_35.safetensors

Each layer file holds either

  * fp16: ``k`` / ``v``                              shape [H_kv, T, D]
  * q8:   ``k_q`` / ``k_scale`` / ``v_q`` / ``v_scale``

K is quantized per (head, block, channel): ``k_scale`` has shape
[1, H_kv, ceil(T / block_size), 1, D]. V is per-token: ``v_scale`` has shape
[1, H_kv, T, 1]. Quantized values are int8 and scales are float32. Both the
tensors and the scales may come with or without a leading batch dimension; the
loader normalizes them.

This module is the cross-implementation side of the saved-prefill path: any
decode backend (the torch one in ``with_torch`` or a future kernel one) reads
the same files.  ``layer()`` dequantises one layer to fp16 (the reference
semantics, and the streaming path for huge contexts); ``layer_quantized()``
returns the stored int8 values so an in-memory q8 cache can be seeded without
a dequantise/re-quantise round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .safetensor_io import SafeTensorsFile



def expand_block_scale(k_scale: torch.Tensor, seq_len: int,
                       block_size: int) -> torch.Tensor:
    """[H, n_blocks, D] -> [H, seq_len, D], one scale vector per token block.

    This is part of the prefill-cache format: K scales are stored per
    (head, block, channel) and must be broadcast over the block's tokens.
    """
    expanded = k_scale.repeat_interleave(block_size, dim=1)[:, :seq_len, :]
    if expanded.shape[1] < seq_len:
        pad = expanded[:, -1:, :].expand(-1, seq_len - expanded.shape[1], -1)
        expanded = torch.cat((expanded, pad), dim=1)
    return expanded


def _ensure_bhtd(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize K/V from [H, T, D] or [B, H, T, D] to [B, H, T, D]."""
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name}: expected batch size 1, got {tuple(tensor.shape)}")
        return tensor
    raise ValueError(f"{name}: expected 3-D or 4-D, got {tuple(tensor.shape)}")


def _k_scale_to_hbd(scale: torch.Tensor) -> torch.Tensor:
    """Normalize K scale to [H, n_blocks, D].

    Observed layout: [1, H, n_blocks, 1, D]. Also accept the simpler
    [H, n_blocks, D] and [B, H, n_blocks, D].
    """
    s = scale
    if s.dim() >= 4 and s.shape[0] == 1:
        s = s[0]
    if s.dim() == 4 and s.shape[2] == 1:
        s = s[:, :, 0, :]
    if s.dim() != 3:
        raise ValueError(f"k_scale: cannot normalize shape {tuple(scale.shape)}")
    return s


def _v_scale_to_ht1(scale: torch.Tensor) -> torch.Tensor:
    """Normalize V scale to [H, T, 1]."""
    s = scale
    if s.dim() >= 4 and s.shape[0] == 1:
        s = s[0]
    if s.dim() == 2:
        s = s.unsqueeze(-1)
    if s.dim() != 3:
        raise ValueError(f"v_scale: cannot normalize shape {tuple(scale.shape)}")
    return s


class SavedKVCache:
    def __init__(self, sample_dir: str | Path) -> None:
        self.dir = Path(sample_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"not a KV cache sample (no meta.json): {self.dir}")
        self.meta = json.loads(meta_path.read_text())
        self._files: dict[str, SafeTensorsFile] = {}
        self._tokens: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return int(self.meta["seq_len"])

    @property
    def num_layers(self) -> int:
        return int(self.meta.get("num_layers", 36))

    @property
    def block_size(self) -> int | None:
        value = self.meta.get("block_size")
        return int(value) if value else None

    def _file(self, stem: str) -> SafeTensorsFile:
        if stem not in self._files:
            self._files[stem] = SafeTensorsFile(self.dir / f"{stem}.safetensors")
        return self._files[stem]

    def tokens(self) -> torch.Tensor:
        if self._tokens is None:
            f = self._file("tokens")
            name = "tokens" if "tokens" in f else f.keys()[0]
            self._tokens = f.get(name).to(torch.long)
        return self._tokens

    def layer(
        self,
        layer_idx: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize one layer to [1, H_kv, T, D] on ``device``."""
        f = self._file(f"layer_{layer_idx:02d}")
        keys = set(f.keys())

        k_name = "k_q" if "k_q" in keys else ("k" if "k" in keys else None)
        v_name = "v_q" if "v_q" in keys else ("v" if "v" in keys else None)
        if k_name is None or v_name is None:
            raise ValueError(f"unrecognized layer file {f.path}: keys={sorted(keys)}")

        k = _ensure_bhtd(f.get(k_name), k_name)
        v = _ensure_bhtd(f.get(v_name), v_name)
        seq_len = k.shape[-2]
        if v.shape[-2] != seq_len:
            raise ValueError(f"K/V length mismatch: {seq_len} vs {v.shape[-2]}")

        if "k_scale" in keys:
            # K: [H, n_blocks, D] -> expand each block's scale over tokens.
            k_scale = _k_scale_to_hbd(f.get("k_scale").to(device).float())
            if k_scale.shape[0] != k.shape[1] or k_scale.shape[-1] != k.shape[-1]:
                raise ValueError(f"unexpected k_scale shape {tuple(k_scale.shape)}")
            num_blocks = k_scale.shape[1]
            block_size = self.block_size or -(-seq_len // num_blocks)
            k_scale = expand_block_scale(k_scale, seq_len, block_size)
            k = k.to(device).float() * k_scale.unsqueeze(0)

            # V: [H, T, 1] per-token scale.
            v_scale = _v_scale_to_ht1(f.get("v_scale").to(device).float())
            if v_scale.shape[0] != v.shape[1]:
                raise ValueError(f"unexpected v_scale shape {tuple(v_scale.shape)}")
            if v_scale.shape[1] == 1:
                v_scale = v_scale.expand(-1, seq_len, -1)
            if v_scale.shape[1] != seq_len:
                raise ValueError(f"v_scale covers {v_scale.shape[1]} tokens, expected {seq_len}")
            v = v.to(device).float() * v_scale.unsqueeze(0)
        else:
            k = k.to(device)
            v = v.to(device)

        return k.to(dtype), v.to(dtype)


    def layer_quantized(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the stored q8 K/V and scales *without* dequantising.

        Shapes after normalisation:
          * ``k_q``, ``v_q``: int8 ``[1, H, T, D]`` (memory-mapped views);
          * ``k_scale``: float32 ``[H, n_blocks, D]``;
          * ``v_scale``: float32 ``[H, T, 1]``.

        This keeps the prefill machine's quantisation intact, which is what a
        host-resident q8 cache should store.
        """
        f = self._file(f"layer_{layer_idx:02d}")
        keys = set(f.keys())
        needed = {"k_q", "v_q", "k_scale", "v_scale"}
        if not needed <= keys:
            raise ValueError(
                f"layer file is not q8 (need {sorted(needed)}): {f.path}"
            )
        k_q = _ensure_bhtd(f.get("k_q"), "k_q")
        v_q = _ensure_bhtd(f.get("v_q"), "v_q")
        k_scale = _k_scale_to_hbd(f.get("k_scale").float())
        v_scale = _v_scale_to_ht1(f.get("v_scale").float())
        return k_q, k_scale, v_q, v_scale

