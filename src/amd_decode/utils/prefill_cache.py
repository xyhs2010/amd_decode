"""Reader for the dense-prefill caches written by ``prefill/prefill_qwen35.py``.

One directory per sample::

    meta.json                  # seq_len, num_layers, layer_types, ...
    tokens.safetensors         # int32 prompt tokens
    layer_00.safetensors ... layer_31.safetensors

Each layer file holds either

  * full attention: ``k`` / ``v``      [1, H_kv, T, D] in the model dtype
  * linear attention: ``conv_state``   [1, 8192, 4]
                      ``recurrent_state`` [1, 32, 128, 128] fp32

The tensors are memory-mapped and read on demand, so a 262144-token cache never
has to fit in RAM.  This module is the cross-implementation side of the
saved-prefill path: every decode backend reads the same files.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .safetensor_io import SafeTensorsFile


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
        return int(self.meta.get("num_layers", 32))

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

    def layer_tensors(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """The raw tensors of one layer file (mmap views, no conversion)."""
        f = self._file(f"layer_{layer_idx:02d}")
        return {name: f.get(name) for name in f.keys()}
