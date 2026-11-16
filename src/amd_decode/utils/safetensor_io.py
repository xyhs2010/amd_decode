"""Minimal, dependency-free safetensors reader.

We intentionally do not depend on the ``safetensors`` package: the format is
small, and reading through ``mmap`` + ``torch.frombuffer`` gives us zero-copy
views that are convenient for both model weights and KV cache files.

The tensors returned by :meth:`SafeTensorsFile.get` are backed by the mapping.
The mapping is kept alive by the tensor's buffer reference, so callers may keep
the tensor after this object is gone.
"""

from __future__ import annotations

import json
import mmap
import struct
import warnings
from pathlib import Path

import torch

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class SafeTensorsFile:
    """Read-only, memory-mapped view of one ``.safetensors`` file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = self.path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack("<Q", self._mm[:8])[0]
        header = json.loads(self._mm[8 : 8 + header_len])
        header.pop("__metadata__", None)
        self._header: dict[str, dict] = header
        self._data_offset = 8 + header_len

    def keys(self) -> list[str]:
        return list(self._header)

    def __contains__(self, name: str) -> bool:
        return name in self._header

    def info(self, name: str) -> dict:
        return self._header[name]

    def get(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        info = self._header[name]
        begin, end = info["data_offsets"]
        view = memoryview(self._mm)[self._data_offset + begin : self._data_offset + end]
        with warnings.catch_warnings():
            # The mapping is read-only and we never mutate the tensor.
            warnings.filterwarnings("ignore", message=".*not writable.*")
            tensor = torch.frombuffer(view, dtype=_DTYPES[info["dtype"]])
        tensor = tensor.reshape(info["shape"])
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor
