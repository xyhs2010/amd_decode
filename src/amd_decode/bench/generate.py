"""Greedy generation on top of a saved dense-prefill KV cache.

The evaluator only needs one method from a decode backend::

    forward(input_ids, kv=None, position_offset=0) -> torch.Tensor

``Qwen3Torch`` satisfies it; a future kernel backend is expected to as well.
The backend may expose ``device`` / ``dtype`` / ``num_layers`` attributes; they
are picked up when present.

``SavedKVCacheView`` is read-only and does not retain newly generated K/V, so
generation uses :class:`SavedPrefixKVCache`: it keeps the saved prefix on disk
and grows a dense in-memory tail for the answer prefix and generated tokens.
The saved prefix's K/V is dequantized one layer at a time and released
immediately, so no layer's full KV is kept resident across steps.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import torch

from ..with_torch.kv_cache_file import SavedKVCache
from ..with_torch.model import KVCache
from .types import EvalSample, Prediction

_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


@runtime_checkable
class DecodeBackend(Protocol):
    """The minimal decode interface needed by the evaluator."""

    def forward(
        self,
        input_ids,
        kv=None,
        position_offset: int = 0,
    ) -> torch.Tensor: ...


class SavedPrefixKVCache:
    """``KVCache`` protocol over a saved prefix plus a growing dense tail.

    ``Qwen3Torch.forward`` checks ``kv.length == position_offset`` before the
    layer loop, calls ``append(layer_idx, k, v)`` once per layer, and expects
    the full K/V back. ``length`` is the saved prefix length plus however many
    tokens the tail currently holds, so it grows automatically after a
    forward pass.
    """

    def __init__(
        self,
        cache: SavedKVCache,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if cache.num_layers < 1:
            raise ValueError("saved cache has no layers")
        self._cache = cache
        self._device = device
        self._dtype = dtype
        self._prefix_len = cache.seq_len
        self._tail = KVCache(cache.num_layers)

    @property
    def length(self) -> int:
        first = self._tail.k[0]
        return self._prefix_len + (0 if first is None else first.shape[-2])

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_k, prefix_v = self._cache.layer(layer_idx, self._device, self._dtype)
        tail_k, tail_v = self._tail.append(layer_idx, k, v)
        return (
            torch.cat((prefix_k, tail_k), dim=-2),
            torch.cat((prefix_v, tail_v), dim=-2),
        )


def postprocess_prediction(text: str) -> str:
    """RULER's ``postprocess_pred``: strip, then map control chars to newlines."""
    return _CONTROL_CHARS.sub("\n", text.strip()).strip()


def resolve_device_dtype(
    decoder: DecodeBackend,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype]:
    if device is None:
        device = getattr(decoder, "device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = getattr(decoder, "dtype", None)
    if dtype is None:
        dtype = torch.float16
    return torch.device(device), dtype


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    """[1, T, V] or [1, V] -> [V] for the last position."""
    if logits.dim() == 3:
        return logits[0, -1]
    if logits.dim() == 2:
        return logits[0]
    raise ValueError(f"unexpected logits shape {tuple(logits.shape)}")


@torch.inference_mode()
def generate(
    decoder: DecodeBackend,
    tokenizer,
    sample: EvalSample,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    max_new_tokens: int | None = None,
) -> Prediction:
    """Greedy-decode an answer after the saved prompt and ``answer_prefix``."""
    device, dtype = resolve_device_dtype(decoder, device, dtype)
    backend_layers = getattr(decoder, "num_layers", None)

    cache = SavedKVCache(sample.cache_path)
    if (
        backend_layers is not None
        and cache.num_layers != backend_layers
    ):
        raise ValueError(
            f"{sample.sample_id}: cache has {cache.num_layers} layers, "
            f"decoder has {backend_layers}"
        )

    prefix_ids = tokenizer.encode(sample.answer_prefix, add_special_tokens=False)
    if not prefix_ids:
        raise ValueError(f"{sample.sample_id}: empty answer_prefix")

    kv = SavedPrefixKVCache(cache, device=device, dtype=dtype)
    limit = sample.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
    if limit < 1:
        raise ValueError(f"{sample.sample_id}: generation budget must be >= 1")

    # The answer prefix is fed as one short prefill on top of the saved prompt.
    logits = decoder.forward(prefix_ids, kv=kv, position_offset=kv.length)
    last = _last_logits(logits)

    generated: list[int] = []
    for _ in range(limit):
        next_id = int(last.argmax().item())
        if next_id in tokenizer.eos_token_ids:
            break
        generated.append(next_id)

        if sample.stop:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if any(stop in text for stop in sample.stop):
                break
        if len(generated) >= limit:
            break

        logits = decoder.forward([next_id], kv=kv, position_offset=kv.length)
        last = _last_logits(logits)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    for stop in sample.stop:
        text = text.split(stop)[0]
    return Prediction(
        sample_id=sample.sample_id,
        text=postprocess_prediction(text),
        generated_ids=tuple(generated),
    )
