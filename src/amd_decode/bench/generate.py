"""Greedy generation over whatever cache the decoder has attached.

The evaluator only needs :class:`amd_decode.DecodeModel`::

    decoder.load_prefill(saved)       # done by the caller, per sample
    decoder.forward(input_ids)        # starts at the cache's length

The backend owns its cache; this module never constructs or inspects one.
"""

from __future__ import annotations

import re

import torch

from .. import DecodeModel
from .types import EvalSample, Prediction

_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def postprocess_prediction(text: str) -> str:
    """RULER's ``postprocess_pred``: strip, then map control chars to newlines."""
    return _CONTROL_CHARS.sub("\n", text.strip()).strip()


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    """[1, T, V] or [1, V] -> [V] for the last position."""
    if logits.dim() == 3:
        return logits[0, -1]
    if logits.dim() == 2:
        return logits[0]
    raise ValueError(f"unexpected logits shape {tuple(logits.shape)}")


@torch.inference_mode()
def generate(
    decoder: DecodeModel,
    tokenizer,
    sample: EvalSample,
    *,
    max_new_tokens: int | None = None,
) -> Prediction:
    """Greedy-decode an answer after the attached prefix and ``answer_prefix``.

    The caller must have loaded the sample's cache with
    ``decoder.load_prefill``; ``forward`` starts at that cache's ``length``.
    """
    prefix_ids = tokenizer.encode(sample.answer_prefix, add_special_tokens=False)
    if not prefix_ids:
        raise ValueError(f"{sample.sample_id}: empty answer_prefix")

    limit = sample.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
    if limit < 1:
        raise ValueError(f"{sample.sample_id}: generation budget must be >= 1")

    # The answer prefix is fed as one short prefill on top of the saved prompt.
    logits = decoder.forward(prefix_ids)
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

        logits = decoder.forward([next_id])
        last = _last_logits(logits)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    for stop in sample.stop:
        text = text.split(stop)[0]
    return Prediction(
        sample_id=sample.sample_id,
        text=postprocess_prediction(text),
        generated_ids=tuple(generated),
    )
