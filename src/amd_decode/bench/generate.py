"""Greedy or sampled generation over whatever cache the decoder has attached.

The evaluator only needs :class:`amd_decode.DecodeModel`::

    decoder.load_prefill(saved)       # done by the caller, per sample
    decoder.forward(input_ids)        # starts at the cache's length

The backend owns its cache; this module never constructs or inspects one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from .. import DecodeModel
from .types import EvalSample, Prediction

_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


@dataclass(frozen=True)
class SamplingConfig:
    """Token selection; ``temperature <= 0`` keeps greedy argmax.

    ``presence_penalty`` subtracts the penalty once per already-generated
    token.  The prompt is not visible here (only the prefill cache is), so
    unlike ``transformers.generate`` the prompt tokens are not penalised.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    seed: int = 0


def _next_token(last: torch.Tensor, generated: list[int],
                sampling: SamplingConfig, generator: torch.Generator) -> int:
    if sampling.temperature <= 0:
        return int(last.argmax().item())

    logits = last.float()
    if sampling.presence_penalty and generated:
        ids = torch.tensor(sorted(set(generated)), device=logits.device)
        logits[ids] -= sampling.presence_penalty
    logits = logits / sampling.temperature

    if sampling.top_k > 0 and sampling.top_k < logits.numel():
        kth = torch.topk(logits, sampling.top_k).values[-1]
        logits = torch.where(logits < kth, -math.inf, logits)
    if sampling.top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True)
        probs = torch.softmax(ordered, dim=-1)
        drop = (probs.cumsum(0) - probs) > sampling.top_p
        ordered = ordered.masked_fill(drop, -math.inf)
        logits = torch.full_like(logits, -math.inf).scatter(0, index, ordered)
    if sampling.min_p > 0:
        probs = torch.softmax(logits, dim=-1)
        logits = logits.masked_fill(
            probs < sampling.min_p * probs.max(), -math.inf)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


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
    sampling: SamplingConfig | None = None,
) -> Prediction:
    """Decode an answer after the attached prefix and ``answer_prefix``.

    The caller must have loaded the sample's cache with
    ``decoder.load_prefill``; ``forward`` starts at that cache's ``length``.
    ``sampling=None`` is greedy.
    """
    prefix_ids = tokenizer.encode(sample.answer_prefix, add_special_tokens=False)
    if not prefix_ids:
        raise ValueError(f"{sample.sample_id}: empty answer_prefix")

    limit = sample.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
    if limit < 1:
        raise ValueError(f"{sample.sample_id}: generation budget must be >= 1")

    # Feed the prefix one token at a time: a multi-token forward takes the
    # dense prefill attention, which is expensive on a 256K cache and OOMs the
    # kernel backends.  Token-by-token decode is equivalent for causal models.
    for token_id in prefix_ids:
        last = _last_logits(decoder.forward([token_id]))

    sampling = sampling or SamplingConfig()
    generator = torch.Generator(device=last.device)
    generator.manual_seed(sampling.seed)

    generated: list[int] = []
    for _ in range(limit):
        next_id = _next_token(last, generated, sampling, generator)
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

    # Qwen3.5 answer rule: score the text after the last ``</think>``.  The
    # token is special, so it is not in the decoded text; trim by id instead.
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    cut = -1
    if len(end_ids) == 1:
        cut = max((i for i, t in enumerate(generated) if t == end_ids[0]), default=-1)
    text = tokenizer.decode(generated[cut + 1 :], skip_special_tokens=True)
    for stop in sample.stop:
        text = text.split(stop)[0]
    return Prediction(
        sample_id=sample.sample_id,
        text=postprocess_prediction(text),
        generated_ids=tuple(generated),
    )
