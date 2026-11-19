"""Tolerance alignment for decoding backends.

A backend is any object implementing :class:`DecodeModel` - four operations:

* ``load_prefill(saved)``: build the cache and seed it from an offline prefill
  (``None`` clears it);
* ``set_hook(hook)``: call ``hook(layer_idx, output)`` after every layer; the
  hook's return value is the captured *feature* for that layer;
* ``forward(input_ids, position_offset=None)``: run one pass, return logits;
* ``get_features()``: features captured during the most recent ``forward``.

The protocol is defined in ``amd_decode``.  ``Qwen35Torch`` implements it
directly.  A future kernel backend only has to provide the same four methods;
``align`` does not know anything else about it.

``align(reference, candidate, prefill, tokens)`` loads the same offline prefill
into both objects and runs the same tokens, comparing every layer feature and
the logits within a fixed relative tolerance (``RTOL`` below).  Each backend
builds its own cache from the same saved directory, so *storage* may differ -
fp16 in VRAM vs fp16 in pinned host memory - while the values are identical.
``prefill=None`` runs the cache-less branch: one plain prefill pass, no cache.

Simplifications, on purpose: features are copied to CPU, so keep ``tokens``
short; the tolerance is fixed here rather than configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from . import DecodeModel

# Per-tensor relative L2 error: ||ref - alt|| / ||ref||.  A per-element
# tolerance is useless here: Qwen3.5 residual streams mix channels of ~1e4 with
# near-zero ones.  3e-2 covers the drift of the int4 FFN kernel, whose dequant
# is closer to the quantized weights than the reference's extra bf16 rounding;
# logits stay within ~5e-3 and greedy argmax matches.
RTOL = 3e-2


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _check(name: str, ref: torch.Tensor, alt: torch.Tensor) -> Check:
    a, b = ref.float(), alt.float()
    err = float((a - b).norm() / b.norm().clamp_min(1e-12))
    return Check(name, err <= RTOL, f"rel={err:.2e}")


@dataclass
class _Step:
    position: int
    features: dict[int, Any] = field(default_factory=dict)
    logits: torch.Tensor | None = None


def _chunks(tokens, decode: bool) -> list[list[int]]:
    ids = torch.as_tensor(tokens).flatten().tolist()
    if not ids:
        raise ValueError("tokens must not be empty")
    return [[tok] for tok in ids] if decode else [ids]


def _run(model, prefill, chunks: list[list[int]]) -> list[_Step]:
    """Run the chunks, capturing every layer feature and the logits on CPU."""
    position = 0 if prefill is None else prefill.seq_len
    model.load_prefill(prefill, position + sum(len(c) for c in chunks))
    model.set_hook(lambda _, out: out.detach().to("cpu"))
    steps: list[_Step] = []
    try:
        for chunk in chunks:
            logits = model.forward(chunk, position_offset=position)
            steps.append(
                _Step(position, dict(model.get_features()),
                      logits.detach().to("cpu"))
            )
            position += len(chunk)
    finally:
        model.set_hook(None)
        model.load_prefill(None)
    return steps


def align(
    reference: DecodeModel,
    candidate: DecodeModel,
    prefill,
    tokens,
    *,
    decode: bool = False,
    verbose: bool = True,
) -> bool:
    """Run ``reference`` and ``candidate`` on the same tokens and compare.

    Args:
        reference: the known-good backend object.
        candidate: the backend under test.
        prefill: an offline prefill (``SavedKVCache``) both backends load, or
            ``None`` for the cache-less branch.
        tokens: 1-D token ids (list or tensor).
        decode: feed one token at a time through the cache instead of one
            prefill pass.
        verbose: print a per-step summary and every mismatching tensor.

    Returns:
        True iff every compared feature and logit is within tolerance.
    """
    if decode and prefill is None:
        raise ValueError("decode=True requires a prefill")

    chunks = _chunks(tokens, decode)
    ref_steps = _run(reference, prefill, chunks)
    alt_steps = _run(candidate, prefill, chunks)

    if verbose:
        print(f"align: decode={decode} "
              f"start_pos={ref_steps[0].position if ref_steps else 0}")

    ok = True
    for step_idx, (ref_step, alt_step) in enumerate(zip(ref_steps, alt_steps)):
        where = f"step {step_idx} (pos {ref_step.position})"
        checks: list[Check] = []

        if ref_step.features.keys() != alt_step.features.keys():
            checks.append(
                Check(
                    f"{where} layers",
                    False,
                    f"captured {sorted(ref_step.features)} != "
                    f"{sorted(alt_step.features)}",
                )
            )
        for layer_idx in ref_step.features.keys() & alt_step.features.keys():
            checks.append(
                _check(
                    f"{where} layer {layer_idx}",
                    ref_step.features[layer_idx],
                    alt_step.features[layer_idx],
                )
            )
        checks.append(_check(f"{where} logits", ref_step.logits, alt_step.logits))

        passed = sum(c.ok for c in checks)
        ok &= passed == len(checks)
        if verbose:
            n_layers = len(ref_step.features)
            layers_ok = sum(c.ok for c in checks if " layer " in c.name)
            logits_ok = checks[-1].ok
            print(
                f"{where}: layers {layers_ok}/{n_layers} within tol, "
                f"logits {'ok' if logits_ok else 'DIFFER'}"
            )
            for check in checks:
                if not check.ok:
                    print(f"  MISMATCH {check.name}: {check.detail}")

    if verbose:
        print("within tolerance" if ok else "MISMATCH")
    return ok
