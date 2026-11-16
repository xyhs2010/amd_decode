"""Byte-level alignment for alternative ``Qwen3Torch._layer`` implementations.

``Qwen3Torch.forward`` calls ``self._layer(...)``, so a subclass only has to
override ``_layer``; nothing in ``model.py`` needs to change::

    class MyQwen3(Qwen3Torch):
        def _layer(self, x, p, cos, sin, kv, layer_idx, past):
            ...  # new maths, same signature

    ok = align("Qwen3-8B", Qwen3Torch, MyQwen3, tokens=[1, 2, 3])

The implementation temporarily wraps ``_layer`` on both instances, so it can
compare the output of *every* layer (and the final logits) instead of only the
end result. That makes a mismatch point at the layer that caused it.

Simplifications, on purpose:
  * per-layer outputs are copied to CPU, so keep ``tokens`` short;
  * exactness means identical raw bytes, not a numerical tolerance;
  * either argument may be a class (built from ``model_dir``) or a ready-made
    instance.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import KVCache, Qwen3Torch


@dataclass
class Check:
    name: str
    exact: bool
    detail: str = ""


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True iff ``a`` and ``b`` have the same shape/dtype and identical bytes."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _check(name: str, ref: torch.Tensor, alt: torch.Tensor) -> Check:
    if ref.shape != alt.shape:
        return Check(name, False, f"shape {tuple(ref.shape)} != {tuple(alt.shape)}")
    if ref.dtype != alt.dtype:
        return Check(name, False, f"dtype {ref.dtype} != {alt.dtype}")
    if _same_bytes(ref, alt):
        return Check(name, True)
    diff = (ref.float() - alt.float()).abs()
    return Check(
        name,
        False,
        f"bytes differ: max|d|={diff.max().item():.3e} mean|d|={diff.mean().item():.3e}",
    )


@contextlib.contextmanager
def _capture_layers(model):
    """Yield a dict filled with each layer's output while the block runs."""
    original = model._layer
    captured: dict[int, torch.Tensor] = {}

    def wrapper(x, p, cos, sin, kv, layer_idx, past):
        out = original(x, p, cos, sin, kv, layer_idx, past)
        captured[layer_idx] = out.detach().to("cpu")
        return out

    # Plain instance attribute: shadows the bound method for this instance
    # only, and (unlike a class attribute) does not receive ``self``.
    model._layer = wrapper
    try:
        yield captured
    finally:
        model._layer = original


@dataclass
class _Step:
    position: int
    layers: dict[int, torch.Tensor] = field(default_factory=dict)
    logits: torch.Tensor | None = None


def _build(model_or_cls, model_dir, device, dtype, weight_bits):
    if isinstance(model_or_cls, type):
        return model_or_cls(
            model_dir, device=device, dtype=dtype, weight_bits=weight_bits
        )
    return model_or_cls


def _chunks(tokens, decode: bool) -> list[list[int]]:
    ids = torch.as_tensor(tokens).flatten().tolist()
    if not ids:
        raise ValueError("tokens must not be empty")
    return [[tok] for tok in ids] if decode else [ids]


def _run(model, chunks: list[list[int]], use_kv: bool) -> list[_Step]:
    kv = KVCache(model.num_layers) if use_kv else None
    steps: list[_Step] = []
    position = 0
    for chunk in chunks:
        with _capture_layers(model) as layers:
            logits = model.forward(chunk, kv=kv, position_offset=position)
        steps.append(_Step(position, dict(layers), logits.detach().to("cpu")))
        position += len(chunk)
    return steps


def align(
    model_dir: str | Path,
    reference,
    candidate,
    tokens,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    weight_bits: int = 4,
    decode: bool = False,
    use_kv: bool = True,
    verbose: bool = True,
) -> bool:
    """Run ``reference`` and ``candidate`` on the same tokens and compare bytes.

    Args:
        model_dir: weights directory, used only when an argument is a class.
        reference: class or instance of the known-good implementation.
        candidate: class or instance of the new implementation.
        tokens: 1-D token ids (list or tensor).
        decode: feed one token at a time through the KV cache instead of one
            prefill pass. ``use_kv`` must be true in this mode.
        use_kv: pass a fresh ``KVCache`` to ``forward``. A fresh cache gives
            the same maths as ``kv=None``; setting this to false tests the
            cache-less branch of ``_layer``.
        verbose: print a per-step summary and every mismatching tensor.

    Returns:
        True iff every compared tensor is byte-for-byte identical.
    """
    if decode and not use_kv:
        raise ValueError("decode=True requires use_kv=True")

    ref_model = _build(reference, model_dir, device, dtype, weight_bits)
    alt_model = _build(candidate, model_dir, device, dtype, weight_bits)
    chunks = _chunks(tokens, decode)

    ref_steps = _run(ref_model, chunks, use_kv)
    alt_steps = _run(alt_model, chunks, use_kv)

    if verbose:
        # Show the models' actual settings: kwargs are ignored for instances.
        print(
            f"align: device={getattr(ref_model, 'device', device)} "
            f"dtype={getattr(ref_model, 'dtype', dtype)} "
            f"weight_bits={getattr(ref_model, 'weight_bits', weight_bits)} "
            f"decode={decode} use_kv={use_kv}"
        )

    ok = True
    for step_idx, (ref_step, alt_step) in enumerate(zip(ref_steps, alt_steps)):
        where = f"step {step_idx} (pos {ref_step.position})"
        checks: list[Check] = []

        if ref_step.layers.keys() != alt_step.layers.keys():
            checks.append(
                Check(
                    f"{where} layers",
                    False,
                    f"captured {sorted(ref_step.layers)} != {sorted(alt_step.layers)}",
                )
            )
        for layer_idx in ref_step.layers.keys() & alt_step.layers.keys():
            checks.append(
                _check(
                    f"{where} layer {layer_idx}",
                    ref_step.layers[layer_idx],
                    alt_step.layers[layer_idx],
                )
            )
        checks.append(_check(f"{where} logits", ref_step.logits, alt_step.logits))

        exact = sum(c.exact for c in checks)
        ok &= exact == len(checks)
        if verbose:
            n_layers = len(ref_step.layers)
            layers_ok = sum(c.exact for c in checks if " layer " in c.name)
            logits_ok = checks[-1].exact
            print(
                f"{where}: layers {layers_ok}/{n_layers} exact, "
                f"logits {'exact' if logits_ok else 'DIFFER'}"
            )
            for check in checks:
                if not check.exact:
                    print(f"  MISMATCH {check.name}: {check.detail}")

    if verbose:
        print("byte-identical" if ok else "MISMATCH")
    return ok
