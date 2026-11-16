"""16K-context decode: 4-bit weights, 16-bit KV cache in host memory.

The context comes from a saved hydra_llm prefill cache (q8 on disk); it is
dequantised once to fp16 and written into a ``HostKVCache``, so the 2.4 GB of
K/V lives in host RAM and none of it in VRAM.  Decode then streams one layer's
4-bit weights at a time, exactly like ``Qwen3Torch``.

The reference path runs the same steps with the stock ``Qwen3Torch`` and a
device ``KVCache`` seeded from the same fp16 K/V.  Both logits (every decode
step) and every layer output of the first step are compared byte-for-byte.

Run from the repo root:

    PYTHONPATH=src .venv/bin/python examples/host_kv_decode.py
    PYTHONPATH=src .venv/bin/python examples/host_kv_decode.py --steps 3
    PYTHONPATH=src .venv/bin/python examples/host_kv_decode.py --sample data/kv_cache/.../L16384/niah_single_2/<id>

The first forward pays a one-time int4 quantisation of the whole model (tens of
seconds); later steps only stream/dequantise one layer at a time.  The two
models share that cache, so the reference run is not charged twice.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch

from amd_decode.with_host import HostQwen3
from amd_decode.with_torch import KVCache, Qwen3Torch
from amd_decode.with_torch.kv_cache_file import SavedKVCache

CONTEXT_LEN = 16384


@contextlib.contextmanager
def capture_layers(model, captured: dict[int, torch.Tensor]):
    """Record every layer output (on CPU) while the block runs."""
    original = model._layer

    def wrapper(x, p, cos, sin, kv, layer_idx, past):
        out = original(x, p, cos, sin, kv, layer_idx, past)
        captured[layer_idx] = out.detach().to("cpu")
        return out

    model._layer = wrapper
    try:
        yield captured
    finally:
        model._layer = original


def find_sample(root: Path, length: int) -> Path:
    pattern = f"*/q8/ruler/L{length}/*/*/meta.json"
    samples = sorted(root.glob(pattern))
    if not samples:
        raise SystemExit(
            f"no saved L{length} cache under {root} (expected {pattern}); "
            "pass --sample explicitly"
        )
    return samples[0].parent


def sync() -> None:
    torch.cuda.synchronize()


def decode(
    model,
    kv,
    first_token: int,
    steps: int,
    capture_step: int | None = None,
):
    """Feed one token at a time; return step logits, tokens, times, layer dump."""
    logits_out: list[torch.Tensor] = []
    tokens: list[int] = []
    times: list[float] = []
    captured: dict[int, torch.Tensor] = {}
    token = first_token
    position = kv.length

    for step in range(steps):
        cm = (
            capture_layers(model, captured)
            if step == capture_step
            else contextlib.nullcontext()
        )
        with cm:
            sync()
            t0 = time.perf_counter()
            logits = model.forward([token], kv=kv, position_offset=position)
            sync()
        times.append(time.perf_counter() - t0)

        last = logits[0, -1].detach()
        logits_out.append(last.to("cpu"))
        token = int(last.argmax())
        tokens.append(token)
        position += 1

    return logits_out, tokens, times, captured


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="data/models/Qwen3-8B-Base")
    parser.add_argument("--cache-root", default="data/kv_cache")
    parser.add_argument("--sample", default=None,
                        help="a saved cache directory (contains meta.json)")
    parser.add_argument("--length", type=int, default=CONTEXT_LEN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-bits", type=int, default=4, choices=(4, 16))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--start-token", type=int, default=None,
                        help="decode input at position seq_len (default: last "
                             "context token)")
    args = parser.parse_args()

    sample_dir = (
        Path(args.sample) if args.sample
        else find_sample(Path(args.cache_root), args.length)
    )
    saved = SavedKVCache(sample_dir)
    seq_len = saved.seq_len
    if args.start_token is not None:
        start_token = args.start_token
    else:
        tokens = saved.tokens()
        start_token = int(tokens[min(seq_len, tokens.numel()) - 1])

    device = torch.device(args.device)
    host = HostQwen3(
        args.model_dir, device=device, dtype=torch.float16,
        weight_bits=args.weight_bits,
    )
    host_kv = host.make_kv_cache(seq_len + args.steps + 1)
    kv_bytes = host_kv.nbytes

    print(f"sample      : {sample_dir}")
    print(f"seq_len     : {seq_len}")
    print(f"model       : {args.model_dir} weight_bits={args.weight_bits} "
          f"dtype={host.dtype}")
    print(f"host KV     : {kv_bytes / 2**30:.2f} GiB, dtype={host.kv_dtype}, "
          f"capacity={host_kv.max_len} tokens, VRAM=0")
    print(f"start token : {start_token} at position {seq_len}")

    # ---- fill the host cache from the saved q8 prefill (dequantised to fp16)
    sync()
    t0 = time.perf_counter()
    for layer_idx in range(host.num_layers):
        k, v = saved.layer(layer_idx, device=device, dtype=host.kv_dtype)
        host_kv.set_layer(layer_idx, k, v)
        del k, v
    sync()
    host_kv.set_length(seq_len)
    print(f"seed host KV: {time.perf_counter() - t0:.2f} s "
          f"({host.num_layers} layers from q8)")

    # ---- decode with host KV
    host_logits, host_tokens, host_times, host_layers = decode(
        host, host_kv, start_token, args.steps, capture_step=0
    )

    # ---- reference: same fp16 K/V on the GPU, stock Qwen3Torch
    ref = Qwen3Torch(
        args.model_dir, device=device, dtype=torch.float16,
        weight_bits=args.weight_bits,
    )
    # Both instances quantise identically; share the 4-bit host tensors so the
    # reference does not double the CPU footprint or the one-time cost.
    ref._int4_cache = host._int4_cache

    ref_kv = KVCache(ref.num_layers)
    for layer_idx in range(ref.num_layers):
        k, v = host_kv.layer_kv(layer_idx)
        # .contiguous() copies the first seq_len tokens host -> device.
        ref_kv.k[layer_idx] = k[:, :, :seq_len].contiguous()
        ref_kv.v[layer_idx] = v[:, :, :seq_len].contiguous()
    ref_kv.length = seq_len
    del k, v
    sync()

    ref_logits, ref_tokens, ref_times, ref_layers = decode(
        ref, ref_kv, start_token, args.steps, capture_step=0
    )

    # ---- compare
    ok = True
    print("\nstep   host-KV decode   device-KV decode   logits   token")
    for step, (ht, rt) in enumerate(zip(host_times, ref_times)):
        logits_ok = same_bytes(host_logits[step], ref_logits[step])
        token_ok = host_tokens[step] == ref_tokens[step]
        ok &= logits_ok and token_ok
        note = " (includes int4 quantisation)" if step == 0 else ""
        print(f"{step:4d} {ht:14.2f}s {rt:17.2f}s   "
              f"{'exact' if logits_ok else 'DIFF':>6s}   "
              f"{'same' if token_ok else 'DIFF':>5s}{note}")

    layer_ok = sum(
        same_bytes(host_layers[i], ref_layers[i]) for i in host_layers
    )
    ok &= layer_ok == len(host_layers) and host_layers.keys() == ref_layers.keys()
    print(f"layer outputs of step 0: {layer_ok}/{len(host_layers)} byte-identical")

    kv_read = host.kv_nbytes_per_token * seq_len
    steady_host = (sorted(host_times[1:])[len(host_times[1:]) // 2]
                   if len(host_times) > 1 else host_times[0])
    steady_ref = (sorted(ref_times[1:])[len(ref_times[1:]) // 2]
                  if len(ref_times) > 1 else ref_times[0])
    print(f"\nKV touched per step: {kv_read / 2**30:.2f} GiB "
          f"(PCIe floor {kv_read / 12e9:.2f} s at 12 GB/s)")
    print(f"steady state: host KV {steady_host:.2f} s/token, "
          f"device KV {steady_ref:.2f} s/token "
          f"(delta {steady_host - steady_ref:+.2f} s)")
    print("byte-identical to Qwen3Torch" if ok else "MISMATCH")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
