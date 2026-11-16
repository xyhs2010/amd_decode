"""16K-context decode with 4-bit weights and an int8 KV cache in host memory.

The saved q8 prefill is copied into pinned host memory *as stored* (int8 plus
the prefill machine's scales) and dequantised one layer at a time on the GPU.
At 16K the cache is 1.1 GiB of host RAM and zero VRAM.

The reference stores the same int8 K/V and scales in VRAM (``storage="device"``),
so both backends run identical quantisation/dequantisation code; only the
storage location differs.  ``align`` compares every layer feature and the
logits byte-for-byte.

Run from the repo root:

    PYTHONPATH=src .venv/bin/python examples/host_kv_decode.py
    PYTHONPATH=src .venv/bin/python examples/host_kv_decode.py --steps 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from amd_decode.align import align
from amd_decode.utils.prefill_cache import SavedKVCache
from amd_decode.with_host import HostQwen3
from amd_decode.with_torch import Qwen3Torch

CONTEXT_LEN = 16384


def find_sample(root: Path) -> Path:
    samples = sorted(root.glob(f"*/q8/ruler/L{CONTEXT_LEN}/*/*/meta.json"))
    if not samples:
        raise SystemExit(
            f"no saved L{CONTEXT_LEN} cache under {root}; pass --sample"
        )
    return samples[0].parent


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def time_decode(model, kv, token: int, steps: int) -> list[float]:
    """Per-token latency of the attended cache (no result comparison)."""
    model.set_kv_cache(kv)
    times: list[float] = []
    try:
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward([token], position_offset=kv.length)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    finally:
        model.set_kv_cache(None)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="data/models/Qwen3-8B-Base")
    parser.add_argument("--cache-root", default="data/kv_cache")
    parser.add_argument("--sample", default=None,
                        help="a saved cache directory (contains meta.json)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--start-token", type=int, default=None)
    args = parser.parse_args()

    sample_dir = Path(args.sample) if args.sample else find_sample(Path(args.cache_root))
    saved = SavedKVCache(sample_dir)
    seq_len = saved.seq_len
    if args.start_token is not None:
        start_token = args.start_token
    else:
        tokens = saved.tokens()
        start_token = int(tokens[min(seq_len, tokens.numel()) - 1])

    device = torch.device(args.device)
    host = HostQwen3(args.model_dir, device=device, dtype=torch.float16,
                     weight_bits=4)
    host_kv = host.make_kv_cache(seq_len + args.steps + 1)

    print(f"sample      : {sample_dir}")
    print(f"seq_len     : {seq_len}")
    print(f"host KV     : {host_kv.nbytes / 2**30:.2f} GiB int8, VRAM=0")
    print(f"start token : {start_token} at position {seq_len}", flush=True)

    t0 = time.perf_counter()
    host_kv.seed_from_prefill(saved)
    torch.cuda.synchronize()
    print(f"seed host KV: {time.perf_counter() - t0:.2f} s")

    # Reference: identical int8 KV in VRAM, stock Qwen3Torch.
    ref = Qwen3Torch(args.model_dir, device=device, dtype=torch.float16,
                     weight_bits=4)
    ref._int4_cache = host._int4_cache
    dev_kv = ref.make_kv_cache(seq_len + args.steps + 1)  # VRAM, q8
    dev_kv.seed_from_prefill(saved)

    prefix_ok = 0
    for layer_idx in range(host.num_layers):
        hk, hv = host_kv.dequantize(layer_idx, seq_len)
        rk, rv = saved.layer(layer_idx, device=device, dtype=torch.float16)
        if same_bytes(hk, rk) and same_bytes(hv, rv):
            prefix_ok += 1
        del hk, hv, rk, rv
    print(f"prefix dequantisation: {prefix_ok}/{host.num_layers} layers "
          f"byte-identical to SavedKVCache.layer", flush=True)

    # Same tokens fed to both backends, one at a time.
    ok = align(
        host, ref, (host_kv, dev_kv), [start_token] * args.steps, decode=True
    )

    # Timing uses fresh logical positions; the appended tokens are overwritten.
    host_kv.set_length(seq_len)
    dev_kv.set_length(seq_len)
    host_times = time_decode(host, host_kv, start_token, args.steps)
    host_kv.set_length(seq_len)
    dev_times = time_decode(ref, dev_kv, start_token, args.steps)

    kv_read = host_kv.bytes_read(seq_len)
    print(f"\nKV touched per step: {kv_read / 2**30:.2f} GiB "
          f"(PCIe floor {kv_read / 12e9:.2f} s at 12 GB/s)")
    print(f"host-q8   : {statistics.median(host_times):.2f} s/token")
    print(f"device-q8 : {statistics.median(dev_times):.2f} s/token")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
