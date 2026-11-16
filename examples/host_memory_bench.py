"""Zero-copy host memory on ROCm: can the GPU use host RAM directly?

Answer for this machine (gfx906 / Radeon Pro VII, ROCm 6.3): yes.  ``hipHostMalloc``
(or a PyTorch pinned tensor) can be turned into a CUDA tensor with
``hipHostGetDevicePointer`` + the private torch storage constructors below, and
GPU kernels then read/write the buffer over PCIe with no copy.  The device
pointer happens to equal the host pointer here.

Two things bite:

* ``torch.empty(..., pin_memory=True)`` is allocated with default (coherent)
  HIP flags.  GPU reads are fast (~12 GB/s) but GPU element-wise *stores* are
  ~0.33 GB/s (4 KB store ~14 us).  ``hipHostMallocNonCoherent`` restores
  ~12 GB/s stores and is what a GPU-written KV cache should use.
* Bulk ``copy_`` (DMA path) is fast in both directions (~12 GB/s); only
  element-wise store kernels are affected by the coherent/non-coherent flag.

Run from the repo root (no ``PYTHONPATH`` needed):

    .venv/bin/python examples/host_memory_bench.py

The wrapper uses private torch APIs (``_construct_storage_from_data_pointer``,
also used by torch.compile's cudagraph trees) because there is no public way to
build a CUDA tensor from a raw device pointer.  Re-check them on torch upgrades.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import time

import torch

HIP_LIBRARY = "/opt/rocm/lib/libamdhip64.so"

HIP_HOST_MALLOC_MAPPED = 0x2
HIP_HOST_MALLOC_WRITE_COMBINED = 0x4
HIP_HOST_MALLOC_COHERENT = 0x40000000
HIP_HOST_MALLOC_NON_COHERENT = 0x80000000
HIP_HOST_REGISTER_MAPPED = 0x2
HIP_MEM_ATTACH_GLOBAL = 0x1

# Best default for a buffer the GPU both reads and writes (KV cache).
HOST_MALLOC_FLAGS = HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_NON_COHERENT


class Hip:
    """Minimal ctypes binding for the HIP host-memory entry points."""

    def __init__(self, path: str = HIP_LIBRARY) -> None:
        self.lib = ctypes.CDLL(path)
        lib = self.lib
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        lib.hipHostMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint
        ]
        lib.hipHostMalloc.restype = ctypes.c_int
        lib.hipHostFree.argtypes = [ctypes.c_void_p]
        lib.hipHostFree.restype = ctypes.c_int
        lib.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint
        ]
        lib.hipHostGetDevicePointer.restype = ctypes.c_int
        lib.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        lib.hipHostRegister.restype = ctypes.c_int
        lib.hipHostUnregister.argtypes = [ctypes.c_void_p]
        lib.hipHostUnregister.restype = ctypes.c_int
        lib.hipMallocManaged.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint
        ]
        lib.hipMallocManaged.restype = ctypes.c_int
        lib.hipFree.argtypes = [ctypes.c_void_p]
        lib.hipFree.restype = ctypes.c_int

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            msg = self.lib.hipGetErrorString(rc).decode()
            raise RuntimeError(f"{what} failed: {msg} (rc={rc})")

    def host_malloc(self, nbytes: int, flags: int = HOST_MALLOC_FLAGS) -> int:
        ptr = ctypes.c_void_p()
        self._check(
            self.lib.hipHostMalloc(ctypes.byref(ptr), nbytes, ctypes.c_uint(flags)),
            "hipHostMalloc",
        )
        return ptr.value

    def host_free(self, ptr: int) -> None:
        self._check(self.lib.hipHostFree(ctypes.c_void_p(ptr)), "hipHostFree")

    def host_device_pointer(self, host_ptr: int) -> int:
        dev = ctypes.c_void_p()
        self._check(
            self.lib.hipHostGetDevicePointer(
                ctypes.byref(dev), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)
            ),
            "hipHostGetDevicePointer",
        )
        return dev.value

    def host_register(self, host_ptr: int, nbytes: int,
                      flags: int = HIP_HOST_REGISTER_MAPPED) -> None:
        self._check(
            self.lib.hipHostRegister(
                ctypes.c_void_p(host_ptr), nbytes, ctypes.c_uint(flags)
            ),
            "hipHostRegister",
        )

    def host_unregister(self, host_ptr: int) -> None:
        self._check(
            self.lib.hipHostUnregister(ctypes.c_void_p(host_ptr)), "hipHostUnregister"
        )

    def malloc_managed(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(
            self.lib.hipMallocManaged(
                ctypes.byref(ptr), nbytes, ctypes.c_uint(HIP_MEM_ATTACH_GLOBAL)
            ),
            "hipMallocManaged",
        )
        return ptr.value

    def free(self, ptr: int) -> None:
        self._check(self.lib.hipFree(ctypes.c_void_p(ptr)), "hipFree")


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def as_cuda_view(ptr: int, shape: tuple[int, ...], dtype: torch.dtype,
                 device: torch.device) -> torch.Tensor:
    """Build a CUDA tensor over a raw device pointer (no ownership, no copy)."""
    if not hasattr(torch._C, "_construct_storage_from_data_pointer"):
        raise RuntimeError("torch has no _construct_storage_from_data_pointer")
    nbytes = math.prod(shape) * _itemsize(dtype)
    strides = []
    acc = 1
    for size in reversed(shape):
        strides.append(acc)
        acc *= size
    strides.reverse()

    storage = torch._C._construct_storage_from_data_pointer(ptr, device, nbytes)
    meta = {
        "nbytes": nbytes,
        "data_ptr": ptr,
        "size": tuple(shape),
        "stride": tuple(strides),
        "dtype": dtype,
        "device": device,
        "storage_offset": 0,
    }
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(meta, storage)


class HostBuffer:
    """A HIP host allocation exposed as both a CPU and a CUDA tensor.

    ``cuda`` is what kernels use: the data stays in host RAM, so every access
    goes over PCIe.  ``cpu`` is a zero-copy CPU view used to initialise and
    verify the contents.  Do not touch ``cpu`` while kernels are in flight.
    """

    def __init__(self, hip: Hip, numel: int, dtype: torch.dtype,
                 device: torch.device, flags: int = HOST_MALLOC_FLAGS) -> None:
        self.hip = hip
        self.nbytes = numel * _itemsize(dtype)
        self.ptr = hip.host_malloc(self.nbytes, flags)
        # Keep the ctypes buffer alive so torch.frombuffer stays valid.
        self._cpu_buffer = (ctypes.c_char * self.nbytes).from_address(self.ptr)
        self.cpu = torch.frombuffer(self._cpu_buffer, dtype=dtype)
        self.cuda = as_cuda_view(self.ptr, (numel,), dtype, device)
        self._freed = False

    def free(self) -> None:
        if not self._freed:
            self.hip.host_free(self.ptr)
            self._freed = True


class Variant:
    def __init__(self, name: str, cuda: torch.Tensor, cpu: torch.Tensor,
                 cleanup=None) -> None:
        self.name = name
        self.cuda = cuda
        self.cpu = cpu
        self._cleanup = cleanup

    def close(self) -> None:
        if self._cleanup is not None:
            self._cleanup()
            self._cleanup = None


def event_time_us(fn, iters: int, warmup: int = 10) -> float:
    """Average GPU time per call (events around a back-to-back enqueue).

    For very small kernels the CPU may not enqueue fast enough, so this is an
    upper bound; the host-vs-device difference is still meaningful because both
    sides pay the same launch cost.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def sync_time_us(fn, iters: int, warmup: int = 5) -> float:
    """Wall-clock latency of one call including launch and synchronise."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def gbps(nbytes: int, us: float) -> float:
    return nbytes / (us * 1e-6) / 1e9


def roundtrip_ok(variant: Variant) -> bool:
    n = variant.cpu.numel()
    variant.cpu.fill_(1.0)
    torch.cuda.synchronize()
    if variant.cuda.sum().item() != n:
        return False
    variant.cuda.fill_(3.0)
    torch.cuda.synchronize()
    return variant.cpu[0].item() == 3.0 and variant.cpu[-1].item() == 3.0


def make_variants(hip: Hip, device: torch.device, numel: int,
                  dtype: torch.dtype) -> list[Variant]:
    nbytes = numel * _itemsize(dtype)
    variants: list[Variant] = []

    # 1. PyTorch pinned memory (this is what pin_memory=True gives you).
    pinned = torch.empty(numel, dtype=dtype, pin_memory=True)
    ptr = hip.host_device_pointer(pinned.data_ptr())
    variants.append(Variant("torch pinned (default flags)",
                            as_cuda_view(ptr, (numel,), dtype, device), pinned))

    # 2. Explicit hipHostMalloc, mapped.
    buf = HostBuffer(hip, numel, dtype, device,
                     HIP_HOST_MALLOC_MAPPED)
    variants.append(Variant("hipHostMalloc(mapped)", buf.cuda, buf.cpu, buf.free))

    # 3. Mapped + non-coherent: fast GPU stores.  Recommended.
    buf = HostBuffer(hip, numel, dtype, device)
    variants.append(Variant("hipHostMalloc(mapped|noncoh)", buf.cuda, buf.cpu, buf.free))

    # 4. Managed memory.  Pages may migrate, so this is not a stable
    #    "host-resident" guarantee; shown for comparison only.
    ptr = hip.malloc_managed(nbytes)
    cpu_buf = (ctypes.c_char * nbytes).from_address(ptr)
    variants.append(Variant("hipMallocManaged",
                            as_cuda_view(ptr, (numel,), dtype, device),
                            torch.frombuffer(cpu_buf, dtype=dtype),
                            lambda p=ptr: hip.free(p)))

    # 5. Register an ordinary pageable buffer after the fact.
    pageable = torch.empty(numel, dtype=dtype)
    hip.host_register(pageable.data_ptr(), nbytes)
    ptr = hip.host_device_pointer(pageable.data_ptr())
    variants.append(Variant("hipHostRegister(pageable)",
                            as_cuda_view(ptr, (numel,), dtype, device), pageable,
                            lambda p=pageable.data_ptr(): hip.host_unregister(p)))
    return variants


def bench_variants(hip: Hip, device: torch.device, dtype: torch.dtype) -> None:
    big_numel = 4 * 1024 * 1024          # 8 MB fp16
    small_numel = 2048                   # 4 KB fp16
    print("== 1. Allocation variants (8 MB buffer, zero-copy) ==")
    print(f"{'variant':32s} {'read 8MB':>11s} {'fill 8MB':>11s} "
          f"{'fill 4KB':>10s} {'ok':>4s}")

    variants = make_variants(hip, device, big_numel, dtype)
    out = torch.empty((), dtype=dtype, device=device)
    try:
        for v in variants:
            small = as_cuda_view(v.cuda.data_ptr(), (small_numel,), dtype, device)
            nbytes = big_numel * _itemsize(dtype)

            read_us = event_time_us(lambda: torch.sum(v.cuda, dim=0, out=out), 10, 3)
            fill_us = event_time_us(lambda: v.cuda.fill_(1.0), 3, 1)
            fill4k_us = event_time_us(lambda: small.fill_(1.0), 50, 5)
            ok = roundtrip_ok(Variant(v.name, small, v.cpu[:small_numel]))
            # roundtrip_ok uses the small view; make sure the big buffer is
            # still sane for the next measurement.
            torch.cuda.synchronize()

            print(f"{v.name:32s} {gbps(nbytes, read_us):10.2f}G "
                  f"{gbps(nbytes, fill_us):10.2f}G {fill4k_us:9.2f}us "
                  f"{'yes' if ok else 'NO':>4s}")
    finally:
        for v in variants:
            v.close()
        del out


def bench_small_reads(hip: Hip, device: torch.device, dtype: torch.dtype) -> None:
    print("\n== 2. Small-packet read latency (GPU sum; events, launch-bound) ==")
    print(f"{'bytes':>8s} {'zero-copy host':>16s} {'device':>10s} "
          f"{'copy+compute':>14s}")
    for nbytes in (4, 64, 1024, 4096, 16384, 65536):
        numel = nbytes // _itemsize(dtype)
        buf = HostBuffer(hip, numel, dtype, device)
        dev = torch.empty(numel, dtype=dtype, device=device)
        out = torch.empty((), dtype=dtype, device=device)
        iters = 200 if nbytes <= 16384 else 50

        def copy_then_sum():
            dev.copy_(buf.cpu)
            torch.sum(dev, dim=0, out=out)

        host_us = event_time_us(lambda: torch.sum(buf.cuda, dim=0, out=out), iters)
        dev_us = event_time_us(lambda: torch.sum(dev, dim=0, out=out), iters)
        copy_us = event_time_us(copy_then_sum, iters)

        print(f"{nbytes:8d} {host_us:15.2f}us {dev_us:9.2f}us {copy_us:13.2f}us")
        del dev, out
        buf.free()


def bench_read_bandwidth(hip: Hip, device: torch.device,
                         dtype: torch.dtype) -> None:
    print("\n== 3. Read bandwidth vs size (GPU sum over the buffer) ==")
    print(f"{'size':>8s} {'zero-copy host':>16s} {'device':>12s} "
          f"{'pinned->device copy':>21s}")
    out = torch.empty((), dtype=dtype, device=device)
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    for mb in (1, 4, 16, 64, 256):
        nbytes = mb * 1024 * 1024
        numel = nbytes // _itemsize(dtype)
        buf = HostBuffer(hip, numel, dtype, device)
        dev = torch.empty(numel, dtype=dtype, device=device)
        iters = 20 if mb <= 16 else 10

        host_us = event_time_us(lambda: torch.sum(buf.cuda, dim=0, out=out), iters, 3)
        dev_us = event_time_us(lambda: torch.sum(dev, dim=0, out=out), iters, 3)
        copy_us = event_time_us(lambda: dev.copy_(buf.cpu), iters, 3)

        cached = "*" if nbytes <= l2_bytes else ""
        print(f"{mb:5d}MB{cached:1s} {gbps(nbytes, host_us):15.2f}G "
              f"{gbps(nbytes, dev_us):11.2f}G {gbps(nbytes, copy_us):20.2f}G")
        del dev, buf
    print(f"  * buffer fits in the {l2_bytes >> 20} MB L2, so the GPU hits cache "
          f"instead of PCIe;")
    print("    >=16 MB is the true host-read bandwidth.")
    del out


def bench_writes(hip: Hip, device: torch.device, dtype: torch.dtype) -> None:
    print("\n== 4. GPU -> host writes (non-coherent mapping) ==")
    print(f"{'size':>8s} {'host fill_':>16s} {'host<-device copy_':>20s} "
          f"{'device fill_':>16s}")
    for nbytes in (4096, 16384, 65536, 1048576, 8 * 1024 * 1024):
        numel = nbytes // _itemsize(dtype)
        buf = HostBuffer(hip, numel, dtype, device)
        src = torch.empty(numel, dtype=dtype, device=device)
        dst_dev = torch.empty(numel, dtype=dtype, device=device)
        iters = 100 if nbytes <= 65536 else 10

        fill_us = event_time_us(lambda: buf.cuda.fill_(1.0), iters, 5)
        copy_us = event_time_us(lambda: buf.cuda.copy_(src), iters, 5)
        dev_us = event_time_us(lambda: dst_dev.fill_(1.0), iters, 5)

        label = f"{nbytes // 1024}KB" if nbytes < 1024 * 1024 else f"{nbytes >> 20}MB"
        print(f"{label:>8s} {fill_us:10.2f}us({gbps(nbytes, fill_us):5.2f}G) "
              f"{copy_us:9.2f}us({gbps(nbytes, copy_us):5.2f}G) "
              f"{dev_us:9.2f}us({gbps(nbytes, dev_us):5.2f}G)")
        del src, dst_dev, buf
    print("  Rows <= 1 MB are launch-bound or L2-cached; 8 MB shows the")
    print("  sustained host-write rate.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this example needs a HIP/ROCm device")
    if torch.version.hip is None:
        raise SystemExit("this example targets ROCm; HIP host allocation is required")

    device = torch.device(args.device)
    if device.index is None:
        # A bare "cuda" device makes the constructed tensors report an
        # unindexed device, which breaks ops that run on cuda:0.
        device = torch.device(device.type, torch.cuda.current_device())
    dtype = getattr(torch, args.dtype)
    torch.cuda.init()

    print(f"torch {torch.__version__}, HIP {torch.version.hip}, "
          f"device {torch.cuda.get_device_name(0)}")
    hip = Hip()

    bench_variants(hip, device, dtype)
    bench_small_reads(hip, device, dtype)
    bench_read_bandwidth(hip, device, dtype)
    bench_writes(hip, device, dtype)

    print("\nNotes")
    print("- Zero-copy host memory works on this gfx906: no H2D copy is needed,")
    print("  reads run at PCIe speed (~12 GB/s, on par with a pinned H2D copy).")
    print("- Small ops are launch-bound; zero-copy costs well under 1 us over")
    print("  device memory, but a copy+compute pair costs ~6-8 us extra.")
    print("- torch pin_memory=True / hipHostMalloc without flags gives coherent")
    print("  memory: ~0.33 GB/s for GPU element-wise stores.  Use")
    print("  hipHostMallocMapped|hipHostMallocNonCoherent for GPU-written caches.")
    print("- Bulk copies (copy_) stay ~12 GB/s in both directions regardless.")
    print("- Synchronise before touching the CPU view; do not access a buffer")
    print("  from host and GPU concurrently.")


if __name__ == "__main__":
    main()
