"""Host memory the GPU can address directly (zero copy) on ROCm.

``hipHostMalloc`` page-locks a host buffer and ``hipHostGetDevicePointer``
returns a device-visible address for it (on this gfx906 it is the same address).
Wrapping that pointer in a CUDA tensor lets kernels read and write host RAM over
PCIe with no explicit copies; ``examples/host_memory_bench.py`` measures the
trade-offs.

Two gotchas, both measured on gfx906 / ROCm 6.3:

* Reads run at PCIe speed (~12 GB/s).  Bulk ``copy_`` is fast in both
  directions, but element-wise GPU *stores* to default (coherent) pinned memory
  are ~0.33 GB/s.  ``hipHostMallocNonCoherent`` restores ~12 GB/s stores, so
  that is the default flag here.
* Synchronise before reading the buffer from the CPU, and never touch it from
  both sides while kernels are in flight.

There is no public torch API for a tensor over a raw device pointer, so this
module uses the same private constructor as torch.compile's cudagraph trees
(``_construct_storage_from_data_pointer``).  The storage has a no-op deleter:
the HIP allocation stays owned by :class:`HostBuffer`.
"""

from __future__ import annotations

import ctypes
import math

import torch

HIP_LIBRARY = "/opt/rocm/lib/libamdhip64.so"

HIP_HOST_MALLOC_MAPPED = 0x2
HIP_HOST_MALLOC_WRITE_COMBINED = 0x4
HIP_HOST_MALLOC_COHERENT = 0x40000000
HIP_HOST_MALLOC_NON_COHERENT = 0x80000000
HIP_HOST_REGISTER_MAPPED = 0x2
HIP_MEM_ATTACH_GLOBAL = 0x1

# Good default for a buffer the GPU both reads and writes (e.g. a KV cache).
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
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        # Ops infer cuda:0; a tensor whose storage says "cuda" (no index)
        # trips copy_/matmul device checks.
        device = torch.device(device.type, torch.cuda.current_device())
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
    """A HIP host allocation exposed as a CUDA tensor (and a CPU view).

    ``cuda`` is what kernels use: the data stays in host RAM.  ``cpu`` is a
    zero-copy CPU view for initialisation and verification.  Do not touch
    ``cpu`` while kernels are in flight.
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

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass
