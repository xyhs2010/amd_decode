// Python bindings for the sparse attention kernels.  Raw pointers and ints
// only, like ``with_hip/hip/bindings.hip``.

#include <pybind11/pybind11.h>

#include <cstdint>

void sparse_attn_split(uintptr_t q, uintptr_t k, uintptr_t v,
                       int64_t k_stride, int64_t v_stride, uintptr_t idx,
                       uintptr_t po, uintptr_t ps, int64_t parts,
                       int64_t max_sel);

void sparse_attn_merge(uintptr_t po, uintptr_t ps, uintptr_t out,
                       int64_t parts);

PYBIND11_MODULE(_sparse_ops, m) {
  m.def("attn_split", &sparse_attn_split, "fp16 decode attention over token ids");
  m.def("attn_merge", &sparse_attn_merge, "merge the attention partials");
}
