// Python bindings for the sparse attention kernels.  Raw pointers and ints
// only, like ``with_hip/hip/bindings.hip``.

#include <pybind11/pybind11.h>

#include <cstdint>

void sparse_attn_split(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t k_tail,
                       uintptr_t v_scale, uintptr_t k_block, uintptr_t idx,
                       uintptr_t po, uintptr_t ps, int64_t n, int64_t prefix,
                       int64_t block_size, int64_t max_len, int64_t n_blocks,
                       int64_t parts, int64_t max_sel);

void sparse_attn_merge(uintptr_t po, uintptr_t ps, uintptr_t out,
                       int64_t parts);

PYBIND11_MODULE(_sparse_ops, m) {
  m.def("attn_split", &sparse_attn_split, "q8 decode attention over token ids");
  m.def("attn_merge", &sparse_attn_merge, "merge the attention partials");
}
