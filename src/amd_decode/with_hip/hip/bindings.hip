// Python bindings for the HIP kernels.  Everything crosses the boundary as
// raw pointers and ints, so the kernel sources stay free of torch headers.

#include <pybind11/pybind11.h>

#include <cstdint>

void q4_matmul(uintptr_t x, uintptr_t q, uintptr_t scale, uintptr_t zero,
               uintptr_t out, int64_t m, int64_t n, int64_t k);

void attn_split(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t k_tail,
                uintptr_t v_scale, uintptr_t k_block, uintptr_t po,
                uintptr_t ps, int64_t n, int64_t prefix, int64_t block_size,
                int64_t max_len, int64_t n_blocks, int64_t parts);

void attn_merge(uintptr_t po, uintptr_t ps, uintptr_t out, int64_t parts);

PYBIND11_MODULE(_ops, m) {
  m.def("q4_matmul", &q4_matmul, "int4 weight GEMV");
  m.def("attn_split", &attn_split, "q8 decode attention, per-warp partials");
  m.def("attn_merge", &attn_merge, "merge the attention partials");
}
