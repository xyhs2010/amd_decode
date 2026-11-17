// Int4 weight-only GEMV: y[m, n] = sum_k x[m, k] * dequant(q[n, k]).
//
// The weight layout is the one produced by with_torch.int4.quantize_int4 /
// pack_int4: q in 0..15, two per byte (even index in the low nibble), with one
// (scale, zero) pair per 128 input features and
//     dequant = (q - 8) * scale + zero.
//
// Factoring the group constants out of the inner loop gives
//     sum_k w*x = scale[n,g] * sum_{k in g} (q-8)*x + zero[n,g] * sum_{k in g} x.
//
// One 32-lane group handles one output row; each lane owns one 128-value group
// per pass, i.e. exactly 64 packed bytes and 128 activations.  Rows are
// distributed over the grid so the kernel also serves short prefills (M > 1),
// reusing the packed weight loads across the M tokens.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kGroup = 128;   // weight values per (scale, zero) pair
constexpr int kMid = 8;       // dequant offset
constexpr int kLanes = 32;    // shuffle width (gfx906 is wave64)
constexpr int kRowsPerBlock = 8;

__global__ void q4_matmul_kernel(const __half* __restrict__ x,
                                 const uint8_t* __restrict__ q,
                                 const __half* __restrict__ scale,
                                 const __half* __restrict__ zero,
                                 __half* __restrict__ out,
                                 int m_tokens, int n_out, int k_in) {
  const int lane = threadIdx.x % kLanes;
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kLanes;
  if (row >= n_out) return;

  const int groups = k_in / kGroup;
  const uint64_t* qrow =
      reinterpret_cast<const uint64_t*>(q + static_cast<size_t>(row) * (k_in / 2));
  const __half* srow = scale + static_cast<size_t>(row) * groups;
  const __half* zrow = zero + static_cast<size_t>(row) * groups;

  for (int m = 0; m < m_tokens; ++m) {
    const __half* xrow = x + static_cast<size_t>(m) * k_in;
    float acc = 0.0f;
    for (int g = lane; g < groups; g += kLanes) {
      const uint64_t* qg = qrow + static_cast<size_t>(g) * 8;  // 8 * 8 B = 128
      const __half* xg = xrow + static_cast<size_t>(g) * kGroup;
      float dot = 0.0f;
      float sum = 0.0f;
      for (int i = 0; i < 8; ++i) {
        uint64_t w = qg[i];
        const __half* xi = xg + i * 16;
        for (int j = 0; j < 16; ++j) {
          float xv = __half2float(xi[j]);
          dot += static_cast<float>(static_cast<int>((w >> (4 * j)) & 0xF) - kMid) * xv;
          sum += xv;
        }
      }
      acc += __half2float(srow[g]) * dot + __half2float(zrow[g]) * sum;
    }
    for (int off = kLanes / 2; off > 0; off >>= 1) {
      acc += __shfl_down(acc, off, kLanes);
    }
    if (lane == 0) out[static_cast<size_t>(m) * n_out + row] = __float2half(acc);
  }
}

}  // namespace

void q4_matmul(uintptr_t x, uintptr_t q, uintptr_t scale, uintptr_t zero,
               uintptr_t out, int64_t m, int64_t n, int64_t k) {
  int grid = static_cast<int>((n + kRowsPerBlock - 1) / kRowsPerBlock);
  q4_matmul_kernel<<<grid, kRowsPerBlock * kLanes>>>(
      reinterpret_cast<const __half*>(x), reinterpret_cast<const uint8_t*>(q),
      reinterpret_cast<const __half*>(scale), reinterpret_cast<const __half*>(zero),
      reinterpret_cast<__half*>(out), static_cast<int>(m), static_cast<int>(n),
      static_cast<int>(k));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}
