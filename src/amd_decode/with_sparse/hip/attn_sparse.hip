// Sparse int8 KV decode attention for Qwen3-8B (32 query / 8 KV heads, D=128).
//
// A copy of ``with_hip/hip/attn.hip`` that attends a per-head list of token ids
// instead of the contiguous ``[0, n)`` range, so a policy can pick any subset.
// ``-1`` marks padding and is skipped.  Dequantisation, the per-block K scales
// of the seeded prefix, the online softmax and the partial/merge split are the
// dense kernel's.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kHeadDim = 128;
constexpr int kGroup = 4;          // query heads per K/V head
constexpr int kKvHeads = 8;
constexpr int kLanes = 32;
constexpr int kDims = kHeadDim / kLanes;  // 4 channels per lane
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kLanes;
constexpr int kWarpTokens = 256;   // id slots per partial, also the stride
constexpr float kInvSqrtD = 0.08838834764831845f;  // 1 / sqrt(128)

// One byte of K/V times its per-channel scale, rounded to fp16.
__device__ __forceinline__ float deq(uint32_t word, int i, float scale) {
  const int8_t q = static_cast<int8_t>((word >> (8 * i)) & 0xFF);
  return __half2float(__float2half(static_cast<float>(q) * scale));
}

__global__ void sparse_split_kernel(
    const __half* __restrict__ q, const uint8_t* __restrict__ k,
    const uint8_t* __restrict__ v, const float* __restrict__ k_tail,
    const float* __restrict__ v_scale, const float* __restrict__ k_block,
    const int32_t* __restrict__ idx, float* __restrict__ po,
    float* __restrict__ ps, int n, int prefix, int block_size, int max_len,
    int n_blocks, int parts, int max_sel) {
  const int h = blockIdx.y;
  const int lane = threadIdx.x & (kLanes - 1);
  const int part = blockIdx.x * kWarps + threadIdx.x / kLanes;
  if (part >= parts) return;

  const int t0 = part * kWarpTokens;
  const int t1 = min(t0 + kWarpTokens, n);
  const int d0 = lane * kDims;
  // This head's token list; the read is warp-uniform, i.e. a broadcast.
  const int32_t* ih = idx + static_cast<size_t>(h) * max_sel;

  __half qreg[kGroup][kDims];
  const __half* qh = q + static_cast<size_t>(h * kGroup) * kHeadDim;
#pragma unroll
  for (int j = 0; j < kGroup; ++j) {
#pragma unroll
    for (int i = 0; i < kDims; ++i) qreg[j][i] = qh[j * kHeadDim + d0 + i];
  }

  float m[kGroup], l[kGroup], acc[kGroup][kDims];
#pragma unroll
  for (int j = 0; j < kGroup; ++j) {
    m[j] = -1e30f;
    l[j] = 0.0f;
#pragma unroll
    for (int i = 0; i < kDims; ++i) acc[j][i] = 0.0f;
  }

  const uint8_t* kh = k + static_cast<size_t>(h) * max_len * kHeadDim;
  const uint8_t* vh = v + static_cast<size_t>(h) * max_len * kHeadDim;
  const float* kth = k_tail + static_cast<size_t>(h) * max_len;
  const float* vsh = v_scale + static_cast<size_t>(h) * max_len;

  float ks[kDims];
  bool have_ks = false;
  int last_blk = -1;
  for (int si = t0; si < t1; ++si) {
    const int t = ih[si];
    if (t < 0) continue;
    if (t < prefix) {
      // Compare block ids, not ``t % block_size``: a gathered list may jump
      // between blocks without landing on a boundary.
      const int blk = t / block_size;
      if (!have_ks || blk != last_blk) {
        const float* sb = k_block
            + (static_cast<size_t>(h) * n_blocks + blk) * kHeadDim + d0;
#pragma unroll
        for (int j = 0; j < kDims; ++j) ks[j] = sb[j];
        have_ks = true;
        last_blk = blk;
      }
    } else {
      have_ks = false;
      const float s = kth[t];
#pragma unroll
      for (int i = 0; i < kDims; ++i) ks[i] = s;
    }

    const uint32_t kw = *reinterpret_cast<const uint32_t*>(
        kh + static_cast<size_t>(t) * kHeadDim + d0);
    const uint32_t vw = *reinterpret_cast<const uint32_t*>(
        vh + static_cast<size_t>(t) * kHeadDim + d0);

    float kf[kDims];
    float sc[kGroup];
#pragma unroll
    for (int i = 0; i < kDims; ++i) kf[i] = deq(kw, i, ks[i]);
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      sc[j] = 0.0f;
#pragma unroll
      for (int i = 0; i < kDims; ++i)
        sc[j] += __half2float(qreg[j][i]) * kf[i];
    }
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1) {
#pragma unroll
      for (int j = 0; j < kGroup; ++j) sc[j] += __shfl_down(sc[j], off, kLanes);
    }
#pragma unroll
    for (int j = 0; j < kGroup; ++j) sc[j] = __shfl(sc[j], 0, kLanes) * kInvSqrtD;

    const float vs = vsh[t];
    float vf[kDims];
#pragma unroll
    for (int i = 0; i < kDims; ++i) vf[i] = deq(vw, i, vs);

#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      if (sc[j] > m[j]) {
        const float corr = __expf(m[j] - sc[j]);
        l[j] *= corr;
#pragma unroll
        for (int i = 0; i < kDims; ++i) acc[j][i] *= corr;
        m[j] = sc[j];
      }
      const float p = __expf(sc[j] - m[j]);
      l[j] += p;
#pragma unroll
      for (int i = 0; i < kDims; ++i) acc[j][i] += p * vf[i];
    }
  }

  const size_t base = (static_cast<size_t>(h * kGroup) * parts) + part;
#pragma unroll
  for (int j = 0; j < kGroup; ++j) {
    const size_t off = base + static_cast<size_t>(j) * parts;
    float* dst = po + off * kHeadDim + d0;
#pragma unroll
    for (int i = 0; i < kDims; ++i) dst[i] = acc[j][i];
    if (lane == 0) {
      ps[off * 2] = m[j];
      ps[off * 2 + 1] = l[j];
    }
  }
}

__global__ void sparse_merge_kernel(const float* __restrict__ po,
                                    const float* __restrict__ ps,
                                    __half* __restrict__ out, int parts) {
  const int hj = blockIdx.x;
  const int d = threadIdx.x;
  const float* stat = ps + static_cast<size_t>(hj) * parts * 2;
  const float* part = po + static_cast<size_t>(hj) * parts * kHeadDim + d;

  float m = -1e30f;
  for (int s = 0; s < parts; ++s) m = fmaxf(m, stat[s * 2]);
  float num = 0.0f;
  float den = 0.0f;
  for (int s = 0; s < parts; ++s) {
    const float e = __expf(stat[s * 2] - m);
    num += e * part[static_cast<size_t>(s) * kHeadDim];
    den += e * stat[s * 2 + 1];
  }
  out[static_cast<size_t>(hj) * kHeadDim + d] = __float2half(num / den);
}

}  // namespace

void sparse_attn_split(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t k_tail,
                       uintptr_t v_scale, uintptr_t k_block, uintptr_t idx,
                       uintptr_t po, uintptr_t ps, int64_t n, int64_t prefix,
                       int64_t block_size, int64_t max_len, int64_t n_blocks,
                       int64_t parts, int64_t max_sel) {
  dim3 grid(static_cast<unsigned>((parts + kWarps - 1) / kWarps), kKvHeads);
  sparse_split_kernel<<<grid, kThreads>>>(
      reinterpret_cast<const __half*>(q), reinterpret_cast<const uint8_t*>(k),
      reinterpret_cast<const uint8_t*>(v),
      reinterpret_cast<const float*>(k_tail),
      reinterpret_cast<const float*>(v_scale),
      reinterpret_cast<const float*>(k_block),
      reinterpret_cast<const int32_t*>(idx), reinterpret_cast<float*>(po),
      reinterpret_cast<float*>(ps), static_cast<int>(n),
      static_cast<int>(prefix), static_cast<int>(block_size),
      static_cast<int>(max_len), static_cast<int>(n_blocks),
      static_cast<int>(parts), static_cast<int>(max_sel));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}

void sparse_attn_merge(uintptr_t po, uintptr_t ps, uintptr_t out,
                       int64_t parts) {
  sparse_merge_kernel<<<kKvHeads * kGroup, kHeadDim>>>(
      reinterpret_cast<const float*>(po), reinterpret_cast<const float*>(ps),
      reinterpret_cast<__half*>(out), static_cast<int>(parts));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}
