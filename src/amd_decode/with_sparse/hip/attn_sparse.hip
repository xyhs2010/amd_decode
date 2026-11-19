// Sparse fp16 KV decode attention for Qwen3.5-9B (16 query / 4 KV heads, D=256).
//
// Attends a per-query-head list of token ids instead of the contiguous
// ``[0, n)`` range, so a policy can pick any subset for every head.  ``-1``
// marks padding and is skipped.  One CTA owns one query head; the 4 query heads
// sharing a K/V head therefore read it independently, which is what allows them
// to select different tokens.
//
// The cache is fp16 in VRAM, so there is no dequantisation and no scale.  One
// warp owns 256 contiguous id slots and keeps an fp32 online-softmax state;
// warps write partials independently and a second kernel merges them.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kHeadDim = 256;
constexpr int kGroup = 4;          // query heads per K/V head
constexpr int kHeads = 16;         // query heads, one id row each
constexpr int kLanes = 32;
constexpr int kDims = kHeadDim / kLanes;  // 8 channels per lane
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kLanes;
constexpr int kWarpTokens = 256;   // id slots per partial, also the stride
constexpr float kInvSqrtD = 0.0625f;  // 1 / sqrt(256)

// 8 contiguous halves, i.e. one lane's slice of a head.
__device__ __forceinline__ void load8(const __half* p, float* out) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
  for (int i = 0; i < kDims; ++i) out[i] = __half2float(h[i]);
}

__global__ void sparse_split_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, int64_t k_stride, int64_t v_stride,
    const int32_t* __restrict__ idx, float* __restrict__ po,
    float* __restrict__ ps, int parts, int max_sel) {
  const int hq = blockIdx.y;          // query head
  const int hkv = hq / kGroup;        // the K/V head it reads
  const int lane = threadIdx.x & (kLanes - 1);
  const int part = blockIdx.x * kWarps + threadIdx.x / kLanes;
  if (part >= parts) return;

  const int t0 = part * kWarpTokens;
  const int t1 = min(t0 + kWarpTokens, max_sel);
  const int d0 = lane * kDims;
  // This head's token list; the read is warp-uniform, i.e. a broadcast.
  const int32_t* ih = idx + static_cast<size_t>(hq) * max_sel;

  float qreg[kDims];
  load8(q + static_cast<size_t>(hq) * kHeadDim + d0, qreg);

  float m = -1e30f, l = 0.0f, acc[kDims];
#pragma unroll
  for (int i = 0; i < kDims; ++i) acc[i] = 0.0f;

  const __half* kh = k + static_cast<size_t>(hkv) * k_stride + d0;
  const __half* vh = v + static_cast<size_t>(hkv) * v_stride + d0;

  for (int si = t0; si < t1; ++si) {
    const int t = ih[si];
    if (t < 0) continue;

    float kf[kDims], vf[kDims];
    load8(kh + static_cast<size_t>(t) * kHeadDim, kf);
    load8(vh + static_cast<size_t>(t) * kHeadDim, vf);

    float sc = 0.0f;
#pragma unroll
    for (int i = 0; i < kDims; ++i) sc += qreg[i] * kf[i];
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1) sc += __shfl_down(sc, off, kLanes);
    sc = __shfl(sc, 0, kLanes) * kInvSqrtD;

    if (sc > m) {
      const float corr = __expf(m - sc);
      l *= corr;
#pragma unroll
      for (int i = 0; i < kDims; ++i) acc[i] *= corr;
      m = sc;
    }
    const float p = __expf(sc - m);
    l += p;
#pragma unroll
    for (int i = 0; i < kDims; ++i) acc[i] += p * vf[i];
  }

  const size_t off = static_cast<size_t>(hq) * parts + part;
  float* dst = po + off * kHeadDim + d0;
#pragma unroll
  for (int i = 0; i < kDims; ++i) dst[i] = acc[i];
  if (lane == 0) {
    ps[off * 2] = m;
    ps[off * 2 + 1] = l;
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

void sparse_attn_split(uintptr_t q, uintptr_t k, uintptr_t v,
                       int64_t k_stride, int64_t v_stride, uintptr_t idx,
                       uintptr_t po, uintptr_t ps, int64_t parts,
                       int64_t max_sel) {
  dim3 grid(static_cast<unsigned>((parts + kWarps - 1) / kWarps), kHeads);
  sparse_split_kernel<<<grid, kThreads>>>(
      reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k),
      reinterpret_cast<const __half*>(v), k_stride, v_stride,
      reinterpret_cast<const int32_t*>(idx), reinterpret_cast<float*>(po),
      reinterpret_cast<float*>(ps), static_cast<int>(parts),
      static_cast<int>(max_sel));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}

void sparse_attn_merge(uintptr_t po, uintptr_t ps, uintptr_t out,
                       int64_t parts) {
  sparse_merge_kernel<<<kHeads, kHeadDim>>>(
      reinterpret_cast<const float*>(po), reinterpret_cast<const float*>(ps),
      reinterpret_cast<__half*>(out), static_cast<int>(parts));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}
