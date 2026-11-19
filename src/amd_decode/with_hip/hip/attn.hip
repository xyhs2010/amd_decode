// Fused fp16 decode attention for Qwen3.5-9B (16 query / 4 KV heads, D=256).
//
// One warp owns a contiguous range of tokens and keeps an online-softmax state
// for the 4 query heads that share its K/V head.  Warps write their partials
// independently and a second kernel merges them, so there is no split-K
// interaction between blocks.  K/V are read straight from the fp16 cache as
// [4][max_len][256] rows; nothing is dequantised or expanded.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kHeadDim = 256;
constexpr int kGroup = 4;          // query heads per K/V head
constexpr int kKvHeads = 4;
constexpr int kLanes = 32;
constexpr int kDims = kHeadDim / kLanes;  // 8 channels per lane
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kLanes;
constexpr int kWarpTokens = 512;   // also the partial stride in the workspace
constexpr float kInvSqrtD = 0.0625f;  // 1 / sqrt(256)

__global__ void attn_split_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, float* __restrict__ po,
    float* __restrict__ ps, int n, int64_t k_head_stride, int64_t k_row_stride,
    int64_t v_head_stride, int64_t v_row_stride, int parts) {
  const int h = blockIdx.y;
  const int lane = threadIdx.x & (kLanes - 1);
  const int part = blockIdx.x * kWarps + threadIdx.x / kLanes;
  if (part >= parts) return;

  const int t0 = part * kWarpTokens;
  const int t1 = min(t0 + kWarpTokens, n);
  const int d0 = lane * kDims;

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

  const __half* kh = k + static_cast<size_t>(h) * k_head_stride;
  const __half* vh = v + static_cast<size_t>(h) * v_head_stride;

  for (int t = t0; t < t1; ++t) {
    // 8 halves = one 16-byte load per lane.
    const __half* krow = kh + static_cast<size_t>(t) * k_row_stride + d0;
    const __half* vrow = vh + static_cast<size_t>(t) * v_row_stride + d0;
    const __half2 k01 = *reinterpret_cast<const __half2*>(krow + 0);
    const __half2 k23 = *reinterpret_cast<const __half2*>(krow + 2);
    const __half2 k45 = *reinterpret_cast<const __half2*>(krow + 4);
    const __half2 k67 = *reinterpret_cast<const __half2*>(krow + 6);
    const __half2 v01 = *reinterpret_cast<const __half2*>(vrow + 0);
    const __half2 v23 = *reinterpret_cast<const __half2*>(vrow + 2);
    const __half2 v45 = *reinterpret_cast<const __half2*>(vrow + 4);
    const __half2 v67 = *reinterpret_cast<const __half2*>(vrow + 6);

    const float kf[kDims] = {
        __low2float(k01), __high2float(k01), __low2float(k23), __high2float(k23),
        __low2float(k45), __high2float(k45), __low2float(k67), __high2float(k67)};
    const float vf[kDims] = {
        __low2float(v01), __high2float(v01), __low2float(v23), __high2float(v23),
        __low2float(v45), __high2float(v45), __low2float(v67), __high2float(v67)};

    float sc[kGroup];
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

__global__ void attn_merge_kernel(const float* __restrict__ po,
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

void attn_split(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t po,
                uintptr_t ps, int64_t n, int64_t k_head_stride,
                int64_t k_row_stride, int64_t v_head_stride,
                int64_t v_row_stride, int64_t parts) {
  dim3 grid(static_cast<unsigned>((parts + kWarps - 1) / kWarps), kKvHeads);
  attn_split_kernel<<<grid, kThreads>>>(
      reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k),
      reinterpret_cast<const __half*>(v), reinterpret_cast<float*>(po),
      reinterpret_cast<float*>(ps), static_cast<int>(n), k_head_stride,
      k_row_stride, v_head_stride, v_row_stride, static_cast<int>(parts));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}

void attn_merge(uintptr_t po, uintptr_t ps, uintptr_t out, int64_t parts) {
  attn_merge_kernel<<<kKvHeads * kGroup, kHeadDim>>>(
      reinterpret_cast<const float*>(po), reinterpret_cast<const float*>(ps),
      reinterpret_cast<__half*>(out), static_cast<int>(parts));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
}
