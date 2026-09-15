#pragma once

// Walsh-Hadamard incoherence for Q4_0 K cache (TurboQuant-style, minus QJL).
// Store: K' = H(S*K) then plain amax/7 nibble quant. Query: Q' = H(S*Q).
// Orthonormal H plus identical signs both sides gives Q'.K' = Q.K, so the
// decode dot needs no inverse rotation. V stays plain RTN (less sensitive).
// Gather (prefill-side) emits original-domain K via the same self-inverse op.

#include <cstdint>

namespace vllm {
namespace wht {

// Fixed pseudo-random signs per (kv head, dim): xorshift hash, no storage,
// identical on store/query/gather paths. Any fixed +-1 pattern spreads
// outliers; determinism keeps prefix-cache and graphs valid.
__device__ __forceinline__ float wht_sign(int head_idx, int d) {
  uint32_t x = static_cast<uint32_t>(head_idx * 2654435761u ^ (d + 1) * 40503u ^ 0x9e3779b9u);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return (x & 1u) ? -1.f : 1.f;
}

// In-place normalized FWHT over sh[0..n), n a power of two. Cooperative over
// threadIdx/blockDim with __syncthreads between stages; safe to call with
// any blockDim >= 1. 1/sqrt(2) per stage keeps H orthonormal (self-inverse).
__device__ __forceinline__ void wht_inplace(float *sh, int n) {
  constexpr float INV_SQRT2 = 0.7071067811865475f;
  for (int stride = 1; stride < n; stride *= 2) {
    for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
      int a = (i / stride) * (stride * 2) + (i % stride);
      int b = a + stride;
      float x = sh[a];
      float y = sh[b];
      sh[a] = (x + y) * INV_SQRT2;
      sh[b] = (x - y) * INV_SQRT2;
    }
    __syncthreads();
  }
}

inline __device__ bool wht_supported(int head_size) {
  return head_size == 256;
}

}  // namespace wht
}  // namespace vllm
