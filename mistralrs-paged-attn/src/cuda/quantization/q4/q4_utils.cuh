#pragma once

// Block-int4 helpers for KV cache: nibbles (q + 8 bias, 2 elems/byte) plus
// fp32 scale per 32 elems in sidecars. Same sidecar scheme as Q8_0, so the
// paged and gather kernels share scale indexing. Plain CUDA/HIP-portable
// ops only so the header compiles under nvcc and hipcc.

#include <cstdint>

namespace vllm {
namespace q4 {

// Number of elems per quantized block. Matches the Q8_0 sidecar grouping.
constexpr int kQ4BlockSize = 32;
// Symmetric range: round(x / d) clamped to [-8, 7], stored + 8.
constexpr float kQ4MaxQ = 7.f;

// Quantize 32 floats to biased nibbles with a single fp32 scale (amax/7).
// Zero-block maps to scale 1 (all-zero output) to avoid div-by-zero.
// Nibble layout inside one byte: even elem in lo, odd elem in hi.
__device__ __forceinline__ void quantize_block_q4_0(const float *vals,
                                                    uint8_t *out,
                                                    float *scale) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < kQ4BlockSize; ++i) {
    float a = vals[i] >= 0.f ? vals[i] : -vals[i];
    amax = amax > a ? amax : a;
  }
  float d = amax / kQ4MaxQ;
  float id = d != 0.f ? 1.f / d : 0.f;
#pragma unroll
  for (int i = 0; i < kQ4BlockSize / 2; ++i) {
    float q0 = vals[2 * i] * id;
    float q1 = vals[2 * i + 1] * id;
    q0 = q0 > kQ4MaxQ ? kQ4MaxQ : (q0 < -8.f ? -8.f : q0);
    q1 = q1 > kQ4MaxQ ? kQ4MaxQ : (q1 < -8.f ? -8.f : q1);
    // Round-half-away (same as the Q8_0 helper), then bias into a nibble.
    uint8_t n0 = static_cast<uint8_t>(
        (q0 >= 0.f ? q0 + 0.5f : q0 - 0.5f) + 8.f);
    uint8_t n1 = static_cast<uint8_t>(
        (q1 >= 0.f ? q1 + 0.5f : q1 - 0.5f) + 8.f);
    out[i] = static_cast<uint8_t>(n0 | (n1 << 4));
  }
  *scale = d != 0.f ? d : 1.f;
}

// Dequantize one biased nibble: x = (nibble - 8) * scale.
__device__ __forceinline__ float dequantize_q4_0(uint8_t nibble, float scale) {
  return (static_cast<float>(nibble) - 8.f) * scale;
}

}  // namespace q4
}  // namespace vllm
