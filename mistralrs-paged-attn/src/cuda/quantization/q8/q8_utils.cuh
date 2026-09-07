#pragma once

// Block-int8 (llama.cpp Q8_0) helpers for KV cache: int8 + fp32 scale per 32
// elems. Plain CUDA/HIP-portable ops only (no arch intrinsics) so the same
// header compiles under nvcc and hipcc.

#include <cstdint>

namespace vllm {
namespace q8 {

// Number of elems per quantized block. Matches llama.cpp QK8_0.
constexpr int kQ8BlockSize = 32;

// Quantize 32 floats to int8 with a single fp32 scale (amax/127).
// Zero-block maps to scale 1 (all-zero output) to avoid div-by-zero.
__device__ __forceinline__ void quantize_block_q8_0(const float *vals,
                                                    int8_t *out,
                                                    float *scale) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < kQ8BlockSize; ++i) {
    float a = vals[i] >= 0.f ? vals[i] : -vals[i];
    amax = amax > a ? amax : a;
  }
  float d = amax / 127.f;
  float id = d != 0.f ? 1.f / d : 0.f;
#pragma unroll
  for (int i = 0; i < kQ8BlockSize; ++i) {
    float q = vals[i] * id;
    q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
    // Round-half-away to match llama.cpp ggml quantize_q8_0.
    out[i] = static_cast<int8_t>(q >= 0.f ? q + 0.5f : q - 0.5f);
  }
  *scale = d != 0.f ? d : 1.f;
}

// Dequantize one element: x = qs * scale.
__device__ __forceinline__ float dequantize_q8_0(int8_t q, float scale) {
  return static_cast<float>(q) * scale;
}

}  // namespace q8
}  // namespace vllm
