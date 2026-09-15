#pragma once

// Lloyd-Max 4-bit codebook for the WHT-rotated KV path. After the signed
// Hadamard, each head-dim coord is ~iid Gaussian (CLT over 256 dims), so the
// amax-normalized block (v = x/amax) is a fixed Gaussian shape. A uniform
// -8..7 grid wastes resolution on the empty tails; a Lloyd-Max grid fit to
// v puts levels where the mass is, cutting 4-bit MSE ~29% (1.5 dB) at the
// same bit width. Levels fit offline (lloydmax_fit.py) to v for x ~
// N(0,1)^32; symmetric, top < 1 so d = amax leaves y = x/d in [-1, 1].

#include <cstdint>

namespace vllm {
namespace q4 {

constexpr float kQ4LmLevels[16] = {
    -0.9688121676445007f, -0.774229109287262f, -0.6181834936141968f,
    -0.4846160411834717f, -0.3652898669242859f, -0.25523099303245544f,
    -0.15097682178020477f, -0.049988243728876114f, 0.049988243728876114f,
    0.15097682178020477f, 0.25523099303245544f, 0.3652898669242859f,
    0.4846160411834717f, 0.6181834936141968f, 0.774229109287262f,
    0.9688121676445007f};

// Nearest-level index for y = x/amax in [-1, 1]. Binary search on the 15
// midpoints; naturally clamps to [0, 15] at the edges.
__device__ __forceinline__ uint8_t nearest_level_q4_lm(float y) {
  int lo = 0;
  int hi = 15;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    float m = 0.5f * (kQ4LmLevels[mid] + kQ4LmLevels[mid + 1]);
    if (y < m) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return static_cast<uint8_t>(lo);
}

// Dequant an LM nibble: x = level[nib] * scale, where scale holds amax.
__device__ __forceinline__ float dequantize_q4_lm(uint8_t nib, float scale) {
  return kQ4LmLevels[nib] * scale;
}

// Unified dequant: LM grid (WHT path, scale = amax) or the plain Q4_0 grid
// (scale = amax/7, nibble - 8 bias). Lm must match how the block was stored.
template <bool Lm>
__device__ __forceinline__ float dequantize_q4(uint8_t nib, float scale) {
  if constexpr (Lm) {
    return kQ4LmLevels[nib] * scale;
  }
  return (static_cast<float>(nib) - 8.f) * scale;
}

}  // namespace q4
}  // namespace vllm
