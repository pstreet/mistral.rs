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

// QJL 1-bit residual magnitude: c = E|r|/amax for the LM grid above (fit
// offline to v = x/amax). Store sign(r) as 1 bit/elem; decode adds +-c.
// Cuts the LM residual MSE ~73% at 1 extra bit/elem (5-bit effective).
constexpr float kQ4ResMag = 0.029516000300645828f;

// Folded 32-entry dequant LUT: index (sbit<<4)|nib gives
// levels[nib] + (sbit ? +c : -c). Bit 0 = r<0, bit 1 = r>=0.
constexpr float kQ4LmResLevels[32] = {
    -0.9983281493186951f, -0.8037450909614563f, -0.6476994752883911f,
    -0.514132022857666f, -0.3948058784008026f, -0.28474700450897217f,
    -0.1804928183555603f, -0.07950424402952194f, 0.020472243428230286f,
    0.12146082520484924f, 0.22571499645709991f, 0.33577385544776917f,
    0.45510002970695496f, 0.5886675119400024f, 0.7447131276130676f,
    0.9392961859703064f, -0.9392961859703064f, -0.7447131276130676f,
    -0.5886675119400024f, -0.45510002970695496f, -0.33577385544776917f,
    -0.22571499645709991f, -0.12146082520484924f, -0.020472243428230286f,
    0.07950424402952194f, 0.1804928183555603f, 0.28474700450897217f,
    0.3948058784008026f, 0.514132022857666f, 0.6476994752883911f,
    0.8037450909614563f, 0.9983281493186951f};

// Residual-bit layout: 32 bits per 32-elem group, 4 bytes, LSB-first.
// Bit i belongs to element i of the group (byte i/8, bit i%8).
__device__ __forceinline__ uint32_t residual_bit_q4(const uint8_t *res4,
                                                   int i) {
  return (static_cast<uint32_t>(res4[i >> 3] >> (i & 7))) & 1u;
}

// LM+residual dequant: x = LUT[(sbit<<4)|nib] * amax.
__device__ __forceinline__ float dequantize_q4_res(uint8_t nib, uint32_t sbit,
                                                  float scale) {
  return kQ4LmResLevels[(sbit << 4) | nib] * scale;
}

// K decode with optional residual: k_res_row points at this
// (block, head, slot) row (row base = k_res + row*G*4), d = head-dim.
// When Lm is false the residual load compiles out (row may be null).
template <bool Lm>
__device__ __forceinline__ float dequant_k_q4_res(const uint8_t *k_res_row,
                                                 int d, uint8_t nib,
                                                 float scale) {
  if constexpr (Lm) {
    int i = d & 31;
    uint32_t s =
        (static_cast<uint32_t>(k_res_row[(d >> 5) * 4 + (i >> 3)] >> (i & 7))) &
        1u;
    return kQ4LmResLevels[(s << 4) | nib] * scale;
  }
  return (static_cast<float>(nib) - 8.f) * scale;
}

// V decode with optional residual: v_res_gbase points at this
// (block, head, group) base (base = v_res + (((b*H+h)*G+g)*B)*4),
// slot = token slot in block, i = head-dim % 32.
template <bool Lm>
__device__ __forceinline__ float dequant_v_q4_res(const uint8_t *v_res_gbase,
                                                 int slot, int i, uint8_t nib,
                                                 float scale) {
  if constexpr (Lm) {
    uint32_t s =
        (static_cast<uint32_t>(v_res_gbase[slot * 4 + (i >> 3)] >> (i & 7))) &
        1u;
    return kQ4LmResLevels[(s << 4) | nib] * scale;
  }
  return (static_cast<float>(nib) - 8.f) * scale;
}

}  // namespace q4
}  // namespace vllm
