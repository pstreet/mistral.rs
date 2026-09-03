// ROCm/HIP FP8 KV-cache conversion utilities.
//
// RDNA has no hardware FP8, so e4m3 <-> float is done in software. Only the
// vector widths the paged-attention kernels instantiate on RDNA are provided
// (VEC_SIZE 1..8 for f32/f16/bf16, plus the scalar write path); anything else
// falls through to the assert below.
#pragma once

#include "attention/attention_dtypes.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>

namespace vllm {
namespace fp8 {
#ifdef ENABLE_FP8

// e4m3: 1 sign, 4 exp (bias 7), 3 mantissa. 0x7F is NaN, max finite 0x7E = 448.
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
  const uint32_t e = (x >> 3) & 0xF;
  const uint32_t m = x & 0x7;
  if (e == 15 && m == 7) {
    return __uint_as_float(0x7FC00000u | ((x & 0x80) << 24));
  }
  if (e == 0) {
    // subnormal: m * 2^-9
    const float s = (x & 0x80) ? -1.f : 1.f;
    return s * (float)m * 0.001953125f;
  }
  // normal: (1 + m/8) * 2^(e - 7); float exp field is (e - 7) + 127 = e + 120
  const uint32_t bits = ((x & 0x80) << 24) | ((e + 120) << 23) | (m << 20);
  return __uint_as_float(bits);
}

__device__ __forceinline__ uint8_t float_to_fp8e4m3(float x) {
  if (x != x) return 0x7Fu;
  const uint32_t sign = (x < 0.f) ? 1u : 0u;
  const float a = fabsf(x);
  if (a == 0.f) return (uint8_t)(sign << 7);
  if (a >= 448.f) return (uint8_t)((sign << 7) | 0x7E);  // satfinite
  const uint32_t bits = __float_as_uint(a);
  const uint32_t exp = (bits >> 23) & 0xFF;
  const uint32_t mant = bits & 0x7FFFFF;
  // a = 2^(exp - 127) * (1.mant); fp8 exponent e8 = exp - 127 + 7 = exp - 120
  int e8 = (int)exp - 120;
  if (e8 >= 1) {
    // normal: keep top 3 mantissa bits, round the 20 discarded (RNE)
    uint32_t m8 = (mant >> 20) & 0x7;
    const uint32_t discarded = mant & 0x7FFFF;
    const uint32_t half = 1u << 19;
    if (discarded > half || (discarded == half && (m8 & 1))) {
      m8 += 1;
      if (m8 == 8) {
        m8 = 0;
        e8 += 1;
      }
    }
    if (e8 > 15) return (uint8_t)((sign << 7) | 0x7E);
    return (uint8_t)((sign << 7) | ((uint32_t)e8 << 3) | m8);
  }
  // subnormal (a < 2^-6): m8 * 2^-9
  int m8 = (int)(a * 512.f + 0.5f);
  if (m8 > 7) m8 = 7;
  return (uint8_t)((sign << 7) | (uint32_t)m8);
}

// fp8 byte -> output scalar, scale applied.
__device__ __forceinline__ float to_f32(uint8_t b, float s) {
  return fp8e4m3_to_float(b) * s;
}
__device__ __forceinline__ uint16_t to_f16(uint8_t b, float s) {
  return float_to_half(fp8e4m3_to_float(b) * s);
}
__device__ __forceinline__ __nv_bfloat16 to_bf16(uint8_t b, float s) {
  return __float2bfloat16(fp8e4m3_to_float(b) * s);
}
// input scalar -> fp8 byte, scale applied.
__device__ __forceinline__ uint8_t from_f32(float x, float s) {
  return float_to_fp8e4m3(x / s);
}
__device__ __forceinline__ uint8_t from_f16(uint16_t x, float s) {
  return float_to_fp8e4m3(half_to_float(x) / s);
}
__device__ __forceinline__ uint8_t from_bf16(__nv_bfloat16 x, float s) {
  return float_to_fp8e4m3(__bfloat162float(x) / s);
}

// Tin is a packed fp8 vector (N bytes), Tout the matching N-element output.
// Signatures must match this primary template for the specializations to bind.
template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin& x, const float scale);

// --- read: fp8 -> f32 ---
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(const uint8_t& x,
                                                                  const float s) {
  return to_f32(x, s);
}
template <>
__inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(
    const uint16_t& x, const float s) {
  union {
    uint16_t u;
    uint8_t b[2];
  } i;
  i.u = x;
  return make_float2(to_f32(i.b[0], s), to_f32(i.b[1], s));
}
template <>
__inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(
    const uint32_t& x, const float s) {
  union {
    uint32_t u;
    uint8_t b[4];
  } i;
  i.u = x;
  return make_float4(to_f32(i.b[0], s), to_f32(i.b[1], s), to_f32(i.b[2], s),
                     to_f32(i.b[3], s));
}

// --- read: fp8 -> f16 ---
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
    const uint8_t& x, const float s) {
  return to_f16(x, s);
}
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(
    const uint16_t& x, const float s) {
  union {
    uint16_t u;
    uint8_t b[2];
  } i;
  i.u = x;
  union {
    uint32_t u32;
    uint16_t u16[2];
  } p;
  p.u16[0] = to_f16(i.b[0], s);
  p.u16[1] = to_f16(i.b[1], s);
  return p.u32;
}
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(
    const uint32_t& x, const float s) {
  union {
    uint32_t u;
    uint8_t b[4];
  } i;
  i.u = x;
  uint2 r;
  r.x = scaled_vec_conversion<uint32_t, uint16_t>(i.b[0] | (i.b[1] << 8), s);
  r.y = scaled_vec_conversion<uint32_t, uint16_t>(i.b[2] | (i.b[3] << 8), s);
  return r;
}
template <>
__inline__ __device__ uint4 scaled_vec_conversion<uint4, uint2>(const uint2& x,
                                                                const float s) {
  union {
    uint2 u;
    uint8_t b[8];
  } i;
  i.u = x;
  uint4 r;
  r.x = scaled_vec_conversion<uint32_t, uint16_t>(i.b[0] | (i.b[1] << 8), s);
  r.y = scaled_vec_conversion<uint32_t, uint16_t>(i.b[2] | (i.b[3] << 8), s);
  r.z = scaled_vec_conversion<uint32_t, uint16_t>(i.b[4] | (i.b[5] << 8), s);
  r.w = scaled_vec_conversion<uint32_t, uint16_t>(i.b[6] | (i.b[7] << 8), s);
  return r;
}

// --- read: fp8 -> bf16 ---
template <>
__inline__ __device__ __nv_bfloat16 scaled_vec_conversion<__nv_bfloat16,
                                                          uint8_t>(
    const uint8_t& x, const float s) {
  return to_bf16(x, s);
}
template <>
__inline__ __device__ __nv_bfloat162 scaled_vec_conversion<__nv_bfloat162,
                                                           uint16_t>(
    const uint16_t& x, const float s) {
  union {
    uint16_t u;
    uint8_t b[2];
  } i;
  i.u = x;
  __nv_bfloat162 r;
  r.x = to_bf16(i.b[0], s);
  r.y = to_bf16(i.b[1], s);
  return r;
}
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t& x, const float s) {
  union {
    uint32_t u;
    uint8_t b[4];
  } i;
  i.u = x;
  bf16_4_t r;
  r.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>(i.b[0] | (i.b[1] << 8),
                                                        s);
  r.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>(i.b[2] | (i.b[3] << 8),
                                                        s);
  return r;
}
template <>
__inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(
    const uint2& x, const float s) {
  union {
    uint2 u;
    uint8_t b[8];
  } i;
  i.u = x;
  bf16_8_t r;
  r.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>(i.b[0] | (i.b[1] << 8),
                                                        s);
  r.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>(i.b[2] | (i.b[3] << 8),
                                                        s);
  r.z = scaled_vec_conversion<__nv_bfloat162, uint16_t>(i.b[4] | (i.b[5] << 8),
                                                        s);
  r.w = scaled_vec_conversion<__nv_bfloat162, uint16_t>(i.b[6] | (i.b[7] << 8),
                                                        s);
  return r;
}

// --- write: f32/f16/bf16 -> fp8 (scalar, N=1) ---
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& x, const float s) {
  return from_f32(x, s);
}
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(
    const uint16_t& x, const float s) {
  return from_f16(x, s);
}
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& x, const float s) {
  return from_bf16(x, s);
}

#endif  // ENABLE_FP8

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
#ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale);
  }
  (void)x;
  (void)scale;
#endif
  assert(false);
  __builtin_unreachable();
}

}  // namespace fp8
}  // namespace vllm
