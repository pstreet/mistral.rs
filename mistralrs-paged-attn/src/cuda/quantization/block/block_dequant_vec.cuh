#pragma once

// Vectorized block-quantized (Q8_0/Q4_0) dequant staging for the paged
// decode path: 4-wide int->float via convertvector, vector scale multiply,
// packed float->bf16 convert. Callers build the float4 lane values (and the
// 4 scales, indexed or contiguous); this header only converts and packs.
//
// Portable intent, ROCm reality: the packed float->bf16 convertvector needs
// __bf16 vector support, which hipcc has but nvcc may not, so the GNU
// vector bits live behind USE_ROCM. Other toolchains keep today's scalar
// from_float path at the call sites.

#include <cstdint>

namespace vllm {
namespace bq {

// True for dtypes with a 4-wide vector dequant path (ROCm bf16 for now).
template <typename Scalar>
inline constexpr bool kUseVecDequant = false;
#ifdef USE_ROCM
template <>
inline constexpr bool kUseVecDequant<__nv_bfloat16> = true;
#endif

#ifdef USE_ROCM
typedef signed char bqv_i8x4 __attribute__((ext_vector_type(4)));
typedef int bqv_i32x4 __attribute__((ext_vector_type(4)));
typedef float bqv_f32x4 __attribute__((ext_vector_type(4)));
typedef float bqv_f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bqv_bf16x2 __attribute__((ext_vector_type(2)));

// Pack 4 floats to 4 XT::bf16 (8 bytes) at dst. dst must hold 8 bytes.
inline __device__ void pack_f32x4_to_bf16(void *dst, bqv_f32x4 v) {
  bqv_f32x2 lo = {v[0], v[1]};
  bqv_f32x2 hi = {v[2], v[3]};
  bqv_bf16x2 blo = __builtin_convertvector(lo, bqv_bf16x2);
  bqv_bf16x2 bhi = __builtin_convertvector(hi, bqv_bf16x2);
  unsigned int lo32;
  unsigned int hi32;
  __builtin_memcpy(&lo32, &blo, 4);
  __builtin_memcpy(&hi32, &bhi, 4);
  unsigned long long packed =
      (static_cast<unsigned long long>(hi32) << 32) | lo32;
  __builtin_memcpy(dst, &packed, 8);
}
#endif

}  // namespace bq
}  // namespace vllm
