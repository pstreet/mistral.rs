// ROCm/HIP FP8 KV-cache conversion utilities.
//
// RDNA3 (gfx110x/gfx115x) has no hardware FP8, so the ROCm build is compiled
// without ENABLE_FP8 and the FP8 KV-cache template instantiations are gated
// out in the kernels. The fp8 namespace is always declared because the kernel
// templates reference fp8::scaled_convert as a dependent call (the namespace
// must resolve at definition time); the body is only reached when an FP8
// instantiation is compiled.
#pragma once

#include "attention/attention_dtypes.h"
#include <assert.h>
#include <stdint.h>

namespace vllm {
namespace fp8 {
#ifdef ENABLE_FP8
// TODO(rocm): port software FP8 e4m3/e5m2 conversion for RDNA.
#endif
template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  (void)x;
  (void)scale;
  assert(false);
  __builtin_unreachable();
}
}  // namespace fp8
}  // namespace vllm
