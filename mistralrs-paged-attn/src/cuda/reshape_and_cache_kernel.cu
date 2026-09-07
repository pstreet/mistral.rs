#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

#ifdef USE_ROCM
#include "quantization/fp8/amd/quant_utils.cuh"
#else
#include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include "quantization/q8/q8_utils.cuh"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

namespace vllm {

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value, // [num_tokens, num_heads, head_size]
    cache_t *__restrict__ key_cache,    // [num_blocks, num_heads, head_size/x,
                                        // block_size, x]
    cache_t *__restrict__ value_cache,  // [num_blocks, num_heads, head_size,
                                        // block_size]
    const int64_t *__restrict__ slot_mapping, // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float *k_scale, const float *v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_idx] =
          vllm::fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, *k_scale);
      value_cache[tgt_value_idx] =
          vllm::fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value,
                                                              *v_scale);
    }
  }
}

#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                        \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>                      \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<KV_T *>(key), reinterpret_cast<KV_T *>(value),      \
          reinterpret_cast<CACHE_T *>(key_cache),                              \
          reinterpret_cast<CACHE_T *>(value_cache), slot_mapping, key_stride,  \
          value_stride, num_heads, head_size, block_size, x,                   \
          reinterpret_cast<const float *>(k_scale),                            \
          reinterpret_cast<const float *>(v_scale));

// Q8_0 block quantize + blocked write. One CUDA block per (token, head);
// threads cooperatively reduce per-32 amax in shared memory, then quantize.
// Scales go to sidecars [num_blocks, num_heads, block_size, head_size/32].
template <typename scalar_t>
__global__ void reshape_and_cache_q8_kernel(
    const scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value, // [num_tokens, num_heads, head_size]
    int8_t *__restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                        // block_size, x], x = 16
    int8_t *__restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                        // block_size]
    float *__restrict__ k_scales,       // [num_blocks, num_heads, block_size,
                                        // head_size/32]
    float *__restrict__ v_scales,       // same layout as k_scales
    const int64_t *__restrict__ slot_mapping, // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x) {
  const int64_t token_head = blockIdx.x;
  const int64_t token_idx = token_head / num_heads;
  const int head_idx = token_head % num_heads;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int G = head_size / vllm::q8::kQ8BlockSize;

  __shared__ float sh_kmax[16];
  __shared__ float sh_vmax[16];
  if (threadIdx.x < 16) {
    sh_kmax[threadIdx.x] = 0.f;
    sh_vmax[threadIdx.x] = 0.f;
  }
  __syncthreads();
  const int64_t key_base = token_idx * key_stride + head_idx * head_size;
  const int64_t value_base = token_idx * value_stride + head_idx * head_size;
  auto to_float = [](scalar_t v) -> float {
    if constexpr (std::is_same<scalar_t, float>::value) {
      return v;
    } else if constexpr (std::is_same<scalar_t, uint16_t>::value) {
      __half h;
      memcpy(&h, &v, sizeof(h));
      return __half2float(h);
    } else {
      return __bfloat162float(v);
    }
  };
  // Phase 1: per-32 amax per tensor via integer atomicMax on non-negative
  // float bits (integer order == float order for values >= 0).
  for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
    float a = to_float(key[key_base + d]);
    a = a >= 0.f ? a : -a;
    atomicMax(reinterpret_cast<unsigned int *>(&sh_kmax[d / 32]),
              __float_as_uint(a));
    float b = to_float(value[value_base + d]);
    b = b >= 0.f ? b : -b;
    atomicMax(reinterpret_cast<unsigned int *>(&sh_vmax[d / 32]),
              __float_as_uint(b));
  }
  __syncthreads();
  if (threadIdx.x < G) {
    float dk = sh_kmax[threadIdx.x] / 127.f;
    float dv = sh_vmax[threadIdx.x] / 127.f;
    const int64_t scale_idx =
        ((block_idx * num_heads + head_idx) * block_size + block_offset) * G +
        threadIdx.x;
    k_scales[scale_idx] = dk != 0.f ? dk : 1.f;
    v_scales[scale_idx] = dv != 0.f ? dv : 1.f;
  }
  __syncthreads();

  // Phase 2: quantize + scattered write.
  for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
    const int c = d / vllm::q8::kQ8BlockSize;
    const int64_t scale_idx =
        ((block_idx * num_heads + head_idx) * block_size + block_offset) * G +
        c;
    float dk = k_scales[scale_idx];
    float dv = v_scales[scale_idx];
    float fk = to_float(key[key_base + d]);
    float fv = to_float(value[value_base + d]);
    float qk = fk / dk;
    float qv = fv / dv;
    qk = qk > 127.f ? 127.f : (qk < -127.f ? -127.f : qk);
    qv = qv > 127.f ? 127.f : (qv < -127.f ? -127.f : qv);
    int8_t oqk = static_cast<int8_t>(qk >= 0.f ? qk + 0.5f : qk - 0.5f);
    int8_t oqv = static_cast<int8_t>(qv >= 0.f ? qv + 0.5f : qv - 0.5f);

    const int x_idx = d / x;
    const int x_offset = d % x;
    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + d * block_size + block_offset;
    key_cache[tgt_key_idx] = oqk;
    value_cache[tgt_value_idx] = oqv;
  }
}

#define CALL_RESHAPE_AND_CACHE_Q8(KV_T)                                        \
  vllm::reshape_and_cache_q8_kernel<KV_T>                                      \
      <<<grid_q8, block_q8, 0, stream>>>(                                      \
          reinterpret_cast<KV_T *>(key), reinterpret_cast<KV_T *>(value),      \
          reinterpret_cast<int8_t *>(key_cache),                               \
          reinterpret_cast<int8_t *>(value_cache), k_scales, v_scales,         \
          slot_mapping, key_stride, value_stride, num_heads, head_size,        \
          block_size, x);

} // namespace vllm

extern "C" void reshape_and_cache(
    void *key,         // [num_tokens, num_heads, head_size]
    void *value,       // [num_tokens, num_heads, head_size]
    void *key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    void *value_cache, // [num_blocks, num_heads, head_size, block_size]
    int64_t *slot_mapping, // [num_tokens]

    int32_t num_tokens, int32_t num_heads, int32_t head_size,
    int32_t block_size, int32_t x, int32_t key_stride, int32_t value_stride,
    cudaStream_t stream,

    uint32_t dtype,       // 0 => f16; 1 => bf16; 2 => f32
    uint32_t cache_dtype, // 0 => f16; 1 => bf16; 2 => f32; 3 => fp8_e4m3
    // 4 => q8_0 block-int8 (served by reshape_and_cache_q8 below, which also
    // takes fp32 per-32 scale sidecars)
    float *k_scale, float *v_scale) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

#ifdef ENABLE_FP8
  if (cache_dtype == 3) {
    // FP8 E4M3 cache
    if (dtype == 0) {
      CALL_RESHAPE_AND_CACHE(uint16_t, uint8_t,
                             vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dtype == 1) {
      CALL_RESHAPE_AND_CACHE(__nv_bfloat16, uint8_t,
                             vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dtype == 2) {
      CALL_RESHAPE_AND_CACHE(float, uint8_t,
                             vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
  } else
#endif
  {
    // Non-FP8 cache
    if (dtype == 0) {
      CALL_RESHAPE_AND_CACHE(uint16_t, uint16_t,
                             vllm::Fp8KVCacheDataType::kAuto);
    } else if (dtype == 1) {
      CALL_RESHAPE_AND_CACHE(__nv_bfloat16, __nv_bfloat16,
                             vllm::Fp8KVCacheDataType::kAuto);
    } else if (dtype == 2) {
      CALL_RESHAPE_AND_CACHE(float, float, vllm::Fp8KVCacheDataType::kAuto);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

// Q8_0 block-quantize write path. Payload is int8 with x = 16 packing;
// scales are fp32 sidecars [num_blocks, num_heads, block_size, head_size/32].
// MVP constraints (checked host-side): head_size % 32 == 0, head_size <= 512.
extern "C" void reshape_and_cache_q8(
    void *key,         // [num_tokens, num_heads, head_size]
    void *value,       // [num_tokens, num_heads, head_size]
    void *key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    void *value_cache, // [num_blocks, num_heads, head_size, block_size]
    float *k_scales,   // [num_blocks, num_heads, block_size, head_size/32]
    float *v_scales,   // same layout as k_scales
    int64_t *slot_mapping, // [num_tokens]

    int32_t num_tokens, int32_t num_heads, int32_t head_size,
    int32_t block_size, int32_t x, int32_t key_stride, int32_t value_stride,
    cudaStream_t stream,

    uint32_t dtype) {  // 0 => f16; 1 => bf16; 2 => f32
  dim3 grid_q8(static_cast<unsigned int>(num_tokens) *
               static_cast<unsigned int>(num_heads));
  // 128 threads cover head_size <= 512 via stride loops (head_size % 32 == 0).
  dim3 block_q8(128);
  if (dtype == 0) {
    CALL_RESHAPE_AND_CACHE_Q8(uint16_t);
  } else if (dtype == 1) {
    CALL_RESHAPE_AND_CACHE_Q8(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_RESHAPE_AND_CACHE_Q8(float);
  }
  CUDA_CHECK(cudaGetLastError());
}
