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
#include "quantization/q4/q4_utils.cuh"
#include "quantization/wht.cuh"

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
  // Scales go to sidecars: k token-major
  // [num_blocks, num_heads, block_size, head_size/32], v transposed to
  // [num_blocks, num_heads, head_size/32, block_size] so paged decode loads
  // one token-vec's scales as a single sector.
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
    float *__restrict__ v_scales,       // [num_blocks, num_heads,
                                         // head_size/32, block_size]
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
    const int64_t v_scale_idx =
        ((block_idx * num_heads + head_idx) * G + threadIdx.x) * block_size +
        block_offset;
    v_scales[v_scale_idx] = dv != 0.f ? dv : 1.f;
  }
  __syncthreads();

  // Phase 2: quantize + scattered write.
  for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
    const int c = d / vllm::q8::kQ8BlockSize;
    const int64_t scale_idx =
        ((block_idx * num_heads + head_idx) * block_size + block_offset) * G +
        c;
    float dk = k_scales[scale_idx];
    float dv = v_scales[((block_idx * num_heads + head_idx) * G + c) *
                            block_size +
                        block_offset];
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

// Q4_0 block quantize + blocked write. One CUDA block per (token, head),
// same amax scheme as Q8_0 with d = amax/7. Nibbles carry a +8 bias, two
// elems per byte: K packs along head-dim (byte holds d, d+1), V packs along
// tokens (byte holds slots 2m, 2m+1). K bytes are block-private so each
// thread writes whole bytes; V bytes are shared across the slot pair, so
// V nibbles merge with atomic And/Or on disjoint bits (zero-init not
// required, recycled blocks safe).
// Scales mirror Q8_0: k token-major, v transposed group-major.
template <typename scalar_t>
__global__ void reshape_and_cache_q4_kernel(
    const scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value, // [num_tokens, num_heads, head_size]
    int8_t *__restrict__ key_cache,     // [num_blocks, num_heads,
                                        // head_size/32, block_size, 16]
    int8_t *__restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                        // block_size/2]
    float *__restrict__ k_scales,       // [num_blocks, num_heads, block_size,
                                        // head_size/32]
    float *__restrict__ v_scales,       // [num_blocks, num_heads,
                                        // head_size/32, block_size]
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
  const int G = head_size / vllm::q4::kQ4BlockSize;

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
  // Phase 1: per-32 amax per tensor, identical to Q8_0.
  // Q4 K incoherence: rotate post-RoPE K by a fixed signed Hadamard before
  // amax/quant so outliers spread (TurboQuant-style, minus QJL). V stays
  // plain RTN. The query side applies the same rotation (pagedattention.cuh)
  // so the dot is exact; gather inverse-rotates. Active only for bf16/256
  // (must match the decode-side gate); otherwise K stages through unrotated.
  __shared__ float wht_k[256];
  constexpr bool kDoWht =
      std::is_same<scalar_t, __nv_bfloat16>::value;
  for (int d = threadIdx.x; d < head_size && d < 256; d += blockDim.x) {
    wht_k[d] = to_float(key[key_base + d]);
  }
  __syncthreads();
  const bool do_wht = kDoWht && vllm::wht::wht_supported(head_size);
  if (do_wht) {
    for (int d = threadIdx.x; d < 256; d += blockDim.x) {
      wht_k[d] *= vllm::wht::wht_sign(head_idx, d);
    }
    __syncthreads();
    vllm::wht::wht_inplace(wht_k, 256);
  }
  // Q4 V incoherence: same signed-Hadamard rotation along head-dim (per
  // token, always available). V packing/scales along slots are unchanged;
  // only the quantized values rotate. Decode un-rotates once on the output
  // vector (pagedattention.cuh); gather inverse-rotates (prefill path).
  __shared__ float wht_v[256];
  for (int d = threadIdx.x; d < head_size && d < 256; d += blockDim.x) {
    wht_v[d] = to_float(value[value_base + d]);
  }
  __syncthreads();
  if (do_wht) {
    for (int d = threadIdx.x; d < 256; d += blockDim.x) {
      wht_v[d] *= vllm::wht::wht_sign(head_idx, d);
    }
    __syncthreads();
    vllm::wht::wht_inplace(wht_v, 256);
  }
  for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
    float a = do_wht ? wht_k[d] : to_float(key[key_base + d]);
    a = a >= 0.f ? a : -a;
    atomicMax(reinterpret_cast<unsigned int *>(&sh_kmax[d / 32]),
              __float_as_uint(a));
    float b = do_wht ? wht_v[d] : to_float(value[value_base + d]);
    b = b >= 0.f ? b : -b;
    atomicMax(reinterpret_cast<unsigned int *>(&sh_vmax[d / 32]),
              __float_as_uint(b));
  }
  __syncthreads();
  if (threadIdx.x < G) {
    // LM codebook (WHT path) keeps d = amax so y = x/d = x/amax matches the
    // grid's fit distribution; plain Q4_0 keeps d = amax/7.
    float dk = do_wht ? sh_kmax[threadIdx.x]
                      : sh_kmax[threadIdx.x] / vllm::q4::kQ4MaxQ;
    float dv = do_wht ? sh_vmax[threadIdx.x]
                      : sh_vmax[threadIdx.x] / vllm::q4::kQ4MaxQ;
    const int64_t scale_idx =
        ((block_idx * num_heads + head_idx) * block_size + block_offset) * G +
        threadIdx.x;
    k_scales[scale_idx] = dk != 0.f ? dk : 1.f;
    const int64_t v_scale_idx =
        ((block_idx * num_heads + head_idx) * G + threadIdx.x) * block_size +
        block_offset;
    v_scales[v_scale_idx] = dv != 0.f ? dv : 1.f;
  }
  __syncthreads();

  // Phase 2: quantize + packed write, two head-dim elems per thread.
  const int64_t v_row_bytes = block_size / 2;
  for (int dd = threadIdx.x * 2; dd < head_size; dd += blockDim.x * 2) {
    const int c = dd / vllm::q4::kQ4BlockSize;
    const int64_t scale_idx =
        ((block_idx * num_heads + head_idx) * block_size + block_offset) * G +
        c;
    const int64_t v_scale_idx =
        ((block_idx * num_heads + head_idx) * G + c) * block_size +
        block_offset;
    float dk = k_scales[scale_idx];
    float dv = v_scales[v_scale_idx];
    float fk0 = do_wht ? wht_k[dd] : to_float(key[key_base + dd]);
    float fk1 = do_wht ? wht_k[dd + 1] : to_float(key[key_base + dd + 1]);
    float fv0 = do_wht ? wht_v[dd] : to_float(value[value_base + dd]);
    float fv1 = do_wht ? wht_v[dd + 1] : to_float(value[value_base + dd + 1]);
    float idk = 1.f / dk;
    float idv = 1.f / dv;
    float yk0 = fk0 * idk;
    float yk1 = fk1 * idk;
    float yv0 = fv0 * idv;
    float yv1 = fv1 * idv;
    uint8_t nk0, nk1, nv0, nv1;
    if (do_wht) {
      nk0 = vllm::q4::nearest_level_q4_lm(yk0);
      nk1 = vllm::q4::nearest_level_q4_lm(yk1);
      nv0 = vllm::q4::nearest_level_q4_lm(yv0);
      nv1 = vllm::q4::nearest_level_q4_lm(yv1);
    } else {
      yk0 = yk0 > 7.f ? 7.f : (yk0 < -8.f ? -8.f : yk0);
      yk1 = yk1 > 7.f ? 7.f : (yk1 < -8.f ? -8.f : yk1);
      yv0 = yv0 > 7.f ? 7.f : (yv0 < -8.f ? -8.f : yv0);
      yv1 = yv1 > 7.f ? 7.f : (yv1 < -8.f ? -8.f : yv1);
      nk0 = static_cast<uint8_t>((yk0 >= 0.f ? yk0 + 0.5f : yk0 - 0.5f) + 8.f);
      nk1 = static_cast<uint8_t>((yk1 >= 0.f ? yk1 + 0.5f : yk1 - 0.5f) + 8.f);
      nv0 = static_cast<uint8_t>((yv0 >= 0.f ? yv0 + 0.5f : yv0 - 0.5f) + 8.f);
      nv1 = static_cast<uint8_t>((yv1 >= 0.f ? yv1 + 0.5f : yv1 - 0.5f) + 8.f);
    }

    // K: 16-byte chunks hold 32 elems; byte (dd/32 chunk, (dd%32)/2).
    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / 32) * block_size * x +
        head_idx * (head_size / 32) * block_size * x +
        (dd / 32) * block_size * x + block_offset * x + (dd % 32) / 2;
    key_cache[tgt_key_idx] =
        static_cast<int8_t>(nk0 | (nk1 << 4));
    // V: bytes (dd, off/2) and (dd+1, off/2); each shared with the paired
    // slot, which owns the other nibble. Merge ours with atomic And/Or on
    // disjoint bits: order-independent and recycled-block safe.
    const int64_t v_byte_base =
        block_idx * num_heads * head_size * v_row_bytes +
        head_idx * head_size * v_row_bytes + block_offset / 2;
    const unsigned int v_shift =
        static_cast<unsigned int>(block_offset & 1) * 4u;
    const int64_t v_byte0 = v_byte_base + dd * v_row_bytes;
    unsigned int *word0 = reinterpret_cast<unsigned int *>(
        value_cache + (v_byte0 & ~3LL));
    unsigned int sh0 =
        static_cast<unsigned int>(v_byte0 & 3LL) * 8u + v_shift;
    atomicAnd(word0, ~(0xFu << sh0));
    atomicOr(word0, static_cast<unsigned int>(nv0) << sh0);
    const int64_t v_byte1 = v_byte0 + v_row_bytes;
    unsigned int *word1 = reinterpret_cast<unsigned int *>(
        value_cache + (v_byte1 & ~3LL));
    unsigned int sh1 =
        static_cast<unsigned int>(v_byte1 & 3LL) * 8u + v_shift;
    atomicAnd(word1, ~(0xFu << sh1));
    atomicOr(word1, static_cast<unsigned int>(nv1) << sh1);
  }
}

#define CALL_RESHAPE_AND_CACHE_Q4(KV_T)                                        \
  vllm::reshape_and_cache_q4_kernel<KV_T>                                      \
      <<<grid_q4, block_q4, 0, stream>>>(                                      \
          reinterpret_cast<KV_T *>(key), reinterpret_cast<KV_T *>(value),      \
          reinterpret_cast<int8_t *>(key_cache),                               \
          reinterpret_cast<int8_t *>(value_cache), k_scales, v_scales,         \
          slot_mapping, key_stride, value_stride, num_heads, head_size,        \
          block_size, x);

// Q4_0 block-quantize write path. Payload is nibbles (x = 16 bytes cover 32
// head-dim elems); scales are fp32 sidecars (k token-major, v transposed).
// MVP constraints (checked host-side): head_size % 32 == 0, head_size <= 512,
// block_size % 2 == 0.
extern "C" void reshape_and_cache_q4(
    void *key,         // [num_tokens, num_heads, head_size]
    void *value,       // [num_tokens, num_heads, head_size]
    void *key_cache,   // [num_blocks, num_heads, head_size/32, block_size, 16]
    void *value_cache, // [num_blocks, num_heads, head_size, block_size/2]
    float *k_scales,   // [num_blocks, num_heads, block_size, head_size/32]
    float *v_scales,   // [num_blocks, num_heads, head_size/32, block_size]
    int64_t *slot_mapping, // [num_tokens]

    int32_t num_tokens, int32_t num_heads, int32_t head_size,
    int32_t block_size, int32_t x, int32_t key_stride, int32_t value_stride,
    cudaStream_t stream,

    uint32_t dtype) {  // 0 => f16; 1 => bf16; 2 => f32
  dim3 grid_q4(static_cast<unsigned int>(num_tokens) *
               static_cast<unsigned int>(num_heads));
  // 128 threads cover head_size <= 512 via paired stride loops.
  dim3 block_q4(128);
  if (dtype == 0) {
    CALL_RESHAPE_AND_CACHE_Q4(uint16_t);
  } else if (dtype == 1) {
    CALL_RESHAPE_AND_CACHE_Q4(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_RESHAPE_AND_CACHE_Q4(float);
  }
  CUDA_CHECK(cudaGetLastError());
}

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
    // takes fp32 per-32 scale sidecars); 5 => q4_0 nibbles (reshape_and_cache_q4)
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
#define CALL_RESHAPE_AND_CACHE_Q8(KV_T)                                        \
  vllm::reshape_and_cache_q8_kernel<KV_T>                                      \
      <<<grid_q8, block_q8, 0, stream>>>(                                      \
          reinterpret_cast<KV_T *>(key), reinterpret_cast<KV_T *>(value),      \
          reinterpret_cast<int8_t *>(key_cache),                               \
          reinterpret_cast<int8_t *>(value_cache), k_scales, v_scales,         \
          slot_mapping, key_stride, value_stride, num_heads, head_size,        \
          block_size, x);
extern "C" void reshape_and_cache_q8(
    void *key,         // [num_tokens, num_heads, head_size]
    void *value,       // [num_tokens, num_heads, head_size]
    void *key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    void *value_cache, // [num_blocks, num_heads, head_size, block_size]
    float *k_scales,   // [num_blocks, num_heads, block_size, head_size/32]
    float *v_scales,   // [num_blocks, num_heads, head_size/32, block_size]
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
