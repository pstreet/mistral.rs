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

// Q8_0 dequantized float -> output scalar. Uses the same ROCm-safe
// conversions as the fp8 utils (float_to_half, __float2bfloat16).
template <typename out_t>
__device__ __forceinline__ out_t q8_out_cast(float v);
template <>
__device__ __forceinline__ uint16_t q8_out_cast<uint16_t>(float v) {
  return float_to_half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16
q8_out_cast<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ float q8_out_cast<float>(float v) {
  return v;
}

/// Gather K and V from paged KV cache into contiguous output tensors.
///
/// One CUDA block per output token, 256 threads cooperatively copy
/// kv_heads * head_size elements for both K and V.
///
/// Uses binary search on cu_seq_lens to find batch_id, avoiding a
/// separate token_to_seq tensor.
///
/// K cache layout: [num_blocks, kv_heads, head_size/x, block_size, x]
/// V cache layout: [num_blocks, kv_heads, head_size, block_size]
/// K/V output:     [num_tokens, kv_heads, head_size]
///
/// K and V dtypes are runtime codes (0/1/2 native, 3 fp8, 4 q8, 5 q4):
/// one instantiation serves every pair. x is the K-side packing.
template <typename out_t>
__global__ void gather_kv_cache_kernel(
    const void *__restrict__ key_cache,   // [num_blocks, kv_heads,
                                          //  head_size/x, block_size, x]
    const void *__restrict__ value_cache, // [num_blocks, kv_heads,
                                          //  head_size, block_size]
    out_t *__restrict__ k_out,         // [num_tokens, kv_heads, head_size]
    out_t *__restrict__ v_out,         // [num_tokens, kv_heads, head_size]
    const float *__restrict__ k_scale, // scalar or nullptr
    const float *__restrict__ v_scale, // scalar or nullptr
    const uint8_t *__restrict__ k_res, // Q4 QJL bits (or nullptr)
    const uint8_t *__restrict__ v_res, // Q4 QJL bits (or nullptr)
    const int32_t *__restrict__ block_table, // [batch, max_blocks]
    const int32_t *__restrict__ cu_seq_lens, // [batch + 1]
    const int32_t num_tokens, const int32_t num_seqs, const int32_t block_size,
    const int32_t block_table_stride, const int32_t num_kv_heads,
    const int32_t head_size, const int32_t x, const uint32_t k_cache_dtype,
    const uint32_t v_cache_dtype) {
  const int32_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

  // Binary search cu_seq_lens to find batch_id.
  // cu_seq_lens is [batch+1] with cumulative token counts.
  // We want the largest i such that cu_seq_lens[i] <= token_id.
  int32_t lo = 0, hi = num_seqs;
  while (lo < hi) {
    int32_t mid = (lo + hi + 1) / 2;
    if (cu_seq_lens[mid] <= token_id) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int32_t batch_id = lo;
  if (batch_id >= num_seqs) {
    return;
  }

  const int32_t batch_offset = token_id - cu_seq_lens[batch_id];
  const int32_t block_table_id = batch_offset / block_size;
  const int32_t slot = batch_offset % block_size;
  const int32_t block_id =
      block_table[batch_id * block_table_stride + block_table_id];

  const int32_t n = num_kv_heads * head_size;
  const int64_t out_base =
      static_cast<int64_t>(token_id) * num_kv_heads * head_size;

  // Precompute strides
  const int64_t k_block_stride =
      static_cast<int64_t>(num_kv_heads) * (head_size / x) * block_size * x;
  const int64_t k_head_stride =
      static_cast<int64_t>(head_size / x) * block_size * x;
  const int64_t v_block_stride =
      static_cast<int64_t>(num_kv_heads) * head_size * block_size;
  const int64_t v_head_stride = static_cast<int64_t>(head_size) * block_size;

  // Q8_0 scale groups per head-dim row. Pure arithmetic: safe to compute even
  // when the sidecar pointers are null (only dereferenced in the Q8 branch).
  const int32_t q8_groups = head_size / vllm::q8::kQ8BlockSize;

  // Q4_0 K incoherence: the cache holds signed-Hadamard-rotated K (see
  // reshape_and_cache). Gather feeds original-domain consumers (prefill),
  // so dequantize each head row then inverse-rotate (H is self-inverse;
  // signs go after the transform). V is plain RTN: handled below as usual.
  // Non-256 head sizes use the identity path in the main loop.
  if (k_cache_dtype == 5) {
    if (head_size == 256) {
      __shared__ float wht_row[256];
      for (int32_t h = 0; h < num_kv_heads; ++h) {
        // QJL residual row for this (block, head, slot), token-major.
        // Null when the sidecars are not allocated (QJL disabled).
        const uint8_t *k_res_row = nullptr;
        if (k_res != nullptr) {
          k_res_row =
              k_res + ((static_cast<int64_t>(block_id) * num_kv_heads + h) *
                           block_size +
                       slot) *
                          q8_groups * 4;
        }
        for (int32_t d = threadIdx.x; d < 256; d += blockDim.x) {
          const int64_t k_q4_idx =
              static_cast<int64_t>(block_id) * k_block_stride +
              h * k_head_stride + (d / 32) * block_size * x + slot * x +
              (d % 32) / 2;
          const uint8_t k_packed =
              reinterpret_cast<const uint8_t *>(key_cache)[k_q4_idx];
          const uint8_t k_nib =
              (d & 1) ? static_cast<uint8_t>((k_packed >> 4) & 0xFu)
                      : static_cast<uint8_t>(k_packed & 0xFu);
          const int64_t scale_row =
              (static_cast<int64_t>(block_id) * num_kv_heads + h) * block_size +
              slot;
          wht_row[d] = vllm::q4::dequant_k_q4_res<true>(
              k_res_row, d, k_nib,
              k_scale[scale_row * q8_groups + d / vllm::q4::kQ4BlockSize]);
        }
        __syncthreads();
        vllm::wht::wht_inplace(wht_row, 256);
        for (int32_t d = threadIdx.x; d < 256; d += blockDim.x) {
          k_out[out_base + h * head_size + d] = q8_out_cast<out_t>(
              wht_row[d] * vllm::wht::wht_sign(h, d));
        }
        __syncthreads();
      }
    }
  }

  // Q4_0 V incoherence (mirror of the K prepass above): the cache holds
  // head-dim-rotated V. Gather feeds original-domain consumers, so dequant
  // each head row then inverse-rotate (signs after the transform).
  if (v_cache_dtype == 5) {
    if (head_size == 256) {
      __shared__ float wht_vrow[256];
      for (int32_t h = 0; h < num_kv_heads; ++h) {
        for (int32_t d = threadIdx.x; d < 256; d += blockDim.x) {
          const int64_t v_q4_idx =
              (static_cast<int64_t>(block_id) * num_kv_heads + h) * head_size *
                  (block_size / 2) +
              d * (block_size / 2) + slot / 2;
          const uint8_t v_packed =
              reinterpret_cast<const uint8_t *>(value_cache)[v_q4_idx];
          const uint8_t v_nib =
              (slot & 1) ? static_cast<uint8_t>((v_packed >> 4) & 0xFu)
                         : static_cast<uint8_t>(v_packed & 0xFu);
          const int64_t v_scale_idx =
              ((static_cast<int64_t>(block_id) * num_kv_heads + h) *
                   q8_groups +
               d / vllm::q4::kQ4BlockSize) *
                  block_size +
              slot;
          // QJL residual group base for this head-dim group, group-major.
          // Null when the sidecars are not allocated (QJL disabled).
          const uint8_t *v_res_gbase = nullptr;
          if (v_res != nullptr) {
            v_res_gbase =
                v_res + (((static_cast<int64_t>(block_id) * num_kv_heads + h) *
                              q8_groups +
                          d / vllm::q4::kQ4BlockSize) *
                         block_size) *
                            4;
          }
          wht_vrow[d] = vllm::q4::dequant_v_q4_res<true>(
              v_res_gbase, slot, d & 31, v_nib, v_scale[v_scale_idx]);
        }
        __syncthreads();
        vllm::wht::wht_inplace(wht_vrow, 256);
        for (int32_t d = threadIdx.x; d < 256; d += blockDim.x) {
          v_out[out_base + h * head_size + d] = q8_out_cast<out_t>(
              wht_vrow[d] * vllm::wht::wht_sign(h, d));
        }
        __syncthreads();
      }
    }
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / head_size;
    const int d = i % head_size;

    // K: [block_id, head_idx, d/x, slot, d%x]
    const int x_idx = d / x;
    const int x_offset = d % x;
    const int64_t k_src_idx = static_cast<int64_t>(block_id) * k_block_stride +
                              head_idx * k_head_stride +
                              x_idx * block_size * x + slot * x + x_offset;

    // V: [block_id, head_idx, d, slot]
    const int64_t v_src_idx = static_cast<int64_t>(block_id) * v_block_stride +
                              head_idx * v_head_stride + d * block_size + slot;

    if (k_cache_dtype == 4) {
      // Q8_0 block-int8: int8 payload, fp32 per-32 scales in sidecars. k is
      // token-major [blocks, heads, block_size, groups]; v is transposed to
      // [blocks, heads, groups, block_size], so the v lookup strides here
      // (prefill-only path; decode reads v scales as one sector).
      const int64_t scale_row = (static_cast<int64_t>(block_id) * num_kv_heads +
                                 head_idx) *
                                    block_size +
                                slot;
      const float k_deq = vllm::q8::dequantize_q8_0(
          reinterpret_cast<const int8_t *>(key_cache)[k_src_idx],
          k_scale[scale_row * q8_groups + d / vllm::q8::kQ8BlockSize]);
      k_out[out_base + i] = q8_out_cast<out_t>(k_deq);
    } else if (k_cache_dtype == 5) {
      // Q4_0 nibbles (+8 bias): K packs along head-dim (byte holds the
      // (d, d+1) pair, d even selects lo). Scales share the Q8_0 scheme (k
      // token-major).
      const int64_t k_q4_idx = static_cast<int64_t>(block_id) *
                                   k_block_stride +
                               head_idx * k_head_stride +
                               (d / 32) * block_size * x + slot * x + (d % 32) / 2;
      const uint8_t k_packed =
          reinterpret_cast<const uint8_t *>(key_cache)[k_q4_idx];
      const uint8_t k_nib =
          (d & 1) ? static_cast<uint8_t>((k_packed >> 4) & 0xFu)
                  : static_cast<uint8_t>(k_packed & 0xFu);
      const int64_t scale_row = (static_cast<int64_t>(block_id) * num_kv_heads +
                                 head_idx) *
                                    block_size +
                                slot;
      const float k_deq = vllm::q4::dequantize_q4<false>(
          k_nib, k_scale[scale_row * q8_groups + d / vllm::q4::kQ4BlockSize]);
      if (head_size != 256) {
        k_out[out_base + i] = q8_out_cast<out_t>(k_deq);
      }  // else K already inverse-rotated by the prepass above
    } else if (k_cache_dtype == 3) {
      k_out[out_base + i] = fp8::scaled_convert<out_t, uint8_t,
                                                Fp8KVCacheDataType::kFp8E4M3>(
          reinterpret_cast<const uint8_t *>(key_cache)[k_src_idx], *k_scale);
    } else {
      k_out[out_base + i] =
          reinterpret_cast<const out_t *>(key_cache)[k_src_idx];
    }

    if (v_cache_dtype == 4) {
      const int64_t v_scale_idx =
          ((static_cast<int64_t>(block_id) * num_kv_heads + head_idx) *
               q8_groups +
           d / vllm::q8::kQ8BlockSize) *
              block_size +
          slot;
      const float v_deq = vllm::q8::dequantize_q8_0(
          reinterpret_cast<const int8_t *>(value_cache)[v_src_idx],
          v_scale[v_scale_idx]);
      v_out[out_base + i] = q8_out_cast<out_t>(v_deq);
    } else if (v_cache_dtype == 5) {
      // Q4_0: V packs along slots (byte holds the (slot, slot+1) pair).
      // v scales are transposed group-major, shared with Q8_0.
      const int64_t v_q4_idx = (static_cast<int64_t>(block_id) * num_kv_heads +
                                head_idx) *
                                   head_size * (block_size / 2) +
                               d * (block_size / 2) + slot / 2;
      const uint8_t v_packed =
          reinterpret_cast<const uint8_t *>(value_cache)[v_q4_idx];
      const uint8_t v_nib =
          (slot & 1) ? static_cast<uint8_t>((v_packed >> 4) & 0xFu)
                     : static_cast<uint8_t>(v_packed & 0xFu);
      const int64_t v_scale_idx =
          ((static_cast<int64_t>(block_id) * num_kv_heads + head_idx) *
               q8_groups +
           d / vllm::q4::kQ4BlockSize) *
              block_size +
          slot;
      const float v_deq =
          vllm::q4::dequantize_q4<false>(v_nib, v_scale[v_scale_idx]);
      if (head_size != 256) {
        v_out[out_base + i] = q8_out_cast<out_t>(v_deq);
      }  // else V already inverse-rotated by the prepass above
    } else if (v_cache_dtype == 3) {
      v_out[out_base + i] = fp8::scaled_convert<out_t, uint8_t,
                                                Fp8KVCacheDataType::kFp8E4M3>(
          reinterpret_cast<const uint8_t *>(value_cache)[v_src_idx], *v_scale);
    } else {
      v_out[out_base + i] =
          reinterpret_cast<const out_t *>(value_cache)[v_src_idx];
    }
  }
}

} // namespace vllm

#define CALL_GATHER_KV_CACHE(OUT_T)                                             \
  vllm::gather_kv_cache_kernel<OUT_T>                                          \
      <<<grid, block, 0, stream>>>(                                            \
          key_cache, value_cache,                                              \
          reinterpret_cast<OUT_T *>(k_out), reinterpret_cast<OUT_T *>(v_out),  \
          reinterpret_cast<const float *>(k_scale),                            \
          reinterpret_cast<const float *>(v_scale),                            \
          reinterpret_cast<const uint8_t *>(k_res),                            \
          reinterpret_cast<const uint8_t *>(v_res), block_table, cu_seq_lens,  \
          num_tokens, num_seqs, block_size, block_table_stride, num_kv_heads,  \
          head_size, x, k_cache_dtype, v_cache_dtype);

extern "C" void gather_kv_cache(
    void *key_cache,   // [num_blocks, kv_heads, head_size/x, block_size, x]
    void *value_cache, // [num_blocks, kv_heads, head_size, block_size]
    void *k_out,       // [num_tokens, kv_heads, head_size]
    void *v_out,       // [num_tokens, kv_heads, head_size]
    void *k_scale,     // scalar or nullptr
    void *v_scale,     // scalar or nullptr
    void *k_res,       // Q4 QJL bits (or nullptr)
    void *v_res,       // Q4 QJL bits (or nullptr)
    const int32_t *block_table, // [batch, max_blocks]
    const int32_t *cu_seq_lens, // [batch + 1]
    int32_t num_tokens, int32_t num_seqs, int32_t block_size,
    int32_t block_table_stride, int32_t num_kv_heads, int32_t head_size,
    int32_t x, cudaStream_t stream,
    uint32_t out_dtype,  // 0 => f16; 1 => bf16; 2 => f32
    // Per-side codes, each 0/1/2 native or 3 fp8_e4m3, 4 q8_0, 5 q4_0.
    // One instantiation serves every (k, v) pair; the kernel branches
    // per side. x is the K-side packing.
    uint32_t k_cache_dtype, uint32_t v_cache_dtype
) {
  if (num_tokens <= 0) {
    return;
  }
  dim3 grid(num_tokens);
  dim3 block(std::min(num_kv_heads * head_size, 512));

  if (out_dtype == 0) {
    CALL_GATHER_KV_CACHE(uint16_t);
  } else if (out_dtype == 1) {
    CALL_GATHER_KV_CACHE(__nv_bfloat16);
  } else if (out_dtype == 2) {
    CALL_GATHER_KV_CACHE(float);
  }
  CUDA_CHECK(cudaGetLastError());
}
