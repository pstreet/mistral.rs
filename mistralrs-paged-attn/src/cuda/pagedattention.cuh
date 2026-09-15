/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdint.h>
#include <stdio.h>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#include "attention/attention_dtypes.h"
#include "attention/attention_utils.cuh"

#ifdef USE_ROCM
#include "quantization/fp8/amd/quant_utils.cuh"
#else
#include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include "quantization/q8/q8_utils.cuh"
#include "quantization/q4/q4_utils.cuh"
#include "quantization/wht.cuh"
#include "quantization/block/block_dequant_vec.cuh"

#include <algorithm>
#include <type_traits>

// Must be a constant expression (used in constexpr math). RDNA targets are
// wave32; HIP's warpSize variable is not a constant expression.
#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

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

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float *red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

inline __device__ float fast_tanh(float x) {
#if defined(__CUDA_ARCH__)
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
  float y;
  asm volatile("tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
  return y;
#else
  return ::tanhf(x);
#endif
#else
  return std::tanh(x);
#endif
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt,
          int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int PARTITION_SIZE = 0> // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float *__restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float *__restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    scalar_t *__restrict__ out,     // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    const scalar_t *__restrict__ q, // [num_seqs, num_heads, head_size]
    const cache_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                         // head_size/x, block_size, x]
    const cache_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                         // head_size, block_size]
    const int num_kv_heads,              // [num_heads]
    const float scale, const float softcapping,
    const uint32_t
        *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const uint32_t *__restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float *k_scale, const float *v_scale,
    const uint8_t *__restrict__ k_res, // Q4 QJL residual bits (or nullptr)
    const uint8_t *__restrict__ v_res, // Q4 QJL residual bits (or nullptr)
    const float *__restrict__ sinks // [num_heads] or nullptr
    ) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const uint32_t context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE
                                       // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in the
  // group has 0, 4, 8, ... th vectors of the query, and the second thread has
  // 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because q is
  // split from a qkv tensor, it may not be contiguous.
  const scalar_t *q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads(); // TODO(naed90): possible speedup if this is replaced with a
                   // memory wall right before we use q_vecs

  // Q4_0 Lloyd-Max codebook gate: the LM grid is only valid on WHT-rotated
  // (Gaussian) data, so it tracks the rotation gate (bf16/256) exactly.
  constexpr bool kLmQ4 = HEAD_SIZE == 256 &&
                         std::is_same<scalar_t, __nv_bfloat16>::value;

  // Q4_0 K incoherence: the cache holds signed-Hadamard-rotated K (see
  // reshape_and_cache). Rotate the staged query identically so Q'.K' = Q.K;
  // the dot below needs no inverse. Q8/FP8/BF16 K is unrotated: skip. Gate
  // must match the store side (bf16/256); other instantiations keep identity.
  if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kQ4_0 && kLmQ4) {
    __shared__ float wht_q[256];
    scalar_t *qflat = reinterpret_cast<scalar_t *>(q_vecs);
    for (int d = thread_idx; d < 256; d += NUM_THREADS) {
      // Gate above pins scalar_t to bf16; direct conversion, no branches.
      wht_q[d] =
          __bfloat162float(qflat[d]) * vllm::wht::wht_sign(kv_head_idx, d);
    }
    __syncthreads();
    vllm::wht::wht_inplace(wht_q, 256);
    for (int d = thread_idx; d < 256; d += NUM_THREADS) {
      from_float(qflat[d], wht_q[d]);
    }
    __syncthreads();
  }

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float *logits = reinterpret_cast<float *>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  // Q8_0 block-int8: one fp32 scale per 32 head-dim elems. k sidecar is
  // [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_SIZE / 32] (token-major: one
  // token's scales are contiguous); v sidecar is transposed to
  // [num_blocks, num_kv_heads, HEAD_SIZE / 32, BLOCK_SIZE] so the decode
  // vec's per-token scales load as one sector. k_scale/v_scale carry the
  // sidecar bases when kv_dt == kQ8_0.
  constexpr int Q8_GROUPS = HEAD_SIZE / 32;
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const uint32_t *block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the thread group size is 4, then the first thread in the
    // group has 0, 4, 8, ... th vectors of the key, and the second thread has
    // 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];
#if defined(USE_ROCM)
      constexpr bool kQ4KStream = kv_dt == Fp8KVCacheDataType::kQ4_0 &&
                                  VEC_SIZE == 8 &&
                                  bq::kUseVecDequant<scalar_t>;
      typename FloatVec<K_vec>::Type qk_acc;
#endif

      // Q8_0/Q4_0 scale row base for this token; the address math is
      // loop-invariant so LICM keeps it in one register, while the scale
      // loads ride L1 across the vec loop instead of occupying 8 registers
      // per thread. Both dtypes share the token-major k sidecar layout.
      const int64_t qk_scale_base =
          (physical_block_number * num_kv_heads + kv_head_idx) * BLOCK_SIZE +
          physical_block_offset;

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t *k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec *>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kQ8_0) {
          using Cache_K_vec = typename vllm::Vec<uint8_t, VEC_SIZE>::Type;
          Cache_K_vec q8_k_packed = *reinterpret_cast<const Cache_K_vec *>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          const int8_t *q8_k_bytes =
              reinterpret_cast<const int8_t *>(&q8_k_packed);
          scalar_t *q8_k_dst = reinterpret_cast<scalar_t *>(&k_vecs[j]);
          // Head-dim base of this vec. The group index is a branchless shift:
          // the old walking compare diverged across threads and chained the
          // unrolled iterations through a loop-carried dependency.
          const int q8_base = vec_idx * VEC_SIZE;
#if defined(USE_ROCM)
          if constexpr (VEC_SIZE == 4 && vllm::bq::kUseVecDequant<scalar_t>) {
            // 4-wide: int8 -> float vector, splat/blend scales, packed
            // float -> bf16 convert. Same values as the scalar loop.
            vllm::bq::bqv_i8x4 qb;
            __builtin_memcpy(&qb, q8_k_bytes, 4);
            vllm::bq::bqv_f32x4 qf = __builtin_convertvector(
                __builtin_convertvector(qb, vllm::bq::bqv_i32x4),
                vllm::bq::bqv_f32x4);
            const float *q8_row = k_scale + qk_scale_base * Q8_GROUPS;
            vllm::bq::bqv_f32x4 sc = {
                q8_row[(q8_base + 0) >> 5],
                q8_row[(q8_base + 1) >> 5],
                q8_row[(q8_base + 2) >> 5],
                q8_row[(q8_base + 3) >> 5],
            };
            vllm::bq::pack_f32x4_to_bf16(q8_k_dst, qf * sc);
          } else
#endif
          {
#pragma unroll
            for (int e = 0; e < VEC_SIZE; ++e) {
              from_float(q8_k_dst[e],
                         vllm::q8::dequantize_q8_0(
                             q8_k_bytes[e],
                             k_scale[qk_scale_base * Q8_GROUPS +
                                     ((q8_base + e) >> 5)]));
            }
          }
        } else if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kQ4_0) {
          // VEC_SIZE/2 bytes hold this vec's VEC_SIZE elems (lo = even elem,
          // hi = odd elem, +8 bias). Byte addresses reuse the Q8 byte math
          // with halved vec granularity; k sidecar stays token-major.
          // Degenerate VEC_SIZE == 1 (fp32, tiny blocks): one byte, one nibble.
          const int q4_vec = vec_idx * VEC_SIZE / 2;
          const int q4_off1 = q4_vec / x;
          const int q4_off2 = q4_vec % x;
          scalar_t *q4_k_dst = reinterpret_cast<scalar_t *>(&k_vecs[j]);
          const int q4_base = vec_idx * VEC_SIZE;
          // QJL residual row for this token (valid when kLmQ4; else null).
          const uint8_t *k_res_row = nullptr;
          if constexpr (kLmQ4) {
            k_res_row = k_res + qk_scale_base * Q8_GROUPS * 4;
          }
#if defined(USE_ROCM)
          if constexpr (kQ4KStream) {
            using Cache_K_vec = typename vllm::Vec<uint8_t, VEC_SIZE / 2>::Type;
            Cache_K_vec q4_k_packed = *reinterpret_cast<const Cache_K_vec *>(
                k_ptr + q4_off1 * BLOCK_SIZE * x + q4_off2);
            const uint8_t *q4_k_nibbles =
                reinterpret_cast<const uint8_t *>(&q4_k_packed);
            K_vec k_cur;
            scalar_t *k_dst = reinterpret_cast<scalar_t *>(&k_cur);
#pragma unroll
            for (int p = 0; p < VEC_SIZE / 2; ++p) {
              const uint8_t packed = q4_k_nibbles[p];
              from_float(k_dst[2 * p],
                         vllm::q4::dequant_k_q4_res<kLmQ4>(
                             k_res_row, q4_base + 2 * p, packed & 0xFu,
                             k_scale[qk_scale_base * Q8_GROUPS +
                                     ((q4_base + 2 * p) >> 5)]));
              from_float(k_dst[2 * p + 1],
                         vllm::q4::dequant_k_q4_res<kLmQ4>(
                             k_res_row, q4_base + 2 * p + 1,
                             (packed >> 4) & 0xFu,
                             k_scale[qk_scale_base * Q8_GROUPS +
                                     ((q4_base + 2 * p + 1) >> 5)]));
            }
            if (j == 0) {
              qk_acc = mul<typename FloatVec<K_vec>::Type, K_vec, K_vec>(
                  q_vecs[thread_group_offset][j], k_cur);
            } else {
              qk_acc = fma(q_vecs[thread_group_offset][j], k_cur, qk_acc);
            }
          } else if constexpr (VEC_SIZE == 4 && vllm::bq::kUseVecDequant<scalar_t>) {
            // 4-wide: 2 bytes -> 4 nibbles -> int/float vectors, indexed
            // scales, packed float -> bf16 convert. Same values as scalar.
            using Cache_K_vec = typename vllm::Vec<uint8_t, 2>::Type;
            Cache_K_vec q4_k_packed = *reinterpret_cast<const Cache_K_vec *>(
                k_ptr + q4_off1 * BLOCK_SIZE * x + q4_off2);
            const uint8_t *q4_nb =
                reinterpret_cast<const uint8_t *>(&q4_k_packed);
            vllm::bq::bqv_f32x4 qf;
            if constexpr (kLmQ4) {
              // Folded residual LUT at scale 1 (the sc vector applies amax).
              qf = {vllm::q4::dequant_k_q4_res<true>(
                        k_res_row, q4_base + 0, q4_nb[0] & 0xFu, 1.f),
                    vllm::q4::dequant_k_q4_res<true>(
                        k_res_row, q4_base + 1, (q4_nb[0] >> 4) & 0xFu, 1.f),
                    vllm::q4::dequant_k_q4_res<true>(
                        k_res_row, q4_base + 2, q4_nb[1] & 0xFu, 1.f),
                    vllm::q4::dequant_k_q4_res<true>(
                        k_res_row, q4_base + 3, (q4_nb[1] >> 4) & 0xFu, 1.f)};
            } else {
              vllm::bq::bqv_i32x4 qi = {
                  static_cast<int>(q4_nb[0] & 0xFu) - 8,
                  static_cast<int>((q4_nb[0] >> 4) & 0xFu) - 8,
                  static_cast<int>(q4_nb[1] & 0xFu) - 8,
                  static_cast<int>((q4_nb[1] >> 4) & 0xFu) - 8,
              };
              qf = __builtin_convertvector(qi, vllm::bq::bqv_f32x4);
            }
            const float *q4_row = k_scale + qk_scale_base * Q8_GROUPS;
            vllm::bq::bqv_f32x4 sc = {
                q4_row[(q4_base + 0) >> 5],
                q4_row[(q4_base + 1) >> 5],
                q4_row[(q4_base + 2) >> 5],
                q4_row[(q4_base + 3) >> 5],
            };
            vllm::bq::pack_f32x4_to_bf16(q4_k_dst, qf * sc);
          } else
#endif
          if constexpr (VEC_SIZE == 1) {
            const uint8_t *q4_k_byte = reinterpret_cast<const uint8_t *>(
                k_ptr + q4_off1 * BLOCK_SIZE * x + q4_off2);
            const uint8_t nib = (q4_base & 1)
                                    ? static_cast<uint8_t>((*q4_k_byte >> 4) & 0xFu)
                                    : static_cast<uint8_t>(*q4_k_byte & 0xFu);
            from_float(q4_k_dst[0],
                       vllm::q4::dequant_k_q4_res<kLmQ4>(
                           k_res_row, q4_base, nib,
                           k_scale[qk_scale_base * Q8_GROUPS + (q4_base >> 5)]));
          } else {
            using Cache_K_vec = typename vllm::Vec<uint8_t, VEC_SIZE / 2>::Type;
            Cache_K_vec q4_k_packed = *reinterpret_cast<const Cache_K_vec *>(
                k_ptr + q4_off1 * BLOCK_SIZE * x + q4_off2);
            const uint8_t *q4_k_nibbles =
                reinterpret_cast<const uint8_t *>(&q4_k_packed);
#pragma unroll
            for (int p = 0; p < VEC_SIZE / 2; ++p) {
              const uint8_t packed = q4_k_nibbles[p];
              from_float(q4_k_dst[2 * p],
                         vllm::q4::dequant_k_q4_res<kLmQ4>(
                             k_res_row, q4_base + 2 * p, packed & 0xFu,
                             k_scale[qk_scale_base * Q8_GROUPS +
                                     ((q4_base + 2 * p) >> 5)]));
              from_float(q4_k_dst[2 * p + 1],
                         vllm::q4::dequant_k_q4_res<kLmQ4>(
                             k_res_row, q4_base + 2 * p + 1,
                             (packed >> 4) & 0xFu,
                             k_scale[qk_scale_base * Q8_GROUPS +
                                     ((q4_base + 2 * p + 1) >> 5)]));
            }
          }
        } else {
          using Cache_K_vec = typename vllm::Vec<cache_t, VEC_SIZE>::Type;
          Cache_K_vec fp8_k_vec = *reinterpret_cast<const Cache_K_vec *>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);

          k_vecs[j] = vllm::fp8::scaled_convert<K_vec, Cache_K_vec, kv_dt>(
              fp8_k_vec, *k_scale);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk;
#if defined(USE_ROCM)
      if constexpr (kQ4KStream) {
        float qk_sum = sum(qk_acc);
#pragma unroll
        for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
          qk_sum += VLLM_SHFL_XOR_SYNC(qk_sum, mask);
        }
        qk = scale * qk_sum;
      } else
#endif
        qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
            q_vecs[thread_group_offset], k_vecs);

      // Apply softcapping
      if (softcapping != 1.0) {
        qk = fast_tanh(qk / softcapping) * softcapping;
      }

      // Add the ALiBi bias if slopes are given.
      qk +=
          (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // For non-partitioned (V1) mode, include the sink in the max.
  // For V2 (partitioned), the sink is handled once in the reduce kernel.
  if (!USE_PARTITIONING && sinks != nullptr) {
    qk_max = fmaxf(qk_max, sinks[head_idx]);
  }

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // For non-partitioned (V1) mode, include the sink in the exp sum.
  if (!USE_PARTITIONING && sinks != nullptr) {
    exp_sum += __expf(sinks[head_idx] - qk_max);
  }

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float *max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float *exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec *>(logits + token_idx -
                                                            start_token_idx));

    const cache_t *v_ptr = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec *>(v_ptr + offset);
        } else if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kQ8_0) {
          using Cache_V_vec = typename vllm::Vec<uint8_t, V_VEC_SIZE>::Type;
          Cache_V_vec q8_v_packed =
              *reinterpret_cast<const Cache_V_vec *>(v_ptr + offset);
          const int8_t *q8_v_bytes =
              reinterpret_cast<const int8_t *>(&q8_v_packed);
          scalar_t *q8_v_dst = reinterpret_cast<scalar_t *>(&v_vec);
          // v scales are group-major, so this (block, head, group) row holds
          // one scale per token and the vec's scales load as one sector.
          const int64_t q8_v_base =
              ((physical_block_number * num_kv_heads + kv_head_idx) *
                   Q8_GROUPS +
               row_idx / 32) *
                  BLOCK_SIZE +
              physical_block_offset;
#if defined(USE_ROCM)
          if constexpr (V_VEC_SIZE % 4 == 0 && vllm::bq::kUseVecDequant<scalar_t>) {
            // 4-wide chunks: int8 -> float vector, contiguous scale vector,
            // packed float -> bf16 convert. Same values as the scalar loop.
            for (int c = 0; c < V_VEC_SIZE; c += 4) {
              vllm::bq::bqv_i8x4 qb;
              __builtin_memcpy(&qb, q8_v_bytes + c, 4);
              vllm::bq::bqv_f32x4 qf = __builtin_convertvector(
                  __builtin_convertvector(qb, vllm::bq::bqv_i32x4),
                  vllm::bq::bqv_f32x4);
              vllm::bq::bqv_f32x4 sc;
              __builtin_memcpy(&sc, v_scale + q8_v_base + c, 16);
              vllm::bq::pack_f32x4_to_bf16(q8_v_dst + c, qf * sc);
            }
          } else
#endif
          {
#pragma unroll
            for (int e = 0; e < V_VEC_SIZE; ++e) {
              from_float(q8_v_dst[e], vllm::q8::dequantize_q8_0(
                                          q8_v_bytes[e], v_scale[q8_v_base + e]));
            }
          }
        } else if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kQ4_0) {
          static_assert(V_VEC_SIZE % 2 == 0, "Q4_0 needs an even V VEC_SIZE");
          // Q4 V packs along tokens: 2 slots per byte, so this vec's
          // V_VEC_SIZE tokens arrive as V_VEC_SIZE/2 contiguous bytes.
          // physical_block_offset is a multiple of V_VEC_SIZE (hence even).
          const int q4_v_offset =
              row_idx * (BLOCK_SIZE / 2) + (physical_block_offset >> 1);
          using Cache_V_vec =
              typename vllm::Vec<uint8_t, V_VEC_SIZE / 2>::Type;
          Cache_V_vec q4_v_packed =
              *reinterpret_cast<const Cache_V_vec *>(v_ptr + q4_v_offset);
          const uint8_t *q4_v_nibbles =
              reinterpret_cast<const uint8_t *>(&q4_v_packed);
          scalar_t *q4_v_dst = reinterpret_cast<scalar_t *>(&v_vec);
          // v scales stay per-token (group-major sidecar, shared with Q8_0).
          const int64_t q4_v_base =
              ((physical_block_number * num_kv_heads + kv_head_idx) *
                   Q8_GROUPS +
               row_idx / 32) *
                  BLOCK_SIZE +
              physical_block_offset;
          // QJL residual group base for this head-dim row (valid when kLmQ4).
          const uint8_t *v_res_gbase = nullptr;
          if constexpr (kLmQ4) {
            v_res_gbase =
                v_res + (((physical_block_number * num_kv_heads + kv_head_idx) *
                              Q8_GROUPS +
                          row_idx / 32) *
                         BLOCK_SIZE) *
                            4;
          }
#if defined(USE_ROCM)
          if constexpr (V_VEC_SIZE % 4 == 0 && vllm::bq::kUseVecDequant<scalar_t>) {
            // 4-wide chunks: 2 nibble bytes -> 4 ints -> float vector,
            // contiguous scale vector, packed convert. Same values as scalar.
            for (int c = 0; c < V_VEC_SIZE; c += 4) {
              const uint8_t *nb = q4_v_nibbles + c / 2;
              vllm::bq::bqv_f32x4 qf;
              if constexpr (kLmQ4) {
                // Folded residual LUT at scale 1 (sc applies amax). All 4
                // lanes share head-dim row_idx; slots are offset+c+lane.
                const int vi = row_idx & 31;
                qf = {vllm::q4::dequant_v_q4_res<true>(
                          v_res_gbase, physical_block_offset + c + 0, vi,
                          nb[0] & 0xFu, 1.f),
                      vllm::q4::dequant_v_q4_res<true>(
                          v_res_gbase, physical_block_offset + c + 1, vi,
                          (nb[0] >> 4) & 0xFu, 1.f),
                      vllm::q4::dequant_v_q4_res<true>(
                          v_res_gbase, physical_block_offset + c + 2, vi,
                          nb[1] & 0xFu, 1.f),
                      vllm::q4::dequant_v_q4_res<true>(
                          v_res_gbase, physical_block_offset + c + 3, vi,
                          (nb[1] >> 4) & 0xFu, 1.f)};
              } else {
                vllm::bq::bqv_i32x4 qi = {
                    static_cast<int>(nb[0] & 0xFu) - 8,
                    static_cast<int>((nb[0] >> 4) & 0xFu) - 8,
                    static_cast<int>(nb[1] & 0xFu) - 8,
                    static_cast<int>((nb[1] >> 4) & 0xFu) - 8,
                };
                qf = __builtin_convertvector(qi, vllm::bq::bqv_f32x4);
              }
              vllm::bq::bqv_f32x4 sc;
              __builtin_memcpy(&sc, v_scale + q4_v_base + c, 16);
              vllm::bq::pack_f32x4_to_bf16(q4_v_dst + c, qf * sc);
            }
          } else
#endif
          {
#pragma unroll
            for (int e = 0; e < V_VEC_SIZE; ++e) {
              const uint8_t packed = q4_v_nibbles[e >> 1];
              const uint8_t nib =
                  (e & 1) ? static_cast<uint8_t>((packed >> 4) & 0xFu)
                          : static_cast<uint8_t>(packed & 0xFu);
              from_float(q4_v_dst[e],
                         vllm::q4::dequant_v_q4_res<kLmQ4>(
                             v_res_gbase, physical_block_offset + e,
                             row_idx & 31, nib, v_scale[q4_v_base + e]));
            }
          }
        } else {
          using Cache_V_vec = typename vllm::Vec<cache_t, V_VEC_SIZE>::Type;
          Cache_V_vec fp8_v_vec =
              *reinterpret_cast<const Cache_V_vec *>(v_ptr + offset);

          v_vec = vllm::fp8::scaled_convert<V_vec, Cache_V_vec, kv_dt>(
              fp8_v_vec, *v_scale);
        }
        if (block_idx == num_context_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t *v_vec_ptr = reinterpret_cast<scalar_t *>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] =
                token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float *out_smem = reinterpret_cast<float *>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float *dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float *src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  // Q4_0 V incoherence: the cache holds head-dim-rotated V, so accs holds
  // A = P.V'. Un-rotate once per output vector: O = S.H(A). Linear, hence
  // exact across v1/v2 partitions (the reduce kernel combines partials
  // linearly). Gate must match the store side (bf16/256).
  if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kQ4_0 && HEAD_SIZE == 256 &&
                std::is_same<scalar_t, __nv_bfloat16>::value) {
    __shared__ float wht_o[256];
    for (int d = thread_idx; d < 256; d += NUM_THREADS) {
      wht_o[d] = 0.f;
    }
    if (warp_idx == 0) {
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          wht_o[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();
    vllm::wht::wht_inplace(wht_o, 256);
    if (warp_idx == 0) {
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] = wht_o[row_idx] * vllm::wht::wht_sign(kv_head_idx, row_idx);
        }
      }
    }
    __syncthreads();
  }
  if (warp_idx == 0) {
    scalar_t *out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt,
          int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t *__restrict__ out,          // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ q,      // [num_seqs, num_heads, head_size]
    const cache_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                         // head_size/x, block_size, x]
    const cache_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                         // head_size, block_size]
    const int num_kv_heads,              // [num_heads]
    const float scale, const float softcapping,
    const uint32_t
        *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const uint32_t *__restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float *k_scale, const float *v_scale,
    const uint8_t *__restrict__ k_res, const uint8_t *__restrict__ v_res,
    const float *__restrict__ sinks) {
  paged_attention_kernel<scalar_t, cache_t, kv_dt, HEAD_SIZE, BLOCK_SIZE,
                         NUM_THREADS>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, softcapping, block_tables, context_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, k_scale, v_scale, k_res, v_res, sinks);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt,
          int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float *__restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float *__restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    scalar_t *__restrict__ tmp_out, // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    const scalar_t *__restrict__ q, // [num_seqs, num_heads, head_size]
    const cache_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                         // head_size/x, block_size, x]
    const cache_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                         // head_size, block_size]
    const int num_kv_heads,              // [num_heads]
    const float scale, const float softcapping,
    const uint32_t
        *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const uint32_t *__restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float *k_scale, const float *v_scale,
    const uint8_t *__restrict__ k_res, const uint8_t *__restrict__ v_res,
    const float *__restrict__ sinks) {
  paged_attention_kernel<scalar_t, cache_t, kv_dt, HEAD_SIZE, BLOCK_SIZE,
                         NUM_THREADS, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      softcapping, block_tables, context_lens, max_num_blocks_per_seq,
      alibi_slopes, q_stride, kv_block_stride, kv_head_stride, k_scale,
      v_scale, k_res, v_res, sinks);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t *__restrict__ out, // [num_seqs, num_heads, head_size]
    const float
        *__restrict__ exp_sums, // [num_seqs, num_heads, max_num_partitions]
    const float
        *__restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    const scalar_t *__restrict__ tmp_out,      // [num_seqs, num_heads,
                                               // max_num_partitions, head_size]
    const uint32_t *__restrict__ context_lens, // [num_seqs]
    const int max_num_partitions,
    const float *__restrict__ sinks // [num_heads] or nullptr
    ) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const uint32_t context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1 && sinks == nullptr) {
    // No need to reduce. Only copy tmp_out to out.
    // (When sinks are present, we must still rescale the single partition.)
    scalar_t *out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t *tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float *shared_max_logits = reinterpret_cast<float *>(shared_mem);
  const float *max_logits_ptr = max_logits +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Include the sink in the global max before rescaling.
  if (sinks != nullptr) {
    max_logit = fmaxf(max_logit, sinks[head_idx]);
  }

  // Load rescaled exp sums to shared memory.
  float *shared_exp_sums =
      reinterpret_cast<float *>(shared_mem + sizeof(float) * num_partitions);
  const float *exp_sums_ptr = exp_sums +
                              seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);

  // Include the sink in the global exp sum.
  if (sinks != nullptr) {
    global_exp_sum += __expf(sinks[head_idx] - max_logit);
  }

  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t *tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t *out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
             inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

} // namespace vllm

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                   \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                        \
      ((void *)vllm::paged_attention_v1_kernel<T, CACHE_T, KV_DT, HEAD_SIZE,   \
                                               BLOCK_SIZE, NUM_THREADS>),      \
      shared_mem_size);                                                        \
  vllm::paged_attention_v1_kernel<T, CACHE_T, KV_DT, HEAD_SIZE, BLOCK_SIZE,    \
                                  NUM_THREADS>                                 \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          reinterpret_cast<T *>(out), reinterpret_cast<T *>(query),            \
          reinterpret_cast<CACHE_T *>(key_cache),                              \
          reinterpret_cast<CACHE_T *>(value_cache), num_kv_heads, scale,       \
          softcapping, block_tables, context_lens, max_num_blocks_per_seq,     \
          reinterpret_cast<float *>(alibi_slopes), q_stride, kv_block_stride,  \
          kv_head_stride, k_scale, v_scale, k_res, v_res, sinks);

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, vllm::Fp8KVCacheDataType KV_DT,
          int BLOCK_SIZE, int NUM_THREADS = 128>
inline void paged_attention_v1_launcher(
    void *out, void *query, void *key_cache, void *value_cache,
    void *__restrict__ alibi_slopes, int num_kv_heads, float scale,
    float softcapping, uint32_t *block_tables, uint32_t *context_lens,
    int max_context_len,

    int num_seqs, int num_heads, int head_size, int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride, cudaStream_t stream,
    const float *k_scale, const float *v_scale,
    const uint8_t *k_res, const uint8_t *v_res,
    const float *sinks) {

  // int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  // assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional. It may be nullptr.

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len =
      DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  switch (head_size) {
  // NOTE(woosuk): To reduce the compilation time, we only compile for the
  // head sizes that we use in the model. However, we can easily extend this
  // to support any head size which is a multiple of 16.
  case 64:
    LAUNCH_PAGED_ATTENTION_V1(64);
    break;
  case 80:
    LAUNCH_PAGED_ATTENTION_V1(80);
    break;
  case 96:
    LAUNCH_PAGED_ATTENTION_V1(96);
    break;
  case 112:
    LAUNCH_PAGED_ATTENTION_V1(112);
    break;
  case 128:
    LAUNCH_PAGED_ATTENTION_V1(128);
    break;
  case 192:
    LAUNCH_PAGED_ATTENTION_V1(192);
    break;
  case 256:
    LAUNCH_PAGED_ATTENTION_V1(256);
    break;
  case 512:
    LAUNCH_PAGED_ATTENTION_V1(512);
    break;
  default:
    break;
  }
}

#define CALL_V1_LAUNCHER(T, CACHE_T, KV_DT, BLOCK_SIZE)                        \
  paged_attention_v1_launcher<T, CACHE_T, KV_DT, BLOCK_SIZE>(                  \
      out, query, key_cache, value_cache, alibi_slopes, num_kv_heads, scale,   \
      softcapping, block_tables, context_lens, max_context_len, num_seqs,      \
      num_heads, head_size, max_num_blocks_per_seq, q_stride, kv_block_stride, \
      kv_head_stride, stream, k_scale, v_scale, k_res, v_res, sinks);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DT)                         \
  switch (block_size) {                                                        \
  case 8:                                                                      \
    CALL_V1_LAUNCHER(T, CACHE_T, KV_DT, 8);                                    \
    break;                                                                     \
  case 16:                                                                     \
    CALL_V1_LAUNCHER(T, CACHE_T, KV_DT, 16);                                   \
    break;                                                                     \
  case 32:                                                                     \
    CALL_V1_LAUNCHER(T, CACHE_T, KV_DT, 32);                                   \
    break;                                                                     \
  default:                                                                     \
    break;                                                                     \
  }


#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                   \
  vllm::paged_attention_v2_kernel<T, CACHE_T, KV_DT, HEAD_SIZE, BLOCK_SIZE,    \
                                  NUM_THREADS, PARTITION_SIZE>                 \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          exp_sums, max_logits, tmp_out_ptr, reinterpret_cast<T *>(query),     \
          reinterpret_cast<CACHE_T *>(key_cache),                              \
          reinterpret_cast<CACHE_T *>(value_cache), num_kv_heads, scale,       \
          softcapping, block_tables, context_lens, max_num_blocks_per_seq,     \
          reinterpret_cast<float *>(alibi_slopes), q_stride, kv_block_stride,  \
          kv_head_stride, k_scale, v_scale, k_res, v_res, sinks);              \
  vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS,            \
                                         PARTITION_SIZE>                       \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                \
          reinterpret_cast<T *>(out), exp_sums, max_logits, tmp_out_ptr,       \
          context_lens, max_num_partitions, sinks);

template <typename T, typename CACHE_T, vllm::Fp8KVCacheDataType KV_DT,
          int BLOCK_SIZE, int NUM_THREADS = 128, int PARTITION_SIZE = 512>
inline void paged_attention_v2_launcher(
    void *out, float *exp_sums, float *max_logits, void *tmp_out, void *query,
    void *key_cache, void *value_cache, void *alibi_slopes, int num_kv_heads,
    float scale, float softcapping, uint32_t *block_tables,
    uint32_t *context_lens, int max_context_len,

    int num_seqs, int num_heads, int head_size, int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride, cudaStream_t stream,
    const float *k_scale, const float *v_scale,
    const uint8_t *k_res, const uint8_t *v_res,
    const float *sinks
) {
  // int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);

  // NOTE: alibi_slopes is optional. It may be nullptr.

  T *tmp_out_ptr = reinterpret_cast<T *>(tmp_out);

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  // For paged attention v2 kernel.
  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);
  // For paged attention v2 reduce kernel.
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);
  switch (head_size) {
  // NOTE(woosuk): To reduce the compilation time, we only compile for the
  // head sizes that we use in the model. However, we can easily extend this
  // to support any head size which is a multiple of 16.
  case 64:
    LAUNCH_PAGED_ATTENTION_V2(64);
    break;
  case 80:
    LAUNCH_PAGED_ATTENTION_V2(80);
    break;
  case 96:
    LAUNCH_PAGED_ATTENTION_V2(96);
    break;
  case 112:
    LAUNCH_PAGED_ATTENTION_V2(112);
    break;
  case 128:
    LAUNCH_PAGED_ATTENTION_V2(128);
    break;
  case 192:
    LAUNCH_PAGED_ATTENTION_V2(192);
    break;
  case 256:
    LAUNCH_PAGED_ATTENTION_V2(256);
    break;
  case 512:
    LAUNCH_PAGED_ATTENTION_V2(512);
    break;
  default:
    break;
  }
}

#define CALL_V2_LAUNCHER(T, CACHE_T, KV_DT, BLOCK_SIZE)                        \
  paged_attention_v2_launcher<T, CACHE_T, KV_DT, BLOCK_SIZE>(                  \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,       \
      alibi_slopes, num_kv_heads, scale, softcapping, block_tables,            \
      context_lens, max_context_len, num_seqs, num_heads, head_size,           \
      max_num_blocks_per_seq, q_stride, kv_block_stride, kv_head_stride,       \
      stream, k_scale, v_scale, k_res, v_res, sinks);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DT)                         \
  switch (block_size) {                                                        \
  case 8:                                                                      \
    CALL_V2_LAUNCHER(T, CACHE_T, KV_DT, 8);                                    \
    break;                                                                     \
  case 16:                                                                     \
    CALL_V2_LAUNCHER(T, CACHE_T, KV_DT, 16);                                   \
    break;                                                                     \
  case 32:                                                                     \
    CALL_V2_LAUNCHER(T, CACHE_T, KV_DT, 32);                                   \
    break;                                                                     \
  default:                                                                     \
    break;                                                                     \
  }


#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
