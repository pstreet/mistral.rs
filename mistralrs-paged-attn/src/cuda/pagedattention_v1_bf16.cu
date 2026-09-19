#include "pagedattention.cuh"
using namespace vllm;
extern "C" void paged_attention_v1_bf16(
    void *out,          // [num_seqs, num_heads, head_size]
    void *query,        // [num_seqs, num_heads, head_size]
    void *key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    void *value_cache,  // [num_blocks, num_heads, head_size, block_size]
    void *alibi_slopes, // [num_heads]
    int32_t num_kv_heads, float scale, float softcapping,
    uint32_t *block_tables, // [num_seqs, max_num_blocks_per_seq]
    uint32_t *context_lens, // [num_seqs]
    int32_t block_size, int32_t max_context_len,

    int32_t num_seqs, int32_t num_heads, int32_t head_size,
    int32_t max_num_blocks_per_seq, int32_t q_stride, int32_t kv_block_stride,
    int32_t kv_head_stride, int32_t v_block_stride, int32_t v_head_stride,
    cudaStream_t stream,

    // Per-side codes, each 0/1/2 native or 3 fp8_e4m3, 4 q8_0, 5 q4_0.
    // One instantiation serves every (k, v) pair; the kernel branches
    // per side at launch-uniform runtime codes.
    uint32_t k_cache_dtype, uint32_t v_cache_dtype,
    float *k_scale, float *v_scale, const uint8_t *k_res, const uint8_t *v_res,
    const float *sinks) {

  CALL_V1_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  CUDA_CHECK(cudaGetLastError());
}
