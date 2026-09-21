#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename T>
__global__ void indexed_row_copy_kernel(const T *src, T *dst,
                                        const uint32_t *dst_rows, int rows,
                                        int64_t row_elements,
                                        int64_t dst_row_capacity) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * row_elements;
  if (idx >= total) {
    return;
  }
  int row = static_cast<int>(idx / row_elements);
  int64_t column = idx - static_cast<int64_t>(row) * row_elements;
  const uint32_t dst_row = dst_rows[row];
  if ((int64_t)dst_row >= dst_row_capacity) {
    return;
  }
  dst[(int64_t)dst_row * row_elements + column] = src[idx];
}

template <typename T>
int launch_indexed_row_copy(const void *src, void *dst,
                            const uint32_t *dst_rows, int rows,
                            int64_t row_elements, int64_t dst_row_capacity,
                            int64_t stream) {
  constexpr int threads = 256;
  int64_t total = static_cast<int64_t>(rows) * row_elements;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  indexed_row_copy_kernel<<<blocks, threads, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const T *>(src), static_cast<T *>(dst), dst_rows, rows,
      row_elements, dst_row_capacity);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int indexed_row_copy_bf16(const void *src, void *dst,
                                      const uint32_t *dst_rows, int rows,
                                      int64_t row_elements,
                                      int64_t dst_row_capacity,
                                      int64_t stream) {
  return launch_indexed_row_copy<__nv_bfloat16>(src, dst, dst_rows, rows,
                                                row_elements, dst_row_capacity,
                                                stream);
}

extern "C" int indexed_row_copy_f16(const void *src, void *dst,
                                     const uint32_t *dst_rows, int rows,
                                     int64_t row_elements,
                                     int64_t dst_row_capacity,
                                     int64_t stream) {
  return launch_indexed_row_copy<__half>(src, dst, dst_rows, rows,
                                         row_elements, dst_row_capacity,
                                         stream);
}

extern "C" int indexed_row_copy_f32(const void *src, void *dst,
                                     const uint32_t *dst_rows, int rows,
                                     int64_t row_elements,
                                     int64_t dst_row_capacity,
                                     int64_t stream) {
  return launch_indexed_row_copy<float>(src, dst, dst_rows, rows,
                                        row_elements, dst_row_capacity,
                                        stream);
}

// Clamp every entry of a u32 index table to [0, capacity). Out-of-range
// entries become `replacement`, except the 0xffffffff pad sentinel which is
// preserved (kernels treat it as a skip row). Bounds the damage a
// stale/garbage slot table can do to downstream indexed writes.
__global__ void clamp_u32_index_table_kernel(uint32_t *table, int len,
                                             uint32_t capacity,
                                             uint32_t replacement) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  const uint32_t v = table[idx];
  if (v != 0xffffffffu && v >= capacity) {
    table[idx] = replacement;
  }
}

extern "C" int clamp_u32_index_table(uint32_t *table, int len,
                                     uint32_t capacity, uint32_t replacement,
                                     int64_t stream) {
  constexpr int threads = 256;
  int blocks = (len + threads - 1) / threads;
  clamp_u32_index_table_kernel<<<blocks, threads, 0,
                                 reinterpret_cast<cudaStream_t>(stream)>>>(
      table, len, capacity, replacement);
  return static_cast<int>(cudaGetLastError());
}
