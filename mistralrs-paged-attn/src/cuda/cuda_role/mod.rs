//! NVIDIA-role paged-attention backend for dual builds, bound to
//! `cuda_backend` types and the nvcc companion. Twins of `backend/`
//! (generated) - same entry shapes, CUDA-role storage and streams.

mod cache;
mod gather_kv;
mod paged_attention;

pub use cache::{copy_blocks, swap_blocks};
pub use gather_kv::gather_kv_cache;
pub use paged_attention::{
    paged_attention, reshape_and_cache, reshape_and_cache_q4, reshape_and_cache_q8,
    PAGED_ATTENTION_V2_PARTITION_SIZE,
};

use candle_core::cuda_backend::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr},
};
use candle_core::cuda_backend::CudaStorage;
use candle_core::{Layout, Result, Storage};

pub(crate) fn as_cuda_storage(storage: &Storage) -> Result<&CudaStorage> {
    match storage {
        Storage::Cuda(s) => Ok(s),
        _ => candle_core::bail!("expected the NVIDIA role's storage"),
    }
}

pub(crate) fn cache_input_layout(
    layout: &Layout,
    name: &str,
    op: &str,
) -> Result<(usize, usize, usize, usize)> {
    let (num_tokens, num_heads, head_size, row_stride) = match *layout.dims() {
        [num_tokens, num_heads, head_size] => {
            (num_tokens, num_heads, head_size, layout.stride()[0])
        }
        [batch, seq_len, num_heads, head_size] => {
            let num_tokens = batch
                .checked_mul(seq_len)
                .ok_or_else(|| candle_core::Error::msg("cache input token count overflow"))?;
            let row_stride = if seq_len == 1 {
                layout.stride()[0]
            } else {
                layout.stride()[1]
            };
            if batch > 1 && seq_len > 1 && layout.stride()[0] != seq_len.saturating_mul(row_stride)
            {
                candle_core::bail!("{op} cannot flatten {name} batch/sequence strides: {layout:?}");
            }
            (num_tokens, num_heads, head_size, row_stride)
        }
        _ => candle_core::bail!("{op} expects rank-3 or rank-4 {name} input, got {layout:?}"),
    };
    if layout.stride()[layout.stride().len() - 1] != 1
        || layout.stride()[layout.stride().len() - 2] != head_size
        || row_stride < num_heads.saturating_mul(head_size)
    {
        candle_core::bail!("{op} expects dense {name} heads, got {layout:?}");
    }
    Ok((num_tokens, num_heads, head_size, row_stride))
}

pub(crate) fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    slice_ptr_on_stream(v, lo, v.stream())
}

pub(crate) fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}
