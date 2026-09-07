mod cache;
#[cfg(not(feature = "rocm"))]
mod context_attention_mla;
#[cfg(not(feature = "rocm"))]
mod fa3;
#[cfg(not(feature = "rocm"))]
mod flash_attn_sinks;
#[cfg(not(feature = "rocm"))]
mod flashinfer;
mod gather_kv;
#[cfg(not(feature = "rocm"))]
mod mla;
mod paged_attention;
#[cfg(not(feature = "rocm"))]
mod scale_update;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr},
};
use candle_core::{Layout, Result};
#[cfg(not(feature = "rocm"))]
pub use context_attention_mla::context_attention_fwd_mla;
#[cfg(not(feature = "rocm"))]
pub use fa3::{
    fa3_fp8_decode, fa3_prepare_decode_metadata, fa3_prepare_paged_metadata, Fa3DecodeMetadata,
    Fa3DecodeParams, Fa3DecodeSchedule, Fa3PagedMetadataLayout, FA3_DECODE_MAX_QUERY_LEN,
    USE_FA3_FP8_PAGED,
};
#[cfg(not(feature = "rocm"))]
pub use flash_attn_sinks::{flash_attn_sinks, flash_attn_sinks_varlen};
#[cfg(not(feature = "rocm"))]
pub use flashinfer::{
    flashinfer_decode, gather_kv_cache_flashinfer, is_flashinfer_cache,
    reshape_and_cache_flashinfer, FlashInferDecodeScratch,
};
pub use gather_kv::gather_kv_cache;
#[cfg(not(feature = "rocm"))]
pub use mla::{concat_and_cache_mla, flashinfer_mla_decode, gather_mla_cache};
pub use paged_attention::{paged_attention, reshape_and_cache, reshape_and_cache_q8};
#[cfg(not(feature = "rocm"))]
pub use scale_update::kv_scale_update;

fn cache_input_layout(
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

pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    slice_ptr_on_stream(v, lo, v.stream())
}

pub fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_input_layout_flattens_uniform_rank_four_rows() -> Result<()> {
        let prefill = Layout::new((2, 3, 2, 4).into(), vec![48, 16, 4, 1], 7);
        assert_eq!(cache_input_layout(&prefill, "key", "test")?, (6, 2, 4, 16));

        let decode = Layout::new((8, 1, 2, 4).into(), vec![24, 8, 4, 1], 5);
        assert_eq!(cache_input_layout(&decode, "value", "test")?, (8, 2, 4, 24));
        Ok(())
    }

    /// CPU mirror of the Q8_0 block math in `quantization/q8/q8_utils.cuh`
    /// (llama.cpp `block_q8_0`): scale = amax/127, round-half-away, symmetric
    /// clamp. The HIP kernel must bit-match this on exact halves.
    fn quantize_block_q8_0_cpu(vals: &[f32; 32]) -> ([i8; 32], f32) {
        let amax = vals.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = if amax != 0. { amax / 127. } else { 1. };
        let id = if d != 0. { 1. / d } else { 0. };
        let mut out = [0i8; 32];
        for (i, &v) in vals.iter().enumerate() {
            let q = (v * id).clamp(-127., 127.);
            out[i] = (if q >= 0. { q + 0.5 } else { q - 0.5 }) as i8;
        }
        (out, d)
    }

    #[test]
    fn q8_block_math_matches_llama_reference_vectors() {
        // All-ones: amax 1 -> d = 1/127, every qs = 127.
        let (qs, d) = quantize_block_q8_0_cpu(&[1.; 32]);
        assert_eq!(d, 1. / 127.);
        assert!(qs.iter().all(|&q| q == 127));
        // Zero block: scale 1, all-zero output (no div-by-zero).
        let (qs, d) = quantize_block_q8_0_cpu(&[0.; 32]);
        assert_eq!((d, qs), (1., [0; 32]));
        // Ramp: max error from 7-bit steps stays under half a step.
        let mut ramp = [0f32; 32];
        for (i, v) in ramp.iter_mut().enumerate() {
            *v = -3. + 6. * i as f32 / 31.;
        }
        let (qs, d) = quantize_block_q8_0_cpu(&ramp);
        assert_eq!(d, 3. / 127.);
        for (i, &q) in qs.iter().enumerate() {
            assert!((q as f32 * d - ramp[i]).abs() <= d / 2. + 1e-6);
        }
        // Outlier: single large value sets the scale, small values survive.
        let mut outlier = [0.01f32; 32];
        outlier[7] = 100.;
        let (qs, d) = quantize_block_q8_0_cpu(&outlier);
        assert_eq!(d, 100. / 127.);
        assert_eq!(qs[7], 127);
        assert!(qs
            .iter()
            .enumerate()
            .all(|(i, &q)| i == 7 || q == 0 || q == 1));
    }
}
