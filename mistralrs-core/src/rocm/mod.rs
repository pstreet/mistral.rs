use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
use candle_core::{DType, Result, Shape, Storage, Tensor};

use crate::cuda::ffi::ck_flash_attn_fwd;

/// Try CK flash attention for ROCm prefill.
///
/// Q, K, V must be BSHD (batch, heads, seq_len, head_dim).
/// `mask_type`: 0 = none, 1 = causal top-left, 2 = causal bottom-right
/// (for gathered prefixes where kv_len > q_len).
/// Returns `Ok(Some(output))` on success, `Ok(None)` if unsupported.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn ck_flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    mask_type: i32,
) -> Result<Option<Tensor>> {
    let (b_sz, n_attn_heads, seq_len, head_dim) = q.dims4()?;
    let (_, n_kv_heads, kv_len, k_head_dim) = k.dims4()?;
    let (_, _, _, v_head_dim) = v.dims4()?;

    if head_dim != k_head_dim || head_dim != v_head_dim {
        return Ok(None);
    }
    if !matches!(head_dim, 128 | 256) {
        return Ok(None);
    }
    if seq_len <= 1 {
        return Ok(None);
    }
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Ok(None);
    }

    let dev = q.device().as_cuda_device()?;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();

    let Storage::Cuda(q_cuda) = &*q_storage else {
        candle_core::bail!("q must be cuda storage");
    };
    let Storage::Cuda(k_cuda) = &*k_storage else {
        candle_core::bail!("k must be cuda storage");
    };
    let Storage::Cuda(v_cuda) = &*v_storage else {
        candle_core::bail!("v must be cuda storage");
    };

    let q_slice = q_cuda.as_cuda_slice::<half::bf16>()?;
    let k_slice = k_cuda.as_cuda_slice::<half::bf16>()?;
    let v_slice = v_cuda.as_cuda_slice::<half::bf16>()?;

    let stream = dev.cuda_stream();
    let stream_i64 = stream.cu_stream() as i64;

    let (q_ptr, _q_guard) = q_slice.device_ptr(&stream);
    let (k_ptr, _k_guard) = k_slice.device_ptr(&stream);
    let (v_ptr, _v_guard) = v_slice.device_ptr(&stream);

    let q_ptr = unsafe { (q_ptr as *const half::bf16).add(q_layout.start_offset()) };
    let k_ptr = unsafe { (k_ptr as *const half::bf16).add(k_layout.start_offset()) };
    let v_ptr = unsafe { (v_ptr as *const half::bf16).add(v_layout.start_offset()) };

    let q_stride = q_layout.stride();
    let k_stride = k_layout.stride();
    let v_stride = v_layout.stride();

    let out_elems = b_sz * n_attn_heads * seq_len * head_dim;
    let mut output = unsafe { dev.alloc::<half::bf16>(out_elems) }?;
    let out_ptr = {
        let (p, _g) = output.device_ptr_mut(&stream);
        p
    };

    let lse_elems = b_sz * n_attn_heads * seq_len;
    let mut lse = unsafe { dev.alloc::<f32>(lse_elems) }?;
    let lse_ptr = {
        let (p, _g) = lse.device_ptr_mut(&stream);
        p
    };

    if std::env::var("MRS_DEBUG_CK").is_ok() {
        eprintln!(
            "[CK FA] b={} q_heads={} kv_heads={} seq_q={} seq_k={} hdim={} mask_type={} scale={}",
            b_sz, n_attn_heads, n_kv_heads, seq_len, kv_len, head_dim, mask_type, softmax_scale
        );
    }

    unsafe {
        ck_flash_attn_fwd(
            q_ptr as *const std::ffi::c_void,
            k_ptr as *const std::ffi::c_void,
            v_ptr as *const std::ffi::c_void,
            out_ptr as *mut std::ffi::c_void,
            lse_ptr as *mut std::ffi::c_void,
            b_sz as i32,
            n_attn_heads as i32,
            n_kv_heads as i32,
            seq_len as i32,
            kv_len as i32,
            head_dim as i32,
            head_dim as i32,
            q_stride[2] as i64,
            k_stride[2] as i64,
            v_stride[2] as i64,
            q_stride[2] as i64,
            q_stride[1] as i64,
            k_stride[1] as i64,
            v_stride[1] as i64,
            q_stride[1] as i64,
            q_stride[0] as i64,
            k_stride[0] as i64,
            v_stride[0] as i64,
            q_stride[0] as i64,
            softmax_scale,
            mask_type,
            stream_i64,
        );
    }

    let out_dims = [b_sz, n_attn_heads, seq_len, head_dim];
    let output_tensor = Tensor::from((
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        }),
        Shape::from_dims(&out_dims),
    ));
    Ok(Some(output_tensor))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    // reference attention: mode 0 = full, 1 = causal top-left, 2 = causal bottom-right
    fn ref_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        mode: u8,
    ) -> candle_core::Result<Tensor> {
        let (_, _, sq, _) = q.dims4()?;
        let (_, _, sk, _) = k.dims4()?;
        let kt = k.transpose(2, 3)?;
        let scores = (scale * &q.matmul(&kt)?)?;
        let mut mask = Vec::with_capacity(sq * sk);
        for i in 0..sq {
            for j in 0..sk {
                let valid = match mode {
                    0 => true,
                    1 => j <= i,
                    2 => j <= i + (sk - sq),
                    _ => unreachable!(),
                };
                mask.push(if valid { 0f32 } else { f32::NEG_INFINITY });
            }
        }
        let mask = Tensor::from_vec(mask, (sq, sk), q.device())?;
        let att = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        att.matmul(v)
    }

    fn check_case(
        dev: &Device,
        sq: usize,
        sk: usize,
        heads: usize,
        dim: usize,
        scale: f64,
        mode: u8,
    ) -> candle_core::Result<()> {
        let q = Tensor::rand(-1.0, 1.0, (1, heads, sq, dim), dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::rand(-1.0, 1.0, (1, heads, sk, dim), dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::rand(-1.0, 1.0, (1, heads, sk, dim), dev)?.to_dtype(DType::BF16)?;
        let out = super::ck_flash_attn(&q, &k, &v, scale as f32, mode as i32)?
            .ok_or_else(|| candle_core::Error::msg("ck_flash_attn returned none"))?;
        let qf = q.to_dtype(DType::F32)?;
        let kf = k.to_dtype(DType::F32)?;
        let vf = v.to_dtype(DType::F32)?;
        let ref_out = ref_attention(&qf, &kf, &vf, scale, mode)?;
        let max_abs = (out.to_dtype(DType::F32)? - &ref_out)?.abs()?.max_all()?;
        let ref_max = ref_out.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
        let rel = max_abs.to_scalar::<f32>()? / ref_max;
        assert!(
            rel < 2e-2,
            "mode={mode} sq={sq} sk={sk}: rel diff {rel} too large"
        );
        Ok(())
    }

    #[test]
    fn ck_flash_attn_matches_reference() -> candle_core::Result<()> {
        let dev = Device::new_cuda(0)?;
        let scale = 1.0f64 / (128f64.sqrt());
        // fresh prompt: kv_len == q_len, causal top-left
        check_case(&dev, 4, 4, 2, 128, scale, 1)?;
        // full (non-causal) attention
        check_case(&dev, 4, 4, 2, 128, scale, 0)?;
        // gathered prefix: kv_len > q_len, causal bottom-right
        check_case(&dev, 4, 12, 2, 128, scale, 2)?;
        check_case(&dev, 7, 19, 2, 128, scale, 2)?;
        Ok(())
    }
}
