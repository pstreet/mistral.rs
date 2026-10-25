use candle_core::hip_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::hip_backend::{CudaStorage, CudaStorageSlice};
use candle_core::{DType, Result, Shape, Storage, Tensor};

use crate::cuda::ffi::{ck_flash_attn_fwd, ck_flash_attn_splitkv_fwd};

/// Try CK flash attention for ROCm prefill.
///
/// Q, K, V must be BSHD (batch, heads, seq_len, head_dim).
/// `mask_type`: 0 = none, 1 = causal top-left, 2 = causal bottom-right
/// (for gathered prefixes where kv_len > q_len).
/// `sliding_window`: None = full attention, Some(w) = causal lookback of w.
/// Returns `Ok(Some(output))` on success, `Ok(None)` if unsupported.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn ck_flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    mask_type: i32,
    sliding_window: Option<usize>,
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
    // CK takes the lookback (W-1); a zero window is meaningless.
    let window_size_left = match sliding_window {
        None => -1,
        Some(0) => return Ok(None),
        Some(w) => i32::try_from(w - 1)
            .map_err(|_| candle_core::Error::msg("sliding window does not fit i32"))?,
    };

    let dev = q.device().as_hip_device()?;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();

    let Storage::Hip(q_cuda) = &*q_storage else {
        candle_core::bail!("q must be hip storage");
    };
    let Storage::Hip(k_cuda) = &*k_storage else {
        candle_core::bail!("k must be hip storage");
    };
    let Storage::Hip(v_cuda) = &*v_storage else {
        candle_core::bail!("v must be hip storage");
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

    if crate::attention::debug_ck() {
        eprintln!(
            "[CK FA] b={} q_heads={} kv_heads={} seq_q={} seq_k={} hdim={} mask_type={} scale={} window={:?}",
            b_sz, n_attn_heads, n_kv_heads, seq_len, kv_len, head_dim, mask_type, softmax_scale, sliding_window
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
            window_size_left,
            stream_i64,
        );
    }

    let out_dims = [b_sz, n_attn_heads, seq_len, head_dim];
    let output_tensor = Tensor::from((
        Storage::Hip(CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        }),
        Shape::from_dims(&out_dims),
    ));
    Ok(Some(output_tensor))
}

#[cfg(feature = "rocm")]
fn hip_sm_count() -> usize {
    use candle_core::hip_backend::cudarc::driver::{result, sys};
    static SM_COUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SM_COUNT.get_or_init(|| {
        (|| {
            let dev = result::device::get(0).ok()?;
            let count = unsafe {
                result::device::get_attribute(
                    dev,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                )
            }
            .ok()?;
            usize::try_from(count).ok().filter(|count| *count > 0)
        })()
        .unwrap_or(40)
    })
}

/// Try CK split-KV paged flash attention (varlen group mode) on the hip role,
/// for draft-step attention over a paged KV cache with FA-style windows.
///
/// `q`: [total_q, nhead_q, hdim] BF16, packed varlen rows. `key_paged`/
/// `value_paged`: [num_pages, page_size, nhead_kv, hdim] BF16. `cu_seqlens_*`:
/// i32 [batch+1] cumulative. `block_tables`: i32 [batch, max_pages_per_seq].
/// `mask_type`: 0 = no mask, 1 = causal top-left, 2 = causal bottom-right
/// (the draft geometry: queries are the trailing block of the kv context).
/// Returns `Ok(None)` on unsupported shapes.
///
/// NOTE: the underlying gfx1151 splitkv instantiation memory-faults on hip
/// (see the TODO dflash entry); nothing calls this from serving paths yet.
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
pub(crate) fn ck_flash_attn_splitkv(
    q: &Tensor,
    key_paged: &Tensor,
    value_paged: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    seqlen_k: Option<&Tensor>,
    block_tables: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>,
    window_right: Option<usize>,
    mask_type: i32,
) -> Result<Option<Tensor>> {
    let (total_q, nhead_q, head_dim) = q.dims3()?;
    let (num_pages, page_size, nhead_kv, k_head_dim) = key_paged.dims4()?;
    let (_, _, _, v_head_dim) = value_paged.dims4()?;
    let (table_batch, max_blocks_per_seq) = block_tables.dims2()?;
    let batch = cu_seqlens_q.dims1()?.saturating_sub(1);

    if head_dim != 128 || k_head_dim != 128 || v_head_dim != 128 {
        return Ok(None);
    }
    // The instantiated kernel tiles KV at kK1 = 32; one tile per page keeps the
    // page navigator on exact boundaries.
    if page_size != 32 || nhead_q % nhead_kv != 0 {
        return Ok(None);
    }
    if batch == 0 || total_q == 0 || table_batch != batch || max_blocks_per_seq == 0 {
        return Ok(None);
    }
    if q.dtype() != DType::BF16
        || key_paged.dtype() != DType::BF16
        || value_paged.dtype() != DType::BF16
    {
        return Ok(None);
    }
    if cu_seqlens_q.dtype() != DType::I32
        || cu_seqlens_k.dtype() != DType::I32
        || block_tables.dtype() != DType::I32
    {
        return Ok(None);
    }
    // FA-style windows: None = unbounded (-1); Some(w) = w - 1 (lookback/side).
    let window_left = match window_left {
        None => -1,
        Some(w) => i32::try_from(w.saturating_sub(1))
            .map_err(|_| candle_core::Error::msg("window does not fit i32"))?,
    };
    let window_right = match window_right {
        None => -1,
        Some(w) => i32::try_from(w.saturating_sub(1))
            .map_err(|_| candle_core::Error::msg("window does not fit i32"))?,
    };

    let dev = q.device().as_hip_device()?;

    let q = q.contiguous()?;
    let key_paged = key_paged.contiguous()?;
    let value_paged = value_paged.contiguous()?;
    let cu_seqlens_q = cu_seqlens_q.contiguous()?;
    let cu_seqlens_k = cu_seqlens_k.contiguous()?;
    let block_tables = block_tables.contiguous()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, _) = key_paged.storage_and_layout();
    let (v_storage, _) = value_paged.storage_and_layout();
    let (cu_q_storage, cu_q_layout) = cu_seqlens_q.storage_and_layout();
    let (cu_k_storage, cu_k_layout) = cu_seqlens_k.storage_and_layout();
    let (bt_storage, bt_layout) = block_tables.storage_and_layout();
    let Storage::Hip(q_cuda) = &*q_storage else {
        candle_core::bail!("q must be hip storage");
    };
    let Storage::Hip(k_cuda) = &*k_storage else {
        candle_core::bail!("k must be hip storage");
    };
    let Storage::Hip(v_cuda) = &*v_storage else {
        candle_core::bail!("v must be hip storage");
    };
    let Storage::Hip(cu_q_cuda) = &*cu_q_storage else {
        candle_core::bail!("cu_seqlens_q must be hip storage");
    };
    let Storage::Hip(cu_k_cuda) = &*cu_k_storage else {
        candle_core::bail!("cu_seqlens_k must be hip storage");
    };
    let Storage::Hip(bt_cuda) = &*bt_storage else {
        candle_core::bail!("block_tables must be hip storage");
    };

    let stream = dev.cuda_stream();
    let stream_i64 = stream.cu_stream() as i64;

    let q_slice = q_cuda.as_cuda_slice::<half::bf16>()?;
    let k_slice = k_cuda.as_cuda_slice::<half::bf16>()?;
    let v_slice = v_cuda.as_cuda_slice::<half::bf16>()?;
    let cu_q_slice = cu_q_cuda.as_cuda_slice::<i32>()?;
    let cu_k_slice = cu_k_cuda.as_cuda_slice::<i32>()?;
    let bt_slice = bt_cuda.as_cuda_slice::<i32>()?;

    let (q_ptr, _q_guard) = q_slice.device_ptr(&stream);
    let (k_ptr, _k_guard) = k_slice.device_ptr(&stream);
    let (v_ptr, _v_guard) = v_slice.device_ptr(&stream);
    let (cu_q_ptr, _cu_q_guard) = cu_q_slice.device_ptr(&stream);
    let (cu_k_ptr, _cu_k_guard) = cu_k_slice.device_ptr(&stream);
    let (bt_ptr, _bt_guard) = bt_slice.device_ptr(&stream);

    let q_ptr = unsafe { (q_ptr as *const half::bf16).add(q_layout.start_offset()) };
    let cu_q_ptr = unsafe { (cu_q_ptr as *const i32).add(cu_q_layout.start_offset()) };
    let cu_k_ptr = unsafe { (cu_k_ptr as *const i32).add(cu_k_layout.start_offset()) };
    let bt_ptr = unsafe { (bt_ptr as *const i32).add(bt_layout.start_offset()) };

    let seqlen_k_ptr = match seqlen_k {
        Some(lens) => {
            let lens = lens.contiguous()?;
            if lens.dtype() != DType::I32 || lens.dims1()? != batch {
                return Ok(None);
            }
            let (lens_storage, lens_layout) = lens.storage_and_layout();
            let Storage::Hip(lens_cuda) = &*lens_storage else {
                candle_core::bail!("seqlen_k must be hip storage");
            };
            let lens_slice = lens_cuda.as_cuda_slice::<i32>()?;
            let (p, _g) = lens_slice.device_ptr(&stream);
            unsafe { (p as *const i32).add(lens_layout.start_offset()) }
        }
        None => std::ptr::null(),
    };

    // Fill the grid when the varlen batch is small: one m-block per draft step.
    // Splits beyond the KV chunk count would hand CTAs empty ranges.
    let mblocks = max_seqlen_q.div_ceil(64).max(1);
    let ctas = batch.saturating_mul(nhead_q).saturating_mul(mblocks).max(1);
    let kv_chunks = max_seqlen_k.div_ceil(page_size).max(1);
    let num_splits = (hip_sm_count().saturating_mul(2) / ctas)
        .clamp(1, 16)
        .min(kv_chunks);

    let out_elems = total_q * nhead_q * head_dim;
    let mut output = unsafe { dev.alloc::<half::bf16>(out_elems) }?;
    let out_ptr = {
        let (p, _g) = output.device_ptr_mut(&stream);
        p
    };
    let lse_elems = nhead_q * total_q;
    let mut lse = unsafe { dev.alloc::<f32>(lse_elems) }?;
    let lse_ptr = {
        let (p, _g) = lse.device_ptr_mut(&stream);
        p
    };
    let o_acc_elems = nhead_q * num_splits * total_q * head_dim;
    let mut o_acc = unsafe { dev.alloc::<f32>(o_acc_elems) }?;
    let o_acc_ptr = {
        let (p, _g) = o_acc.device_ptr_mut(&stream);
        p
    };
    let lse_acc_elems = nhead_q * num_splits * total_q;
    let mut lse_acc = unsafe { dev.alloc::<f32>(lse_acc_elems) }?;
    let lse_acc_ptr = {
        let (p, _g) = lse_acc.device_ptr_mut(&stream);
        p
    };

    if crate::attention::debug_ck() {
        eprintln!(
            "[CK splitkv] batch={batch} total_q={total_q} heads={nhead_q}/{nhead_kv} hdim={head_dim} \
             pages={num_pages} page={page_size} splits={num_splits} mask={mask_type} \
             window=({window_left},{window_right}) max_q={max_seqlen_q} max_k={max_seqlen_k}"
        );
    }

    let status = unsafe {
        ck_flash_attn_splitkv_fwd(
            q_ptr as *const std::ffi::c_void,
            k_ptr as *const std::ffi::c_void,
            v_ptr as *const std::ffi::c_void,
            out_ptr as *mut std::ffi::c_void,
            lse_ptr as *mut std::ffi::c_void,
            o_acc_ptr as *mut std::ffi::c_void,
            lse_acc_ptr as *mut std::ffi::c_void,
            cu_q_ptr,
            cu_k_ptr,
            seqlen_k_ptr,
            bt_ptr,
            max_blocks_per_seq as i64,
            num_pages as i32,
            page_size as i32,
            num_splits as i32,
            batch as i32,
            total_q as i32,
            max_seqlen_q as i32,
            max_seqlen_k as i32,
            nhead_q as i32,
            nhead_kv as i32,
            head_dim as i32,
            softmax_scale,
            window_left,
            window_right,
            mask_type,
            stream_i64,
        )
    };
    if status < 0.0 {
        candle_core::bail!("ck_flash_attn_splitkv_fwd rejected the shape");
    }

    let out_dims = [total_q, nhead_q, head_dim];
    let output_tensor = Tensor::from((
        Storage::Hip(CudaStorage {
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

    // reference attention: mode 0 = full, 1 = causal top-left, 2 = causal bottom-right.
    // `window` bounds the causal lookback, mirroring CK's left-window.
    fn ref_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        mode: u8,
        window: Option<usize>,
    ) -> candle_core::Result<Tensor> {
        let (_, _, sq, _) = q.dims4()?;
        let (_, _, sk, _) = k.dims4()?;
        let kt = k.transpose(2, 3)?;
        let scores = (scale * &q.matmul(&kt)?)?;
        let mut mask = Vec::with_capacity(sq * sk);
        let offset = if mode == 2 { sk.saturating_sub(sq) } else { 0 };
        for i in 0..sq {
            for j in 0..sk {
                let causal_ok = match mode {
                    0 => true,
                    1 => j <= i,
                    2 => j <= i + offset,
                    _ => unreachable!(),
                };
                let window_ok = match window {
                    None => true,
                    // `causal_ok` is checked first, so the subtraction cannot underflow.
                    Some(w) => !causal_ok || i + offset - j < w,
                };
                mask.push(if causal_ok && window_ok {
                    0f32
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        let mask = Tensor::from_vec(mask, (sq, sk), q.device())?;
        let att = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        att.matmul(v)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn check_case(
        dev: &Device,
        sq: usize,
        sk: usize,
        heads: usize,
        dim: usize,
        scale: f64,
        mode: u8,
        window: Option<usize>,
    ) -> candle_core::Result<()> {
        let q = Tensor::rand(-1.0, 1.0, (1, heads, sq, dim), dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::rand(-1.0, 1.0, (1, heads, sk, dim), dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::rand(-1.0, 1.0, (1, heads, sk, dim), dev)?.to_dtype(DType::BF16)?;
        let out = super::ck_flash_attn(&q, &k, &v, scale as f32, mode as i32, window)?
            .ok_or_else(|| candle_core::Error::msg("ck_flash_attn returned none"))?;
        let qf = q.to_dtype(DType::F32)?;
        let kf = k.to_dtype(DType::F32)?;
        let vf = v.to_dtype(DType::F32)?;
        let ref_out = ref_attention(&qf, &kf, &vf, scale, mode, window)?;
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
        let dev = Device::new_hip(0)?;
        let scale = 1.0f64 / (128f64.sqrt());
        // fresh prompt: kv_len == q_len, causal top-left
        check_case(&dev, 4, 4, 2, 128, scale, 1, None)?;
        // full (non-causal) attention
        check_case(&dev, 4, 4, 2, 128, scale, 0, None)?;
        // gathered prefix: kv_len > q_len, causal bottom-right
        check_case(&dev, 4, 12, 2, 128, scale, 2, None)?;
        check_case(&dev, 7, 19, 2, 128, scale, 2, None)?;
        // causal sliding window: the mask carries causality, the window bounds it
        check_case(&dev, 16, 16, 2, 128, scale, 1, Some(4))?;
        check_case(&dev, 16, 16, 2, 128, scale, 1, Some(8))?;
        check_case(&dev, 12, 20, 2, 128, scale, 2, Some(6))?;
        Ok(())
    }

    // Reference attention over a paged kv cache: gathers each sequence's pages,
    // then applies the FA-style window in the bottom-right frame (queries are
    // the trailing q_len rows of the kv context).
    fn paged_ref_case(
        pages_k_f32: &Tensor,
        pages_v_f32: &Tensor,
        page_ids: &[usize],
        kv_len: usize,
        q_f32: &Tensor,
        scale: f64,
        window_left: Option<usize>,
        window_right: Option<usize>,
        mask_type: i32,
    ) -> candle_core::Result<Tensor> {
        let page = pages_k_f32.dim(1)?;
        let nhead_kv = pages_k_f32.dim(2)?;
        let hdim = pages_k_f32.dim(3)?;
        let (q_len, nhead_q, _) = q_f32.dims3()?;
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for pid in page_ids {
            ks.push(pages_k_f32.narrow(0, *pid, 1)?);
            vs.push(pages_v_f32.narrow(0, *pid, 1)?);
        }
        let k_seq = Tensor::cat(&ks, 0)?
            .reshape((page_ids.len() * page, nhead_kv, hdim))?
            .narrow(0, 0, kv_len)?;
        let v_seq = Tensor::cat(&vs, 0)?
            .reshape((page_ids.len() * page, nhead_kv, hdim))?
            .narrow(0, 0, kv_len)?;
        let group = nhead_q / nhead_kv;
        let offset = kv_len.saturating_sub(q_len);
        let mut outs = Vec::with_capacity(nhead_q);
        for h in 0..nhead_q {
            let h_kv = h / group;
            let q_h = q_f32.narrow(1, h, 1)?.squeeze(1)?;
            let k_h = k_seq.narrow(1, h_kv, 1)?.squeeze(1)?;
            let v_h = v_seq.narrow(1, h_kv, 1)?.squeeze(1)?;
            let scores = (scale * &q_h.matmul(&k_h.transpose(0, 1)?)?)?;
            let mut mask = Vec::with_capacity(q_len * kv_len);
            for i in 0..q_len {
                for j in 0..kv_len {
                    let causal_ok = match mask_type {
                        1 => j <= i,
                        2 => j <= i + offset,
                        _ => true,
                    };
                    let window_ok = window_left
                        .map_or(true, |w| i + offset >= j || i + offset - j <= w)
                        && window_right.map_or(true, |w| j <= i + offset || j - (i + offset) <= w);
                    mask.push(if causal_ok && window_ok {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            }
            let mask = Tensor::from_vec(mask, (q_len, kv_len), q_f32.device())?;
            let att = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
            outs.push(att.matmul(&v_h)?);
        }
        let out = Tensor::cat(&outs, 1)?;
        Ok(out)
    }

    #[allow(clippy::cast_precision_loss)]
    fn check_paged_case(
        dev: &Device,
        nhead_q: usize,
        nhead_kv: usize,
        seqs: &[(usize, usize)], // (q_len, kv_len)
        window: Option<usize>,
        window_right: Option<usize>,
        mask_type: i32,
    ) -> candle_core::Result<()> {
        let page = 32usize;
        let hdim = 128usize;
        let scale = 1.0f64 / (hdim as f64).sqrt();
        let mut page_ids = Vec::new();
        for (_, kv_len) in seqs {
            for _ in 0..kv_len.div_ceil(page) {
                page_ids.push(page_ids.len());
            }
        }
        let num_pages = page_ids.len();
        let max_blocks = seqs.iter().map(|(_, kv)| kv.div_ceil(page)).max().unwrap();
        let total_q: usize = seqs.iter().map(|(q, _)| *q).sum();
        let max_kv = seqs.iter().map(|(_, kv)| *kv).max().unwrap();

        let pages_k_f32 = Tensor::rand(-1.0, 1.0, (num_pages, page, nhead_kv, hdim), &Device::Cpu)?;
        let pages_v_f32 = Tensor::rand(-1.0, 1.0, (num_pages, page, nhead_kv, hdim), &Device::Cpu)?;
        let q_f32 = Tensor::rand(-1.0, 1.0, (total_q, nhead_q, hdim), &Device::Cpu)?;

        let k_paged = pages_k_f32.to_device(dev)?.to_dtype(DType::BF16)?;
        let v_paged = pages_v_f32.to_device(dev)?.to_dtype(DType::BF16)?;
        let q = q_f32.to_device(dev)?.to_dtype(DType::BF16)?;

        let mut cu_q = vec![0i32];
        let mut cu_k = vec![0i32];
        for &(q_len, kv_len) in seqs {
            cu_q.push(cu_q.last().unwrap() + q_len as i32);
            cu_k.push(cu_k.last().unwrap() + kv_len as i32);
        }
        let cu_seqlens_q = Tensor::from_vec(cu_q.clone(), (seqs.len() + 1,), dev)?;
        let cu_seqlens_k = Tensor::from_vec(cu_k.clone(), (seqs.len() + 1,), dev)?;
        let mut table = Vec::with_capacity(seqs.len() * max_blocks);
        let mut ids = page_ids.iter();
        for (_, kv_len) in seqs {
            let mut blocks = Vec::new();
            for _ in 0..kv_len.div_ceil(page) {
                blocks.push(*ids.next().unwrap() as i32);
            }
            blocks.resize(max_blocks, 0);
            table.extend(blocks);
        }
        let block_tables = Tensor::from_vec(table, (seqs.len(), max_blocks), dev)?;
        let lens: Vec<i32> = seqs.iter().map(|&(_, kv)| kv as i32).collect();
        let seqlen_k = Tensor::from_vec(lens, (seqs.len(),), dev)?;

        let out = super::ck_flash_attn_splitkv(
            &q,
            &k_paged,
            &v_paged,
            &cu_seqlens_q,
            &cu_seqlens_k,
            Some(&seqlen_k),
            &block_tables,
            seqs.iter().map(|s| s.0).max().unwrap(),
            max_kv,
            scale as f32,
            window,
            window_right,
            mask_type,
        )?
        .ok_or_else(|| candle_core::Error::msg("ck_flash_attn_splitkv returned none"))?;

        let mut refs = Vec::new();
        let mut q_rows = 0usize;
        let mut ids = page_ids.iter();
        for &(q_len, kv_len) in seqs {
            let mut seq_ids = Vec::new();
            for _ in 0..kv_len.div_ceil(page) {
                seq_ids.push(*ids.next().unwrap());
            }
            refs.push(paged_ref_case(
                &pages_k_f32,
                &pages_v_f32,
                &seq_ids,
                kv_len,
                &q_f32.narrow(0, q_rows, q_len)?,
                scale,
                window.map(|w| w - 1),
                window_right.map(|w| w - 1),
                mask_type,
            )?);
            q_rows += q_len;
        }
        let ref_out = Tensor::cat(&refs, 0)?;
        let max_abs = (out.to_dtype(DType::F32)? - &ref_out)?.abs()?.max_all()?;
        let ref_max = ref_out.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
        let rel = max_abs.to_scalar::<f32>()? / ref_max;
        assert!(
            rel < 2e-2,
            "mask={mask_type} seqs={seqs:?}: rel diff {rel} too large"
        );
        Ok(())
    }

    #[test]
    #[ignore = "the gfx1151 splitkv kernel memory-faults on hip in its base path; see TODO dflash entry"]
    fn ck_splitkv_paged_varlen_matches_reference() -> candle_core::Result<()> {
        let dev = Device::new_hip(0)?;
        // bottom-right causal, no window; kv shorter than one page
        check_paged_case(&dev, 4, 2, &[(16, 20)], None, None, 2)?;
        // causal with a left window, kv across 3 pages
        check_paged_case(&dev, 4, 2, &[(16, 67)], Some(24), None, 2)?;
        // non-causal local window (both sides), 2 pages
        check_paged_case(&dev, 4, 2, &[(16, 33)], Some(8), Some(8), 0)?;
        // varlen batch: same window across sequences with ragged kv
        check_paged_case(&dev, 4, 2, &[(16, 20), (16, 67)], Some(24), None, 2)?;
        // tiny heads force max splits through the combine kernel
        check_paged_case(&dev, 2, 1, &[(16, 100)], Some(40), None, 2)?;
        Ok(())
    }
}
