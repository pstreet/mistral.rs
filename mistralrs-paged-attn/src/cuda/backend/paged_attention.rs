use crate::cuda::backend::{cache_input_layout, slice_ptr};
use crate::cuda::ffi;
use crate::cuda::ffi::{
    paged_attention_v1_bf16, paged_attention_v1_f16, paged_attention_v1_f32,
    paged_attention_v2_bf16, paged_attention_v2_f16, paged_attention_v2_f32,
};
use crate::BlockQuantKind;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
#[cfg(not(feature = "rocm"))]
use candle_core::cuda::cudarc::driver::DeviceSlice;
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::ffi::c_int;
use std::sync::{Mutex, OnceLock};

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

type WsMap = Mutex<HashMap<candle::cuda_backend::DeviceId, &'static Mutex<WorkspaceSlot>>>;

static PAGED_ATTN_V2_WORKSPACE: OnceLock<WsMap> = OnceLock::new();

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

// Tokens per vLLM v2 attention partition. Must match PARTITION_SIZE in
// pagedattention.cuh. The v2 grid and reduce stride derive from
// ceil(max_context_len / PARTITION_SIZE), so any host code that keys cached
// launches (e.g. CUDA graphs) must treat this as part of the launch shape.
pub const PAGED_ATTENTION_V2_PARTITION_SIZE: usize = 512;

fn validate_side_scales(
    cache_dtype: DType,
    scale: Option<&Tensor>,
    side: &str,
    op: &str,
) -> Result<()> {
    match (cache_dtype, scale) {
        (DType::F8E4M3, Some(scale)) => {
            if scale.dtype() != DType::F32 || scale.elem_count() != 1 {
                candle::bail!("{op} requires a scalar f32 {side} scale for an f8e4m3 cache");
            }
        }
        (DType::F8E4M3, None) => {
            candle::bail!("{op} requires an explicit {side} scale for an f8e4m3 cache");
        }
        (DType::U8, Some(scale)) => {
            // Block-quantized (Q8_0/Q4_0): fp32 per-32 scale sidecar, not scalar.
            if scale.dtype() != DType::F32 {
                candle::bail!(
                    "{op} requires an f32 {side} scale sidecar for a block-quantized cache"
                );
            }
        }
        (DType::U8, None) => {
            candle::bail!(
                "{op} requires an explicit {side} scale sidecar for a block-quantized cache"
            );
        }
        (_, None) => {}
        _ => {
            candle::bail!("{op} only accepts a {side} scale for an f8e4m3 or block-quantized cache")
        }
    }
    Ok(())
}

fn workspace_ensure(
    dev: &candle::cuda_backend::CudaDevice,
    bytes: usize,
) -> Result<(u64, std::sync::MutexGuard<'static, WorkspaceSlot>)> {
    let map = PAGED_ATTN_V2_WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let device_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&device_key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(bytes.max(1))? };
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot {
                    slice,
                    cap: bytes.max(1),
                })));
                guard.insert(device_key, leaked);
                leaked
            }
        }
    };

    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < bytes {
        slot.slice = unsafe { dev.alloc::<u8>(bytes)? };
        slot.cap = bytes;
    }
    let ptr = slot.slice.device_ptr(slot.slice.stream()).0;
    Ok((ptr, slot))
}

struct PagedAttention {
    softmax_scale: f32,
    softcapping: f32,

    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    alibi_slopes: Option<Tensor>,
    max_context_len: usize,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    k_res: Option<Tensor>,
    v_res: Option<Tensor>,
    sinks: Option<Tensor>,
    k_quant: Option<BlockQuantKind>,
    v_quant: Option<BlockQuantKind>,
}

impl PagedAttention {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dtype = q.dtype();
        validate_side_scales(
            self.key_cache.dtype(),
            self.k_scale.as_ref(),
            "K",
            "paged_attention",
        )?;
        validate_side_scales(
            self.value_cache.dtype(),
            self.v_scale.as_ref(),
            "V",
            "paged_attention",
        )?;
        let k_cache_dtype = crate::side_cache_dtype(self.key_cache.dtype(), self.k_quant)
            .ok_or_else(|| {
                candle::Error::msg(format!(
                    "cache dtype {:?} is not supported",
                    self.key_cache.dtype()
                ))
            })?;
        let v_cache_dtype = crate::side_cache_dtype(self.value_cache.dtype(), self.v_quant)
            .ok_or_else(|| {
                candle::Error::msg(format!(
                    "cache dtype {:?} is not supported",
                    self.value_cache.dtype()
                ))
            })?;
        // U8 covers both block-quantized payloads; the layers disambiguate
        // per side. FP8 needs kernel support compiled in.
        if (k_cache_dtype == 3 || v_cache_dtype == 3) && !crate::cuda::USE_FP8 {
            candle::bail!("FP8 is not supported on this system.");
        }

        let dev = q.device();
        let out_shape = q_l.shape().clone();

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Cuda(kc) => kc,
            _ => candle::bail!("key_cache must be a cuda tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Cuda(vc) => vc,
            _ => candle::bail!("value_cache must be a cuda tensor"),
        };

        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Cuda(bt) => bt,
            _ => candle::bail!("block_tables must be a cuda tensor"),
        };

        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Cuda(cl) => cl,
            _ => candle::bail!("context_lens must be a cuda tensor"),
        };

        let q_rank = q_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();

        if q_rank != 3 {
            candle::bail!(
                "paged-attention expects `q` tensor to be of rank 3 \
                (q: {q_l:?})"
            )
        }

        if kc_rank != 5 {
            candle::bail!(
                "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
            )
        }

        if vc_rank != 4 {
            candle::bail!(
                "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
            )
        }

        // Get cuda slices for all tensors
        let q = q.as_cuda_slice::<T>()?;
        let (kc_ptr, _kc_guard) = if k_cache_dtype == 3 {
            slice_ptr(kc.as_cuda_slice::<F8E4M3>()?, kc_l.start_offset())
        } else if matches!(k_cache_dtype, 4 | 5) {
            slice_ptr(kc.as_cuda_slice::<u8>()?, kc_l.start_offset())
        } else {
            slice_ptr(kc.as_cuda_slice::<T>()?, kc_l.start_offset())
        };
        let (vc_ptr, _vc_guard) = if v_cache_dtype == 3 {
            slice_ptr(vc.as_cuda_slice::<F8E4M3>()?, vc_l.start_offset())
        } else if matches!(v_cache_dtype, 4 | 5) {
            slice_ptr(vc.as_cuda_slice::<u8>()?, vc_l.start_offset())
        } else {
            slice_ptr(vc.as_cuda_slice::<T>()?, vc_l.start_offset())
        };
        let cl = cl.as_cuda_slice::<u32>()?; // Should be i32!
        let bt = bt.as_cuda_slice::<u32>()?; // Should be i32!

        // Get cuda views for all tensors
        let q = q.slice(q_l.start_offset()..);
        let cl = cl.slice(cl_l.start_offset()..);
        let bt = bt.slice(bt_l.start_offset()..);

        let alibi_s_ptr = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            let (alibi_s, alibi_s_l) = alibi_slopes.storage_and_layout();
            let alibi_s = match &*alibi_s {
                Storage::Cuda(alibi_s) => alibi_s,
                _ => candle::bail!("context_lens must be a cuda tensor"),
            };
            let alibi_s = alibi_s.as_cuda_slice::<f32>()?;
            let (alibi_s_ptr, _alibi_s_guard) = slice_ptr(alibi_s, alibi_s_l.start_offset());
            alibi_s_ptr as *const std::ffi::c_void
        } else {
            std::ptr::null()
        };

        // FP8 global scales and block-quantized sidecars alike ride these
        // pointers; validation above already matched each side. Each side
        // resolves independently: a native side passes null while a
        // quantized/FP8 other side passes its tensor (joint nulling here
        // faults the quantized side's kernel reads on split caches).
        let _ks_storage = self.k_scale.as_ref().map(|ks| ks.storage_and_layout());
        let (k_scale_ptr, _ks_guard) = match &_ks_storage {
            Some((s, l)) => {
                let s = match &**s {
                    Storage::Cuda(s) => s,
                    _ => candle::bail!("k_scale must be a cuda tensor"),
                };
                let (p, g) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
                (p as *const f32, Some(g))
            }
            None => (std::ptr::null(), None),
        };
        let _vs_storage = self.v_scale.as_ref().map(|vs| vs.storage_and_layout());
        let (v_scale_ptr, _vs_guard) = match &_vs_storage {
            Some((s, l)) => {
                let s = match &**s {
                    Storage::Cuda(s) => s,
                    _ => candle::bail!("v_scale must be a cuda tensor"),
                };
                let (p, g) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
                (p as *const f32, Some(g))
            }
            None => (std::ptr::null(), None),
        };

        // QJL residual bit packs (Q4 LM path); null unless Q4 sidecars exist.
        // Independent per side like the scales above.
        let _kr_storage = self.k_res.as_ref().map(|kr| kr.storage_and_layout());
        let (k_res_ptr, _kr_guard) = match &_kr_storage {
            Some((s, l)) => {
                let s = match &**s {
                    Storage::Cuda(s) => s,
                    _ => candle::bail!("k_res must be a cuda tensor"),
                };
                let (p, g) = slice_ptr(s.as_cuda_slice::<u8>()?, l.start_offset());
                (p as *const u8, Some(g))
            }
            None => (std::ptr::null(), None),
        };
        let _vr_storage = self.v_res.as_ref().map(|vr| vr.storage_and_layout());
        let (v_res_ptr, _vr_guard) = match &_vr_storage {
            Some((s, l)) => {
                let s = match &**s {
                    Storage::Cuda(s) => s,
                    _ => candle::bail!("v_res must be a cuda tensor"),
                };
                let (p, g) = slice_ptr(s.as_cuda_slice::<u8>()?, l.start_offset());
                (p as *const u8, Some(g))
            }
            None => (std::ptr::null(), None),
        };

        let sinks_ptr = if let Some(sinks) = self.sinks.as_ref() {
            let (s, s_l) = sinks.storage_and_layout();
            let s = match &*s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("sinks must be a cuda tensor"),
            };
            let s = s.as_cuda_slice::<f32>()?;
            let (s_ptr, _s_guard) = slice_ptr(s, s_l.start_offset());
            s_ptr as *const f32
        } else {
            std::ptr::null()
        };

        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
        if !(head_size == 64
            || head_size == 80
            || head_size == 96
            || head_size == 112
            || head_size == 128
            || head_size == 192
            || head_size == 256
            || head_size == 512)
        {
            candle_core::bail!("`head_size` must be one of 64, 80, 96, 112, 128, 192, 256 or 512");
        }

        let (num_seqs_bt, max_num_blocks_per_seq) = bt_l.shape().dims2()?;

        if num_seqs_bt != num_seqs {
            candle::bail!(
                "shape mismatch block_tables {:?}, expected {:?}",
                bt_l.shape(),
                (num_seqs, max_num_blocks_per_seq)
            )
        }

        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        // Q4_0 packs 2 elems per byte: 16-byte chunks cover 32 head-dim elems.
        let expected_chunks = match self.k_quant {
            Some(BlockQuantKind::Q4_0) => head_size / 32,
            _ => head_size / x,
        };
        if head_size_kc != expected_chunks {
            candle::bail!(
                "shape mismatch key_cache {:?}, expected {:?}",
                kc_l.shape(),
                (num_blocks, num_kv_heads, expected_chunks, block_size, x)
            )
        }

        // Q4_0 packs 2 slots per byte along the block.
        let expected_v_block = match self.v_quant {
            Some(BlockQuantKind::Q4_0) => block_size / 2,
            _ => block_size,
        };
        if (num_blocks, num_kv_heads, head_size, expected_v_block) != vc_l.shape().dims4()? {
            candle::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }

        if (num_seqs) != cl_l.shape().dims1()? {
            candle::bail!(
                "shape mismatch context_lens {:?}, expected {:?}",
                cl_l.shape(),
                (num_seqs)
            )
        }

        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];
        let v_block_stride = vc_l.stride()[0];
        let v_head_stride = vc_l.stride()[1];

        let partition_size = PAGED_ATTENTION_V2_PARTITION_SIZE;
        let effective_max_context_len =
            (max_num_blocks_per_seq * block_size).min(self.max_context_len);
        let max_num_partitions = effective_max_context_len.div_ceil(partition_size);
        let use_v1 = (max_num_partitions == 1 || num_seqs * num_heads > 512)
            && partition_size % block_size == 0;

        let elem_count = out_shape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }?;

        let (out_ptr, out_guard) = out.device_ptr(out.stream());
        let (q_ptr, _q_guard) = q.device_ptr(q.stream());
        let (bt_ptr, _bt_guard) = bt.device_ptr(bt.stream());
        let (cl_ptr, _cl_guard) = cl.device_ptr(cl.stream());

        if use_v1 {
            let paged_attention_v1_func = match dtype {
                DType::F16 => paged_attention_v1_f16,
                DType::BF16 => paged_attention_v1_bf16,
                DType::F32 => paged_attention_v1_f32,
                dtype => candle::bail!("dtype {dtype:?} is not supported"),
            };
            unsafe {
                paged_attention_v1_func(
                    out_ptr as *const std::ffi::c_void,
                    q_ptr as *const std::ffi::c_void,
                    kc_ptr as *const std::ffi::c_void,
                    vc_ptr as *const std::ffi::c_void,
                    alibi_s_ptr,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    self.softcapping,
                    bt_ptr as *const i32,
                    cl_ptr as *const i32,
                    block_size as c_int,
                    effective_max_context_len as c_int,
                    num_seqs as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    max_num_blocks_per_seq as c_int,
                    q_stride as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    v_block_stride as c_int,
                    v_head_stride as c_int,
                    dev.cuda_stream().cu_stream(),
                    k_cache_dtype,
                    v_cache_dtype,
                    k_scale_ptr,
                    v_scale_ptr,
                    k_res_ptr,
                    v_res_ptr,
                    sinks_ptr,
                )
            }
        } else {
            let tmp_out_shape = Shape::from((num_seqs, num_heads, max_num_partitions, head_size));
            let exp_sums_shape = Shape::from((num_seqs, num_heads, max_num_partitions));
            let tmp_out_bytes = tmp_out_shape.elem_count() * std::mem::size_of::<T>();
            let exp_sums_bytes = exp_sums_shape.elem_count() * std::mem::size_of::<f32>();
            let tmp_out_offset = 0;
            let exp_sums_offset = align_up(tmp_out_offset + tmp_out_bytes, 16);
            let max_logits_offset = align_up(exp_sums_offset + exp_sums_bytes, 16);
            let workspace_bytes = max_logits_offset + exp_sums_bytes;
            let (workspace_ptr, _workspace_guard) = workspace_ensure(dev, workspace_bytes)?;

            let tmp_out_ptr = (workspace_ptr + tmp_out_offset as u64) as *mut std::ffi::c_void;
            let exp_sums_ptr = (workspace_ptr + exp_sums_offset as u64) as *mut f32;
            let max_logits_ptr = (workspace_ptr + max_logits_offset as u64) as *mut f32;

            let paged_attention_v2_func = match dtype {
                DType::F16 => paged_attention_v2_f16,
                DType::BF16 => paged_attention_v2_bf16,
                DType::F32 => paged_attention_v2_f32,
                dtype => candle::bail!("dtype {dtype:?} is not supported"),
            };
            unsafe {
                paged_attention_v2_func(
                    out_ptr as *const std::ffi::c_void,
                    exp_sums_ptr as *const f32,
                    max_logits_ptr as *const f32,
                    tmp_out_ptr as *const std::ffi::c_void,
                    q_ptr as *const std::ffi::c_void,
                    kc_ptr as *const std::ffi::c_void,
                    vc_ptr as *const std::ffi::c_void,
                    alibi_s_ptr,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    self.softcapping,
                    bt_ptr as *const i32,
                    cl_ptr as *const i32,
                    block_size as c_int,
                    effective_max_context_len as c_int,
                    num_seqs as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    max_num_blocks_per_seq as c_int,
                    q_stride as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    v_block_stride as c_int,
                    v_head_stride as c_int,
                    dev.cuda_stream().cu_stream(),
                    k_cache_dtype,
                    v_cache_dtype,
                    k_scale_ptr,
                    v_scale_ptr,
                    k_res_ptr,
                    v_res_ptr,
                    sinks_ptr,
                )
            }
        }

        drop(out_guard);

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, out_shape))
    }
}

impl candle::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for paged-attention")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l),
            dt => candle::bail!("paged-attention is only supported for f32/f16/bf16 ({dt:?})"),
        }
    }
}

/// PagedAttention layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors key_cache and value_cache
/// with fewer heads than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(num_sequences, num_heads_q, head_size)`.
/// * `key_cache` - Key cache paged tensor of shape `(num_blocks, num_heads_kv, head_size / x, block_size, x)`
///   with `x` being the size of an element in bytes.
/// * `value_cache` - Value cache paged tensor of shape `(num_blocks, num_heads_kv, head_size, block_size)`.
/// * `block_tables` - Padded table associating blocks to each sequence of shape `(num_sequences, max_context_len // block_size)`
/// * `context_lens` - Tensor associating lengths to each sequence of shape `(num_sequences)`
/// * `max_context_len` - Max of `context_len`
/// * `softmax_scale` - scaling factor
/// * `softcapping`- Softcapping value as in Gemma 2. Using 1.0 means do nothing.
/// * `alibi_slopes`- Optional alibi slopes, `(num_heads_q)`.
///
/// The resulting tensor has dimensions `(num_sequences, num_heads_q, head_size)`.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    k_res: Option<&Tensor>,
    v_res: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    sinks: Option<&Tensor>,
    k_quant: Option<BlockQuantKind>,
    v_quant: Option<BlockQuantKind>,
) -> Result<Tensor> {
    let op = PagedAttention {
        softmax_scale,
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        context_lens: context_lens.clone(),
        max_context_len,
        softcapping,
        alibi_slopes: alibi_slopes.cloned(),
        k_scale: k_scale.cloned(),
        v_scale: v_scale.cloned(),
        k_res: k_res.cloned(),
        v_res: v_res.cloned(),
        sinks: sinks
            .map(|s| s.to_dtype(candle_core::DType::F32))
            .transpose()?,
        k_quant,
        v_quant,
    };
    q.apply_op1(op)
}

#[allow(clippy::too_many_arguments)]
fn update_cache<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    write_k: bool,
    write_v: bool,
) -> Result<()> {
    let dtype = key.dtype();

    let internal_type = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };

    // Masked-off sides are skipped below (codes default native; the kernel
    // never touches them). A masked-off side may hold any dtype, including
    // a quantized cache written by its own masked call.
    let k_cache_dtype = if write_k {
        match key_cache.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            DType::F8E4M3 => 3,
            DType::U8 => candle::bail!(
                "reshape_and_cache only writes native/fp8 K sides; quantized K goes through reshape_and_cache_q8/q4"
            ),
            dtype => candle::bail!("cache dtype {dtype:?} is not supported"),
        }
    } else {
        0
    };
    let v_cache_dtype = if write_v {
        match value_cache.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            DType::F8E4M3 => 3,
            DType::U8 => candle::bail!(
                "reshape_and_cache only writes native/fp8 V sides; quantized V goes through reshape_and_cache_q8/q4"
            ),
            dtype => candle::bail!("cache dtype {dtype:?} is not supported"),
        }
    } else {
        0
    };
    if write_k {
        validate_side_scales(key_cache.dtype(), k_scale, "K", "reshape_and_cache")?;
    }
    if write_v {
        validate_side_scales(value_cache.dtype(), v_scale, "V", "reshape_and_cache")?;
    }

    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Cuda(k) => k,
        _ => candle::bail!("key must be a cuda tensor"),
    };

    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Cuda(v) => v,
        _ => candle::bail!("value must be a cuda tensor"),
    };

    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Cuda(kc) => kc,
        _ => candle::bail!("key_cache must be a cuda tensor"),
    };

    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Cuda(vc) => vc,
        _ => candle::bail!("value_cache must be a cuda tensor"),
    };

    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("slot_mapping must be a cuda tensor"),
    };

    let kc_rank = kc_l.stride().len();
    let vc_rank = vc_l.stride().len();

    if kc_rank != 5 {
        candle::bail!(
            "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
        )
    }

    if vc_rank != 4 {
        candle::bail!(
            "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
        )
    }

    let dev = k.device();

    // Get cuda slices for all tensors
    let k = k.as_cuda_slice::<T>()?;
    let v = v.as_cuda_slice::<T>()?;
    let s = s.as_cuda_slice::<i64>()?;

    // Slice each side in its own dtype: native sides read as the input
    // type, fp8 sides as fp8 bytes. Masked-off sides pass null.
    if (write_k && k_cache_dtype == 3 || write_v && v_cache_dtype == 3) && !crate::cuda::USE_FP8 {
        candle::bail!("FP8 is not supported on this system.");
    }
    let (kc_ptr, _kc_guard) = if !write_k {
        (0, None)
    } else if k_cache_dtype == 3 {
        let (p, g) = slice_ptr(kc.as_cuda_slice::<F8E4M3>()?, kc_l.start_offset());
        (p, Some(g))
    } else {
        let (p, g) = slice_ptr(kc.as_cuda_slice::<T>()?, kc_l.start_offset());
        (p, Some(g))
    };
    let (vc_ptr, _vc_guard) = if !write_v {
        (0, None)
    } else if v_cache_dtype == 3 {
        let (p, g) = slice_ptr(vc.as_cuda_slice::<F8E4M3>()?, vc_l.start_offset());
        (p, Some(g))
    } else {
        let (p, g) = slice_ptr(vc.as_cuda_slice::<T>()?, vc_l.start_offset());
        (p, Some(g))
    };

    // Get cuda views for all tensors
    let k = k.slice(k_l.start_offset()..);
    let v = v.slice(v_l.start_offset()..);
    let s = s.slice(s_l.start_offset()..);

    let _ks_storage = k_scale.map(|ks| ks.storage_and_layout());
    let (k_scale_ptr, _ks_guard) = match &_ks_storage {
        Some((s, l)) => {
            if !crate::cuda::USE_FP8 {
                candle::bail!("FP8 is not supported on this system.");
            }
            let s = match &**s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("k_scale must be a cuda tensor"),
            };
            let (p, g) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
            (p as *const f32, Some(g))
        }
        None => (std::ptr::null(), None),
    };
    let _vs_storage = v_scale.map(|vs| vs.storage_and_layout());
    let (v_scale_ptr, _vs_guard) = match &_vs_storage {
        Some((s, l)) => {
            if !crate::cuda::USE_FP8 {
                candle::bail!("FP8 is not supported on this system.");
            }
            let s = match &**s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("v_scale must be a cuda tensor"),
            };
            let (p, g) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
            (p as *const f32, Some(g))
        }
        None => (std::ptr::null(), None),
    };

    let (num_tokens, num_heads, head_size, key_stride) =
        cache_input_layout(k_l, "key", "paged-attention")?;
    let (value_tokens, value_heads, value_head_size, value_stride) =
        cache_input_layout(v_l, "value", "paged-attention")?;
    if (num_tokens, num_heads, head_size) != (value_tokens, value_heads, value_head_size) {
        candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }

    let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
    if write_k && (num_heads_kc != num_heads || head_size_kc != head_size / x) {
        candle::bail!(
            "shape mismatch value_cache {:?}, expected {:?}",
            vc_l.shape(),
            (num_blocks, num_heads, head_size / x, block_size, x)
        )
    }

    if write_v && (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
        candle::bail!(
            "shape mismatch key_cache {:?} and value_cache {:?}",
            kc_l.shape(),
            vc_l.shape()
        )
    }

    if (num_tokens) != s_l.shape().dims1()? {
        candle::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            s_l.shape(),
            (num_tokens)
        )
    }

    let key_stride = c_int::try_from(key_stride).map_err(candle::Error::wrap)?;
    let value_stride = c_int::try_from(value_stride).map_err(candle::Error::wrap)?;

    let (k_ptr, _k_guard) = k.device_ptr(k.stream());
    let (v_ptr, _v_guard) = v.device_ptr(v.stream());
    let (s_ptr, _s_guard) = s.device_ptr(s.stream());

    unsafe {
        ffi::reshape_and_cache(
            k_ptr as *const core::ffi::c_void,
            v_ptr as *const core::ffi::c_void,
            kc_ptr as *const core::ffi::c_void,
            vc_ptr as *const core::ffi::c_void,
            s_ptr as *const core::ffi::c_long,
            num_tokens as c_int,
            num_heads as c_int,
            head_size as c_int,
            block_size as c_int,
            x as c_int,
            key_stride,
            value_stride,
            dev.cuda_stream().cu_stream(),
            internal_type,
            k_cache_dtype,
            v_cache_dtype,
            write_k,
            write_v,
            k_scale_ptr,
            v_scale_ptr,
        )
    }
    Ok(())
}

/// Insert key and values with Q8_0 block quantization (llama.cpp Q8_0: int8 +
/// fp32 scale per 32 elems) inside the paged cache.
///
/// * `key_cache`/`value_cache` - int8 (U8) paged tensors with the same blocked
///   shapes as the unquantized path and `x = 16`.
/// * `k_scales`/`v_scales` - fp32 sidecars: k shaped
///   `(num_blocks, num_heads, block_size, head_size / 32)`, v transposed to
///   `(num_blocks, num_heads, head_size / 32, block_size)`, written by the
///   kernel. Must be zero-initialized (padding slots are skipped, never read).
///
/// Sides write independently for mixed K/V caches: a masked-off side's cache
/// pointer is ignored and its scales may be `None`.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_q8(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scales: Option<&Tensor>,
    v_scales: Option<&Tensor>,
    slot_mapping: &Tensor,
    write_k: bool,
    write_v: bool,
) -> Result<()> {
    const Q8_BLOCK: usize = 32;
    const Q8_MAX_HEAD: usize = 512;
    const Q8_X: usize = 16;

    let dtype = key.dtype();
    let internal_type = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };
    if write_k && key_cache.dtype() != DType::U8 {
        candle::bail!(
            "reshape_and_cache_q8 requires an int8 (U8) K cache, got {:?}",
            key_cache.dtype(),
        );
    }
    if write_v && value_cache.dtype() != DType::U8 {
        candle::bail!(
            "reshape_and_cache_q8 requires an int8 (U8) V cache, got {:?}",
            value_cache.dtype(),
        );
    }
    for (name, t) in [("k_scales", k_scales), ("v_scales", v_scales)] {
        if let Some(t) = t {
            if t.dtype() != DType::F32 {
                candle::bail!(
                    "reshape_and_cache_q8 requires f32 {name}, got {:?}",
                    t.dtype()
                );
            }
        }
    }
    if write_k && k_scales.is_none() {
        candle::bail!("reshape_and_cache_q8 needs k_scales to write K");
    }
    if write_v && v_scales.is_none() {
        candle::bail!("reshape_and_cache_q8 needs v_scales to write V");
    }

    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Cuda(k) => k,
        _ => candle::bail!("key must be a cuda tensor"),
    };
    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Cuda(v) => v,
        _ => candle::bail!("value must be a cuda tensor"),
    };
    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Cuda(kc) => kc,
        _ => candle::bail!("key_cache must be a cuda tensor"),
    };
    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Cuda(vc) => vc,
        _ => candle::bail!("value_cache must be a cuda tensor"),
    };
    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("slot_mapping must be a cuda tensor"),
    };
    let (num_tokens, num_heads, head_size, key_stride) =
        cache_input_layout(k_l, "key", "reshape_and_cache_q8")?;
    let (value_tokens, value_heads, value_head_size, value_stride) =
        cache_input_layout(v_l, "value", "reshape_and_cache_q8")?;
    if (num_tokens, num_heads, head_size) != (value_tokens, value_heads, value_head_size) {
        candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }
    if head_size % Q8_BLOCK != 0 || head_size > Q8_MAX_HEAD {
        candle::bail!(
            "reshape_and_cache_q8 requires head_size % 32 == 0 and <= 512, got {head_size}"
        );
    }
    let groups = head_size / Q8_BLOCK;

    // Masked-off sides skip validation entirely below. x rides to the
    // kernel but only the written Q side uses it (both quant writers pin 16).
    let (num_blocks, block_size, x) = {
        let (num_blocks, _, _, block_size, x) = kc_l.shape().dims5()?;
        (num_blocks, block_size, x)
    };
    if write_k {
        let (_, num_heads_kc, head_size_kc, _, _) = kc_l.shape().dims5()?;
        if x != Q8_X {
            candle::bail!("reshape_and_cache_q8 requires x = 16 packing, got {x}");
        }
        if num_heads_kc != num_heads || head_size_kc != head_size / x {
            candle::bail!(
                "shape mismatch key_cache {:?}, expected {:?}",
                kc_l.shape(),
                (num_blocks, num_heads, head_size / x, block_size, x)
            )
        }
        let expected_k_scales = [num_blocks, num_heads, block_size, groups];
        match k_scales {
            Some(ks) if ks.shape().dims() == expected_k_scales => {}
            Some(ks) => candle::bail!(
                "shape mismatch k_scales {:?}, expected {:?}",
                ks.shape(),
                expected_k_scales
            ),
            None => unreachable!("k_scales checked above"),
        }
    }
    if write_v {
        // v sidecar is transposed group-major for single-sector decode loads.
        let expected_v_scales = [num_blocks, num_heads, groups, block_size];
        if vc_l.shape().dims4()? != (num_blocks, num_heads, head_size, block_size) {
            candle::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }
        match v_scales {
            Some(vs) if vs.shape().dims() == expected_v_scales => {}
            Some(vs) => candle::bail!(
                "shape mismatch v_scales {:?}, expected {:?}",
                vs.shape(),
                expected_v_scales
            ),
            None => unreachable!("v_scales checked above"),
        }
    }
    if num_tokens != s_l.shape().dims1()? {
        candle::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            s_l.shape(),
            num_tokens
        )
    }

    let dev = k.device();

    // Masked-off sides pass null; the kernel never touches them. (The cache
    // tensors always exist, but slicing a native cache as u8 would fail.)
    let (kc_ptr, _kc_guard) = if write_k {
        let (p, g) = slice_ptr(kc.as_cuda_slice::<u8>()?, kc_l.start_offset());
        (p, Some(g))
    } else {
        (0, None)
    };
    let (vc_ptr, _vc_guard) = if write_v {
        let (p, g) = slice_ptr(vc.as_cuda_slice::<u8>()?, vc_l.start_offset());
        (p, Some(g))
    } else {
        (0, None)
    };
    let s_i64 = s.as_cuda_slice::<i64>()?;
    let (s_ptr, _s_guard) = slice_ptr(s_i64, s_l.start_offset());

    let _ks_storage = k_scales.map(|ks| ks.storage_and_layout());
    let (ks_ptr, _ks_guard) = match &_ks_storage {
        Some((s, l)) => {
            let s = match &**s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("Q8 k_scales must be a cuda tensor"),
            };
            let (p, g) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
            (p, Some(g))
        }
        None => (0, None),
    };
    let _vs_storage = v_scales.map(|vs| vs.storage_and_layout());
    let (vs_ptr, _vs_guard) = match &_vs_storage {
        Some((s, l)) => {
            let s = match &**s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("Q8 v_scales must be a cuda tensor"),
            };
            let (p, g) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
            (p, Some(g))
        }
        None => (0, None),
    };

    let key_stride = c_int::try_from(key_stride).map_err(candle::Error::wrap)?;
    let value_stride = c_int::try_from(value_stride).map_err(candle::Error::wrap)?;

    // Typed input pointers for the FFI call (k/v are &CudaStorage from above).
    let (k_in_ptr, _k_in_guard, v_in_ptr, _v_in_guard): (u64, _, u64, _) = match dtype {
        DType::F16 => {
            let kk = k.as_cuda_slice::<f16>()?;
            let vv = v.as_cuda_slice::<f16>()?;
            let (a, ag) = slice_ptr(kk, k_l.start_offset());
            let (b, bg) = slice_ptr(vv, v_l.start_offset());
            (a, ag, b, bg)
        }
        DType::BF16 => {
            let kk = k.as_cuda_slice::<bf16>()?;
            let vv = v.as_cuda_slice::<bf16>()?;
            let (a, ag) = slice_ptr(kk, k_l.start_offset());
            let (b, bg) = slice_ptr(vv, v_l.start_offset());
            (a, ag, b, bg)
        }
        DType::F32 => {
            let kk = k.as_cuda_slice::<f32>()?;
            let vv = v.as_cuda_slice::<f32>()?;
            let (a, ag) = slice_ptr(kk, k_l.start_offset());
            let (b, bg) = slice_ptr(vv, v_l.start_offset());
            (a, ag, b, bg)
        }
        _ => unreachable!("dtype checked above"),
    };

    unsafe {
        ffi::reshape_and_cache_q8(
            k_in_ptr as *const core::ffi::c_void,
            v_in_ptr as *const core::ffi::c_void,
            kc_ptr as *const core::ffi::c_void,
            vc_ptr as *const core::ffi::c_void,
            ks_ptr as *mut f32,
            vs_ptr as *mut f32,
            s_ptr as *const core::ffi::c_long,
            num_tokens as c_int,
            num_heads as c_int,
            head_size as c_int,
            block_size as c_int,
            x as c_int,
            key_stride,
            value_stride,
            dev.cuda_stream().cu_stream(),
            internal_type,
            write_k,
            write_v,
        )
    }
    Ok(())
}
/// Insert key and values with Q4_0 block quantization (nibbles + fp32 scale
/// per 32 elems) inside the paged cache.
///
/// * `key_cache` - U8 paged tensor shaped
///   `(num_blocks, num_heads, head_size / 32, block_size, 16)` (16-byte
///   chunks hold 32 elems).
/// * `value_cache` - U8 paged tensor shaped
///   `(num_blocks, num_heads, head_size, block_size / 2)` (2 slots per byte).
/// * `k_scales` - fp32 sidecar shaped
///   `(num_blocks, num_heads, block_size, head_size / 32)`, written by the
///   kernel. Must be zero-initialized (padding slots are skipped, never read).
/// * `v_scales` - fp32 sidecar shaped
///   `(num_blocks, num_heads, head_size / 32, block_size)`, written by the
///   kernel. Must be zero-initialized.
///
/// Sides write independently for mixed K/V caches: a masked-off side's cache
/// pointer is ignored and its scales may be `None`.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_q4(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scales: Option<&Tensor>,
    v_scales: Option<&Tensor>,
    k_res: Option<&Tensor>,
    v_res: Option<&Tensor>,
    slot_mapping: &Tensor,
    write_k: bool,
    write_v: bool,
) -> Result<()> {
    const Q4_BLOCK: usize = 32;
    const Q4_MAX_HEAD: usize = 512;
    const Q4_X: usize = 16;

    let dtype = key.dtype();
    let internal_type = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };
    if write_k && key_cache.dtype() != DType::U8 {
        candle::bail!(
            "reshape_and_cache_q4 requires a nibble (U8) K cache, got {:?}",
            key_cache.dtype(),
        );
    }
    if write_v && value_cache.dtype() != DType::U8 {
        candle::bail!(
            "reshape_and_cache_q4 requires a nibble (U8) V cache, got {:?}",
            value_cache.dtype(),
        );
    }
    for (name, t) in [("k_scales", k_scales), ("v_scales", v_scales)] {
        if let Some(t) = t {
            if t.dtype() != DType::F32 {
                candle::bail!(
                    "reshape_and_cache_q4 requires f32 {name}, got {:?}",
                    t.dtype()
                );
            }
        }
    }
    if write_k && k_scales.is_none() {
        candle::bail!("reshape_and_cache_q4 needs k_scales to write K");
    }
    if write_v && v_scales.is_none() {
        candle::bail!("reshape_and_cache_q4 needs v_scales to write V");
    }
    for (name, t) in [("k_res", k_res), ("v_res", v_res)] {
        if let Some(t) = t {
            if t.dtype() != DType::U8 {
                candle::bail!(
                    "reshape_and_cache_q4 requires u8 {name}, got {:?}",
                    t.dtype()
                );
            }
        }
    }

    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Cuda(k) => k,
        _ => candle::bail!("key must be a cuda tensor"),
    };
    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Cuda(v) => v,
        _ => candle::bail!("value must be a cuda tensor"),
    };
    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Cuda(kc) => kc,
        _ => candle::bail!("key_cache must be a cuda tensor"),
    };
    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Cuda(vc) => vc,
        _ => candle::bail!("value_cache must be a cuda tensor"),
    };
    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("slot_mapping must be a cuda tensor"),
    };
    let kr_storage = k_res.map(|kr| kr.storage_and_layout());
    let vr_storage = v_res.map(|vr| vr.storage_and_layout());

    let (num_tokens, num_heads, head_size, key_stride) =
        cache_input_layout(k_l, "key", "reshape_and_cache_q4")?;
    let (value_tokens, value_heads, value_head_size, value_stride) =
        cache_input_layout(v_l, "value", "reshape_and_cache_q4")?;
    if (num_tokens, num_heads, head_size) != (value_tokens, value_heads, value_head_size) {
        candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }
    if head_size % Q4_BLOCK != 0 || head_size > Q4_MAX_HEAD {
        candle::bail!(
            "reshape_and_cache_q4 requires head_size % 32 == 0 and <= 512, got {head_size}"
        );
    }
    let groups = head_size / Q4_BLOCK;

    let (num_blocks, block_size) = {
        let (num_blocks, _, _, block_size, _) = kc_l.shape().dims5()?;
        (num_blocks, block_size)
    };
    // x rides to the kernel but only a written Q side uses it (Q4 pins 16).
    let x = kc_l.shape().dims5()?.4;
    if write_k {
        let (_, num_heads_kc, head_size_kc, _, _) = kc_l.shape().dims5()?;
        if x != Q4_X {
            candle::bail!("reshape_and_cache_q4 requires x = 16 packing, got {x}");
        }
        if num_heads_kc != num_heads || head_size_kc != head_size / 32 {
            candle::bail!(
                "shape mismatch key_cache {:?}, expected {:?}",
                kc_l.shape(),
                (num_blocks, num_heads, head_size / 32, block_size, x)
            )
        }
        match k_scales {
            Some(ks) if ks.shape().dims() == [num_blocks, num_heads, block_size, groups] => {}
            Some(ks) => candle::bail!(
                "shape mismatch k_scales {:?}, expected {:?}",
                ks.shape(),
                [num_blocks, num_heads, block_size, groups]
            ),
            None => unreachable!("k_scales checked above"),
        }
        if let Some(k_res) = k_res {
            if k_res.shape().dims() != [num_blocks, num_heads, block_size, groups, 4] {
                candle::bail!(
                    "shape mismatch k_res {:?}, expected {:?}",
                    k_res.shape(),
                    [num_blocks, num_heads, block_size, groups, 4]
                );
            }
        }
    }
    if write_v {
        if vc_l.shape().dims4()? != (num_blocks, num_heads, head_size, block_size / 2) {
            candle::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }
        match v_scales {
            Some(vs) if vs.shape().dims() == [num_blocks, num_heads, groups, block_size] => {}
            Some(vs) => candle::bail!(
                "shape mismatch v_scales {:?}, expected {:?}",
                vs.shape(),
                [num_blocks, num_heads, groups, block_size]
            ),
            None => unreachable!("v_scales checked above"),
        }
        if let Some(v_res) = v_res {
            if v_res.shape().dims() != [num_blocks, num_heads, groups, block_size, 4] {
                candle::bail!(
                    "shape mismatch v_res {:?}, expected {:?}",
                    v_res.shape(),
                    [num_blocks, num_heads, groups, block_size, 4]
                );
            }
        }
        if block_size % 2 != 0 {
            candle::bail!("reshape_and_cache_q4 requires an even block_size, got {block_size}");
        }
    }
    if num_tokens != s_l.shape().dims1()? {
        candle::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            s_l.shape(),
            num_tokens
        )
    }

    let dev = k.device();

    // Masked-off sides pass null; the kernel never touches them. (The cache
    // tensors always exist, but slicing a native cache as u8 would fail.)
    let (kc_ptr, _kc_guard) = if write_k {
        let (p, g) = slice_ptr(kc.as_cuda_slice::<u8>()?, kc_l.start_offset());
        (p, Some(g))
    } else {
        (0, None)
    };
    let (vc_ptr, _vc_guard) = if write_v {
        let (p, g) = slice_ptr(vc.as_cuda_slice::<u8>()?, vc_l.start_offset());
        (p, Some(g))
    } else {
        (0, None)
    };
    let s_i64 = s.as_cuda_slice::<i64>()?;
    let (s_ptr, _s_guard) = slice_ptr(s_i64, s_l.start_offset());

    let _ks_storage = k_scales.map(|ks| ks.storage_and_layout());
    let (ks_ptr, _ks_guard) = match &_ks_storage {
        Some((s, l)) => {
            let s = match &**s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("Q4 k_scales must be a cuda tensor"),
            };
            let (p, g) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
            (p, Some(g))
        }
        None => (0, None),
    };
    let _vs_storage = v_scales.map(|vs| vs.storage_and_layout());
    let (vs_ptr, _vs_guard) = match &_vs_storage {
        Some((s, l)) => {
            let s = match &**s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("Q4 v_scales must be a cuda tensor"),
            };
            let (p, g) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
            (p, Some(g))
        }
        None => (0, None),
    };

    let key_stride = c_int::try_from(key_stride).map_err(candle::Error::wrap)?;
    let value_stride = c_int::try_from(value_stride).map_err(candle::Error::wrap)?;

    // Typed input pointers for the FFI call (k/v are &CudaStorage from above).
    let (k_in_ptr, _k_in_guard, v_in_ptr, _v_in_guard): (u64, _, u64, _) = match dtype {
        DType::F16 => {
            let kk = k.as_cuda_slice::<f16>()?;
            let vv = v.as_cuda_slice::<f16>()?;
            let (a, ag) = slice_ptr(kk, k_l.start_offset());
            let (b, bg) = slice_ptr(vv, v_l.start_offset());
            (a, ag, b, bg)
        }
        DType::BF16 => {
            let kk = k.as_cuda_slice::<bf16>()?;
            let vv = v.as_cuda_slice::<bf16>()?;
            let (a, ag) = slice_ptr(kk, k_l.start_offset());
            let (b, bg) = slice_ptr(vv, v_l.start_offset());
            (a, ag, b, bg)
        }
        DType::F32 => {
            let kk = k.as_cuda_slice::<f32>()?;
            let vv = v.as_cuda_slice::<f32>()?;
            let (a, ag) = slice_ptr(kk, k_l.start_offset());
            let (b, bg) = slice_ptr(vv, v_l.start_offset());
            (a, ag, b, bg)
        }
        _ => unreachable!("dtype checked above"),
    };

    // QJL residual sidecars; null when QJL is disabled (store skips write).
    let (kr_ptr, _kr_guard) = if let Some((ref s, l)) = kr_storage {
        let s = match &**s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("Q4 k_res must be a cuda tensor"),
        };
        let (ptr, guard) = slice_ptr(s.as_cuda_slice::<u8>()?, l.start_offset());
        (ptr as *mut u8, Some(guard))
    } else {
        (std::ptr::null_mut(), None)
    };
    let (vr_ptr, _vr_guard) = if let Some((ref s, l)) = vr_storage {
        let s = match &**s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("Q4 v_res must be a cuda tensor"),
        };
        let (ptr, guard) = slice_ptr(s.as_cuda_slice::<u8>()?, l.start_offset());
        (ptr as *mut u8, Some(guard))
    } else {
        (std::ptr::null_mut(), None)
    };

    unsafe {
        ffi::reshape_and_cache_q4(
            k_in_ptr as *const core::ffi::c_void,
            v_in_ptr as *const core::ffi::c_void,
            kc_ptr as *const core::ffi::c_void,
            vc_ptr as *const core::ffi::c_void,
            ks_ptr as *mut f32,
            vs_ptr as *mut f32,
            kr_ptr as *mut u8,
            vr_ptr as *mut u8,
            s_ptr as *const core::ffi::c_long,
            num_tokens as c_int,
            num_heads as c_int,
            head_size as c_int,
            block_size as c_int,
            x as c_int,
            key_stride,
            value_stride,
            dev.cuda_stream().cu_stream(),
            internal_type,
            write_k,
            write_v,
        )
    }
    Ok(())
}
/// Insert key and values at the provided slot mapping inside the key value paged cache
///
/// # Arguments
///
/// * `key` - Key tensor shaped `(num_tokens, num_heads, head_size)` or `(batch, seq_len, num_heads, head_size)`.
/// * `value` - Value tensor with the same logical shape as `key`.
/// * `key_cache` - Key cache paged tensor of shape `(num_blocks, num_heads, head_size / x, block_size, x)`
///   with `x` being the size of an element in bytes.
/// * `value_cache` - Value cache paged tensor of shape `(num_blocks, num_heads, head_size, block_size)`.
/// * `slot_mapping` - Mapping associating a slot to each token of shape `(num_tokens)`.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    write_k: bool,
    write_v: bool,
) -> Result<()> {
    match key.dtype() {
        DType::F16 => update_cache::<f16>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
            write_k,
            write_v,
        ),
        DType::BF16 => update_cache::<bf16>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
            write_k,
            write_v,
        ),
        DType::F32 => update_cache::<f32>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
            write_k,
            write_v,
        ),
        dt => {
            candle::bail!("reshape_and_cache is only supported for f32, f16 and bf16 ({dt:?})")
        }
    }
}
