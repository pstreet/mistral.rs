//! Dual-build (`cuda+rocm`) Hip implementation of [`super::fast_mmvq`].
//!
//! Same contract as `fast_mmvq.rs`, bound to the hip backend types with the
//! kernels resolved through the companion plugin. The decode dispatch routes
//! here for Hip weights; unsupported shapes fall back to dequant.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use candle_core::hip_backend::{
    cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut, SyncOnDrop},
    CudaDevice, CudaStorage, DeviceId,
};
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::{
    utils::{hip_slice_ptr_mut_on_stream, hip_slice_ptr_on_stream},
    GluActivationType,
};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36; // 2 halves (4 bytes) + QK8_1 int8 = 4 + 32 = 36
const MATRIX_ROW_PADDING: usize = 512;

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

fn output_shape(xs: &Tensor, nrows: usize) -> Shape {
    let mut out_dims = xs.dims().to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = nrows;
    Shape::from(out_dims)
}

// Mirror of fast_mmvq::supports - keep in sync.
pub fn supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

/// Maximum flattened batch handled by the Hip launcher table.
pub const MMVQ_MAX_BATCH: usize = 8;

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
}

struct WorkspaceGuard<'a> {
    slot: MutexGuard<'static, WorkspaceSlot>,
    stream: &'a CudaStream,
}

impl WorkspaceGuard<'_> {
    fn ptr_mut(&mut self) -> (u64, SyncOnDrop<'_>) {
        self.slot.slice.device_ptr_mut(self.stream)
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: DeviceId,
    stream: usize,
    thread: ThreadId,
    capacity: usize,
}

type WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<WorkspaceSlot>>>;

static WORKSPACE: OnceLock<WsMap> = OnceLock::new();

fn workspace_ensure<'a>(
    dev: &CudaDevice,
    bytes: usize,
    stream: &'a CudaStream,
) -> Result<WorkspaceGuard<'a>> {
    let map = WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let capacity = bytes.max(1).next_power_of_two();
    let key = WorkspaceKey {
        device: dev.id(),
        stream: stream.cu_stream() as usize,
        thread: std::thread::current().id(),
        capacity,
    };
    let workspace_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(capacity)? };
                // CUDA graph replay requires process-stable workspace addresses.
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot { slice })));
                guard.insert(key, leaked);
                leaked
            }
        }
    };
    Ok(WorkspaceGuard {
        slot: workspace_mtx.lock().unwrap(),
        stream,
    })
}

// Launcher dispatch by weight and output dtype.

type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

fn plain_launcher_bf16(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_bf16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_bf16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_bf16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_bf16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_bf16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_bf16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_bf16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_bf16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_bf16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_bf16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f16(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f32(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f32_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f32_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f32_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f32_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f32_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f32_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f32_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f32_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f32_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f32_plain,
        _ => return None,
    };
    Some(f)
}

type FusedGluLauncher = unsafe extern "C" fn(
    vx_gate: *const std::ffi::c_void,
    vx_up: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

type FusedQkvLauncher = unsafe extern "C" fn(
    vx_q: *const std::ffi::c_void,
    vx_k: *const std::ffi::c_void,
    vx_v: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    q_dst: *mut std::ffi::c_void,
    k_dst: *mut std::ffi::c_void,
    v_dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_q: i32,
    nrows_k: i32,
    nrows_v: i32,
    stride_col_y: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

fn fused_glu_launcher(input_ty: DType, dtype: GgmlDType) -> Option<FusedGluLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_bf16_fused_glu),

        (DType::F16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f16_fused_glu),
        (DType::F16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f16_fused_glu),
        (DType::F16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f16_fused_glu),

        (DType::F32, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f32_fused_glu),
        (DType::F32, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f32_fused_glu),
        (DType::F32, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f32_fused_glu),
        _ => None,
    }
}

fn fused_qkv_launcher(input_ty: DType, dtype: GgmlDType) -> Option<FusedQkvLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_bf16_fused_qkv),

        (DType::F16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f16_fused_qkv),
        (DType::F16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f16_fused_qkv),

        (DType::F32, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f32_fused_qkv),
        (DType::F32, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f32_fused_qkv),
        _ => None,
    }
}

pub fn supports_fused_glu(input_ty: DType, dtype: GgmlDType) -> bool {
    fused_glu_launcher(input_ty, dtype).is_some()
}

/// Compute `w @ xs^T` where `w` is a Q8_1-quantizable GGUF weight tensor and
/// `xs` is a contiguous BF16 / F16 / F32 activation on the same Hip device.
///
/// The product of the leading input dimensions must be in `1..=8`.
///
/// Output has the same leading dimensions as `xs` with the last axis replaced
/// by `w.shape().dims2()?.0` (nrows of the weight).
///
/// The output dtype matches the input dtype (BF16 → BF16, F16 → F16, F32 → F32).
pub fn plain(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    let dtype = w.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmvq: unsupported quant dtype {dtype:?}");
    }
    let Device::Hip(dev) = w.device() else {
        candle_core::bail!("fast_mmvq: weight must live on Hip");
    };
    if !xs.device().same_device(&w.device()) {
        candle_core::bail!("fast_mmvq: input and weight are on different devices");
    }
    let (nrows, ncols) = w.shape().dims2()?;

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("fast_mmvq: input dtype must be BF16, F16, or F32, got {input_ty:?}");
    }

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Hip(xs_hip) = &*xs_storage else {
        candle_core::bail!("fast_mmvq: input must live on Hip");
    };
    let xs_offset = xs_layout.start_offset();

    let stream_ptr = stream.cu_stream();
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let (weight_ptr, _weight_guard) = w.device_ptr_with_guard(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_hip.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = hip_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = plain_launcher_bf16(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Hip(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_hip.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = hip_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = plain_launcher_f16(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Hip(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_hip.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = hip_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = plain_launcher_f32(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Hip(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        _ => unreachable!(),
    }
}

pub fn fused_glu(
    gate_w: &QTensor,
    up_w: &QTensor,
    xs: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    let dtype = gate_w.dtype();
    if dtype != up_w.dtype() {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up dtype mismatch {:?} vs {:?}",
            dtype,
            up_w.dtype()
        );
    }
    let Some(launcher) = fused_glu_launcher(xs.dtype(), dtype) else {
        candle_core::bail!("fast_mmvq fused_glu: unsupported dtype combination");
    };

    let Device::Hip(dev) = gate_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: gate weight must live on Hip");
    };
    let Device::Hip(up_dev) = up_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: up weight must live on Hip");
    };
    if dev.id() != up_dev.id() {
        candle_core::bail!("fast_mmvq fused_glu: gate/up weights are on different Hip devices");
    }
    if !xs.device().same_device(&gate_w.device()) {
        candle_core::bail!("fast_mmvq fused_glu: input and weights are on different devices");
    }

    let (nrows, ncols) = gate_w.shape().dims2()?;
    let (up_nrows, up_ncols) = up_w.shape().dims2()?;
    if (nrows, ncols) != (up_nrows, up_ncols) {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up shape mismatch [{nrows}, {ncols}] vs [{up_nrows}, {up_ncols}]"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq fused_glu: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq fused_glu: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq fused_glu: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmvq fused_glu: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Hip(xs_hip) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_glu: input must live on Hip");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream();
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let (gate_ptr, _gate_guard) = gate_w.device_ptr_with_guard(&stream)?;
    let (up_ptr, _up_guard) = up_w.device_ptr_with_guard(&stream)?;
    let gate_ptr = gate_ptr as *const std::ffi::c_void;
    let up_ptr = up_ptr as *const std::ffi::c_void;
    let activation = activation as i32;

    match input_ty {
        DType::BF16 => {
            let slice = xs_hip.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = hip_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Hip(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_hip.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = hip_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Hip(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_hip.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = hip_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Hip(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        _ => unreachable!(),
    }
}

/// Compute Q, K, and V matvecs with one input quantization pass and one MMVQ
/// kernel. The result tensors match the unfused `plain` outputs for each
/// projection and preserve the input dtype.
pub fn fused_qkv(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: &QTensor,
    xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dtype = q_w.dtype();
    if dtype != k_w.dtype() || dtype != v_w.dtype() {
        candle_core::bail!(
            "fast_mmvq fused_qkv: q/k/v dtype mismatch {:?}, {:?}, {:?}",
            dtype,
            k_w.dtype(),
            v_w.dtype()
        );
    }
    let Some(launcher) = fused_qkv_launcher(xs.dtype(), dtype) else {
        candle_core::bail!("fast_mmvq fused_qkv: unsupported dtype combination");
    };

    let Device::Hip(dev) = q_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: q weight must live on Hip");
    };
    let Device::Hip(k_dev) = k_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: k weight must live on Hip");
    };
    let Device::Hip(v_dev) = v_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: v weight must live on Hip");
    };
    if dev.id() != k_dev.id() || dev.id() != v_dev.id() {
        candle_core::bail!("fast_mmvq fused_qkv: q/k/v weights are on different Hip devices");
    }
    if !xs.device().same_device(&q_w.device()) {
        candle_core::bail!("fast_mmvq fused_qkv: input and weights are on different devices");
    }

    let (q_nrows, ncols) = q_w.shape().dims2()?;
    let (k_nrows, k_ncols) = k_w.shape().dims2()?;
    let (v_nrows, v_ncols) = v_w.shape().dims2()?;
    if ncols != k_ncols || ncols != v_ncols {
        candle_core::bail!(
            "fast_mmvq fused_qkv: q/k/v ncols mismatch {ncols}, {k_ncols}, {v_ncols}"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq fused_qkv: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq fused_qkv: shape mismatch: weight ncols {ncols} vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq fused_qkv: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmvq fused_qkv: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Hip(xs_hip) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_qkv: input must live on Hip");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream();
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let (q_ptr, _q_guard) = q_w.device_ptr_with_guard(&stream)?;
    let (k_ptr, _k_guard) = k_w.device_ptr_with_guard(&stream)?;
    let (v_ptr, _v_guard) = v_w.device_ptr_with_guard(&stream)?;
    let q_ptr = q_ptr as *const std::ffi::c_void;
    let k_ptr = k_ptr as *const std::ffi::c_void;
    let v_ptr = v_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_hip.as_cuda_slice::<half::bf16>()?;
            let mut q_out = unsafe { dev.alloc::<half::bf16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::bf16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::bf16>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = hip_slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = hip_slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = hip_slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F16 => {
            let slice = xs_hip.as_cuda_slice::<half::f16>()?;
            let mut q_out = unsafe { dev.alloc::<half::f16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::f16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::f16>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = hip_slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = hip_slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = hip_slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F32 => {
            let slice = xs_hip.as_cuda_slice::<f32>()?;
            let mut q_out = unsafe { dev.alloc::<f32>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<f32>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<f32>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = hip_slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = hip_slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = hip_slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Hip(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QMatMul};
    use candle_core::Module as _;
    use std::sync::Arc;

    use crate::{GgufMatMul, QuantMethod, QuantMethodConfig};

    const HIDDEN: usize = 256;
    const INTERMEDIATE: usize = 512;
    const TOLERANCE: f32 = 5e-3;

    fn patterned(shape: impl Into<Shape>, salt: usize, scale: f32) -> Result<Tensor> {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|index| {
                let value = (index.wrapping_mul(37) + salt.wrapping_mul(19)) % 211;
                (value as f32 / 105.0 - 1.0) * scale
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        // Dtypes may differ (fused preserves the input dtype, the chute
        // returns F32); compare values in F32.
        let actual = actual
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(expected.iter().all(|value| value.is_finite()));
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs() / (1.0 + expected.abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= TOLERANCE,
            "relative error {max_error} exceeds {TOLERANCE}"
        );
        Ok(())
    }

    fn chute_reference(w: &Arc<QTensor>, xs: &Tensor) -> Result<Tensor> {
        QMatMul::from_arc(w.clone())?.forward(xs)
    }

    fn dequant_reference(w: &Arc<QTensor>, xs: &Tensor) -> Result<Tensor> {
        let weights = w.dequantize(xs.device())?.to_dtype(DType::F32)?;
        let (rows, _) = w.shape().dims2()?;
        let flat = xs.to_dtype(DType::F32)?.flatten(0, xs.rank() - 2)?;
        let out = flat.matmul(&weights.t()?)?;
        let mut shape: Vec<usize> = xs.dims()[..xs.rank() - 1].to_vec();
        shape.push(rows);
        out.reshape(shape)
    }

    #[test]
    fn hip_mmvq_plain_matches_dequant_chute() -> Result<()> {
        let hip = Device::new_hip(0)?;
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q4K] {
            let w = Arc::new(QTensor::quantize_onto(
                &patterned((INTERMEDIATE, HIDDEN), 11, 0.03)?,
                dtype,
                &hip,
            )?);
            for input_ty in [DType::BF16, DType::F16, DType::F32] {
                let xs = patterned((2, 3, HIDDEN), 3, 0.2)?
                    .to_dtype(input_ty)?
                    .to_device(&hip)?;
                assert_close(&plain(&w, &xs)?, &chute_reference(&w, &xs)?)?;
            }
        }
        Ok(())
    }

    #[test]
    fn hip_mmvq_fused_glu_matches_plains() -> Result<()> {
        let hip = Device::new_hip(0)?;
        let gate = QTensor::quantize_onto(
            &patterned((INTERMEDIATE, HIDDEN), 11, 0.03)?,
            GgmlDType::Q4K,
            &hip,
        )?;
        let up = QTensor::quantize_onto(
            &patterned((INTERMEDIATE, HIDDEN), 29, 0.03)?,
            GgmlDType::Q4K,
            &hip,
        )?;
        let xs = patterned((2, 3, HIDDEN), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&hip)?;
        let expected = crate::fused_glu(
            &plain(&gate, &xs)?,
            &plain(&up, &xs)?,
            GluActivationType::Silu,
        )?;
        assert_close(
            &fused_glu(&gate, &up, &xs, GluActivationType::Silu)?,
            &expected,
        )
    }

    #[test]
    fn hip_mmvq_fused_qkv_matches_plains() -> Result<()> {
        let hip = Device::new_hip(0)?;
        let q = QTensor::quantize_onto(&patterned((384, HIDDEN), 11, 0.03)?, GgmlDType::Q4K, &hip)?;
        let k = QTensor::quantize_onto(&patterned((256, HIDDEN), 29, 0.03)?, GgmlDType::Q4K, &hip)?;
        let v = QTensor::quantize_onto(&patterned((128, HIDDEN), 47, 0.03)?, GgmlDType::Q4K, &hip)?;
        let xs = patterned((1, 2, 2, HIDDEN), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&hip)?;

        let (q_out, k_out, v_out) = fused_qkv(&q, &k, &v, &xs)?;
        assert_close(&q_out, &plain(&q, &xs)?)?;
        assert_close(&k_out, &plain(&k, &xs)?)?;
        assert_close(&v_out, &plain(&v, &xs)?)
    }

    #[test]
    fn hip_mmvq_real_shapes_match_chute() -> Result<()> {
        // Qwen3.6-35B GDN in_proj_qkv: k=2048, out=8192, Q4K, decode-sized
        // batch. Small-shape tests pass while these sizes NaN in serving.
        let hip = Device::new_hip(0)?;
        const K: usize = 2048;
        let cases: [(GgmlDType, usize, usize); 4] = [
            (GgmlDType::Q4K, 4096, 2048),
            (GgmlDType::Q4K, 2048, 2048),
            (GgmlDType::Q6K, 512, 2048),
            (GgmlDType::Q5K, 2048, 2048),
        ];
        for (dtype, rows, k) in cases {
            let w = Arc::new(QTensor::quantize_onto(
                &patterned((rows, k), 11, 0.03)?,
                dtype,
                &hip,
            )?);
            let xs = patterned((1, 1, k), 3, 0.2)?
                .to_dtype(DType::BF16)?
                .to_device(&hip)?;
            let got = plain(&w, &xs)?;
            let got_max = got
                .abs()?
                .max_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;
            assert!(
                got_max.is_finite(),
                "plain {dtype:?} rows={rows} k={k} produced non-finite output"
            );
            assert_close(&got, &dequant_reference(&w, &xs)?)?;
        }

        let q = Arc::new(QTensor::quantize_onto(
            &patterned((4096, K), 11, 0.03)?,
            GgmlDType::Q4K,
            &hip,
        )?);
        let k_w = Arc::new(QTensor::quantize_onto(
            &patterned((2048, K), 29, 0.03)?,
            GgmlDType::Q4K,
            &hip,
        )?);
        let v = Arc::new(QTensor::quantize_onto(
            &patterned((2048, K), 47, 0.03)?,
            GgmlDType::Q4K,
            &hip,
        )?);
        let xs = patterned((1, 1, K), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&hip)?;
        let (q_out, k_out, v_out) = fused_qkv(&q, &k_w, &v, &xs)?;
        for (name, out) in [("q", &q_out), ("k", &k_out), ("v", &v_out)] {
            let max = out
                .abs()?
                .max_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;
            assert!(max.is_finite(), "fused_qkv {name} non-finite");
        }
        assert_close(&q_out, &plain(&q, &xs)?)?;
        assert_close(&k_out, &plain(&k_w, &xs)?)?;
        assert_close(&v_out, &plain(&v, &xs)?)
    }

    #[test]
    fn hip_gguf_matmul_f32_gate_projection_matches_cpu() -> Result<()> {
        // Qwen3.6-35B stores the GDN beta/alpha gates as F32 [64, 2048]
        // while the working dense Qwen3.5-4B stores them Q8_0: this exact
        // projection NaNs in dual-build serving at decode.
        let hip = Device::new_hip(0)?;
        for (dtype, rows, k) in [
            (GgmlDType::F32, 64, 2048),
            (GgmlDType::F32, 64, 2560),
            (GgmlDType::Q8_0, 64, 2048),
        ] {
            let w = Arc::new(QTensor::quantize_onto(
                &patterned((rows, k), 11, 0.03)?,
                dtype,
                &hip,
            )?);
            for input_ty in [DType::BF16, DType::F32] {
                let xs = patterned((1, 1, k), 3, 0.2)?
                    .to_dtype(input_ty)?
                    .to_device(&hip)?;
                let got = crate::GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: w.clone(),
                    b: None,
                })?
                .forward(&xs)?;
                let got_max = got
                    .abs()?
                    .max_all()?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
                assert!(
                    got_max.is_finite(),
                    "{dtype:?} [{rows}, {k}] {input_ty:?} produced non-finite output"
                );
                assert_close(&got, &dequant_reference(&w, &xs)?)?;
            }
        }
        Ok(())
    }

    #[test]
    fn hip_unquant_f32_gate_projection_matches_cpu() -> Result<()> {
        // F32 GGUF tensors skip the packed path and land as dense
        // UnquantLinear weights; the Qwen3.6 GDN b/a gates take this path.
        let hip = Device::new_hip(0)?;
        for (rows, k) in [(32usize, 2048usize), (64, 2048), (32, 2560)] {
            let weight = patterned((rows, k), 11, 0.03)?
                .to_device(&hip)?
                .to_dtype(DType::F32)?;
            let layer = crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                candle_nn::Linear::new(weight, None),
            ))?;
            let xs = patterned((1, 1, k), 3, 0.2)?
                .to_dtype(DType::F32)?
                .to_device(&hip)?;
            let got = layer.forward(&xs)?;
            let got_max = got
                .abs()?
                .max_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?;
            assert!(
                got_max.is_finite(),
                "unquant F32 [{rows}, {k}] produced non-finite output"
            );
            let expected = crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                candle_nn::Linear::new(patterned((rows, k), 11, 0.03)?, None),
            ))?;
            assert_close(&got, &expected.forward(&xs.to_device(&Device::Cpu)?)?)?;
        }
        Ok(())
    }
}
