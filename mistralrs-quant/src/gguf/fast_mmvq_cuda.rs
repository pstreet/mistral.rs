//! NVIDIA-role twin of [`super::fast_mmvq`] (dual builds): same contract
//! as the device module, typed on the upstream cuda backend with the kernels
//! resolved through the nvcc companion (`ffi_cuda`). Serves dense decode
//! batches 1-8 (plain/fused_glu/fused_qkv) for the NVIDIA role.
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use candle_core::cuda_backend::{
    cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut, SyncOnDrop},
    CudaDevice, CudaStorage, DeviceId,
};
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi_cuda;
use crate::{
    utils::{nv_slice_ptr_mut_on_stream, nv_slice_ptr_on_stream},
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

/// Maximum flattened batch handled by the role launcher table.
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
        GgmlDType::Q4_0 => ffi_cuda::cuda_launch_mmvq_gguf_q4_0_bf16_plain,
        GgmlDType::Q4_1 => ffi_cuda::cuda_launch_mmvq_gguf_q4_1_bf16_plain,
        GgmlDType::Q5_0 => ffi_cuda::cuda_launch_mmvq_gguf_q5_0_bf16_plain,
        GgmlDType::Q5_1 => ffi_cuda::cuda_launch_mmvq_gguf_q5_1_bf16_plain,
        GgmlDType::Q8_0 => ffi_cuda::cuda_launch_mmvq_gguf_q8_0_bf16_plain,
        GgmlDType::Q2K => ffi_cuda::cuda_launch_mmvq_gguf_q2_k_bf16_plain,
        GgmlDType::Q3K => ffi_cuda::cuda_launch_mmvq_gguf_q3_k_bf16_plain,
        GgmlDType::Q4K => ffi_cuda::cuda_launch_mmvq_gguf_q4_k_bf16_plain,
        GgmlDType::Q5K => ffi_cuda::cuda_launch_mmvq_gguf_q5_k_bf16_plain,
        GgmlDType::Q6K => ffi_cuda::cuda_launch_mmvq_gguf_q6_k_bf16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f16(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi_cuda::cuda_launch_mmvq_gguf_q4_0_f16_plain,
        GgmlDType::Q4_1 => ffi_cuda::cuda_launch_mmvq_gguf_q4_1_f16_plain,
        GgmlDType::Q5_0 => ffi_cuda::cuda_launch_mmvq_gguf_q5_0_f16_plain,
        GgmlDType::Q5_1 => ffi_cuda::cuda_launch_mmvq_gguf_q5_1_f16_plain,
        GgmlDType::Q8_0 => ffi_cuda::cuda_launch_mmvq_gguf_q8_0_f16_plain,
        GgmlDType::Q2K => ffi_cuda::cuda_launch_mmvq_gguf_q2_k_f16_plain,
        GgmlDType::Q3K => ffi_cuda::cuda_launch_mmvq_gguf_q3_k_f16_plain,
        GgmlDType::Q4K => ffi_cuda::cuda_launch_mmvq_gguf_q4_k_f16_plain,
        GgmlDType::Q5K => ffi_cuda::cuda_launch_mmvq_gguf_q5_k_f16_plain,
        GgmlDType::Q6K => ffi_cuda::cuda_launch_mmvq_gguf_q6_k_f16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f32(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi_cuda::cuda_launch_mmvq_gguf_q4_0_f32_plain,
        GgmlDType::Q4_1 => ffi_cuda::cuda_launch_mmvq_gguf_q4_1_f32_plain,
        GgmlDType::Q5_0 => ffi_cuda::cuda_launch_mmvq_gguf_q5_0_f32_plain,
        GgmlDType::Q5_1 => ffi_cuda::cuda_launch_mmvq_gguf_q5_1_f32_plain,
        GgmlDType::Q8_0 => ffi_cuda::cuda_launch_mmvq_gguf_q8_0_f32_plain,
        GgmlDType::Q2K => ffi_cuda::cuda_launch_mmvq_gguf_q2_k_f32_plain,
        GgmlDType::Q3K => ffi_cuda::cuda_launch_mmvq_gguf_q3_k_f32_plain,
        GgmlDType::Q4K => ffi_cuda::cuda_launch_mmvq_gguf_q4_k_f32_plain,
        GgmlDType::Q5K => ffi_cuda::cuda_launch_mmvq_gguf_q5_k_f32_plain,
        GgmlDType::Q6K => ffi_cuda::cuda_launch_mmvq_gguf_q6_k_f32_plain,
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
        (DType::BF16, GgmlDType::Q4_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q4_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_1_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_1_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q8_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q8_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q2K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q2_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q3K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q3_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q4K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q6K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q6_k_bf16_fused_glu),

        (DType::F16, GgmlDType::Q4_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q4_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_1_f16_fused_glu),
        (DType::F16, GgmlDType::Q5_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q5_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_1_f16_fused_glu),
        (DType::F16, GgmlDType::Q8_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q8_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q2K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q2_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q3K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q3_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q4K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q5K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q6K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q6_k_f16_fused_glu),

        (DType::F32, GgmlDType::Q4_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q4_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_1_f32_fused_glu),
        (DType::F32, GgmlDType::Q5_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q5_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_1_f32_fused_glu),
        (DType::F32, GgmlDType::Q8_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q8_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q2K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q2_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q3K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q3_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q4K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q5K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q6K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q6_k_f32_fused_glu),
        _ => None,
    }
}

fn fused_qkv_launcher(input_ty: DType, dtype: GgmlDType) -> Option<FusedQkvLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgmlDType::Q4_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q4_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_1_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_1_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q8_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q8_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q2K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q2_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q3K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q3_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q4K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q6K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q6_k_bf16_fused_qkv),

        (DType::F16, GgmlDType::Q4_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q4_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_1_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_1_f16_fused_qkv),
        (DType::F16, GgmlDType::Q8_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q8_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q2K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q2_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q3K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q3_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q4K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q6K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q6_k_f16_fused_qkv),

        (DType::F32, GgmlDType::Q4_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q4_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_1_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5_1) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_1_f32_fused_qkv),
        (DType::F32, GgmlDType::Q8_0) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q8_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q2K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q2_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q3K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q3_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q4K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q4_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q5_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q6K) => Some(ffi_cuda::cuda_launch_mmvq_gguf_q6_k_f32_fused_qkv),
        _ => None,
    }
}

pub fn supports_fused_glu(input_ty: DType, dtype: GgmlDType) -> bool {
    fused_glu_launcher(input_ty, dtype).is_some()
}

/// Compute `w @ xs^T` where `w` is a Q8_1-quantizable GGUF weight tensor and
/// `xs` is a contiguous BF16 / F16 / F32 activation on the same CUDA-role device.
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
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("fast_mmvq: weight must live on the CUDA role");
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
    let Storage::Cuda(xs_nv) = &*xs_storage else {
        candle_core::bail!("fast_mmvq: input must live on the CUDA role");
    };
    let xs_offset = xs_layout.start_offset();

    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let weight_ptr = w.device_ptr()? as *const std::ffi::c_void;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_nv.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = nv_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_bf16(
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
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_nv.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = nv_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_f16(
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
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_nv.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = nv_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_f32(
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
                Storage::Cuda(out_storage),
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

    let Device::Cuda(dev) = gate_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: gate weight must live on the CUDA role");
    };
    let Device::Cuda(up_dev) = up_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: up weight must live on the CUDA role");
    };
    if dev.id() != up_dev.id() {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up weights are on different CUDA-role devices"
        );
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
    let Storage::Cuda(xs_nv) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_glu: input must live on the CUDA role");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let gate_ptr = gate_w.device_ptr()? as *const std::ffi::c_void;
    let up_ptr = up_w.device_ptr()? as *const std::ffi::c_void;
    let gate_ptr = gate_ptr as *const std::ffi::c_void;
    let up_ptr = up_ptr as *const std::ffi::c_void;
    let activation = activation as i32;

    match input_ty {
        DType::BF16 => {
            let slice = xs_nv.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = nv_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_bf16(
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
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_nv.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = nv_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_f16(
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
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_nv.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = nv_slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_f32(
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
                Storage::Cuda(out_storage),
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

    let Device::Cuda(dev) = q_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: q weight must live on the CUDA role");
    };
    let Device::Cuda(k_dev) = k_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: k weight must live on the CUDA role");
    };
    let Device::Cuda(v_dev) = v_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: v weight must live on the CUDA role");
    };
    if dev.id() != k_dev.id() || dev.id() != v_dev.id() {
        candle_core::bail!("fast_mmvq fused_qkv: q/k/v weights are on different CUDA-role devices");
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
    let Storage::Cuda(xs_nv) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_qkv: input must live on the CUDA role");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let q_ptr = q_w.device_ptr()? as *const std::ffi::c_void;
    let k_ptr = k_w.device_ptr()? as *const std::ffi::c_void;
    let v_ptr = v_w.device_ptr()? as *const std::ffi::c_void;
    let q_ptr = q_ptr as *const std::ffi::c_void;
    let k_ptr = k_ptr as *const std::ffi::c_void;
    let v_ptr = v_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_nv.as_cuda_slice::<half::bf16>()?;
            let mut q_out = unsafe { dev.alloc::<half::bf16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::bf16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::bf16>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = nv_slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = nv_slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = nv_slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_bf16(
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
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F16 => {
            let slice = xs_nv.as_cuda_slice::<half::f16>()?;
            let mut q_out = unsafe { dev.alloc::<half::f16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::f16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::f16>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = nv_slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = nv_slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = nv_slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_f16(
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
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F32 => {
            let slice = xs_nv.as_cuda_slice::<f32>()?;
            let mut q_out = unsafe { dev.alloc::<f32>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<f32>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<f32>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = nv_slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = nv_slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = nv_slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = nv_slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi_cuda::cuda_launch_mmvq_gguf_quantize_q8_1_f32(
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
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
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
    /// Capture probe for the dense-decode route: the model-scale graph
    /// faults replaying a 256-element bmul whose K operand came from the
    /// MMVQ k-projection inside the capture. Reproduces plain()+bmul as a
    /// graph to dissect the wild pointer at small scale.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    /// Multi-bucket collision probe: the engine precaptures several graph
    /// buckets whose memalloc placeholder VAs come from the SHARED capture
    /// pool; with AUTO_FREE each launch re-materializes its own graph's
    /// allocations over the same pool blocks. Captures two graphs
    /// (batch-1 and batch-2 shaped) and interleaves launches - if one
    /// exec's launch stomps the other's output VAs, the earlier tensor
    /// reads back corrupt.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn mmvq_two_graph_capture_collision() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const NROWS: usize = 256;
        const NCOLS: usize = 512;

        let dev = Device::new_cuda(0)?;
        let host: Vec<f32> = (0..NROWS * NCOLS)
            .map(|i| ((i as f32 * 0.7).sin() * 0.5).abs())
            .collect();
        let w_cpu = Tensor::from_vec(host, (NROWS, NCOLS), &Device::Cpu)?;
        let w = QTensor::quantize_onto(&w_cpu, GgmlDType::Q4K, &dev)?;
        let cos_host: Vec<f32> = (0..NROWS).map(|i| (i as f32 * 0.02).cos() * 0.5).collect();
        let cos = Tensor::from_vec(cos_host, (1, 4, 64), &dev)?.to_dtype(DType::BF16)?;

        let mk_xs = |batch: usize| -> Result<Tensor> {
            let n = batch * NCOLS;
            let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
            Ok(Tensor::from_vec(v, (batch, 1, NCOLS), &dev)?.to_dtype(DType::BF16)?)
        };
        let run = |xs: &Tensor| -> Result<Tensor> {
            let k = plain(&w, xs)?;
            let kb = k.reshape((xs.dim(0)?, 4, 64))?;
            Ok(kb.broadcast_mul(&cos)?.flatten_all()?)
        };

        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager_a = run(&mk_xs(1)?)?;
        let eager_b = run(&mk_xs(2)?)?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };

        let capture = |xs: &Tensor| -> Result<(Tensor, *mut core::ffi::c_void)> {
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(candle_core::Error::wrap)?;
            let out = run(xs)?;
            let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
            let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
            unsafe {
                let r = sys::cuStreamEndCapture(
                    stream.cu_stream(),
                    core::ptr::from_mut(&mut graph).cast(),
                );
                assert_eq!(r as i32, 0, "end capture");
                let r = sys::cuGraphInstantiateWithFlags(
                    core::ptr::from_mut(&mut exec).cast(),
                    graph.cast(),
                    1, // AUTO_FREE_ON_LAUNCH, as the engine
                );
                assert_eq!(r as i32, 0, "instantiate");
                let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
                assert_eq!(r as i32, 0, "upload");
            }
            Ok((out, exec))
        };
        let launch = |exec: *mut core::ffi::c_void| -> Result<()> {
            unsafe {
                let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
                assert_eq!(r as i32, 0, "launch");
            }
            Ok(())
        };
        let max_diff = |a: &Tensor, b: &Tensor| -> Result<f32> {
            let av = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let bv = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            Ok(av
                .iter()
                .zip(&bv)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max))
        };

        // Precapture order: bucket A (batch 1) then bucket B (batch 2),
        // both from the shared pool.
        let (out_a, exec_a) = capture(&mk_xs(1)?)?;
        let (out_b, exec_b) = capture(&mk_xs(2)?)?;
        unsafe { stream.context().enable_event_tracking() };

        // Launch A fresh, check.
        launch(exec_a)?;
        dev.synchronize()?;
        let d = max_diff(&out_a, &eager_a)?;
        eprintln!("[collision-probe] A after launch A: {d}");
        assert!(d < 1e-3, "A corrupted after its own launch: {d}");

        // Launch B (re-materializes ITS VAs), then read A's tensor: if the
        // execs overlap VAs, B's kernels just stomped A's output.
        launch(exec_b)?;
        dev.synchronize()?;
        let d_b = max_diff(&out_b, &eager_b)?;
        eprintln!("[collision-probe] B after launch B: {d_b}");
        assert!(d_b < 1e-3, "B corrupted after its own launch: {d_b}");
        let d_a = max_diff(&out_a, &eager_a)?;
        eprintln!("[collision-probe] A after launch B: {d_a}");
        assert!(
            d_a < 1e-3,
            "COLLISION: A's output stomped by B's launch: {d_a}"
        );

        // And the reverse direction.
        launch(exec_a)?;
        dev.synchronize()?;
        let d_b2 = max_diff(&out_b, &eager_b)?;
        eprintln!("[collision-probe] B after launch A: {d_b2}");
        assert!(
            d_b2 < 1e-3,
            "COLLISION: B's output stomped by A's launch: {d_b2}"
        );
        Ok(())
    }

    /// cuBLAS-in-capture probe: the model's lm_head is BF16 (unquantized),
    /// so the captured decode step still contains one cublas GEMM - the
    /// last non-companion kernel family in the graph. Reproduces
    /// matmul(cublas) + broadcast bmul under capture; if the replay
    /// corrupts, the remaining model-scale fault is the cublas path.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn cublas_matmul_bmul_capture_launch_matches_eager() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const NROWS: usize = 4096;
        const NCOLS: usize = 2048;

        let dev = Device::new_cuda(0)?;
        let w_host: Vec<f32> = (0..NROWS * NCOLS)
            .map(|i| ((i as f32 * 1.7).sin() * 0.25).abs())
            .collect();
        let w = Tensor::from_vec(w_host, (NROWS, NCOLS), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&dev)?;
        let x_host: Vec<f32> = (0..NCOLS).map(|i| (i as f32 * 0.01).sin()).collect();
        let xs = Tensor::from_vec(x_host, (1, 1, NCOLS), &dev)?.to_dtype(DType::BF16)?;
        let scale_host: Vec<f32> = (0..NROWS)
            .map(|i| 1.0 + (i as f32 * 0.001).cos() * 0.1)
            .collect();
        let scale = Tensor::from_vec(scale_host, (1, 1, NROWS), &dev)?.to_dtype(DType::BF16)?;

        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let run = |x: &Tensor| -> Result<Tensor> {
            let logits = x.broadcast_mul(&xs)?; // [1,1,NCOLS]
            let out = logits.broadcast_mul(&xs)?;
            let mm = out.reshape((1, NCOLS))?.matmul(&w.t()?)?;
            mm.reshape((1, 1, NROWS))?.broadcast_mul(&scale)
        };
        let eager = run(&xs)?.flatten_all()?.to_dtype(DType::F32)?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run(&xs)?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
            let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "launch");
            stream.context().enable_event_tracking();
        }
        dev.synchronize()?;

        let b = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let a = eager.to_vec1::<f32>()?;
        let bad: Vec<(usize, f32, f32)> = a
            .iter()
            .zip(b.iter())
            .enumerate()
            .filter(|(_, (x, y))| (**x - **y).abs() > 1e-2)
            .map(|(i, (x, y))| (i, *x, *y))
            .collect();
        eprintln!(
            "[cublas-probe] mismatching: {} of {}; largest: {:?}",
            bad.len(),
            a.len(),
            bad.iter()
                .max_by(|x, y| (x.1 - x.2).abs().partial_cmp(&(y.1 - y.2).abs()).unwrap())
        );
        assert!(bad.is_empty(), "cublas-in-capture replay corrupt");
        Ok(())
    }

    /// Relaunch probe: temp-allocating chain (malloc + memfree nodes) under
    /// capture with engine-exact flags, launched twice plus once more after
    /// pool churn. Green means malloc-node relaunch is sound at small scale.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn bmul_malloc_relaunch_stability() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const N: usize = 256;

        let dev = Device::new_cuda(0)?;
        let x_host: Vec<f32> = (0..N)
            .map(|i| ((i as f32 * 0.7).sin() * 0.5).abs())
            .collect();
        let s_host: Vec<f32> = (0..N)
            .map(|i| 1.0 + (i as f32 * 0.02).cos() * 0.1)
            .collect();
        let x = Tensor::from_vec(x_host, (1, 1, N), &dev)?.to_dtype(DType::BF16)?;
        let scale = Tensor::from_vec(s_host, (1, 1, N), &dev)?.to_dtype(DType::BF16)?;

        // Chain with capture-time temps (malloc nodes) AND a dropped temp
        // (memfree node), mirroring the eager rope chain's allocation
        // pattern inside the model capture.
        let run = || -> Result<Tensor> {
            let t1 = x.broadcast_mul(&scale)?;
            let cvt = t1.to_dtype(DType::F32)?;
            let t2 = cvt.to_dtype(DType::BF16)?;
            drop(cvt);
            t2.broadcast_mul(&scale)
        };

        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run()?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run()?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
        }

        let check = |tag: &str| -> Result<()> {
            unsafe {
                let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
                assert_eq!(r as i32, 0, "launch {tag}");
            }
            dev.synchronize()?;
            let got = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let max_diff = eager
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[relaunch-probe] {tag}: max_diff={max_diff}");
            assert!(max_diff < 1e-2, "{tag} diverged: {max_diff}");
            Ok(())
        };

        check("launch1")?;
        check("launch2")?;

        // Pool churn between launches: allocate + free several large blocks
        // (the next request's prefill), then relaunch.
        {
            let mut churn = Vec::new();
            for _ in 0..8 {
                churn.push(Tensor::zeros((1 << 20,), DType::U8, &dev)?);
            }
            dev.synchronize()?;
            drop(churn);
            dev.synchronize()?;
        }
        check("launch3-after-churn")?;

        // Placement-shift probe: hold GBs DURING the relaunch so the
        // graph's malloc nodes cannot land on their launch-1 VAs. If a
        // node baked a concrete VA at upload instead of re-resolving from
        // its malloc dependency, this relaunch reads stale memory.
        {
            let mut hold = Vec::new();
            for _ in 0..6 {
                hold.push(Tensor::zeros((512 << 20,), DType::U8, &dev)?);
            }
            dev.synchronize()?;
            check("launch4-under-pressure")?;
            drop(hold);
            dev.synchronize()?;
        }
        check("launch5-after-release")?;

        unsafe {
            stream.context().enable_event_tracking();
        }
        Ok(())
    }

    /// cuBLAS relaunch probe: GEMM + temp allocs under capture with
    /// engine-exact flags, relaunched after GB-scale pool churn.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn cublas_malloc_relaunch_stability() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const NROWS: usize = 2048;
        const NCOLS: usize = 1024;

        let dev = Device::new_cuda(0)?;
        let w_host: Vec<f32> = (0..NROWS * NCOLS)
            .map(|i| ((i as f32 * 1.7).sin() * 0.25).abs())
            .collect();
        let w = Tensor::from_vec(w_host, (NROWS, NCOLS), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&dev)?;
        let x_host: Vec<f32> = (0..NCOLS).map(|i| (i as f32 * 0.01).sin()).collect();
        let xs = Tensor::from_vec(x_host, (1, NCOLS), &dev)?.to_dtype(DType::BF16)?;
        let scale_host: Vec<f32> = (0..NROWS)
            .map(|i| 1.0 + (i as f32 * 0.001).cos() * 0.1)
            .collect();
        let scale = Tensor::from_vec(scale_host, (1, NROWS), &dev)?.to_dtype(DType::BF16)?;

        let run = || -> Result<Tensor> {
            let mm = xs.matmul(&w.t()?)?;
            let cvt = mm.to_dtype(DType::F32)?;
            let back = cvt.to_dtype(DType::BF16)?;
            drop(cvt);
            back.broadcast_mul(&scale)
        };

        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run()?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run()?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
        }

        let check = |tag: &str| -> Result<()> {
            unsafe {
                let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
                assert_eq!(r as i32, 0, "launch {tag}");
            }
            dev.synchronize()?;
            let got = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let max_diff = eager
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[cublas-relaunch-probe] {tag}: max_diff={max_diff}");
            assert!(max_diff < 1e-1, "{tag} diverged: {max_diff}");
            Ok(())
        };

        check("launch1")?;
        {
            let mut churn = Vec::new();
            for _ in 0..4 {
                churn.push(Tensor::zeros((512 << 20,), DType::U8, &dev)?);
            }
            dev.synchronize()?;
            drop(churn);
            dev.synchronize()?;
        }
        check("launch2-after-gb-churn")?;

        unsafe {
            stream.context().enable_event_tracking();
        }
        Ok(())
    }

    /// Node-count relaunch probe: hundreds of temp-allocating chains in
    /// one capture, then relaunch. Green means node count is innocent.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn many_node_relaunch_stability() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const N: usize = 256;
        const CHAINS: usize = 400;

        let dev = Device::new_cuda(0)?;
        let x_host: Vec<f32> = (0..N)
            .map(|i| ((i as f32 * 0.7).sin() * 0.5).abs())
            .collect();
        let s_host: Vec<f32> = (0..N)
            .map(|i| 1.0 + (i as f32 * 0.02).cos() * 0.1)
            .collect();
        let x = Tensor::from_vec(x_host, (1, 1, N), &dev)?.to_dtype(DType::BF16)?;
        let scale = Tensor::from_vec(s_host, (1, 1, N), &dev)?.to_dtype(DType::BF16)?;

        let run = || -> Result<Tensor> {
            let mut acc = x.clone();
            for _ in 0..CHAINS {
                let t = acc.broadcast_mul(&scale)?;
                let cvt = t.to_dtype(DType::F32)?;
                let back = cvt.to_dtype(DType::BF16)?;
                drop(cvt);
                drop(t);
                acc = back.broadcast_mul(&scale)?;
            }
            Ok(acc)
        };

        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run()?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run()?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
            let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "launch1");
            let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "launch2");
            stream.context().enable_event_tracking();
        }
        dev.synchronize()?;

        let got = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let max_diff = eager
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[many-node-probe] after 2 launches: max_diff={max_diff}");
        assert!(max_diff < 1e-1, "many-node relaunch diverged: {max_diff}");
        Ok(())
    }

    /// Double-instantiate probe: a malloc-node graph can only be
    /// instantiated once on this driver (second instantiate returns
    /// CUDA_ERROR_NOT_SUPPORTED). Pins that restriction so the engine
    /// never depends on re-instantiation; exec A must stay healthy.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn double_instantiate_stability() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const N: usize = 256;
        const CHAINS: usize = 200;

        let dev = Device::new_cuda(0)?;
        let x_host: Vec<f32> = (0..N)
            .map(|i| ((i as f32 * 0.7).sin() * 0.5).abs())
            .collect();
        let s_host: Vec<f32> = (0..N)
            .map(|i| 1.0 + (i as f32 * 0.02).cos() * 0.1)
            .collect();
        let x = Tensor::from_vec(x_host, (1, 1, N), &dev)?.to_dtype(DType::BF16)?;
        let scale = Tensor::from_vec(s_host, (1, 1, N), &dev)?.to_dtype(DType::BF16)?;

        let run = || -> Result<Tensor> {
            let mut acc = x.clone();
            for _ in 0..CHAINS {
                let t = acc.broadcast_mul(&scale)?;
                let cvt = t.to_dtype(DType::F32)?;
                let back = cvt.to_dtype(DType::BF16)?;
                drop(cvt);
                drop(t);
                acc = back.broadcast_mul(&scale)?;
            }
            Ok(acc)
        };

        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run()?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run()?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec_a: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec_b: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec_a).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate A");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec_b).cast(),
                graph.cast(),
                1,
            );
            // Single-use instantiate: the second instantiate of a
            // malloc-node graph is NOT_SUPPORTED on this driver.
            assert_eq!(r as i32, 801, "instantiate B must fail with NOT_SUPPORTED");
            let r = sys::cuGraphUpload(exec_a.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload A");
        }

        let check = |tag: &str, exec: *mut core::ffi::c_void| -> Result<()> {
            unsafe {
                let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
                assert_eq!(r as i32, 0, "launch {tag}");
            }
            dev.synchronize()?;
            let got = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let max_diff = eager
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[double-instantiate-probe] {tag}: max_diff={max_diff}");
            assert!(max_diff < 1e-1, "{tag} diverged: {max_diff}");
            Ok(())
        };

        check("exec-A", exec_a)?;

        unsafe {
            stream.context().enable_event_tracking();
        }
        Ok(())
    }

    /// Companion-relaunch probe: MMVQ companion + bmul chain under
    /// capture with engine-exact flags, relaunched twice.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn companion_mmvq_relaunch_stability() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const NROWS: usize = 256;
        const NCOLS: usize = 512;

        let dev = Device::new_cuda(0)?;
        let host: Vec<f32> = (0..NROWS * NCOLS)
            .map(|i| ((i as f32 * 0.7).sin() * 0.5).abs())
            .collect();
        let w_cpu = Tensor::from_vec(host, (NROWS, NCOLS), &Device::Cpu)?;
        let w = QTensor::quantize_onto(&w_cpu, GgmlDType::Q4K, &dev)?;

        let x_host: Vec<f32> = (0..NCOLS).map(|i| (i as f32 * 0.01).sin()).collect();
        let xs = Tensor::from_vec(x_host, (1, 1, NCOLS), &dev)?.to_dtype(DType::BF16)?;
        let cos_host: Vec<f32> = (0..NROWS).map(|i| (i as f32 * 0.02).cos() * 0.5).collect();
        let cos = Tensor::from_vec(cos_host, (1, 4, 64), &dev)?.to_dtype(DType::BF16)?;

        let run = || -> Result<Tensor> {
            let k = plain(&w, &xs)?;
            let k4 = k.reshape((1, 4, 64))?;
            let m = (k4 * &cos)?;
            let cvt = m.to_dtype(DType::F32)?;
            let back = cvt.to_dtype(DType::BF16)?;
            drop(cvt);
            back.reshape((1, 1, NROWS))
        };

        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run()?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run()?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
        }

        let check = |tag: &str| -> Result<()> {
            unsafe {
                let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
                assert_eq!(r as i32, 0, "launch {tag}");
            }
            dev.synchronize()?;
            let got = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let max_diff = eager
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[companion-relaunch-probe] {tag}: max_diff={max_diff}");
            assert!(max_diff < 1e-1, "{tag} diverged: {max_diff}");
            Ok(())
        };

        check("launch1")?;
        check("launch2")?;

        unsafe {
            stream.context().enable_event_tracking();
        }
        Ok(())
    }

    /// htod-memcpy probe: htod copy into a capture temp (memcpy dest =
    /// malloc VA) under capture with engine-exact flags, relaunched with
    /// rewritten staging. Green exonerates the memcpy patch class.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn htod_memcpy_relaunch_stability() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const N: usize = 256;

        let dev = Device::new_cuda(0)?;
        let x_host: Vec<f32> = (0..N)
            .map(|i| ((i as f32 * 0.7).sin() * 0.5).abs())
            .collect();
        let s_host: Vec<f32> = (0..N)
            .map(|i| 1.0 + (i as f32 * 0.02).cos() * 0.1)
            .collect();
        let x = Tensor::from_vec(x_host, (1, 1, N), &dev)?.to_dtype(DType::BF16)?;

        // Host staging buffer, mimicking the engine's pinned host staging:
        // fresh host data per launch (replay uploads new values).
        let mut staging = s_host.clone();

        let run_once = |staging: &[f32]| -> Result<Tensor> {
            let scale = Tensor::from_vec(staging.to_vec(), (1, 1, N), &Device::Cpu)?
                .to_dtype(DType::BF16)?
                .to_device(&dev)?;
            let t = x.broadcast_mul(&scale)?;
            drop(scale);
            t.broadcast_mul(&x)
        };

        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run_once(&staging)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run_once(&staging)?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
        }

        let check = |tag: &str| -> Result<()> {
            unsafe {
                let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
                assert_eq!(r as i32, 0, "launch {tag}");
            }
            dev.synchronize()?;
            let got = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let max_diff = eager
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("[htod-relaunch-probe] {tag}: max_diff={max_diff}");
            assert!(max_diff < 1e-1, "{tag} diverged: {max_diff}");
            Ok(())
        };

        check("launch1")?;
        // Rewrite the staging buffer (new replay values) and relaunch:
        // the memcpy node must re-read the host pointer, not a stale VA.
        for v in staging.iter_mut() {
            *v = -*v;
        }
        check("launch2-new-staging")?;

        unsafe {
            stream.context().enable_event_tracking();
        }
        Ok(())
    }

    /// Capture probe for the dense-decode route: the model-scale graph
    /// faults replaying a 256-element bmul whose K operand came from the
    /// MMVQ k-projection inside the capture. Reproduces plain()+bmul as a
    /// graph to dissect the wild pointer at small scale.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn mmvq_plain_bmul_capture_launch_matches_eager() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const NROWS: usize = 256;
        const NCOLS: usize = 512;

        let dev = Device::new_cuda(0)?;
        let host: Vec<f32> = (0..NROWS * NCOLS)
            .map(|i| ((i as f32 * 0.7).sin() * 0.5).abs())
            .collect();
        let w_cpu = Tensor::from_vec(host, (NROWS, NCOLS), &Device::Cpu)?;
        let w = QTensor::quantize_onto(&w_cpu, GgmlDType::Q4K, &dev)?;

        let x_host: Vec<f32> = (0..NCOLS).map(|i| (i as f32 * 0.01).sin()).collect();
        let xs = Tensor::from_vec(x_host, (1, 1, NCOLS), &dev)?.to_dtype(DType::BF16)?;
        let cos_host: Vec<f32> = (0..NROWS).map(|i| (i as f32 * 0.02).cos() * 0.5).collect();
        let cos = Tensor::from_vec(cos_host, (1, 4, 64), &dev)?.to_dtype(DType::BF16)?;

        let run = || -> Result<Tensor> {
            let k = plain(&w, &xs)?;
            if std::env::var("MMVQ_PROBE_PLAIN_ONLY").is_ok() {
                return k.reshape((1, 1, NROWS));
            }
            let k4 = k.reshape((1, 4, 64))?;
            (k4 * &cos)?.reshape((1, 1, NROWS))
        };

        // Mirror the engine: the htod cache populates CUDA_PARAM_CACHE during
        // the warmup so capture reads fixed-address params buffers.
        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run()?.flatten_all()?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;

        // Mirror the engine's pool guard exactly: release threshold u64::MAX
        // plus the three reuse attributes on the device mempool, active for
        // the capture (the engine arms it after warmup, before capture).
        {
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            let r = unsafe { sys::cuDeviceGetMemPool(&mut pool, 0) };
            assert_eq!(r as i32, 0, "get mempool");
            let mut threshold = u64::MAX;
            let r = unsafe {
                sys::cuMemPoolSetAttribute(
                    pool.cast(),
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    (&mut threshold as *mut u64).cast(),
                )
            };
            assert_eq!(r as i32, 0, "release threshold");
            for attr in [
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            ] {
                let mut enabled = 1i32;
                let r = unsafe {
                    sys::cuMemPoolSetAttribute(pool.cast(), attr, (&mut enabled as *mut i32).cast())
                };
                assert_eq!(r as i32, 0, "reuse attr");
            }
        }

        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run()?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                0,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
            let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "launch");
            stream.context().enable_event_tracking();
        }
        dev.synchronize()?;

        let a = eager.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let b = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let bad: Vec<(usize, f32, f32)> = a
            .iter()
            .zip(b.iter())
            .enumerate()
            .filter(|(_, (x, y))| (**x - **y).abs() > 1e-2)
            .map(|(i, (x, y))| (i, *x, *y))
            .collect();
        let mut sorted = bad.clone();
        sorted.sort_by(|x, y| (y.1 - y.2).abs().partial_cmp(&(x.1 - x.2).abs()).unwrap());
        eprintln!(
            "[mmvq-capture-probe] mismatching elements: {} of {}; largest 8 (idx, eager, replayed): {:?}",
            bad.len(),
            a.len(),
            &sorted[..sorted.len().min(8)]
        );
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-3, "eager vs replay diff {max_diff}");
        Ok(())
    }

    fn hip_mmvq_plain_matches_dequant_chute() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q4K] {
            let w = Arc::new(QTensor::quantize_onto(
                &patterned((INTERMEDIATE, HIDDEN), 11, 0.03)?,
                dtype,
                &device,
            )?);
            for input_ty in [DType::BF16, DType::F16, DType::F32] {
                let xs = patterned((2, 3, HIDDEN), 3, 0.2)?
                    .to_dtype(input_ty)?
                    .to_device(&device)?;
                assert_close(&plain(&w, &xs)?, &chute_reference(&w, &xs)?)?;
            }
        }
        Ok(())
    }

    #[test]
    fn hip_mmvq_fused_glu_matches_plains() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let gate = QTensor::quantize_onto(
            &patterned((INTERMEDIATE, HIDDEN), 11, 0.03)?,
            GgmlDType::Q4K,
            &device,
        )?;
        let up = QTensor::quantize_onto(
            &patterned((INTERMEDIATE, HIDDEN), 29, 0.03)?,
            GgmlDType::Q4K,
            &device,
        )?;
        let xs = patterned((2, 3, HIDDEN), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&device)?;
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
        let device = Device::new_cuda(0)?;
        let q = QTensor::quantize_onto(
            &patterned((384, HIDDEN), 11, 0.03)?,
            GgmlDType::Q4K,
            &device,
        )?;
        let k = QTensor::quantize_onto(
            &patterned((256, HIDDEN), 29, 0.03)?,
            GgmlDType::Q4K,
            &device,
        )?;
        let v = QTensor::quantize_onto(
            &patterned((128, HIDDEN), 47, 0.03)?,
            GgmlDType::Q4K,
            &device,
        )?;
        let xs = patterned((1, 2, 2, HIDDEN), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&device)?;

        let (q_out, k_out, v_out) = fused_qkv(&q, &k, &v, &xs)?;
        assert_close(&q_out, &plain(&q, &xs)?)?;
        assert_close(&k_out, &plain(&k, &xs)?)?;
        assert_close(&v_out, &plain(&v, &xs)?)
    }

    #[test]
    fn hip_mmvq_real_shapes_match_chute() -> Result<()> {
        // Qwen3.6-35B GDN in_proj_qkv: k=2048, out=8192, Q4K, decode-sized
        // batch. Small-shape tests pass while these sizes NaN in serving.
        let device = Device::new_cuda(0)?;
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
                &device,
            )?);
            let xs = patterned((1, 1, k), 3, 0.2)?
                .to_dtype(DType::BF16)?
                .to_device(&device)?;
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
            &device,
        )?);
        let k_w = Arc::new(QTensor::quantize_onto(
            &patterned((2048, K), 29, 0.03)?,
            GgmlDType::Q4K,
            &device,
        )?);
        let v = Arc::new(QTensor::quantize_onto(
            &patterned((2048, K), 47, 0.03)?,
            GgmlDType::Q4K,
            &device,
        )?);
        let xs = patterned((1, 1, K), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&device)?;
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
        let device = Device::new_cuda(0)?;
        for (dtype, rows, k) in [
            (GgmlDType::F32, 64, 2048),
            (GgmlDType::F32, 64, 2560),
            (GgmlDType::Q8_0, 64, 2048),
        ] {
            let w = Arc::new(QTensor::quantize_onto(
                &patterned((rows, k), 11, 0.03)?,
                dtype,
                &device,
            )?);
            for input_ty in [DType::BF16, DType::F32] {
                let xs = patterned((1, 1, k), 3, 0.2)?
                    .to_dtype(input_ty)?
                    .to_device(&device)?;
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
        let device = Device::new_cuda(0)?;
        for (rows, k) in [(32usize, 2048usize), (64, 2048), (32, 2560)] {
            let weight = patterned((rows, k), 11, 0.03)?
                .to_device(&device)?
                .to_dtype(DType::F32)?;
            let layer = crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                candle_nn::Linear::new(weight, None),
            ))?;
            let xs = patterned((1, 1, k), 3, 0.2)?
                .to_dtype(DType::F32)?
                .to_device(&device)?;
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
