//! Dual-build (`cuda+rocm`) AMD-role implementation of the indexed MoE
//! forward. The gather-path entries (`qtensor_indexed_moe_forward`,
//! `qmatmul_indexed_moe_forward`) are real: they quantize the input to
//! Q8_1 once and launch the hipcc-built indexed MoE kernels (same FFI
//! surface, dlsym-resolved from the companion plugin). The dev-typed
//! grouped/decode/reduce entries stay stubbed until the mistralrs-core
//! call sites become role-aware; their signatures keep the NVIDIA-typed
//! devices so `forward_grouped` keeps compiling in dual builds.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::hip_backend::cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr};
use candle_core::hip_backend::{CudaDType, CudaDevice, CudaStorage};
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::utils::{
    hip_slice_ptr, hip_slice_ptr_mut_on_stream, hip_slice_ptr_on_stream, hip_u32_ptrs,
};

pub const ACT_GELU_PYTORCH_TANH: i32 = 0;
pub const ACT_SILU: i32 = 1;

// Constants matching candle's quantized CUDA implementation
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;
const CUDA_GRID_YZ_LIMIT: usize = 65_535;
const MOE_REDUCE_THREADS: usize = 256;

struct U8WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

struct DispatchWorkspaceSlot {
    slice: CudaSlice<u32>,
    cap: usize,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: candle_core::hip_backend::DeviceId,
    stream: usize,
}

type U8WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<U8WorkspaceSlot>>>;

struct F32WorkspaceSlot {
    slice: CudaSlice<f32>,
    cap: usize,
}

type F32WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<F32WorkspaceSlot>>>;

static MOE_Q8_WORKSPACE: OnceLock<U8WsMap> = OnceLock::new();
static MOE_DECODE_F32_WORKSPACE: OnceLock<F32WsMap> = OnceLock::new();

type DispatchWsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<DispatchWorkspaceSlot>>>;

static MOE_DISPATCH_WORKSPACE: OnceLock<DispatchWsMap> = OnceLock::new();

fn dispatch_workspace_ensure(
    dev: &CudaDevice,
    len: usize,
) -> Result<(u64, std::sync::MutexGuard<'static, DispatchWorkspaceSlot>)> {
    let len = len.max(1);
    let map = MOE_DISPATCH_WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = workspace_key(dev);
    let device_mtx: &'static Mutex<DispatchWorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u32>(len)? };
                let leaked = Box::leak(Box::new(Mutex::new(DispatchWorkspaceSlot {
                    slice,
                    cap: len,
                })));
                guard.insert(key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < len {
        slot.slice = unsafe { dev.alloc::<u32>(len)? };
        slot.cap = len;
    }
    let ptr = {
        let (ptr, guard) = hip_slice_ptr(&slot.slice, 0);
        drop(guard);
        ptr
    };
    Ok((ptr, slot))
}

fn workspace_key(dev: &CudaDevice) -> WorkspaceKey {
    WorkspaceKey {
        device: dev.id(),
        stream: dev.cuda_stream().cu_stream() as usize,
    }
}

fn u8_workspace_ensure(
    dev: &CudaDevice,
    len: usize,
) -> Result<std::sync::MutexGuard<'static, U8WorkspaceSlot>> {
    let len = len.max(1);
    let map = MOE_Q8_WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = workspace_key(dev);
    let device_mtx: &'static Mutex<U8WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(len)? };
                let leaked = Box::leak(Box::new(Mutex::new(U8WorkspaceSlot { slice, cap: len })));
                guard.insert(key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < len {
        slot.slice = unsafe { dev.alloc::<u8>(len)? };
        slot.cap = len;
    }
    Ok(slot)
}

fn f32_workspace_ensure(
    dev: &CudaDevice,
    len: usize,
) -> Result<std::sync::MutexGuard<'static, F32WorkspaceSlot>> {
    let len = len.max(1);
    let map = MOE_DECODE_F32_WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = workspace_key(dev);
    let device_mtx: &'static Mutex<F32WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<f32>(len)? };
                let leaked = Box::leak(Box::new(Mutex::new(F32WorkspaceSlot { slice, cap: len })));
                guard.insert(key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < len {
        slot.slice = unsafe { dev.alloc::<f32>(len)? };
        slot.cap = len;
    }
    Ok(slot)
}

fn check_hip_launch(status: i32, kernel: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        candle_core::bail!("{kernel} HIP launch failed with status {status}")
    }
}

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn indexed_moe_weight_dtype(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q8_1
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

fn q8_1_bytes(num_rows: usize, k_padded: usize) -> usize {
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let num_blocks_per_row = k_padded / q8_1_block_size;
    num_rows * num_blocks_per_row * q8_1_type_size
}

/// Quantize f32 input to Q8_1 format for use with quantized matmul kernels.
fn quantize_q8_1(
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;

    let cuda_stream = dev.cuda_stream();
    let stream = cuda_stream.cu_stream();

    const CHUNK_SIZE: usize = 65535;
    let mut rows_processed = 0;
    while rows_processed < total_rows {
        let remaining_rows = total_rows - rows_processed;
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        let src_start_elem = rows_processed * k;

        let q8_1_block_size = GgmlDType::Q8_1.block_size();
        let q8_1_type_size = GgmlDType::Q8_1.type_size();
        let num_blocks_per_row = kx_padded / q8_1_block_size;
        let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

        let dst_start_byte = rows_processed * dst_row_size_bytes;

        let (src_ptr, _src_guard) = hip_slice_ptr_on_stream(src, src_start_elem, &cuda_stream);
        let (dst_ptr, _dst_guard) = hip_slice_ptr_mut_on_stream(dst, dst_start_byte, &cuda_stream);

        unsafe {
            ffi::launch_quantize_q8_1(
                src_ptr as *const f32,
                dst_ptr as *mut std::ffi::c_void,
                k as i32,
                kx_padded as i32,
                num_blocks as i32,
                rows_in_chunk as i32,
                stream,
            );
        }

        rows_processed += rows_in_chunk;
    }

    Ok(())
}
/// Quantize a contiguous [rows, k] activation to Q8_1 into `input_quant`.
/// Fused half->Q8_1 kernels handle BF16/F16 directly, avoiding a cast kernel.
fn quantize_tensor_into_q8_1(
    xs_contig: &Tensor,
    input_quant: &mut CudaSlice<u8>,
    dev: &CudaDevice,
) -> Result<()> {
    // Use fused half->Q8_1 kernels when input is BF16/F16 (avoids separate cast kernel)
    if xs_contig.dtype() == candle_core::DType::BF16 || xs_contig.dtype() == candle_core::DType::F16
    {
        let (xs_storage, xs_layout) = xs_contig.storage_and_layout();
        let Storage::Hip(xs_hip) = &*xs_storage else {
            candle_core::bail!("expected Hip tensor");
        };
        let num_rows = xs_contig.dim(0)?;
        let k = xs_contig.dim(1)?;
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let cuda_stream = dev.cuda_stream();
        let stream = cuda_stream.cu_stream();
        let (out_ptr, _og) = hip_slice_ptr_mut_on_stream(input_quant, 0, &cuda_stream);
        if xs_contig.dtype() == candle_core::DType::BF16 {
            let xs_slice = xs_hip.as_cuda_slice::<half::bf16>()?;
            let (xs_ptr, _xg) =
                hip_slice_ptr_on_stream(xs_slice, xs_layout.start_offset(), &cuda_stream);
            unsafe {
                ffi::launch_quantize_q8_1_bf16(
                    xs_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    k_padded as i32,
                    num_rows as i32,
                    stream,
                );
            }
        } else {
            let xs_slice = xs_hip.as_cuda_slice::<half::f16>()?;
            let (xs_ptr, _xg) =
                hip_slice_ptr_on_stream(xs_slice, xs_layout.start_offset(), &cuda_stream);
            unsafe {
                ffi::launch_quantize_q8_1_f16(
                    xs_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    k_padded as i32,
                    num_rows as i32,
                    stream,
                );
            }
        }
    } else {
        let xs_f32 = xs_contig.to_dtype(candle_core::DType::F32)?;
        let (xs_storage, xs_layout) = xs_f32.storage_and_layout();
        let Storage::Hip(xs_hip) = &*xs_storage else {
            candle_core::bail!("expected Hip tensor");
        };
        let xs_slice = xs_hip.as_cuda_slice::<f32>()?;
        assert!(xs_layout.start_offset() == 0);
        let num_rows = xs_contig.dim(0)?;
        let k = xs_contig.dim(1)?;
        quantize_q8_1(xs_slice, input_quant, k, num_rows, dev)?;
    }
    Ok(())
}

/// Quantize a [rows, k] activation to Q8_1 in the per-device workspace.
/// The returned guard keeps the workspace slot (and its buffer) alive
/// until the consumer kernel has been launched.
fn quantize_input_q8_1_workspace(
    xs: &Tensor,
    dev: &CudaDevice,
) -> Result<(
    std::sync::MutexGuard<'static, U8WorkspaceSlot>,
    u64,
    usize,
    usize,
)> {
    let xs_contig = xs.contiguous()?;
    let num_rows = xs_contig.dim(0)?;
    let k = xs_contig.dim(1)?;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes = q8_1_bytes(num_rows, k_padded);

    let mut slot = u8_workspace_ensure(dev, y_size_in_bytes)?;
    quantize_tensor_into_q8_1(&xs_contig, &mut slot.slice, dev)?;
    // The workspace buffer lives in a leaked per-device slot, so the raw
    // pointer stays valid for the process lifetime; the slot guard only
    // serializes reuse of the buffer across calls on the same stream.
    let ptr = {
        let (ptr, guard) = hip_slice_ptr(&slot.slice, 0);
        drop(guard);
        ptr
    };
    Ok((slot, ptr, k, k_padded))
}

/// Perform indexed MoE forward pass with fused Q8_1 input quantization.
#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_fused_q8_1_input(
    weight_ptr: u64,
    w_shape: &Shape,
    w_dtype: GgmlDType,
    input: &Tensor,
    in_shape: &Shape,
    ids: &CudaSlice<u32>,
    idx_shape: &Shape,
    dev: &CudaDevice,
) -> Result<(CudaStorage, Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    let total_rows = batch * input_dim1;
    let input = input.reshape((total_rows, k))?;
    let (_quant_ws_guard, inputs_ptr, quant_k, k_padded) =
        quantize_input_q8_1_workspace(&input, dev)?;
    assert!(quant_k == k, "K mismatch");

    // Output buffer - zero-initialize to prevent NaN from uninitialized memory
    let outsize = batch * topk * n;
    let out = dev.alloc_zeros::<f32>(outsize)?;

    let stream = dev.cuda_stream().cu_stream();

    let n_i32 = n as i32;
    let k_i32 = k as i32;
    let batch_i32 = batch as i32;
    let topk_i32 = topk as i32;
    let k_padded_i32 = k_padded as i32;
    let input_dim1_i32 = input_dim1 as i32;

    let (indices_ptr, _indices_guard) = hip_slice_ptr(ids, 0);
    let (outputs_ptr, _outputs_guard) = hip_slice_ptr(&out, 0);

    unsafe {
        let weights_ptr = weight_ptr as *const std::ffi::c_void;
        let inputs_ptr = inputs_ptr as *const std::ffi::c_void;
        let indices_ptr = indices_ptr as *const u32;
        let outputs_ptr = outputs_ptr as *mut f32;

        macro_rules! launch {
            ($entry:ident) => {{
                ffi::$entry(
                    weights_ptr,
                    inputs_ptr,
                    indices_ptr,
                    outputs_ptr,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                )
            }};
        }

        match w_dtype {
            GgmlDType::Q4_0 => launch!(launch_indexed_moe_forward_q4_0_q8_1),
            GgmlDType::Q4_1 => launch!(launch_indexed_moe_forward_q4_1_q8_1),
            GgmlDType::Q5_0 => launch!(launch_indexed_moe_forward_q5_0_q8_1),
            GgmlDType::Q5_1 => launch!(launch_indexed_moe_forward_q5_1_q8_1),
            GgmlDType::Q8_0 => launch!(launch_indexed_moe_forward_q8_0_q8_1),
            GgmlDType::Q8_1 => launch!(launch_indexed_moe_forward_q8_1_q8_1),
            GgmlDType::Q2K => launch!(launch_indexed_moe_forward_q2k_q8_1),
            GgmlDType::Q3K => launch!(launch_indexed_moe_forward_q3k_q8_1),
            GgmlDType::Q4K => launch!(launch_indexed_moe_forward_q4k_q8_1),
            GgmlDType::Q5K => launch!(launch_indexed_moe_forward_q5k_q8_1),
            GgmlDType::Q6K => launch!(launch_indexed_moe_forward_q6k_q8_1),
            _ => candle_core::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
        }
    }

    drop(_indices_guard);
    drop(_outputs_guard);

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;

    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

/// Perform indexed MoE forward pass on a QTensor (hip role).
pub fn qtensor_indexed_moe_forward(qtensor: &QTensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let dtype = qtensor.dtype();

    if !indexed_moe_weight_dtype(dtype) {
        candle_core::bail!(
            "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
            dtype
        );
    }

    let Device::Hip(dev) = qtensor.device() else {
        candle_core::bail!("indexed_moe_forward requires the Hip device for weights");
    };

    let (x_storage, _x_layout) = x.storage_and_layout();
    let Storage::Hip(_) = &*x_storage else {
        candle_core::bail!("indexed_moe_forward requires the Hip device for input");
    };

    let (ids_storage, _ids_layout) = ids.storage_and_layout();
    let Storage::Hip(ids_hip) = &*ids_storage else {
        candle_core::bail!("indexed_moe_forward requires the Hip device for indices");
    };

    let weight_ptr = qtensor.device_ptr()? as u64;

    let ids_slice = ids_hip.as_cuda_slice::<u32>()?;

    let (storage, out_shape) = indexed_moe_forward_fused_q8_1_input(
        weight_ptr,
        qtensor.shape(),
        dtype,
        x,
        x.shape(),
        ids_slice,
        ids.shape(),
        &dev,
    )?;

    Ok(Tensor::from((Storage::Hip(storage), out_shape)))
}

/// Perform indexed MoE forward pass on a QMatMul (hip role).
pub fn qmatmul_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        QMatMul::Tensor(_) | QMatMul::TensorF16(_) => {
            candle_core::bail!(
                "indexed_moe_forward is only supported for quantized tensors (QTensor)"
            )
        }
    }
}

/// Fused MoE decode on the hip role: quantize input, fused
/// gate+up+activation+multiply, quantize intermediate, fused
/// down+aggregate. Takes plain tensors and extracts the Hip device and
/// slices itself so the core call sites stay device-agnostic.
///
/// # Safety
/// `topk_weights_flat` must be an F32 Hip tensor holding `batch * topk`
/// values on the weights' device; `topk_ids_flat` must be U32.
#[allow(clippy::too_many_arguments)]
pub unsafe fn indexed_moe_fused_decode_hip(
    gate_qt: &QTensor,
    up_qt: &QTensor,
    down_qt: &QTensor,
    xs_flat: &Tensor,
    topk_ids_flat: &Tensor,
    topk_weights_flat: &Tensor,
    batch: usize,
    topk: usize,
    act_type: i32,
) -> Result<Tensor> {
    let gate_up_dtype = gate_qt.dtype();
    if up_qt.dtype() != gate_up_dtype {
        candle_core::bail!("fused MoE decode needs matching gate/up dtypes");
    }
    if !indexed_moe_weight_dtype(gate_up_dtype) {
        candle_core::bail!("unsupported dtype for fused MoE decode: {gate_up_dtype:?}");
    }
    let down_dtype = down_qt.dtype();
    if !indexed_moe_weight_dtype(down_dtype) {
        candle_core::bail!("unsupported dtype for fused MoE decode: {down_dtype:?}");
    }

    let Device::Hip(dev) = gate_qt.device() else {
        candle_core::bail!("fused MoE decode requires Hip weights");
    };

    let (_, n_gate, k_gate) = gate_qt.shape().dims3()?;
    let (_, n_down, k_down) = down_qt.shape().dims3()?;
    let hidden_size = k_gate;
    let intermediate_size = n_gate;
    if n_down != hidden_size || k_down != intermediate_size {
        candle_core::bail!("fused MoE decode shape mismatch");
    }

    let topk_ids_flat = topk_ids_flat.contiguous()?;
    if topk_ids_flat.elem_count() != batch * topk {
        candle_core::bail!("fused MoE decode routing does not match batch * topk");
    }
    let (ids_storage, ids_layout) = topk_ids_flat.storage_and_layout();
    let Storage::Hip(ids_hip) = &*ids_storage else {
        candle_core::bail!("fused MoE decode requires Hip routing ids");
    };
    if topk_weights_flat.dtype() != candle_core::DType::F32 {
        candle_core::bail!("fused MoE decode weights must be F32");
    }
    let topk_weights_flat = topk_weights_flat.contiguous()?;
    if topk_weights_flat.elem_count() != batch * topk {
        candle_core::bail!("fused MoE decode weights do not match routing");
    }
    let (tw_storage, tw_layout) = topk_weights_flat.storage_and_layout();
    let Storage::Hip(tw_hip) = &*tw_storage else {
        candle_core::bail!("fused MoE decode requires Hip routing weights");
    };

    let xs_contig = xs_flat.contiguous()?;
    let input_rows = xs_contig.dim(0)?;
    let k = xs_contig.dim(1)?;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let input_q8_bytes = q8_1_bytes(input_rows, k_padded);
    let intermediate_rows = batch * topk;
    let k_down_padded = pad(intermediate_size, MATRIX_ROW_PADDING);
    let intermediate_q8_bytes = q8_1_bytes(intermediate_rows, k_down_padded);

    let mut q8_workspace = u8_workspace_ensure(&dev, input_q8_bytes.max(intermediate_q8_bytes))?;
    quantize_tensor_into_q8_1(&xs_contig, &mut q8_workspace.slice, &dev)?;

    let cuda_stream = dev.cuda_stream();
    let stream = cuda_stream.cu_stream();

    let gate_up_outsize = batch * topk * intermediate_size;
    let gate_up_workspace = f32_workspace_ensure(&dev, gate_up_outsize)?;

    let gate_ptr = gate_qt.device_ptr()? as *const std::ffi::c_void;
    let up_ptr = up_qt.device_ptr()? as *const std::ffi::c_void;

    {
        let (inp_ptr, _ig) = hip_slice_ptr(&q8_workspace.slice, 0);
        let ids_slice = ids_hip.as_cuda_slice::<u32>()?;
        let (ids_ptr, _idg) =
            hip_slice_ptr_on_stream(ids_slice, ids_layout.start_offset(), &cuda_stream);
        let (out_ptr, _og) = hip_slice_ptr(&gate_up_workspace.slice, 0);

        type FusedGateUpFn = unsafe extern "C" fn(
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *const u32,
            *mut f32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            *mut std::ffi::c_void,
        );

        macro_rules! launch_gate_up {
            ($entry:ident) => {
                ffi::$entry(
                    gate_ptr,
                    up_ptr,
                    inp_ptr as *const std::ffi::c_void,
                    ids_ptr as *const u32,
                    out_ptr as *mut f32,
                    intermediate_size as i32,
                    k as i32,
                    batch as i32,
                    topk as i32,
                    k_padded as i32,
                    act_type,
                    stream,
                )
            };
        }

        match gate_up_dtype {
            GgmlDType::Q8_0 => launch_gate_up!(launch_moe_gemv_fused_gate_up_q8_0_q8_1),
            GgmlDType::Q4_0 => launch_gate_up!(launch_moe_gemv_fused_gate_up_q4_0_q8_1),
            GgmlDType::Q4_1 => launch_gate_up!(launch_moe_gemv_fused_gate_up_q4_1_q8_1),
            GgmlDType::Q5_0 => launch_gate_up!(launch_moe_gemv_fused_gate_up_q5_0_q8_1),
            GgmlDType::Q5_1 => launch_gate_up!(launch_moe_gemv_fused_gate_up_q5_1_q8_1),
            GgmlDType::Q8_1 => launch_gate_up!(launch_moe_gemv_fused_gate_up_q8_1_q8_1),
            GgmlDType::Q2K => launch_gate_up!(launch_moe_gemv_fused_gate_up_q2k_q8_1),
            GgmlDType::Q3K => launch_gate_up!(launch_moe_gemv_fused_gate_up_q3k_q8_1),
            GgmlDType::Q4K => launch_gate_up!(launch_moe_gemv_fused_gate_up_q4k_q8_1),
            GgmlDType::Q5K => launch_gate_up!(launch_moe_gemv_fused_gate_up_q5k_q8_1),
            GgmlDType::Q6K => launch_gate_up!(launch_moe_gemv_fused_gate_up_q6k_q8_1),
            _ => candle_core::bail!("unsupported dtype for fused MoE decode: {gate_up_dtype:?}"),
        }
    }

    quantize_q8_1(
        &gate_up_workspace.slice,
        &mut q8_workspace.slice,
        intermediate_size,
        intermediate_rows,
        &dev,
    )?;

    let final_outsize = batch * hidden_size;
    let final_out = dev.alloc_zeros::<f32>(final_outsize)?;
    let down_ptr = down_qt.device_ptr()? as *const std::ffi::c_void;

    {
        let (inp_ptr, _ig) = hip_slice_ptr(&q8_workspace.slice, 0);
        let ids_slice = ids_hip.as_cuda_slice::<u32>()?;
        let (ids_ptr, _idg) =
            hip_slice_ptr_on_stream(ids_slice, ids_layout.start_offset(), &cuda_stream);
        let tw_slice = tw_hip.as_cuda_slice::<f32>()?;
        let (tw_ptr, _wg) =
            hip_slice_ptr_on_stream(tw_slice, tw_layout.start_offset(), &cuda_stream);
        let (out_ptr, _og) = hip_slice_ptr(&final_out, 0);

        type DownAggregateFn = unsafe extern "C" fn(
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *const u32,
            *const f32,
            *mut f32,
            i32,
            i32,
            i32,
            i32,
            i32,
            *mut std::ffi::c_void,
        );

        macro_rules! launch_down {
            ($entry:ident) => {
                ffi::$entry(
                    down_ptr,
                    inp_ptr as *const std::ffi::c_void,
                    ids_ptr as *const u32,
                    tw_ptr as *const f32,
                    out_ptr as *mut f32,
                    hidden_size as i32,
                    intermediate_size as i32,
                    batch as i32,
                    topk as i32,
                    k_down_padded as i32,
                    stream,
                )
            };
        }

        match down_dtype {
            GgmlDType::Q8_0 => launch_down!(launch_moe_gemv_down_aggregate_q8_0_q8_1),
            GgmlDType::Q4_0 => launch_down!(launch_moe_gemv_down_aggregate_q4_0_q8_1),
            GgmlDType::Q4_1 => launch_down!(launch_moe_gemv_down_aggregate_q4_1_q8_1),
            GgmlDType::Q5_0 => launch_down!(launch_moe_gemv_down_aggregate_q5_0_q8_1),
            GgmlDType::Q5_1 => launch_down!(launch_moe_gemv_down_aggregate_q5_1_q8_1),
            GgmlDType::Q8_1 => launch_down!(launch_moe_gemv_down_aggregate_q8_1_q8_1),
            GgmlDType::Q2K => launch_down!(launch_moe_gemv_down_aggregate_q2k_q8_1),
            GgmlDType::Q3K => launch_down!(launch_moe_gemv_down_aggregate_q3k_q8_1),
            GgmlDType::Q4K => launch_down!(launch_moe_gemv_down_aggregate_q4k_q8_1),
            GgmlDType::Q5K => launch_down!(launch_moe_gemv_down_aggregate_q5k_q8_1),
            GgmlDType::Q6K => launch_down!(launch_moe_gemv_down_aggregate_q6k_q8_1),
            _ => candle_core::bail!("unsupported dtype for fused MoE decode: {down_dtype:?}"),
        }
    }

    let out_shape: Shape = vec![batch, hidden_size].into();
    Ok(Tensor::from((
        Storage::Hip(CudaStorage::wrap_cuda_slice(final_out, dev.clone())),
        out_shape,
    )))
}

// ================= hip-role grouped MoE entries =================
// Same contracts as the dev-typed entries below, but device-free: the Hip
// device comes from the weights and routing tables are plain tensors, so
// the core call sites stay device-agnostic.

fn wrap_hip_u32_table(slice: CudaSlice<u32>, dev: &CudaDevice, len: usize) -> Tensor {
    Tensor::from((
        Storage::Hip(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
        Shape::from(len),
    ))
}

/// Build expert dispatch tables on the hip role: expert_bounds plus
/// token ids sorted by expert (and the inverse source permutation).
pub fn moe_dispatch_build_hip(
    topk_ids_flat: &Tensor,
    total_assignments: usize,
    num_experts: usize,
    topk: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dev: &CudaDevice = match topk_ids_flat.device() {
        Device::Hip(dev) => dev,
        _ => candle_core::bail!("moe dispatch build requires Hip routing ids"),
    };
    let topk_ids_flat = topk_ids_flat.contiguous()?;
    if topk_ids_flat.elem_count() != total_assignments {
        candle_core::bail!("moe dispatch build routing does not match total assignments");
    }
    let expert_bounds = unsafe { dev.alloc::<u32>(num_experts + 1) }?;
    let sorted_token_ids = unsafe { dev.alloc::<u32>(total_assignments) }?;
    let sorted_source_ids = unsafe { dev.alloc::<u32>(total_assignments) }?;

    let cuda_stream = dev.cuda_stream();
    let stream = cuda_stream.cu_stream();
    let (dispatch_ws_ptr, _dispatch_ws_guard) = dispatch_workspace_ensure(dev, 2 * num_experts)?;

    {
        hip_u32_ptrs!(topk_ids_flat, &cuda_stream, topk_ptr, _topk_guard);
        let (bounds_ptr, _bounds_guard) = hip_slice_ptr(&expert_bounds, 0);
        let (sorted_ptr, _sorted_guard) = hip_slice_ptr(&sorted_token_ids, 0);
        let (source_ptr, _source_guard) = hip_slice_ptr(&sorted_source_ids, 0);
        let counts_ptr = dispatch_ws_ptr as *mut i32;
        let cursors_ptr = unsafe { counts_ptr.add(num_experts) };

        unsafe {
            ffi::launch_moe_dispatch(
                topk_ptr as *const i32,
                bounds_ptr as *mut i32,
                sorted_ptr as *mut i32,
                source_ptr as *mut i32,
                total_assignments as i32,
                num_experts as i32,
                topk as i32,
                counts_ptr,
                cursors_ptr,
                stream,
            );
        }
    }

    Ok((
        wrap_hip_u32_table(expert_bounds, dev, num_experts + 1),
        wrap_hip_u32_table(sorted_token_ids, dev, total_assignments),
        wrap_hip_u32_table(sorted_source_ids, dev, total_assignments),
    ))
}

/// Quantize input to Q8_1 on the hip role, returning an owned buffer.
pub fn quantize_input_q8_1_hip(xs: &Tensor) -> Result<(Tensor, usize, usize)> {
    let dev: &CudaDevice = match xs.device() {
        Device::Hip(dev) => dev,
        _ => candle_core::bail!("quantize_input_q8_1_hip requires a Hip tensor"),
    };
    let xs_contig = xs.contiguous()?;
    let num_rows = xs_contig.dim(0)?;
    let k = xs_contig.dim(1)?;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes = q8_1_bytes(num_rows, k_padded);

    let mut input_quant = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_tensor_into_q8_1(&xs_contig, &mut input_quant, dev)?;

    let out = Tensor::from((
        Storage::Hip(CudaStorage::wrap_cuda_slice(input_quant, dev.clone())),
        Shape::from(y_size_in_bytes),
    ));
    Ok((out, k, k_padded))
}

/// Grouped MoE GEMM on pre-quantized Q8_1 input, hip role.
#[allow(clippy::too_many_arguments)]
pub fn grouped_moe_gemm_prequantized_hip(
    qtensor: &QTensor,
    input_quant: &Tensor,
    k: usize,
    k_padded: usize,
    expert_bounds: &Tensor,
    sorted_token_ids: &Tensor,
    topk_weights: Option<&Tensor>,
    total_assignments: usize,
    topk: usize,
    num_experts: usize,
    input_dim1: usize,
) -> Result<Tensor> {
    let dtype = qtensor.dtype();
    let (_, n, k_w) = qtensor.shape().dims3()?;
    if k != k_w {
        candle_core::bail!("grouped MoE GEMM K mismatch");
    }
    if !indexed_moe_weight_dtype(dtype) {
        candle_core::bail!("unsupported dtype for grouped MoE GEMM: {dtype:?}");
    }

    let Device::Hip(dev) = qtensor.device() else {
        candle_core::bail!("grouped MoE GEMM requires Hip weights");
    };

    let topk_weights = match topk_weights {
        Some(tw) => Some(tw.to_dtype(DType::F32)?.contiguous()?),
        None => None,
    };
    let num_tokens = total_assignments / topk;
    let out_rows = if topk_weights.is_some() {
        num_tokens
    } else {
        total_assignments
    };
    let out = dev.alloc_zeros::<f32>(out_rows * n)?;

    let cuda_stream = dev.cuda_stream();
    let stream = cuda_stream.cu_stream();
    let weight_ptr = qtensor.device_ptr()? as *const std::ffi::c_void;

    {
        let (iq_storage, _) = input_quant.storage_and_layout();
        let Storage::Hip(iq_hip) = &*iq_storage else {
            candle_core::bail!("grouped MoE GEMM requires Hip quantized input");
        };
        let iq_slice = iq_hip.as_cuda_slice::<u8>()?;
        let (inputs_ptr, _ig) = hip_slice_ptr_on_stream(iq_slice, 0, &cuda_stream);
        hip_u32_ptrs!(expert_bounds, &cuda_stream, bounds_ptr, _bg);
        hip_u32_ptrs!(sorted_token_ids, &cuda_stream, sorted_ptr, _sg);
        let (out_ptr, _og) = hip_slice_ptr(&out, 0);
        let topk_w_ptr = match &topk_weights {
            Some(tw) => {
                let (tw_storage, tw_layout) = tw.storage_and_layout();
                let Storage::Hip(tw_hip) = &*tw_storage else {
                    candle_core::bail!("grouped MoE GEMM requires Hip routing weights");
                };
                let tw_slice = tw_hip.as_cuda_slice::<f32>()?;
                let (tw_ptr, _wg) =
                    hip_slice_ptr_on_stream(tw_slice, tw_layout.start_offset(), &cuda_stream);
                tw_ptr as *const f32
            }
            None => std::ptr::null(),
        };

        macro_rules! launch_grouped {
            ($entry:ident) => {
                ffi::$entry(
                    weight_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    bounds_ptr as *const i32,
                    sorted_ptr as *const i32,
                    topk_w_ptr,
                    out_ptr as *mut f32,
                    n as i32,
                    k as i32,
                    k_padded as i32,
                    num_experts as i32,
                    topk as i32,
                    input_dim1 as i32,
                    stream,
                )
            };
        }

        unsafe {
            match dtype {
                GgmlDType::Q8_0 => launch_grouped!(launch_moe_grouped_gemm_q8_0),
                GgmlDType::Q4_0 => launch_grouped!(launch_moe_grouped_gemm_q4_0),
                GgmlDType::Q4_1 => launch_grouped!(launch_moe_grouped_gemm_q4_1),
                GgmlDType::Q5_0 => launch_grouped!(launch_moe_grouped_gemm_q5_0),
                GgmlDType::Q5_1 => launch_grouped!(launch_moe_grouped_gemm_q5_1),
                GgmlDType::Q8_1 => launch_grouped!(launch_moe_grouped_gemm_q8_1),
                GgmlDType::Q2K => launch_grouped!(launch_moe_grouped_gemm_q2k),
                GgmlDType::Q3K => launch_grouped!(launch_moe_grouped_gemm_q3k),
                GgmlDType::Q4K => launch_grouped!(launch_moe_grouped_gemm_q4k),
                GgmlDType::Q5K => launch_grouped!(launch_moe_grouped_gemm_q5k),
                GgmlDType::Q6K => launch_grouped!(launch_moe_grouped_gemm_q6k),
                _ => candle_core::bail!("unsupported dtype: {dtype:?}"),
            }
        }
    }

    let out_shape: Shape = vec![out_rows, n].into();
    Ok(Tensor::from((
        Storage::Hip(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        out_shape,
    )))
}

/// Reduce flat per-assignment MoE outputs into per-token F32 outputs, hip role.
///
/// # Safety
/// `topk_weights` must hold `num_tokens * topk` F32 values on Hip.
pub unsafe fn moe_weighted_reduce_flat_hip(
    inputs: &Tensor,
    topk_weights: &Tensor,
    num_tokens: usize,
    topk: usize,
) -> Result<Tensor> {
    let expected_assignments = num_tokens.checked_mul(topk).ok_or_else(|| {
        candle_core::Error::msg("moe_weighted_reduce_flat_hip: route count overflow")
    })?;
    let (total_assignments, hidden) = inputs.dims2()?;
    if total_assignments != expected_assignments {
        candle_core::bail!(
            "moe_weighted_reduce_flat_hip: input rows {total_assignments} do not match num_tokens={num_tokens} * topk={topk}"
        );
    }
    if inputs.dtype() != DType::F32 {
        candle_core::bail!(
            "moe_weighted_reduce_flat_hip: input dtype must be F32, got {:?}",
            inputs.dtype()
        );
    }
    let dev: &CudaDevice = match inputs.device() {
        Device::Hip(dev) => dev,
        _ => candle_core::bail!("moe_weighted_reduce_flat_hip: input must live on Hip"),
    };

    let inputs = inputs.contiguous()?;
    let (storage, layout) = inputs.storage_and_layout();
    let Storage::Hip(hip) = &*storage else {
        candle_core::bail!("moe_weighted_reduce_flat_hip: input must live on Hip");
    };
    let input_slice = hip.as_cuda_slice::<f32>()?;
    let topk_weights = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .contiguous()?;
    if topk_weights.elem_count() != expected_assignments {
        candle_core::bail!("moe_weighted_reduce_flat_hip: weights do not match routing");
    }
    let (tw_storage, tw_layout) = topk_weights.storage_and_layout();
    let Storage::Hip(tw_hip) = &*tw_storage else {
        candle_core::bail!("moe_weighted_reduce_flat_hip: weights must live on Hip");
    };
    let tw_slice = tw_hip.as_cuda_slice::<f32>()?;

    let out = unsafe { dev.alloc::<f32>(num_tokens * hidden)? };
    let cuda_stream = dev.cuda_stream();
    let stream = cuda_stream.cu_stream();

    {
        let (input_ptr, _ig) =
            hip_slice_ptr_on_stream(input_slice, layout.start_offset(), &cuda_stream);
        let (weights_ptr, _wg) =
            hip_slice_ptr_on_stream(tw_slice, tw_layout.start_offset(), &cuda_stream);
        let (out_ptr, _og) = hip_slice_ptr(&out, 0);
        let status = ffi::launch_moe_weighted_reduce_flat(
            input_ptr as *const std::ffi::c_void,
            weights_ptr as *const f32,
            out_ptr as *mut std::ffi::c_void,
            num_tokens as i32,
            hidden as i32,
            topk as i32,
            stream,
        );
        check_hip_launch(status, "moe_weighted_reduce_flat")?;
    }

    Ok(Tensor::from((
        Storage::Hip(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from((num_tokens, hidden)),
    )))
}

/// Reduce flat per-assignment MoE outputs into BF16 per-token outputs, hip role.
///
/// # Safety
/// `topk_weights` must hold `num_tokens * topk` F32 values on Hip.
pub unsafe fn moe_weighted_reduce_flat_bf16_hip(
    inputs: &Tensor,
    topk_weights: &Tensor,
    num_tokens: usize,
    topk: usize,
) -> Result<Tensor> {
    let (total_assignments, hidden) = inputs.dims2()?;
    if total_assignments != num_tokens * topk {
        candle_core::bail!(
            "moe_weighted_reduce_flat_bf16_hip: input rows {total_assignments} do not match num_tokens={num_tokens} * topk={topk}"
        );
    }
    if inputs.dtype() != DType::F32 {
        candle_core::bail!(
            "moe_weighted_reduce_flat_bf16_hip: input dtype must be F32, got {:?}",
            inputs.dtype()
        );
    }
    let dev: &CudaDevice = match inputs.device() {
        Device::Hip(dev) => dev,
        _ => candle_core::bail!("moe_weighted_reduce_flat_bf16_hip: input must live on Hip"),
    };

    let inputs = inputs.contiguous()?;
    let (storage, layout) = inputs.storage_and_layout();
    let Storage::Hip(hip) = &*storage else {
        candle_core::bail!("moe_weighted_reduce_flat_bf16_hip: input must live on Hip");
    };
    let input_slice = hip.as_cuda_slice::<f32>()?;
    let topk_weights = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let (tw_storage, tw_layout) = topk_weights.storage_and_layout();
    let Storage::Hip(tw_hip) = &*tw_storage else {
        candle_core::bail!("moe_weighted_reduce_flat_bf16_hip: weights must live on Hip");
    };
    let tw_slice = tw_hip.as_cuda_slice::<f32>()?;

    let out = unsafe { dev.alloc::<half::bf16>(num_tokens * hidden)? };
    let cuda_stream = dev.cuda_stream();
    let stream = cuda_stream.cu_stream();

    {
        let (input_ptr, _ig) =
            hip_slice_ptr_on_stream(input_slice, layout.start_offset(), &cuda_stream);
        let (weights_ptr, _wg) =
            hip_slice_ptr_on_stream(tw_slice, tw_layout.start_offset(), &cuda_stream);
        let (out_ptr, _og) = hip_slice_ptr(&out, 0);
        let status = ffi::launch_moe_weighted_reduce_flat_bf16(
            input_ptr as *const std::ffi::c_void,
            weights_ptr as *const f32,
            out_ptr as *mut std::ffi::c_void,
            num_tokens as i32,
            hidden as i32,
            topk as i32,
            stream,
        );
        check_hip_launch(status, "moe_weighted_reduce_flat_bf16")?;
    }

    Ok(Tensor::from((
        Storage::Hip(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from((num_tokens, hidden)),
    )))
}

/// Reduce flat per-assignment MoE outputs keeping the input dtype, hip role.
pub fn moe_weighted_reduce_flat_same_dtype_hip(
    inputs: &Tensor,
    topk_weights: &Tensor,
    num_tokens: usize,
    topk: usize,
) -> Result<Tensor> {
    let routes = num_tokens
        .checked_mul(topk)
        .ok_or_else(|| candle_core::Error::msg("typed MoE reduction route count overflow"))?;
    if num_tokens == 0 || topk == 0 {
        candle_core::bail!("typed MoE reduction dimensions must be nonzero");
    }
    let dev: &CudaDevice = match inputs.device() {
        Device::Hip(dev) => dev,
        _ => candle_core::bail!("typed MoE reduction input must live on Hip"),
    };
    let topk_weights = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .contiguous()?;
    if topk_weights.elem_count() != routes {
        candle_core::bail!("typed MoE reduction weights do not match routing");
    }
    let (weights_storage, weights_layout) = topk_weights.storage_and_layout();
    let Storage::Hip(weights_hip) = &*weights_storage else {
        candle_core::bail!("typed MoE reduction weights must live on Hip");
    };
    let weights_slice = weights_hip.as_cuda_slice::<f32>()?;
    match inputs.dtype() {
        DType::F32 => unsafe {
            typed_reduce_hip::<f32>(
                inputs,
                weights_slice,
                weights_layout.start_offset(),
                num_tokens,
                topk,
                dev,
                ffi::launch_moe_weighted_reduce_flat,
                "moe_weighted_reduce_flat",
            )
        },
        DType::F16 => unsafe {
            typed_reduce_hip::<half::f16>(
                inputs,
                weights_slice,
                weights_layout.start_offset(),
                num_tokens,
                topk,
                dev,
                ffi::launch_moe_weighted_reduce_flat_f16_input,
                "moe_weighted_reduce_flat_f16_input",
            )
        },
        DType::BF16 => unsafe {
            typed_reduce_hip::<half::bf16>(
                inputs,
                weights_slice,
                weights_layout.start_offset(),
                num_tokens,
                topk,
                dev,
                ffi::launch_moe_weighted_reduce_flat_bf16_input,
                "moe_weighted_reduce_flat_bf16_input",
            )
        },
        dtype => candle_core::bail!("typed MoE reduction does not support {dtype:?}"),
    }
}

type TypedReduceHipFn = unsafe extern "C" fn(
    *const std::ffi::c_void,
    *const f32,
    *mut std::ffi::c_void,
    i32,
    i32,
    i32,
    *mut std::ffi::c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
unsafe fn typed_reduce_hip<T: CudaDType + DeviceRepr>(
    inputs: &Tensor,
    topk_weights: &CudaSlice<f32>,
    topk_weights_offset: usize,
    num_tokens: usize,
    topk: usize,
    dev: &CudaDevice,
    launch: TypedReduceHipFn,
    kernel: &str,
) -> Result<Tensor> {
    let expected_assignments = num_tokens
        .checked_mul(topk)
        .ok_or_else(|| candle_core::Error::msg(format!("{kernel}: route count overflow")))?;
    let (total_assignments, hidden) = inputs.dims2()?;
    if total_assignments != expected_assignments {
        candle_core::bail!(
            "{kernel}: input rows {total_assignments} do not match num_tokens={num_tokens} * topk={topk}"
        );
    }
    if hidden == 0 {
        candle_core::bail!("{kernel}: hidden dimension must be nonzero");
    }
    if hidden.div_ceil(MOE_REDUCE_THREADS) > CUDA_GRID_YZ_LIMIT {
        candle_core::bail!("{kernel}: hidden dimension exceeds the grid limit");
    }
    let inputs = inputs.contiguous()?;
    let (storage, layout) = inputs.storage_and_layout();
    let Storage::Hip(hip) = &*storage else {
        candle_core::bail!("{kernel}: input must live on Hip");
    };
    let input_slice = hip.as_cuda_slice::<T>()?;
    let output_len = num_tokens
        .checked_mul(hidden)
        .ok_or_else(|| candle_core::Error::msg(format!("{kernel}: output size overflow")))?;
    let mut out = unsafe { dev.alloc::<T>(output_len)? };
    let cuda_stream = dev.cuda_stream();
    let stream = cuda_stream.cu_stream();

    {
        let (input_ptr, _ig) =
            hip_slice_ptr_on_stream(input_slice, layout.start_offset(), &cuda_stream);
        let (weights_ptr, _wg) =
            hip_slice_ptr_on_stream(topk_weights, topk_weights_offset, &cuda_stream);
        let (out_ptr, _og) = hip_slice_ptr_mut_on_stream(&mut out, 0, &cuda_stream);
        let status = launch(
            input_ptr as *const std::ffi::c_void,
            weights_ptr as *const f32,
            out_ptr as *mut std::ffi::c_void,
            num_tokens as i32,
            hidden as i32,
            topk as i32,
            stream,
        );
        check_hip_launch(status, kernel)?;
    }

    Ok(Tensor::from((
        Storage::Hip(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from((num_tokens, hidden)),
    )))
}
// The entries below take the NVIDIA-role `CudaDevice` because the
// mistralrs-core call sites (`forward_grouped`, `forward_decode`) extract
// it via `as_cuda_device()` and never run on the hip role. They stay
// bailing until those call sites become role-aware (stage 2).

#[allow(dead_code)]
pub struct IndexedMoeLoraWeights<'a> {
    gate: &'a QTensor,
    up: &'a QTensor,
    down: &'a QTensor,
}

impl<'a> IndexedMoeLoraWeights<'a> {
    pub fn new(gate: &'a QTensor, up: &'a QTensor, down: &'a QTensor) -> Self {
        Self { gate, up, down }
    }
}

/// Hip-role routing for indexed MoE LoRA decode: plain tensors, the
/// device comes from the weights.
pub struct IndexedMoeRoutingHip<'a> {
    topk_ids: &'a Tensor,
    batch: usize,
    topk: usize,
    num_experts: usize,
}

impl<'a> IndexedMoeRoutingHip<'a> {
    pub fn new(topk_ids: &'a Tensor, batch: usize, topk: usize, num_experts: usize) -> Self {
        Self {
            topk_ids,
            batch,
            topk,
            num_experts,
        }
    }
}

const MOE_OUTPUT_F32: i32 = 0;
const MOE_OUTPUT_F16: i32 = 1;
const MOE_OUTPUT_BF16: i32 = 2;

fn moe_output_type_hip(dtype: DType) -> Option<i32> {
    match dtype {
        DType::F32 => Some(MOE_OUTPUT_F32),
        DType::F16 => Some(MOE_OUTPUT_F16),
        DType::BF16 => Some(MOE_OUTPUT_BF16),
        _ => None,
    }
}

fn q8_1_bytes_checked(num_rows: usize, k_padded: usize) -> Result<usize> {
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let num_blocks_per_row = k_padded
        .checked_div(q8_1_block_size)
        .ok_or_else(|| candle_core::Error::msg("indexed MoE LoRA block size overflow"))?;
    num_rows
        .checked_mul(num_blocks_per_row)
        .and_then(|elements| elements.checked_mul(q8_1_type_size))
        .ok_or_else(|| candle_core::Error::msg("indexed MoE LoRA size overflow"))
}

trait MoeLoraOutputHip: CudaDType + DeviceRepr {}

impl MoeLoraOutputHip for f32 {}
impl MoeLoraOutputHip for half::f16 {}
impl MoeLoraOutputHip for half::bf16 {}

type GateUpPairHipFn = unsafe extern "C" fn(
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *const u32,
    *mut std::ffi::c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *mut std::ffi::c_void,
) -> i32;

type LoraDownHipFn = unsafe extern "C" fn(
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *const u32,
    *mut std::ffi::c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *mut std::ffi::c_void,
) -> i32;

fn lora_gate_up_launcher_hip(dtype: GgmlDType) -> Option<GateUpPairHipFn> {
    match dtype {
        GgmlDType::Q8_0 => Some(ffi::launch_moe_gemv_gate_up_pair_q8_0_q8_1),
        GgmlDType::Q4_0 => Some(ffi::launch_moe_gemv_gate_up_pair_q4_0_q8_1),
        GgmlDType::Q4_1 => Some(ffi::launch_moe_gemv_gate_up_pair_q4_1_q8_1),
        GgmlDType::Q5_0 => Some(ffi::launch_moe_gemv_gate_up_pair_q5_0_q8_1),
        GgmlDType::Q5_1 => Some(ffi::launch_moe_gemv_gate_up_pair_q5_1_q8_1),
        GgmlDType::Q8_1 => Some(ffi::launch_moe_gemv_gate_up_pair_q8_1_q8_1),
        GgmlDType::Q2K => Some(ffi::launch_moe_gemv_gate_up_pair_q2k_q8_1),
        GgmlDType::Q3K => Some(ffi::launch_moe_gemv_gate_up_pair_q3k_q8_1),
        GgmlDType::Q4K => Some(ffi::launch_moe_gemv_gate_up_pair_q4k_q8_1),
        GgmlDType::Q5K => Some(ffi::launch_moe_gemv_gate_up_pair_q5k_q8_1),
        GgmlDType::Q6K => Some(ffi::launch_moe_gemv_gate_up_pair_q6k_q8_1),
        _ => None,
    }
}

fn lora_down_launcher_hip(dtype: GgmlDType) -> Option<LoraDownHipFn> {
    match dtype {
        GgmlDType::Q8_0 => Some(ffi::launch_moe_gemv_lora_down_q8_0_q8_1),
        GgmlDType::Q4_0 => Some(ffi::launch_moe_gemv_lora_down_q4_0_q8_1),
        GgmlDType::Q4_1 => Some(ffi::launch_moe_gemv_lora_down_q4_1_q8_1),
        GgmlDType::Q5_0 => Some(ffi::launch_moe_gemv_lora_down_q5_0_q8_1),
        GgmlDType::Q5_1 => Some(ffi::launch_moe_gemv_lora_down_q5_1_q8_1),
        GgmlDType::Q8_1 => Some(ffi::launch_moe_gemv_lora_down_q8_1_q8_1),
        GgmlDType::Q2K => Some(ffi::launch_moe_gemv_lora_down_q2k_q8_1),
        GgmlDType::Q3K => Some(ffi::launch_moe_gemv_lora_down_q3k_q8_1),
        GgmlDType::Q4K => Some(ffi::launch_moe_gemv_lora_down_q4k_q8_1),
        GgmlDType::Q5K => Some(ffi::launch_moe_gemv_lora_down_q5k_q8_1),
        GgmlDType::Q6K => Some(ffi::launch_moe_gemv_lora_down_q6k_q8_1),
        _ => None,
    }
}

pub struct IndexedMoeLoraDecodeHip<'a> {
    weights: IndexedMoeLoraWeights<'a>,
    routing: IndexedMoeRoutingHip<'a>,
    dev: CudaDevice,
    hidden: usize,
    intermediate: usize,
    gate_up_launch: GateUpPairHipFn,
    down_launch: LoraDownHipFn,
}

impl<'a> IndexedMoeLoraDecodeHip<'a> {
    pub fn new(
        weights: IndexedMoeLoraWeights<'a>,
        routing: IndexedMoeRoutingHip<'a>,
    ) -> Result<Option<Self>> {
        let gate_dtype = weights.gate.dtype();
        if weights.up.dtype() != gate_dtype {
            return Ok(None);
        }
        let Some(gate_up_launch) = lora_gate_up_launcher_hip(gate_dtype) else {
            return Ok(None);
        };
        let Some(down_launch) = lora_down_launcher_hip(weights.down.dtype()) else {
            return Ok(None);
        };
        let (num_experts, intermediate, hidden) = weights.gate.shape().dims3()?;
        if num_experts == 0 || intermediate == 0 || hidden == 0 {
            candle_core::bail!("indexed MoE LoRA weight dimensions must be nonzero");
        }
        if num_experts != routing.num_experts {
            candle_core::bail!("indexed MoE LoRA expert count does not match routing");
        }
        if weights.up.shape().dims3()? != (num_experts, intermediate, hidden)
            || weights.down.shape().dims3()? != (num_experts, hidden, intermediate)
        {
            candle_core::bail!("indexed MoE LoRA weight geometry does not match");
        }
        let Device::Hip(dev) = weights.gate.device() else {
            candle_core::bail!("indexed MoE LoRA weights must live on Hip");
        };
        for tensor in [weights.up, weights.down] {
            let Device::Hip(weight_dev) = tensor.device() else {
                candle_core::bail!("indexed MoE LoRA weights must live on Hip");
            };
            if weight_dev.id() != dev.id() {
                candle_core::bail!("indexed MoE LoRA weights must share a Hip device");
            }
        }
        let Device::Hip(routes_dev) = routing.topk_ids.device() else {
            candle_core::bail!("indexed MoE LoRA routes must live on Hip");
        };
        if routes_dev.id() != dev.id() {
            candle_core::bail!("indexed MoE LoRA routes must share a Hip device");
        }
        if routing.batch == 0 || routing.topk == 0 {
            candle_core::bail!("indexed MoE LoRA routing dimensions must be nonzero");
        }
        let routes = routing
            .batch
            .checked_mul(routing.topk)
            .ok_or_else(|| candle_core::Error::msg("indexed MoE LoRA route count overflow"))?;
        if routing.topk_ids.elem_count() < routes {
            candle_core::bail!("indexed MoE LoRA route buffer is too small");
        }
        for dim in [
            num_experts,
            intermediate,
            hidden,
            routing.batch,
            routing.topk,
        ] {
            i32::try_from(dim)?;
        }
        i32::try_from(pad(hidden, MATRIX_ROW_PADDING))?;
        i32::try_from(pad(intermediate, MATRIX_ROW_PADDING))?;
        intermediate
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::msg("indexed MoE LoRA output size overflow"))?;

        Ok(Some(Self {
            weights,
            routing,
            dev,
            hidden,
            intermediate,
            gate_up_launch,
            down_launch,
        }))
    }

    fn validate_input(&self, input: &Tensor, rows: usize, features: usize) -> Result<Tensor> {
        if input.dims2()? != (rows, features) {
            candle_core::bail!("indexed MoE LoRA input shape does not match weights");
        }
        let Device::Hip(input_dev) = input.device() else {
            candle_core::bail!("indexed MoE LoRA input must live on Hip");
        };
        if input_dev.id() != self.dev.id() {
            candle_core::bail!("indexed MoE LoRA input must share a Hip device");
        }
        input.contiguous()
    }

    fn gate_up_t<T: MoeLoraOutputHip>(&self, input: &Tensor, output_type: i32) -> Result<Tensor> {
        let input = self.validate_input(input, self.routing.batch, self.hidden)?;
        let k_padded = pad(self.hidden, MATRIX_ROW_PADDING);
        let q8_bytes = q8_1_bytes_checked(self.routing.batch, k_padded)?;
        let mut q8 = u8_workspace_ensure(&self.dev, q8_bytes)?;
        quantize_tensor_into_q8_1(&input, &mut q8.slice, &self.dev)?;

        let output_len = self
            .routing
            .batch
            .checked_mul(self.routing.topk)
            .and_then(|routes| routes.checked_mul(self.intermediate))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| candle_core::Error::msg("indexed MoE LoRA output size overflow"))?;
        let mut output = unsafe { self.dev.alloc::<T>(output_len)? };
        let cuda_stream = self.dev.cuda_stream();
        let stream = cuda_stream.cu_stream();
        let gate_ptr = self.weights.gate.device_ptr()? as *const std::ffi::c_void;
        let up_ptr = self.weights.up.device_ptr()? as *const std::ffi::c_void;

        {
            let (input_ptr, _ig) = hip_slice_ptr(&q8.slice, 0);
            hip_u32_ptrs!(self.routing.topk_ids, &cuda_stream, ids_ptr, _idg);
            let (output_ptr, _og) = hip_slice_ptr_mut_on_stream(&mut output, 0, &cuda_stream);
            let status = unsafe {
                (self.gate_up_launch)(
                    gate_ptr,
                    up_ptr,
                    input_ptr as *const std::ffi::c_void,
                    ids_ptr as *const u32,
                    output_ptr as *mut std::ffi::c_void,
                    self.intermediate as i32,
                    self.hidden as i32,
                    self.routing.batch as i32,
                    self.routing.topk as i32,
                    k_padded as i32,
                    self.routing.num_experts as i32,
                    output_type,
                    stream,
                )
            };
            check_hip_launch(status, "moe_gemv_gate_up_pair")?;
        }

        Ok(Tensor::from((
            Storage::Hip(CudaStorage::wrap_cuda_slice(output, self.dev.clone())),
            Shape::from((self.routing.batch, self.routing.topk, self.intermediate * 2)),
        )))
    }

    pub fn gate_up(&self, input: &Tensor) -> Result<Option<Tensor>> {
        let Some(output_type) = moe_output_type_hip(input.dtype()) else {
            return Ok(None);
        };
        match input.dtype() {
            DType::F32 => self.gate_up_t::<f32>(input, output_type).map(Some),
            DType::F16 => self.gate_up_t::<half::f16>(input, output_type).map(Some),
            DType::BF16 => self.gate_up_t::<half::bf16>(input, output_type).map(Some),
            _ => Ok(None),
        }
    }

    fn down_t<T: MoeLoraOutputHip>(&self, input: &Tensor, output_type: i32) -> Result<Tensor> {
        let routes = self.routing.batch * self.routing.topk;
        let input = self.validate_input(input, routes, self.intermediate)?;
        let k_padded = pad(self.intermediate, MATRIX_ROW_PADDING);
        let q8_bytes = q8_1_bytes_checked(routes, k_padded)?;
        let mut q8 = u8_workspace_ensure(&self.dev, q8_bytes)?;
        quantize_tensor_into_q8_1(&input, &mut q8.slice, &self.dev)?;

        let output_len = routes
            .checked_mul(self.hidden)
            .ok_or_else(|| candle_core::Error::msg("indexed MoE LoRA output size overflow"))?;
        let mut output = unsafe { self.dev.alloc::<T>(output_len)? };
        let cuda_stream = self.dev.cuda_stream();
        let stream = cuda_stream.cu_stream();
        let down_ptr = self.weights.down.device_ptr()? as *const std::ffi::c_void;

        {
            let (input_ptr, _ig) = hip_slice_ptr(&q8.slice, 0);
            hip_u32_ptrs!(self.routing.topk_ids, &cuda_stream, ids_ptr, _idg);
            let (output_ptr, _og) = hip_slice_ptr_mut_on_stream(&mut output, 0, &cuda_stream);
            let status = unsafe {
                (self.down_launch)(
                    down_ptr,
                    input_ptr as *const std::ffi::c_void,
                    ids_ptr as *const u32,
                    output_ptr as *mut std::ffi::c_void,
                    self.hidden as i32,
                    self.intermediate as i32,
                    self.routing.batch as i32,
                    self.routing.topk as i32,
                    k_padded as i32,
                    self.routing.num_experts as i32,
                    output_type,
                    stream,
                )
            };
            check_hip_launch(status, "moe_gemv_lora_down")?;
        }

        Ok(Tensor::from((
            Storage::Hip(CudaStorage::wrap_cuda_slice(output, self.dev.clone())),
            Shape::from((self.routing.batch, self.routing.topk, self.hidden)),
        )))
    }

    pub fn down(&self, input: &Tensor) -> Result<Option<Tensor>> {
        let Some(output_type) = moe_output_type_hip(input.dtype()) else {
            return Ok(None);
        };
        match input.dtype() {
            DType::F32 => self.down_t::<f32>(input, output_type).map(Some),
            DType::F16 => self.down_t::<half::f16>(input, output_type).map(Some),
            DType::BF16 => self.down_t::<half::bf16>(input, output_type).map(Some),
            _ => Ok(None),
        }
    }
}

use candle_core::cuda::{cudarc::driver::CudaSlice as NvCudaSlice, CudaDevice as NvCudaDevice};

#[allow(dead_code)]
pub struct IndexedMoeRouting<'a> {
    topk_ids: &'a NvCudaSlice<u32>,
    batch: usize,
    topk: usize,
    num_experts: usize,
    dev: &'a NvCudaDevice,
}

impl<'a> IndexedMoeRouting<'a> {
    pub fn new(
        topk_ids: &'a NvCudaSlice<u32>,
        batch: usize,
        topk: usize,
        num_experts: usize,
        dev: &'a NvCudaDevice,
    ) -> Self {
        Self {
            topk_ids,
            batch,
            topk,
            num_experts,
            dev,
        }
    }
}

#[allow(dead_code)]
pub struct IndexedMoeLoraDecode<'a> {
    weights: IndexedMoeLoraWeights<'a>,
    routing: IndexedMoeRouting<'a>,
    hidden: usize,
    intermediate: usize,
}

impl<'a> IndexedMoeLoraDecode<'a> {
    pub fn new(
        weights: IndexedMoeLoraWeights<'a>,
        routing: IndexedMoeRouting<'a>,
    ) -> Result<Option<Self>> {
        let _ = (weights, routing);
        Ok(None)
    }

    pub fn gate_up(&self, _input: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    pub fn down(&self, _input: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

macro_rules! hip_bail {
    ($name:literal) => {
        candle_core::bail!(concat!(
            $name,
            " needs role-aware call sites in mistralrs-core (stage 2 of the MoE port)"
        ))
    };
}

pub fn moe_dispatch_build(
    _topk_ids_flat: &NvCudaSlice<u32>,
    _total_assignments: usize,
    _num_experts: usize,
    _topk: usize,
    _dev: &NvCudaDevice,
) -> Result<(NvCudaSlice<u32>, NvCudaSlice<u32>, NvCudaSlice<u32>)> {
    hip_bail!("cuda::moe_dispatch_build")
}

pub unsafe fn moe_weighted_reduce_flat(
    _inputs: &Tensor,
    _topk_weights: *const f32,
    _num_tokens: usize,
    _topk: usize,
    _dev: &NvCudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::moe_weighted_reduce_flat")
}

pub unsafe fn moe_weighted_reduce_flat_bf16(
    _inputs: &Tensor,
    _topk_weights: *const f32,
    _num_tokens: usize,
    _topk: usize,
    _dev: &NvCudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::moe_weighted_reduce_flat_bf16")
}

pub fn moe_weighted_reduce_flat_same_dtype(
    _inputs: &Tensor,
    _topk_weights: &Tensor,
    _num_tokens: usize,
    _topk: usize,
    _dev: &NvCudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::moe_weighted_reduce_flat_same_dtype")
}

pub fn quantize_input_q8_1(
    _xs: &Tensor,
    _dev: &NvCudaDevice,
) -> Result<(NvCudaSlice<u8>, usize, usize)> {
    hip_bail!("cuda::quantize_input_q8_1")
}

pub fn grouped_moe_gemm_prequantized(
    _qtensor: &QTensor,
    _input_quant: &NvCudaSlice<u8>,
    _k: usize,
    _k_padded: usize,
    _expert_bounds: &NvCudaSlice<u32>,
    _sorted_token_ids: &NvCudaSlice<u32>,
    _topk_weights: Option<(*const f32, usize)>,
    _total_assignments: usize,
    _topk: usize,
    _num_experts: usize,
    _input_dim1: usize,
    _dev: &NvCudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::grouped_moe_gemm_prequantized")
}

pub unsafe fn indexed_moe_fused_decode(
    _gate_qt: &QTensor,
    _up_qt: &QTensor,
    _down_qt: &QTensor,
    _xs_flat: &Tensor,
    _topk_ids: &NvCudaSlice<u32>,
    _topk_weights_ptr: *const f32,
    _batch: usize,
    _topk: usize,
    _act_type: i32,
    _dev: &NvCudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::indexed_moe_fused_decode")
}
