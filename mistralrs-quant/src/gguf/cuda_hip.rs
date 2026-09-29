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

use candle_core::hip_backend::cudarc::driver::{CudaSlice, DevicePtr};
use candle_core::hip_backend::{CudaDevice, CudaStorage};
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::utils::{hip_slice_ptr, hip_slice_ptr_mut_on_stream, hip_slice_ptr_on_stream};

pub const ACT_GELU_PYTORCH_TANH: i32 = 0;
pub const ACT_SILU: i32 = 1;

// Constants matching candle's quantized CUDA implementation
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

struct U8WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: candle_core::hip_backend::DeviceId,
    stream: usize,
}

type U8WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<U8WorkspaceSlot>>>;

static MOE_Q8_WORKSPACE: OnceLock<U8WsMap> = OnceLock::new();

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
/// Quantize a [rows, k] activation to Q8_1 in the per-device workspace.
/// The returned guard keeps the workspace slot (and its buffer) alive
/// until the consumer kernel has been launched.
fn quantize_input_q8_1_hip(
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
    let input_quant = &mut slot.slice;

    {
        // Use fused half->Q8_1 kernels when input is BF16/F16 (avoids separate cast kernel)
        if xs_contig.dtype() == candle_core::DType::BF16
            || xs_contig.dtype() == candle_core::DType::F16
        {
            let (xs_storage, xs_layout) = xs_contig.storage_and_layout();
            let Storage::Hip(xs_hip) = &*xs_storage else {
                candle_core::bail!("expected Hip tensor");
            };
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
            quantize_q8_1(xs_slice, input_quant, k, num_rows, dev)?;
        }
    }

    drop(input_quant);
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
    let (_quant_ws_guard, inputs_ptr, quant_k, k_padded) = quantize_input_q8_1_hip(&input, dev)?;
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

// ================= stubs: dev-typed entries =================
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
