//! NVIDIA-role twin of the fused MoE decode path for dual builds.
//!
//! Same flow as `cuda.rs`'s `indexed_moe_fused_decode`, bound to the
//! upstream `cuda_backend` device types and dlsym-ing the nvcc companion
//! (`ffi_cuda`). Scope: the decode path only - quantize to Q8_1, fused
//! gate_up with activation, down projection with topk-weighted
//! aggregation. Grouped prefill and the gather path stay on their
//! existing (eager / hip) routes until their companion slices land.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::ffi_cuda;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, SyncOnDrop,
};
use candle_core::cuda_backend::{CudaDevice, CudaStorage};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Result, Shape, Storage, Tensor};

pub const MATRIX_ROW_PADDING: usize = 512;
const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;

struct U8WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

struct F32WorkspaceSlot {
    slice: CudaSlice<f32>,
    cap: usize,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: candle_core::cuda_backend::DeviceId,
    stream: usize,
}

type U8WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<U8WorkspaceSlot>>>;
type F32WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<F32WorkspaceSlot>>>;

static MOE_DECODE_Q8_WORKSPACE: OnceLock<U8WsMap> = OnceLock::new();
static MOE_DECODE_F32_WORKSPACE: OnceLock<F32WsMap> = OnceLock::new();

fn workspace_key(dev: &CudaDevice) -> WorkspaceKey {
    WorkspaceKey {
        device: dev.id(),
        stream: dev.cuda_stream().cu_stream() as usize,
    }
}

fn u8_workspace_ensure(
    ws: &'static OnceLock<U8WsMap>,
    dev: &CudaDevice,
    len: usize,
) -> Result<std::sync::MutexGuard<'static, U8WorkspaceSlot>> {
    let len = len.max(1);
    let map = ws.get_or_init(|| Mutex::new(HashMap::new()));
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
    ws: &'static OnceLock<F32WsMap>,
    dev: &CudaDevice,
    len: usize,
) -> Result<std::sync::MutexGuard<'static, F32WorkspaceSlot>> {
    let len = len.max(1);
    let map = ws.get_or_init(|| Mutex::new(HashMap::new()));
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

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn q8_1_bytes(num_rows: usize, k_padded: usize) -> usize {
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let num_blocks_per_row = k_padded / q8_1_block_size;
    num_rows * num_blocks_per_row * q8_1_type_size
}

fn slice_ptr<T: DeviceRepr>(v: &CudaSlice<T>, lo: usize) -> (u64, SyncOnDrop<'_>) {
    slice_ptr_on_stream(v, lo, v.stream())
}

fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

fn slice_ptr_mut_on_stream<'a, T: DeviceRepr>(
    v: &'a mut CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

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
    let stream = cuda_stream.cu_stream() as *mut std::ffi::c_void;

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

        let (src_ptr, _src_guard) = slice_ptr_on_stream(src, src_start_elem, &cuda_stream);
        let (dst_ptr, _dst_guard) = slice_ptr_mut_on_stream(dst, dst_start_byte, &cuda_stream);

        unsafe {
            ffi_cuda::cuda_launch_quantize_q8_1(
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

/// Fused MoE decode for the NVIDIA role of a dual build: quantizes the
/// activations to Q8_1 once, runs the fused gate+up+activation GEMV per
/// (token, expert) pair, then the down projection with topk-weighted
/// aggregation. All device-side, so it is CUDA-graph capturable.
#[allow(clippy::too_many_arguments)]
pub unsafe fn indexed_moe_fused_decode_cuda(
    gate_qt: &QTensor,
    up_qt: &QTensor,
    down_qt: &QTensor,
    xs_flat: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    batch: usize,
    topk: usize,
    act_type: i32,
) -> Result<Tensor> {
    let topk_ids_flat = topk_ids.flatten_all()?.contiguous()?;
    let (ti_storage, ti_layout) = topk_ids_flat.storage_and_layout();
    let Storage::Cuda(ti_cuda) = &*ti_storage else {
        candle_core::bail!("fused MoE decode expects CUDA topk ids");
    };
    let ti_u32_slice = ti_cuda.as_cuda_slice::<u32>()?;
    assert!(ti_layout.start_offset() == 0, "expected contiguous tensor");

    let tw_f32 = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let (tw_storage, tw_layout) = tw_f32.storage_and_layout();
    let Storage::Cuda(tw_cuda) = &*tw_storage else {
        candle_core::bail!("fused MoE decode expects CUDA topk weights");
    };
    let tw_slice = tw_cuda.as_cuda_slice::<f32>()?;
    let cuda_stream = tw_cuda.device().cuda_stream();
    let (topk_weights_ptr, _twg) =
        slice_ptr_on_stream(tw_slice, tw_layout.start_offset(), &cuda_stream);

    let dev = tw_cuda.device();
    indexed_moe_fused_decode_cuda_inner(
        gate_qt,
        up_qt,
        down_qt,
        xs_flat,
        ti_u32_slice,
        topk_weights_ptr as *const f32,
        batch,
        topk,
        act_type,
        dev,
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn indexed_moe_fused_decode_cuda_inner(
    gate_qt: &QTensor,
    up_qt: &QTensor,
    down_qt: &QTensor,
    xs_flat: &Tensor,
    topk_ids: &CudaSlice<u32>,
    topk_weights_ptr: *const f32,
    batch: usize,
    topk: usize,
    act_type: i32,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let gate_up_dtype = gate_qt.dtype();
    assert!(up_qt.dtype() == gate_up_dtype);
    let down_dtype = down_qt.dtype();

    let (_, n_gate, k_gate) = gate_qt.shape().dims3()?;
    let (_, n_down, k_down) = down_qt.shape().dims3()?;

    // gate and up: [E, intermediate, hidden], down: [E, hidden, intermediate]
    let hidden_size = k_gate;
    let intermediate_size = n_gate;
    assert!(n_down == hidden_size);
    assert!(k_down == intermediate_size);

    let xs_contig = xs_flat.contiguous()?;
    let input_rows = xs_contig.dim(0)?;
    let k = xs_contig.dim(1)?;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let input_q8_bytes = q8_1_bytes(input_rows, k_padded);
    let intermediate_rows = batch * topk;
    let k_down_padded = pad(intermediate_size, MATRIX_ROW_PADDING);
    let intermediate_q8_bytes = q8_1_bytes(intermediate_rows, k_down_padded);

    let mut q8_workspace = u8_workspace_ensure(
        &MOE_DECODE_Q8_WORKSPACE,
        dev,
        input_q8_bytes.max(intermediate_q8_bytes),
    )?;

    // Step 1: Quantize input to Q8_1 (shared between gate and up).
    // The role twin takes the already-contiguous activation and a caller
    // that guarantees f32 (the dispatch narrows the dtype first).
    let xs_f32 = xs_contig.to_dtype(DType::F32)?;
    let (xs_storage, xs_layout) = xs_f32.storage_and_layout();
    let xs_cuda = match &*xs_storage {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("expected CUDA tensor"),
    };
    let xs_slice = xs_cuda.as_cuda_slice::<f32>()?;
    assert!(xs_layout.start_offset() == 0);
    quantize_q8_1(xs_slice, &mut q8_workspace.slice, k, input_rows, dev)?;

    let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    // Step 2: Fused gate+up+activation+multiply
    let gate_up_outsize = batch * topk * intermediate_size;
    let gate_up_workspace = f32_workspace_ensure(&MOE_DECODE_F32_WORKSPACE, dev, gate_up_outsize)?;

    let gate_ptr = gate_qt.device_ptr()? as *const std::ffi::c_void;
    let up_ptr = up_qt.device_ptr()? as *const std::ffi::c_void;

    {
        let (inp_ptr, _ig) = slice_ptr(&q8_workspace.slice, 0);
        let (ids_ptr, _idg) = slice_ptr(topk_ids, 0);
        let (out_ptr, _og) = slice_ptr(&gate_up_workspace.slice, 0);

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

        let launch_fn: FusedGateUpFn = match gate_up_dtype {
            GgmlDType::Q8_0 => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q8_0_q8_1,
            GgmlDType::Q4_0 => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q4_0_q8_1,
            GgmlDType::Q4_1 => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q4_1_q8_1,
            GgmlDType::Q5_0 => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q5_0_q8_1,
            GgmlDType::Q5_1 => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q5_1_q8_1,
            GgmlDType::Q8_1 => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q8_1_q8_1,
            GgmlDType::Q2K => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q2k_q8_1,
            GgmlDType::Q3K => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q3k_q8_1,
            GgmlDType::Q4K => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q4k_q8_1,
            GgmlDType::Q5K => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q5k_q8_1,
            GgmlDType::Q6K => ffi_cuda::cuda_launch_moe_gemv_fused_gate_up_q6k_q8_1,
            _ => candle_core::bail!("unsupported dtype for fused MoE decode: {gate_up_dtype:?}"),
        };

        unsafe {
            launch_fn(
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
            );
        }
    }
    // Step 3: Quantize the intermediate output for down projection.
    // gate_up_out is [batch * topk, intermediate_size] in f32; the input Q8_1
    // bytes and the intermediate Q8_1 bytes share the q8 workspace (sized to
    // the max), so the read (f32 workspace) and write (q8 workspace) never
    // alias.
    quantize_q8_1(
        &gate_up_workspace.slice,
        &mut q8_workspace.slice,
        intermediate_size,
        intermediate_rows,
        dev,
    )?;
    drop(gate_up_workspace);

    // Step 4: Fused down+aggregate with topk_weights and atomicAdd
    let final_outsize = batch * hidden_size;
    let final_out = unsafe { dev.alloc_zeros::<f32>(final_outsize)? };

    let down_ptr = down_qt.device_ptr()? as *const std::ffi::c_void;

    {
        let (inp_ptr, _ig) = slice_ptr(&q8_workspace.slice, 0);
        let (ids_ptr, _idg) = slice_ptr(topk_ids, 0);
        let (out_ptr, _og) = slice_ptr(&final_out, 0);

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

        let launch_fn: DownAggregateFn = match down_dtype {
            GgmlDType::Q8_0 => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q8_0_q8_1,
            GgmlDType::Q4_0 => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q4_0_q8_1,
            GgmlDType::Q4_1 => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q4_1_q8_1,
            GgmlDType::Q5_0 => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q5_0_q8_1,
            GgmlDType::Q5_1 => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q5_1_q8_1,
            GgmlDType::Q8_1 => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q8_1_q8_1,
            GgmlDType::Q2K => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q2k_q8_1,
            GgmlDType::Q3K => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q3k_q8_1,
            GgmlDType::Q4K => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q4k_q8_1,
            GgmlDType::Q5K => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q5k_q8_1,
            GgmlDType::Q6K => ffi_cuda::cuda_launch_moe_gemv_down_aggregate_q6k_q8_1,
            _ => candle_core::bail!("unsupported dtype for fused MoE decode: {down_dtype:?}"),
        };

        unsafe {
            launch_fn(
                down_ptr,
                inp_ptr as *const std::ffi::c_void,
                ids_ptr as *const u32,
                topk_weights_ptr,
                out_ptr as *mut f32,
                hidden_size as i32,
                intermediate_size as i32,
                batch as i32,
                topk as i32,
                k_down_padded as i32,
                stream,
            );
        }
    }

    let out_shape: Shape = vec![batch, hidden_size].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(final_out, dev.clone())),
        out_shape,
    )))
}

#[cfg(all(test, feature = "cuda", feature = "rocm"))]
mod tests {
    use super::*;

    fn w_alike(w: &Tensor, experts: usize) -> Vec<Tensor> {
        std::iter::repeat_with(|| w.clone()).take(experts).collect()
    }

    fn qtensor(
        rows: usize,
        cols: usize,
        experts: usize,
        phase: f32,
        dev: &Tensor,
    ) -> Result<QTensor> {
        let host: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * phase).sin() * 0.5).abs())
            .collect();
        let w = Tensor::from_vec(host, (rows, cols), &candle_core::Device::Cpu)?;
        let stacked = Tensor::stack(&w_alike(&w, experts), 0)?.contiguous()?;
        candle_core::quantized::QTensor::quantize_onto(&stacked, GgmlDType::Q4_0, dev.device())
    }

    /// Capture probe for the fused decode path: eager warm-up, then the
    /// same twin call inside a stream capture; replay must match the eager
    /// output. Isolates the nvcc MoE companion kernels as a graph node.
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn fused_decode_capture_launch_matches_eager() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;

        const EXPERTS: usize = 3;
        const HIDDEN: usize = 64;
        const INTERMEDIATE: usize = 96;
        const TOKENS: usize = 2;
        const TOPK: usize = 2;

        let dev = Tensor::zeros((1,), DType::F32, &candle_core::Device::new_cuda(0)?)?;
        let device = dev.device();
        let gate = qtensor(INTERMEDIATE, HIDDEN, EXPERTS, 0.1, &dev)?;
        let up = qtensor(INTERMEDIATE, HIDDEN, EXPERTS, 0.3, &dev)?;
        let down = qtensor(HIDDEN, INTERMEDIATE, EXPERTS, 0.7, &dev)?;

        let xs_host: Vec<f32> = (0..TOKENS * HIDDEN)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let xs = Tensor::from_vec(xs_host, (TOKENS, HIDDEN), device)?;
        let ids = Tensor::from_vec(
            (0..TOKENS * TOPK)
                .map(|i| (i % EXPERTS) as u32)
                .collect::<Vec<_>>(),
            (TOKENS, TOPK),
            device,
        )?;
        let weights = Tensor::from_vec(vec![0.5f32; TOKENS * TOPK], (TOKENS, TOPK), device)?;

        let eager = unsafe {
            indexed_moe_fused_decode_cuda(&gate, &up, &down, &xs, &ids, &weights, TOKENS, TOPK, 0)?
        };
        device.synchronize()?;

        let cuda_dev = match device {
            candle_core::Device::Cuda(d) => d,
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
        let replayed = unsafe {
            indexed_moe_fused_decode_cuda(&gate, &up, &down, &xs, &ids, &weights, TOKENS, TOPK, 0)?
        };
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
        device.synchronize()?;

        let eager_v = eager.to_vec2::<f32>()?;
        let replay_v = replayed.to_vec2::<f32>()?;
        let max_diff = eager_v
            .iter()
            .zip(replay_v.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-3, "eager vs replay diff {max_diff}");
        Ok(())
    }
}
