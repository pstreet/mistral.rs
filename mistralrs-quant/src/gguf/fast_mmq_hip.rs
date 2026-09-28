//! Dual-build (`cuda+rocm`) Hip implementation of [`super::fast_mmq`].
//!
//! Same contract as `fast_mmq.rs`, bound to the hip backend types with the
//! kernels resolved through the companion plugin. Handles batch > 8
//! (complement to `fast_mmvq_hip` which handles batch 1-8). The prefill
//! dispatch routes here for Hip weights; MoE grouped entries stay on the
//! dequant/CPU fallback (grouped-fusion follow-up).

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use candle_core::hip_backend::{
    cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, SyncOnDrop},
    CudaDType, CudaDevice, CudaStorage, DeviceId,
};
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, GpuArch, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::{
    utils::{hip_slice_ptr_mut_on_stream, hip_slice_ptr_on_stream},
    GluActivationType,
};

const QK8_1: usize = 32;
const BLOCK_Q8_1_MMQ_SIZE: usize = 4 * QK8_1 + 4 * 4; // 128 qs + 16 scale bytes = 144
const MATRIX_ROW_PADDING: usize = 512;
const MMQ_X_MAX: usize = 128;
const MMQ_Y_MAX: usize = 128;

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

fn wrap_hip_output<T: CudaDType + DeviceRepr>(
    out: CudaSlice<T>,
    dev: &CudaDevice,
    shape: Shape,
) -> Tensor {
    Tensor::from((
        Storage::Hip(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        shape,
    ))
}

// Mirror of fast_mmq::supports - keep in sync.
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

// Mirror of fast_mmq::dequant_handoff_rows - keep in sync.
pub fn dequant_handoff_rows(dtype: GgmlDType, arch: GpuArch) -> Option<usize> {
    if matches!(arch, GpuArch::Rocm { gfx: (11, _, _) })
        && matches!(dtype, GgmlDType::Q2K | GgmlDType::Q6K)
    {
        Some(candle_core::quantized::cuda::dequant_f16_min_rows())
    } else {
        None
    }
}

// Mirror of fast_mmq::batch_supported - keep in sync.
pub fn batch_supported(dtype: GgmlDType, flat_batch: usize, arch: GpuArch) -> bool {
    if !supports(dtype) {
        return false;
    }
    match dequant_handoff_rows(dtype, arch) {
        Some(max) => flat_batch <= max,
        None => true,
    }
}

pub fn device_arch(device: &Device) -> Result<GpuArch> {
    match device {
        Device::Cuda(dev) => Ok(GpuArch::resolve(dev)?),
        Device::Hip(dev) => Ok(GpuArch::resolve_hip(dev)?),
        _ => candle_core::bail!("fast_mmq device_arch requires a GPU device"),
    }
}

/// qk (block quantization size) per dtype.
fn qk_for(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            32
        }
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 256,
        _ => unreachable!(),
    }
}

// ds_layout mapping: which Q8_1_mmq scale layout to use per weight type.
// D4 = scale only, DS4 = scale+partial_sum, D2S6 = 2 scales + 6 partial_sums
enum DsLayout {
    D4,
    DS4,
    D2S6,
}

fn ds_layout_for(dtype: GgmlDType) -> DsLayout {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 => DsLayout::DS4,
        GgmlDType::Q5_0 => DsLayout::D4,
        GgmlDType::Q5_1 => DsLayout::DS4,
        GgmlDType::Q8_0 => DsLayout::D4,
        GgmlDType::Q2K => DsLayout::D2S6,
        GgmlDType::Q3K => DsLayout::D4,
        GgmlDType::Q4K | GgmlDType::Q5K => DsLayout::DS4,
        GgmlDType::Q6K => DsLayout::D4,
        _ => unreachable!(),
    }
}

type QuantizeLauncher = unsafe extern "C" fn(
    x: *const std::ffi::c_void,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    s02: i64,
    s03: i64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    stream: *mut std::ffi::c_void,
);

type QuantizeGluF32Launcher = unsafe extern "C" fn(
    gate: *const f32,
    up: *const f32,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    ne00: i64,
    s01: i64,
    ne0: i64,
    ne1: i64,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

type QuantizeGluLauncher = unsafe extern "C" fn(
    gate: *const std::ffi::c_void,
    up: *const std::ffi::c_void,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    ne0: i64,
    ne1: i64,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

fn quantize_launcher(layout: DsLayout) -> QuantizeLauncher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_q8_1_D4,
        DsLayout::DS4 => ffi::launch_mmq_quantize_q8_1_DS4,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_q8_1_D2S6,
    }
}

fn quantize_glu_f32_launcher(layout: DsLayout) -> QuantizeGluF32Launcher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_glu_q8_1_D4_f32,
        DsLayout::DS4 => ffi::launch_mmq_quantize_glu_q8_1_DS4_f32,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_glu_q8_1_D2S6_f32,
    }
}

fn quantize_glu_launcher(layout: DsLayout) -> QuantizeGluLauncher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_glu_q8_1_D4,
        DsLayout::DS4 => ffi::launch_mmq_quantize_glu_q8_1_DS4,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_glu_q8_1_D2S6,
    }
}

type MmqLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_y: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
    type_dst: i32,
    stream: *mut std::ffi::c_void,
);

fn mmq_launcher(dtype: GgmlDType) -> Option<MmqLauncher> {
    let f: MmqLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmq_gguf_q4_0,
        GgmlDType::Q4_1 => ffi::launch_mmq_gguf_q4_1,
        GgmlDType::Q5_0 => ffi::launch_mmq_gguf_q5_0,
        GgmlDType::Q5_1 => ffi::launch_mmq_gguf_q5_1,
        GgmlDType::Q8_0 => ffi::launch_mmq_gguf_q8_0,
        GgmlDType::Q2K => ffi::launch_mmq_gguf_q2_k,
        GgmlDType::Q3K => ffi::launch_mmq_gguf_q3_k,
        GgmlDType::Q4K => ffi::launch_mmq_gguf_q4_k,
        GgmlDType::Q5K => ffi::launch_mmq_gguf_q5_k,
        GgmlDType::Q6K => ffi::launch_mmq_gguf_q6_k,
        _ => return None,
    };
    Some(f)
}

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

static MMQ_WORKSPACE: OnceLock<WsMap> = OnceLock::new();
static FIXUP_WORKSPACE: OnceLock<WsMap> = OnceLock::new();

#[derive(Clone, Copy)]
struct DeviceInfo {
    arch: GpuArch,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
}

static DEVICE_INFO: OnceLock<Mutex<HashMap<DeviceId, DeviceInfo>>> = OnceLock::new();

fn get_device_info(dev: &CudaDevice) -> Result<DeviceInfo> {
    use candle_core::hip_backend::cudarc::driver::{result, sys};
    let map = DEVICE_INFO.get_or_init(|| Mutex::new(HashMap::new()));
    let key = dev.id();
    let mut guard = map.lock().unwrap();
    if let Some(info) = guard.get(&key) {
        return Ok(*info);
    }
    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(8);
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    let nsm = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .unwrap_or(1);
    let smpbo = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
    }
    .unwrap_or(49152);
    let warp_size = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        )
    }
    .unwrap_or(32);
    let info = DeviceInfo {
        // Kernel-bridge encoding the llama.cpp launchers expect; policy code
        // matches on `arch` instead.
        arch: GpuArch::resolve_hip(dev)?,
        cc: major * 100 + minor * 10,
        nsm,
        smpbo: smpbo as i64,
        warp_size,
    };
    guard.insert(key, info);
    Ok(info)
}

fn fixup_workspace_bytes(dev: &CudaDevice) -> Result<usize> {
    Ok(get_device_info(dev)?.nsm as usize * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>())
}

fn workspace_ensure<'a>(
    ws: &'static OnceLock<WsMap>,
    dev: &CudaDevice,
    bytes: usize,
    stream: &'a CudaStream,
) -> Result<WorkspaceGuard<'a>> {
    let map = ws.get_or_init(|| Mutex::new(HashMap::new()));
    let capacity = bytes.max(1).next_power_of_two();
    let key = WorkspaceKey {
        device: dev.id(),
        stream: stream.cu_stream() as usize,
        thread: std::thread::current().id(),
        capacity,
    };
    let device_mtx: &'static Mutex<WorkspaceSlot> = {
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
    let slot = device_mtx.lock().unwrap();
    Ok(WorkspaceGuard { slot, stream })
}

struct DenseMmqRun<'a> {
    weights: &'a [&'a QTensor],
    xs: &'a Tensor,
    dev: &'a CudaDevice,
    stream: &'a CudaStream,
    scratch_ptr: *mut std::ffi::c_void,
    fixup_ptr: *mut std::ffi::c_void,
    quantize: QuantizeLauncher,
    launcher: MmqLauncher,
    device_info: DeviceInfo,
    k: usize,
    k_padded: usize,
    batch_size: usize,
    qk: usize,
    type_x: i32,
}

impl DenseMmqRun<'_> {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        xs_slice: &CudaSlice<T>,
        xs_offset: usize,
    ) -> Result<Vec<Tensor>> {
        let stream_ptr = self.stream.cu_stream();
        let (xs_ptr, _xs_guard) = hip_slice_ptr_on_stream(xs_slice, xs_offset, self.stream);
        unsafe {
            (self.quantize)(
                xs_ptr as *const std::ffi::c_void,
                std::ptr::null(),
                self.scratch_ptr,
                self.type_x,
                self.k as i64,
                self.k as i64,
                0,
                0,
                self.k_padded as i64,
                self.batch_size as i64,
                1,
                1,
                stream_ptr,
            );
        }

        let mut outputs = Vec::with_capacity(self.weights.len());
        for weight in self.weights {
            let (nrows, _) = weight.shape().dims2()?;
            let (weight_ptr, _weight_guard) = weight.device_ptr_with_guard(self.stream)?;
            let mut out = unsafe { self.dev.alloc::<T>(nrows * self.batch_size)? };
            if std::env::var("MISTRALRS_MMQ_DEBUG").is_ok() {
                let d = &self.device_info;
                eprintln!(
                    "[mmqq] w_rows={nrows} k={} b={} cc={} nsm={} smpbo={} warp={}",
                    self.k, self.batch_size, d.cc, d.nsm, d.smpbo, d.warp_size
                );
            }
            {
                let (out_ptr, _out_guard) = hip_slice_ptr_mut_on_stream(&mut out, 0, self.stream);
                unsafe {
                    (self.launcher)(
                        self.fixup_ptr,
                        weight_ptr as *const std::ffi::c_void,
                        self.scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        self.k as i64,
                        nrows as i64,
                        self.batch_size as i64,
                        (self.k / self.qk) as i64,
                        nrows as i64,
                        self.device_info.cc,
                        self.device_info.nsm,
                        self.device_info.smpbo,
                        self.device_info.warp_size,
                        self.type_x,
                        stream_ptr,
                    );
                }
            }
            outputs.push(wrap_hip_output(out, self.dev, output_shape(self.xs, nrows)));
        }
        Ok(outputs)
    }
}

struct DenseGluDownRun<'a> {
    down: &'a QTensor,
    gate: &'a Tensor,
    dev: &'a CudaDevice,
    stream: &'a CudaStream,
    scratch_ptr: *mut std::ffi::c_void,
    fixup_ptr: *mut std::ffi::c_void,
    quantize: QuantizeGluLauncher,
    launcher: MmqLauncher,
    device_info: DeviceInfo,
    k: usize,
    k_padded: usize,
    batch_size: usize,
    qk: usize,
    type_x: i32,
    activation: i32,
}

impl DenseGluDownRun<'_> {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        gate_slice: &CudaSlice<T>,
        gate_offset: usize,
        up_slice: &CudaSlice<T>,
        up_offset: usize,
    ) -> Result<Tensor> {
        let stream_ptr = self.stream.cu_stream();
        let (gate_ptr, _gate_guard) = hip_slice_ptr_on_stream(gate_slice, gate_offset, self.stream);
        let (up_ptr, _up_guard) = hip_slice_ptr_on_stream(up_slice, up_offset, self.stream);
        unsafe {
            (self.quantize)(
                gate_ptr as *const std::ffi::c_void,
                up_ptr as *const std::ffi::c_void,
                std::ptr::null(),
                self.scratch_ptr,
                self.type_x,
                self.k as i64,
                self.k as i64,
                self.k_padded as i64,
                self.batch_size as i64,
                self.activation,
                stream_ptr,
            );
        }

        let (nrows, _) = self.down.shape().dims2()?;
        let (weight_ptr, _weight_guard) = self.down.device_ptr_with_guard(self.stream)?;
        let mut out = unsafe { self.dev.alloc::<T>(nrows * self.batch_size)? };
        {
            let (out_ptr, _out_guard) = hip_slice_ptr_mut_on_stream(&mut out, 0, self.stream);
            unsafe {
                (self.launcher)(
                    self.fixup_ptr,
                    weight_ptr as *const std::ffi::c_void,
                    self.scratch_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    self.k as i64,
                    nrows as i64,
                    self.batch_size as i64,
                    (self.k / self.qk) as i64,
                    nrows as i64,
                    self.device_info.cc,
                    self.device_info.nsm,
                    self.device_info.smpbo,
                    self.device_info.warp_size,
                    self.type_x,
                    stream_ptr,
                );
            }
        }
        Ok(wrap_hip_output(
            out,
            self.dev,
            output_shape(self.gate, nrows),
        ))
    }
}

fn shared_lhs(weights: &[&QTensor], xs: &Tensor) -> Result<Vec<Tensor>> {
    let Some(first) = weights.first() else {
        candle_core::bail!("fast_mmq shared_lhs: at least one weight is required");
    };
    let dtype = first.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq shared_lhs: unsupported quant dtype {dtype:?}");
    }
    let Device::Hip(dev) = first.device() else {
        candle_core::bail!("fast_mmq shared_lhs: weights must live on Hip");
    };
    let (_, ncols) = first.shape().dims2()?;
    for weight in &weights[1..] {
        if weight.dtype() != dtype {
            candle_core::bail!("fast_mmq shared_lhs: weight dtype mismatch");
        }
        let Device::Hip(weight_dev) = weight.device() else {
            candle_core::bail!("fast_mmq shared_lhs: weights must live on Hip");
        };
        if weight_dev.id() != dev.id() {
            candle_core::bail!("fast_mmq shared_lhs: weights are on different Hip devices");
        }
        let (_, weight_ncols) = weight.shape().dims2()?;
        if weight_ncols != ncols {
            candle_core::bail!(
                "fast_mmq shared_lhs: weight ncols mismatch {ncols} vs {weight_ncols}"
            );
        }
    }
    if !xs.device().same_device(&first.device()) {
        candle_core::bail!("fast_mmq shared_lhs: input and weights are on different devices");
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmq shared_lhs: input must have at least one dimension");
    };
    let batch_size = batch_dims.iter().product::<usize>();
    if batch_size == 0 {
        candle_core::bail!("fast_mmq shared_lhs: batch size must be greater than zero");
    }
    if k != ncols {
        candle_core::bail!(
            "fast_mmq shared_lhs: weight ncols {ncols} does not match input tail {k}"
        );
    }

    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq shared_lhs: k={k} not divisible by qk={qk}");
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmq shared_lhs: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Hip(xs_hip) = &*xs_storage else {
        candle_core::bail!("fast_mmq shared_lhs: input must live on Hip");
    };
    let xs_offset = xs_layout.start_offset();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        _ => unreachable!(),
    };

    let stream = dev.cuda_stream();
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);
    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = batch_size * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = MMQ_X_MAX * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let mut workspace = workspace_ensure(&MMQ_WORKSPACE, &dev, workspace_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    let fixup_bytes = fixup_workspace_bytes(&dev)?;
    let mut fixup_workspace = workspace_ensure(&FIXUP_WORKSPACE, &dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_workspace.ptr_mut();
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let run = DenseMmqRun {
        weights,
        xs: &xs,
        dev: &dev,
        stream: &stream,
        scratch_ptr,
        fixup_ptr,
        quantize: quantize_launcher(ds_layout_for(dtype)),
        launcher: mmq_launcher(dtype).expect("supports() checked"),
        device_info: get_device_info(&dev)?,
        k,
        k_padded,
        batch_size,
        qk,
        type_x,
    };
    match input_ty {
        DType::BF16 => run.launch(xs_hip.as_cuda_slice::<half::bf16>()?, xs_offset),
        DType::F16 => run.launch(xs_hip.as_cuda_slice::<half::f16>()?, xs_offset),
        DType::F32 => run.launch(xs_hip.as_cuda_slice::<f32>()?, xs_offset),
        _ => unreachable!(),
    }
}

fn down_from_glu(
    down: &QTensor,
    gate: &Tensor,
    up: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    let dtype = down.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq down_from_glu: unsupported quant dtype {dtype:?}");
    }
    let Device::Hip(dev) = down.device() else {
        candle_core::bail!("fast_mmq down_from_glu: weight must live on Hip");
    };
    if gate.shape() != up.shape() {
        candle_core::bail!(
            "fast_mmq down_from_glu: gate/up shape mismatch {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }
    if gate.dtype() != up.dtype() {
        candle_core::bail!(
            "fast_mmq down_from_glu: gate/up dtype mismatch {:?} vs {:?}",
            gate.dtype(),
            up.dtype()
        );
    }
    if !gate.device().same_device(&down.device()) || !up.device().same_device(&down.device()) {
        candle_core::bail!("fast_mmq down_from_glu: tensors are on different devices");
    }

    let Some((&k, batch_dims)) = gate.dims().split_last() else {
        candle_core::bail!("fast_mmq down_from_glu: input must have at least one dimension");
    };
    let batch_size = batch_dims.iter().product::<usize>();
    if batch_size == 0 {
        candle_core::bail!("fast_mmq down_from_glu: batch size must be greater than zero");
    }
    let (_, ncols) = down.shape().dims2()?;
    if k != ncols {
        candle_core::bail!(
            "fast_mmq down_from_glu: weight ncols {ncols} does not match input tail {k}"
        );
    }
    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq down_from_glu: k={k} not divisible by qk={qk}");
    }

    let input_ty = gate.dtype();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        other => candle_core::bail!(
            "fast_mmq down_from_glu: input dtype must be BF16, F16, or F32, got {other:?}"
        ),
    };
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let Storage::Hip(gate_hip) = &*gate_storage else {
        candle_core::bail!("fast_mmq down_from_glu: gate must live on Hip");
    };
    let (up_storage, up_layout) = up.storage_and_layout();
    let Storage::Hip(up_hip) = &*up_storage else {
        candle_core::bail!("fast_mmq down_from_glu: up must live on Hip");
    };

    let stream = dev.cuda_stream();
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);
    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = batch_size * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = MMQ_X_MAX * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let mut workspace = workspace_ensure(&MMQ_WORKSPACE, &dev, workspace_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    let fixup_bytes = fixup_workspace_bytes(&dev)?;
    let mut fixup_workspace = workspace_ensure(&FIXUP_WORKSPACE, &dev, fixup_bytes, &stream)?;
    let (fixup_ptr, _fixup_guard) = fixup_workspace.ptr_mut();
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let run = DenseGluDownRun {
        down,
        gate: &gate,
        dev: &dev,
        stream: &stream,
        scratch_ptr,
        fixup_ptr,
        quantize: quantize_glu_launcher(ds_layout_for(dtype)),
        launcher: mmq_launcher(dtype).expect("supports() checked"),
        device_info: get_device_info(&dev)?,
        k,
        k_padded,
        batch_size,
        qk,
        type_x,
        activation: activation as i32,
    };
    match input_ty {
        DType::BF16 => run.launch(
            gate_hip.as_cuda_slice::<half::bf16>()?,
            gate_layout.start_offset(),
            up_hip.as_cuda_slice::<half::bf16>()?,
            up_layout.start_offset(),
        ),
        DType::F16 => run.launch(
            gate_hip.as_cuda_slice::<half::f16>()?,
            gate_layout.start_offset(),
            up_hip.as_cuda_slice::<half::f16>()?,
            up_layout.start_offset(),
        ),
        DType::F32 => run.launch(
            gate_hip.as_cuda_slice::<f32>()?,
            gate_layout.start_offset(),
            up_hip.as_cuda_slice::<f32>()?,
            up_layout.start_offset(),
        ),
        _ => unreachable!(),
    }
}

/// Compute one GGUF-quantized projection while preserving the input dtype.
pub fn plain(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    let mut outputs = shared_lhs(&[w], xs)?;
    Ok(outputs.pop().expect("one weight produces one output"))
}

/// Compute Q, K, and V projections with one activation quantization pass.
pub(crate) fn fused_qkv(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: &QTensor,
    xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut outputs = shared_lhs(&[q_w, k_w, v_w], xs)?;
    let v = outputs.pop().expect("three weights produce three outputs");
    let k = outputs.pop().expect("three weights produce three outputs");
    let q = outputs.pop().expect("three weights produce three outputs");
    Ok((q, k, v))
}

/// Compute gate and up projections with one activation quantization pass.
pub(crate) fn fused_glu(
    gate_w: &QTensor,
    up_w: &QTensor,
    xs: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    if gate_w.shape() != up_w.shape() {
        candle_core::bail!(
            "fast_mmq fused_glu: gate/up shape mismatch {:?} vs {:?}",
            gate_w.shape(),
            up_w.shape()
        );
    }
    let mut outputs = shared_lhs(&[gate_w, up_w], xs)?;
    let up = outputs.pop().expect("two weights produce two outputs");
    let gate = outputs.pop().expect("two weights produce two outputs");
    crate::fused_glu(&gate, &up, activation)
}

pub(crate) fn fused_ffn(
    gate_w: &QTensor,
    up_w: &QTensor,
    down_w: &QTensor,
    xs: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    if gate_w.shape() != up_w.shape() {
        candle_core::bail!(
            "fast_mmq fused_ffn: gate/up shape mismatch {:?} vs {:?}",
            gate_w.shape(),
            up_w.shape()
        );
    }
    let mut outputs = shared_lhs(&[gate_w, up_w], xs)?;
    let up = outputs.pop().expect("two weights produce two outputs");
    let gate = outputs.pop().expect("two weights produce two outputs");
    down_from_glu(down_w, &gate, &up, activation)
}

macro_rules! hip_bail_moe {
    ($name:literal) => {
        candle_core::bail!(concat!(
            $name,
            " on the Hip backend stays on the MoE CPU fallback; grouped fusion is a follow-up"
        ))
    };
}

pub fn grouped(
    _weight: &QTensor,
    _xs: &Tensor,
    _ids_src: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _ids_dst: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _expert_bounds: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _total_assignments: usize,
    _ncols_max: usize,
    _num_experts: usize,
    _dev: &candle_core::cuda::CudaDevice,
) -> Result<Tensor> {
    hip_bail_moe!("fast_mmq::grouped")
}

pub fn grouped_from_glu_pair(
    _weight: &QTensor,
    _gate: &Tensor,
    _up: &Tensor,
    _ids_src: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _ids_dst: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _expert_bounds: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _total_assignments: usize,
    _ncols_max: usize,
    _num_experts: usize,
    _activation: i32,
    _dev: &candle_core::cuda::CudaDevice,
) -> Result<Tensor> {
    hip_bail_moe!("fast_mmq::grouped_from_glu_pair")
}

pub fn grouped_from_glu_sorted_pair(
    _weight: &QTensor,
    _gate: &Tensor,
    _up: &Tensor,
    _ids_dst: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _expert_bounds: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _total_assignments: usize,
    _ncols_max: usize,
    _num_experts: usize,
    _activation: i32,
    _dev: &candle_core::cuda::CudaDevice,
) -> Result<Tensor> {
    hip_bail_moe!("fast_mmq::grouped_from_glu_sorted_pair")
}

pub fn grouped_from_glu_packed(
    _weight: &QTensor,
    _gate_up: &Tensor,
    _ids_src: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _ids_dst: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _expert_bounds: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _total_assignments: usize,
    _ncols_max: usize,
    _num_experts: usize,
    _activation: i32,
    _dev: &candle_core::cuda::CudaDevice,
) -> Result<Tensor> {
    hip_bail_moe!("fast_mmq::grouped_from_glu_packed")
}

pub fn grouped_pair_packed(
    _gate: &QTensor,
    _up: &QTensor,
    _xs: &Tensor,
    _ids_src: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _ids_dst: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _expert_bounds: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _total_assignments: usize,
    _topk: usize,
    _num_experts: usize,
    _dev: &candle_core::cuda::CudaDevice,
) -> Result<Tensor> {
    hip_bail_moe!("fast_mmq::grouped_pair_packed")
}

pub fn grouped_pair(
    _gate: &QTensor,
    _up: &QTensor,
    _xs: &Tensor,
    _ids_src: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _ids_dst: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _expert_bounds: &candle_core::cuda::cudarc::driver::CudaSlice<u32>,
    _total_assignments: usize,
    _topk: usize,
    _num_experts: usize,
    _dev: &candle_core::cuda::CudaDevice,
) -> Result<(Tensor, Tensor)> {
    hip_bail_moe!("fast_mmq::grouped_pair")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QMatMul};
    use candle_core::Module as _;
    use std::sync::Arc;

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

    #[test]
    fn hip_mmq_plain_matches_dequant_chute() -> Result<()> {
        let hip = Device::new_hip(0)?;
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q4K] {
            let w = Arc::new(QTensor::quantize_onto(
                &patterned((INTERMEDIATE, HIDDEN), 11, 0.03)?,
                dtype,
                &hip,
            )?);
            let xs = patterned((64, HIDDEN), 3, 0.2)?
                .to_dtype(DType::BF16)?
                .to_device(&hip)?;
            assert_close(&plain(&w, &xs)?, &chute_reference(&w, &xs)?)?;
        }
        Ok(())
    }

    #[test]
    fn hip_mmq_fused_qkv_matches_plains() -> Result<()> {
        let hip = Device::new_hip(0)?;
        let q =
            QTensor::quantize_onto(&patterned((384, HIDDEN), 11, 0.03)?, GgmlDType::Q4K, &hip)?;
        let k =
            QTensor::quantize_onto(&patterned((256, HIDDEN), 29, 0.03)?, GgmlDType::Q4K, &hip)?;
        let v =
            QTensor::quantize_onto(&patterned((128, HIDDEN), 47, 0.03)?, GgmlDType::Q4K, &hip)?;
        let xs = patterned((64, HIDDEN), 3, 0.2)?
            .to_dtype(DType::BF16)?
            .to_device(&hip)?;

        let (q_out, k_out, v_out) = fused_qkv(&q, &k, &v, &xs)?;
        assert_close(&q_out, &plain(&q, &xs)?)?;
        assert_close(&k_out, &plain(&k, &xs)?)?;
        assert_close(&v_out, &plain(&v, &xs)?)
    }

    #[test]
    fn hip_mmq_fused_glu_matches_plains() -> Result<()> {
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
        let xs = patterned((64, HIDDEN), 3, 0.2)?
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
}
