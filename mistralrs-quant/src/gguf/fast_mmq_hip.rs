//! Dual-build (`cuda+rocm`) stand-in for [`super::fast_mmq`].
//!
//! The real launcher layer needs `QTensor`-on-hip (`QStorage::Hip`, S2) and
//! is gated out in dual builds. This module keeps the public surface
//! compiling: pure policy fns are mirrored here (keep in sync with
//! `fast_mmq.rs`), launcher entries bail with a clear S2 pointer. The
//! serving dispatch degrades to the dequant path on Hip automatically
//! (`is_cuda()` is false), so nothing here runs in production today.
//! S2 fleshes this file out with real Hip launchers.

use candle_core::cuda::{cudarc::driver::CudaSlice, CudaDevice};
use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, GpuArch, Result, Tensor,
};

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

macro_rules! hip_bail {
    ($name:literal) => {
        candle_core::bail!(concat!(
            $name,
            " on the Hip backend needs S2 (QStorage::Hip); the dispatch routes to dequant"
        ))
    };
}

pub fn plain(_w: &QTensor, _xs: &Tensor) -> Result<Tensor> {
    hip_bail!("fast_mmq::plain")
}

pub(crate) fn fused_qkv(
    _q_w: &QTensor,
    _k_w: &QTensor,
    _v_w: &QTensor,
    _xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    hip_bail!("fast_mmq::fused_qkv")
}

pub(crate) fn fused_glu(
    _gate_w: &QTensor,
    _up_w: &QTensor,
    _xs: &Tensor,
    _activation: crate::GluActivationType,
) -> Result<Tensor> {
    hip_bail!("fast_mmq::fused_glu")
}

pub(crate) fn fused_ffn(
    _gate_w: &QTensor,
    _up_w: &QTensor,
    _down_w: &QTensor,
    _xs: &Tensor,
    _activation: crate::GluActivationType,
) -> Result<Tensor> {
    hip_bail!("fast_mmq::fused_ffn")
}

pub fn grouped(
    _weight: &QTensor,
    _xs: &Tensor,
    _ids_src: &CudaSlice<u32>,
    _ids_dst: &CudaSlice<u32>,
    _expert_bounds: &CudaSlice<u32>,
    _total_assignments: usize,
    _ncols_max: usize,
    _num_experts: usize,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("fast_mmq::grouped")
}

pub fn grouped_from_glu_pair(
    _weight: &QTensor,
    _gate: &Tensor,
    _up: &Tensor,
    _ids_src: &CudaSlice<u32>,
    _ids_dst: &CudaSlice<u32>,
    _expert_bounds: &CudaSlice<u32>,
    _total_assignments: usize,
    _ncols_max: usize,
    _num_experts: usize,
    _activation: i32,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("fast_mmq::grouped_from_glu_pair")
}

pub fn grouped_from_glu_sorted_pair(
    _weight: &QTensor,
    _gate: &Tensor,
    _up: &Tensor,
    _ids_dst: &CudaSlice<u32>,
    _expert_bounds: &CudaSlice<u32>,
    _total_assignments: usize,
    _ncols_max: usize,
    _num_experts: usize,
    _activation: i32,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("fast_mmq::grouped_from_glu_sorted_pair")
}

pub fn grouped_from_glu_packed(
    _weight: &QTensor,
    _gate_up: &Tensor,
    _ids_src: &CudaSlice<u32>,
    _ids_dst: &CudaSlice<u32>,
    _expert_bounds: &CudaSlice<u32>,
    _total_assignments: usize,
    _ncols_max: usize,
    _num_experts: usize,
    _activation: i32,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("fast_mmq::grouped_from_glu_packed")
}

pub fn grouped_pair_packed(
    _gate: &QTensor,
    _up: &QTensor,
    _xs: &Tensor,
    _ids_src: &CudaSlice<u32>,
    _ids_dst: &CudaSlice<u32>,
    _expert_bounds: &CudaSlice<u32>,
    _total_assignments: usize,
    _topk: usize,
    _num_experts: usize,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("fast_mmq::grouped_pair_packed")
}

pub fn grouped_pair(
    _gate: &QTensor,
    _up: &QTensor,
    _xs: &Tensor,
    _ids_src: &CudaSlice<u32>,
    _ids_dst: &CudaSlice<u32>,
    _expert_bounds: &CudaSlice<u32>,
    _total_assignments: usize,
    _topk: usize,
    _num_experts: usize,
    _dev: &CudaDevice,
) -> Result<(Tensor, Tensor)> {
    hip_bail!("fast_mmq::grouped_pair")
}
