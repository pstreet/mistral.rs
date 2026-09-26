//! Dual-build (`cuda+rocm`) stand-in for [`super::fast_mmvq`].
//!
//! Same contract as [`super::fast_mmq_hip`]: pure policy mirrored (keep in
//! sync with `fast_mmvq.rs`), launcher entries bail. The decode dispatch
//! degrades to dequant on Hip automatically. S2 fleshes this file out.

use super::ffi;
use crate::GluActivationType;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Result, Tensor,
};

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

/// Maximum flattened batch handled by the CUDA launcher table.
pub const MMVQ_MAX_BATCH: usize = 8;

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

macro_rules! hip_bail {
    ($name:literal) => {
        candle_core::bail!(concat!(
            $name,
            " on the Hip backend needs S2 (QStorage::Hip); the dispatch routes to dequant"
        ))
    };
}

pub fn plain(_w: &QTensor, _xs: &Tensor) -> Result<Tensor> {
    hip_bail!("fast_mmvq::plain")
}

pub fn fused_glu(
    _gate_w: &QTensor,
    _up_w: &QTensor,
    _xs: &Tensor,
    _activation: GluActivationType,
) -> Result<Tensor> {
    hip_bail!("fast_mmvq::fused_glu")
}

pub fn fused_qkv(
    _q_w: &QTensor,
    _k_w: &QTensor,
    _v_w: &QTensor,
    _xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    hip_bail!("fast_mmvq::fused_qkv")
}
