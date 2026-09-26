//! Dual-build (`cuda+rocm`) stand-in for [`super::cuda`].
//!
//! Same contract as [`super::fast_mmq_hip`]: the MoE/indexed launcher layer
//! needs `QTensor`-on-hip (`QStorage::Hip`, S2) and is gated out in dual
//! builds. This module keeps the re-exported surface compiling (types are
//! plain data holders; launcher entries bail). S2 fleshes this file out.

use candle_core::{
    cuda::{cudarc::driver::CudaSlice, CudaDevice},
    quantized::{QMatMul, QTensor},
    Result, Tensor,
};

pub const ACT_GELU_PYTORCH_TANH: i32 = 0;
pub const ACT_SILU: i32 = 1;

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

#[allow(dead_code)]
pub struct IndexedMoeRouting<'a> {
    topk_ids: &'a CudaSlice<u32>,
    batch: usize,
    topk: usize,
    num_experts: usize,
    dev: &'a CudaDevice,
}

impl<'a> IndexedMoeRouting<'a> {
    pub fn new(
        topk_ids: &'a CudaSlice<u32>,
        batch: usize,
        topk: usize,
        num_experts: usize,
        dev: &'a CudaDevice,
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
        // MoE decode on Hip needs S2; the single-vendor constructor also
        // returns Ok(None) for unsupported dtypes, so callers handle it.
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
            " on the Hip backend needs S2 (QStorage::Hip); the dispatch routes to dequant"
        ))
    };
}

pub fn qtensor_indexed_moe_forward(
    _qtensor: &QTensor,
    _x: &Tensor,
    _ids: &Tensor,
) -> Result<Tensor> {
    hip_bail!("cuda::qtensor_indexed_moe_forward")
}

pub fn qmatmul_indexed_moe_forward(
    _qmatmul: &QMatMul,
    _x: &Tensor,
    _ids: &Tensor,
) -> Result<Tensor> {
    hip_bail!("cuda::qmatmul_indexed_moe_forward")
}

pub fn moe_dispatch_build(
    _topk_ids_flat: &CudaSlice<u32>,
    _total_assignments: usize,
    _num_experts: usize,
    _topk: usize,
    _dev: &CudaDevice,
) -> Result<(CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u32>)> {
    hip_bail!("cuda::moe_dispatch_build")
}

pub unsafe fn moe_weighted_reduce_flat(
    _inputs: &Tensor,
    _topk_weights: *const f32,
    _num_tokens: usize,
    _topk: usize,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::moe_weighted_reduce_flat")
}

pub unsafe fn moe_weighted_reduce_flat_bf16(
    _inputs: &Tensor,
    _topk_weights: *const f32,
    _num_tokens: usize,
    _topk: usize,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::moe_weighted_reduce_flat_bf16")
}

pub fn moe_weighted_reduce_flat_same_dtype(
    _inputs: &Tensor,
    _topk_weights: &Tensor,
    _num_tokens: usize,
    _topk: usize,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::moe_weighted_reduce_flat_same_dtype")
}

pub fn quantize_input_q8_1(
    _xs: &Tensor,
    _dev: &CudaDevice,
) -> Result<(CudaSlice<u8>, usize, usize)> {
    hip_bail!("cuda::quantize_input_q8_1")
}

pub fn grouped_moe_gemm_prequantized(
    _qtensor: &QTensor,
    _input_quant: &CudaSlice<u8>,
    _k: usize,
    _k_padded: usize,
    _expert_bounds: &CudaSlice<u32>,
    _sorted_token_ids: &CudaSlice<u32>,
    _topk_weights: Option<(*const f32, usize)>,
    _total_assignments: usize,
    _topk: usize,
    _num_experts: usize,
    _input_dim1: usize,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::grouped_moe_gemm_prequantized")
}

pub unsafe fn indexed_moe_fused_decode(
    _gate_qt: &QTensor,
    _up_qt: &QTensor,
    _down_qt: &QTensor,
    _xs_flat: &Tensor,
    _topk_ids: &CudaSlice<u32>,
    _topk_weights_ptr: *const f32,
    _batch: usize,
    _topk: usize,
    _act_type: i32,
    _dev: &CudaDevice,
) -> Result<Tensor> {
    hip_bail!("cuda::indexed_moe_fused_decode")
}
