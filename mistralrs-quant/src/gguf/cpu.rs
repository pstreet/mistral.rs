//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{QMatMul, QTensor},
    DType, Result, Tensor,
};
use candle_nn::Linear;
use std::sync::Arc;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Perform indexed MoE forward pass on a QTensor by dequantizing and using UnquantLinear.
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
/// Group-by-expert gather matmul: for each distinct expert selected in
/// `ids`, matmul its routed input rows against that expert's weight slice.
///
/// `w`: (n_experts, out, in); `x`: (tokens, x_t, in) with `x_t` of 1
/// (broadcast across slots) or `t` (per-slot rows); `ids`: (tokens, t).
/// Output: (tokens, t, out).
///
/// This is the memory-sound replacement for the materializing gather
/// (`index_select` of expert rows per (token, slot) pair), which costs
/// O(tokens * t * expert_size) memory - a 2900-token prompt with 8
/// experts and 768x2048 f32 weights is 146 GB, OOM on any GPU. The
/// grouped path copies activations only: O(tokens * t * (in + out));
/// expert weights are narrow views, never copied. The id routing runs
/// on the host (ids is tokens * t u32 - kilobytes) and each distinct
/// expert is one small GEMM.
fn grouped_gather_mm(w: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (tokens, x_t, k) = x.dims3()?;
    let (ids_tokens, t) = ids.dims2()?;
    if ids_tokens != tokens {
        candle_core::bail!("grouped gather: {tokens} input tokens vs {ids_tokens} id rows");
    }
    if x_t != 1 && x_t != t {
        candle_core::bail!("grouped gather: {x_t} input rows per token must be 1 or the {t} slots");
    }
    let (n_experts, out_f, in_f) = w.dims3()?;
    if in_f != k {
        candle_core::bail!("grouped gather: weight in-features {in_f} vs input {k}");
    }
    let device = x.device();

    // One row per (token, slot); broadcast the single row when x_t == 1.
    let rows = if x_t == 1 {
        x.broadcast_as((tokens, t, k))?.reshape((tokens * t, k))?
    } else {
        x.reshape((tokens * t, k))?
    };

    // Host-side routing: ids is tiny (tokens * t u32).
    let ids_host: Vec<u32> = ids.flatten_all()?.to_vec1::<u32>()?;
    let mut positions: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
    for (slot, e) in ids_host.iter().enumerate() {
        let e = *e as usize;
        if e >= n_experts {
            candle_core::bail!("grouped gather: expert id {e} out of range ({n_experts})");
        }
        positions[e].push(slot as u32);
    }

    // Zeroed accumulator: each slot is routed to exactly one expert,
    // so add == set. (An in-place scatter would avoid a full-accumulator
    // copy per expert, but candle's scatter_add_set enforces stricter
    // rank alignment than index_add and rejects this shape; the copies
    // are memcpy-class traffic, dwarfed by the GEMMs.) `tokens * t`
    // slots always exist (t >= 1), so the accumulator is never empty.
    let mut out = Tensor::zeros((tokens * t, out_f), x.dtype(), device)?;
    for (e, pos) in positions.iter().enumerate() {
        if pos.is_empty() {
            continue;
        }
        let idx = Tensor::from_vec(pos.clone(), (pos.len(),), device)?;
        let rows_e = rows.index_select(&idx, 0)?;
        // w[e] is a narrow view - no weight copy.
        let w_e = w.narrow(0, e, 1)?.squeeze(0)?.t()?;
        let contrib = rows_e.matmul(&w_e)?;
        out = out.index_add(&idx, &contrib, 0)?;
    }
    out.reshape((tokens, t, out_f))
}

/// Shape-normalizing wrapper around [`grouped_gather_mm`]: returns
/// `Ok(None)` for layouts outside the common MoE forms so callers keep
/// their existing fallback. Handles the same 4D per-slot and 5D
/// broadcast forms the materializing path does, with matching output
/// shapes.
fn try_grouped_gather_mm(w: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Option<Tensor>> {
    let (n_experts, out_f, in_f) = match w.dims3() {
        Ok(dims) => dims,
        Err(_) => return Ok(None),
    };
    let (rows, ids2, out_shape): (Tensor, Tensor, Option<Vec<usize>>) = match *x.dims() {
        [tokens, x_t, k] => match *ids.dims() {
            [ids_tokens, t] if ids_tokens == tokens && (x_t == 1 || x_t == t) => {
                // Already normalized; output keeps the 3D form.
                (x.clone(), ids.clone(), None)
            }
            _ => return Ok(None),
        },
        [b, s, xt, h] => match *ids.dims() {
            [ib, is, t] if ib == b && is == s && (xt == 1 || xt == t) => (
                x.reshape((b * s, xt, h))?,
                ids.reshape((b * s, t))?,
                Some(vec![b, s, t, out_f]),
            ),
            _ => return Ok(None),
        },
        [b, s, 1, 1, h] => match *ids.dims() {
            [ib, is, t] if ib == b && is == s => (
                x.reshape((b * s, 1, h))?,
                ids.reshape((b * s, t))?,
                Some(vec![b, s, t, out_f]),
            ),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    let (tokens, _, k) = rows.dims3()?;
    if k != in_f || out_f == 0 || n_experts == 0 {
        return Ok(None);
    }
    // Routing reads ids as U32 on the host; other id dtypes (and
    // zero-slot ids) keep the materializing fallback.
    if ids2.dtype() != DType::U32 || ids2.elem_count() == 0 {
        return Ok(None);
    }
    let out = grouped_gather_mm(w, &rows, &ids2)?;
    Ok(Some(match out_shape {
        Some(shape) => out.reshape(shape)?,
        None => out,
    }))
}

pub fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    // Repacked per-expert gemv path; falls back to dequantize-and-gather only for
    // layouts the packed kernels cannot serve. Normalize the metal/cpu 4D/5D input
    // shapes to the (tokens, x_t, hidden) form the kernel expects.
    {
        let (x3, ids2, out_shape): (Tensor, Tensor, Option<Vec<usize>>) = match *x.dims() {
            [b, s, xt, h] => {
                let (ib, is, t) = ids.dims3()?;
                if ib == b && is == s {
                    (
                        x.reshape((b * s, xt, h))?,
                        ids.reshape((b * s, t))?,
                        Some(vec![b, s, t, 0]),
                    )
                } else {
                    (x.clone(), ids.clone(), None)
                }
            }
            [b, s, 1, 1, h] => {
                let (ib, is, t) = ids.dims3()?;
                if ib == b && is == s {
                    (
                        x.reshape((b * s, 1, h))?,
                        ids.reshape((b * s, t))?,
                        Some(vec![b, s, t, 0]),
                    )
                } else {
                    (x.clone(), ids.clone(), None)
                }
            }
            [_, _, _] => (x.clone(), ids.clone(), Some(vec![])),
            _ => (x.clone(), ids.clone(), None),
        };
        if let Some(shape) = out_shape.filter(|_| x3.rank() == 3 && ids2.rank() == 2) {
            if let Some(out) = qtensor.indexed_gemv(&x3, &ids2)? {
                return if shape.is_empty() {
                    Ok(out)
                } else {
                    let n_out = out.dim(2)?;
                    out.reshape((shape[0], shape[1], shape[2], n_out))
                };
            }
        }
    }

    let device = x.device();

    // Dequantize all weights to f32
    let weights = qtensor.dequantize(device)?;

    // Dequantized weights are F32 while activations may be BF16/F16: the
    // gather matmul needs matching dtypes, so run it in F32 and cast back.
    // The grouped path avoids the O(tokens*t*expert_size) materializing
    // gather (a 2900-token prompt would copy 146 GB of expert weights).
    let input_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    if let Some(out) = try_grouped_gather_mm(&weights, &x_f32, ids)? {
        return out.to_dtype(input_dtype);
    }

    // Fallback for layouts outside the grouped forms: materializing.
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;
    unquant.gather_forward(&x_f32, ids)?.to_dtype(input_dtype)
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for CPU/Metal GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn cpu_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // For non-quantized tensors, use UnquantLinear directly. The
            // gather matmul needs matching dtypes: on mismatch run it in
            // F32 and restore the input dtype.
            if t.dtype() == x.dtype() {
                if let Some(out) = try_grouped_gather_mm(t, x, ids)? {
                    return Ok(out);
                }
                let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
                    t.clone(),
                    None,
                )))?;
                unquant.gather_forward(x, ids)
            } else {
                let input_dtype = x.dtype();
                let w_f32 = t.to_dtype(DType::F32)?;
                let x_f32 = x.to_dtype(DType::F32)?;
                if let Some(out) = try_grouped_gather_mm(&w_f32, &x_f32, ids)? {
                    return out.to_dtype(input_dtype);
                }
                let unquant =
                    UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(w_f32, None)))?;
                unquant.gather_forward(&x_f32, ids)?.to_dtype(input_dtype)
            }
        }
    }
}

#[cfg(test)]
mod grouped_gather_tests {
    use super::*;
    use candle_core::Device;

    fn reference_gather(w: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        // The materializing path, as a reference: UnquantLinear::gather_forward.
        let unquant =
            UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(w.clone(), None)))?;
        unquant.gather_forward(x, ids)
    }

    fn compare(got: &Tensor, want: &Tensor, label: &str) -> Result<()> {
        let g: Vec<f32> = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let w: Vec<f32> = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(g.len(), w.len(), "{label} length");
        for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            // Relative: the grouped path quantizes activations (Q8_1) on the
            // fused role, so large-magnitude outputs carry ~1e-6 relative
            // accumulation noise; an absolute 1e-4 flaked at |value| ~ 76.
            // Routing/ordering bugs still produce O(1) garbage here.
            let diff = (a - b).abs();
            let budget = 1e-4 * (1.0 + b.abs());
            assert!(
                diff < budget,
                "{label}[{i}]: grouped {a} vs materialized {b}"
            );
        }
        Ok(())
    }

    fn case(
        dev: &Device,
        n_experts: usize,
        out_f: usize,
        in_f: usize,
        tokens: usize,
        x_t: usize,
        t: usize,
        ids_pattern: &[u32],
    ) -> Result<()> {
        let mk = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i % 13) as f32 * 0.37 - 0.7) * 0.3)
                .collect()
        };
        let w = Tensor::from_vec(
            (0..n_experts * out_f * in_f)
                .map(|i| ((i % 23) as f32 * 0.11 - 0.8) * 0.2)
                .collect(),
            (n_experts, out_f, in_f),
            dev,
        )?;
        let x = Tensor::from_vec(mk(tokens * x_t * in_f), (tokens, x_t, in_f), dev)?;
        let ids_data: Vec<u32> = (0..tokens * t)
            .map(|i| ids_pattern[i % ids_pattern.len()])
            .collect();
        let ids = Tensor::from_vec(ids_data, (tokens, t), dev)?;
        let got = grouped_gather_mm(&w, &x, &ids)?;
        let want = reference_gather(&w, &x, &ids)?;
        compare(&got, &want, &format!("experts={n_experts} x_t={x_t} t={t}"))
    }

    #[test]
    fn grouped_matches_materialized_cpu() -> Result<()> {
        let dev = Device::Cpu;
        // Per-slot rows, uneven routing across experts.
        case(&dev, 4, 16, 32, 5, 2, 2, &[0, 3, 3, 1, 2])?;
        // Broadcast form: one row per token, t experts.
        case(&dev, 4, 16, 32, 5, 1, 3, &[3, 0, 1, 3, 2])?;
        // All tokens to one expert; other experts empty.
        case(&dev, 8, 8, 16, 4, 1, 2, &[5, 5, 5, 5])?;
        // Single slot.
        case(&dev, 3, 12, 24, 6, 1, 1, &[2, 0, 1, 2, 0, 1])?;
        Ok(())
    }

    #[test]
    fn try_grouped_normalizes_4d_5d() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::from_vec(
            (0..4 * 8 * 16)
                .map(|i| (i % 7) as f32 * 0.2 - 0.6)
                .collect(),
            (4, 8, 16),
            &dev,
        )?;
        let x5 = Tensor::from_vec(
            (0..2 * 3 * 16).map(|i| (i % 5) as f32 * 0.1).collect(),
            (2, 3, 1, 1, 16),
            &dev,
        )?;
        let ids3 = Tensor::from_vec(
            vec![1u32, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
            (2, 3, 3),
            &dev,
        )?;
        let got5 = try_grouped_gather_mm(&w, &x5, &ids3)?
            .ok_or_else(|| candle_core::Error::msg("5D form not grouped"))?;
        assert_eq!(got5.dims(), vec![2, 3, 3, 8]);

        let x4 = Tensor::from_vec(
            (0..2 * 3 * 3 * 16).map(|i| (i % 5) as f32 * 0.1).collect(),
            (2, 3, 3, 16),
            &dev,
        )?;
        let got4 = try_grouped_gather_mm(&w, &x4, &ids3)?
            .ok_or_else(|| candle_core::Error::msg("4D form not grouped"))?;
        assert_eq!(got4.dims(), vec![2, 3, 3, 8]);
        Ok(())
    }

    #[test]
    fn grouped_large_scale_matches_cpu() -> Result<()> {
        // Prefill scale on the serving GPU role (1138 tokens, 8 slots,
        // real 16B dims): grouped-vs-grouped across backends. The
        // materializing reference would OOM here (146 GB), so cross-
        // backend agreement is the check; small sizes already proved
        // grouped == materialized above.
        let dev = crate::utils::test_gpu_device();
        let (n_experts, out_f, in_f, tokens, t) = (64usize, 768, 2048, 1138, 8);
        let mk = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i % 23) as f32 * 0.37 - 1.1) * 0.05)
                .collect()
        };
        let w_data: Vec<f32> = mk(n_experts * out_f * in_f);
        let x_data: Vec<f32> = mk(tokens * in_f);
        let ids_data: Vec<u32> = (0..tokens * t)
            .map(|i| (i * 7 % n_experts) as u32)
            .collect();
        let run = |d: &Device| -> Result<Tensor> {
            let w = Tensor::from_vec(w_data.clone(), (n_experts, out_f, in_f), d)?;
            let x = Tensor::from_vec(x_data.clone(), (tokens, 1, in_f), d)?;
            let ids = Tensor::from_vec(ids_data.clone(), (tokens, t), d)?;
            grouped_gather_mm(&w, &x, &ids)
        };
        let gpu = run(&dev)?;
        let cpu = run(&Device::Cpu)?;
        compare(&gpu, &cpu, "large-scale gpu vs cpu")
    }

    #[test]
    fn grouped_gather_serves_test_gpu() -> Result<()> {
        // The same kernel bar on the serving GPU role of this build shape
        // (hip in dual, the single GPU otherwise; MISTRALRS_TEST_DEVICE
        // can point it at the non-serving role).
        let dev = crate::utils::test_gpu_device();
        case(&dev, 4, 16, 32, 5, 2, 2, &[0, 3, 3, 1, 2])?;
        case(&dev, 4, 16, 32, 5, 1, 3, &[3, 0, 1, 3, 2])?;
        Ok(())
    }
}
