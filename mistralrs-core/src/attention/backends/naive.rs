#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::attention::{chunked_attention, SdpaParams};

/// Not *really* sure why this is necessary but it is.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    if matches!(device, Device::Cpu) {
        return Ok(());
    }

    // MRS_ATTENTION_SYNC: 0 never, 1 always, unset = sync only under memory pressure.
    static FORCE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let force = *FORCE.get_or_init(|| {
        std::env::var("MRS_ATTENTION_SYNC")
            .ok()
            .and_then(|v| match v.as_str() {
                "0" => Some(Some(false)),
                "1" => Some(Some(true)),
                _ => None,
            })
            .flatten()
    });

    // If less that 4 GB available, synchronize
    #[cfg(target_pointer_width = "64")]
    const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
    #[cfg(not(target_pointer_width = "64"))]
    const FOUR_GIB: usize = usize::MAX;
    let avail = MemoryUsage.query(device)?.available();
    let do_sync = force.unwrap_or(avail < FOUR_GIB);
    if do_sync {
        if std::env::var("MRS_ATTENTION_DEBUG").is_ok() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static ONCE: AtomicBool = AtomicBool::new(false);
            if !ONCE.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[attn-sync] syncing: avail={}MB (threshold 4096MB, force={:?})",
                    avail / (1024 * 1024),
                    force
                );
            }
        }
        device.synchronize()?;
    }
    Ok(())
}

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    // Use chunked attention with a closure that captures the necessary parameters
    chunked_attention(q, k, v, mask, |q_chunk, k, v, mask_chunk| {
        let mut att =
            MatMul.matmul_affine_mul(q_chunk, &k.t()?, sdpa_params.softmax_scale.into())?;

        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?;
            att = att.tanh()?;
            att = (att * softcap as f64)?;
        }

        if let Some(mask) = mask_chunk {
            att = att.broadcast_add(mask)?;
        }

        // Compute softmax in F32 for precision (BF16 exp() loses information).
        let att_dtype = att.dtype();
        if att_dtype == candle_core::DType::BF16 || att_dtype == candle_core::DType::F16 {
            att = att.to_dtype(candle_core::DType::F32)?;
        }
        att = candle_nn::ops::softmax_last_dim(&att)?;
        if att.dtype() != att_dtype {
            att = att.to_dtype(att_dtype)?;
        }
        MatMul.matmul(&att, v)
    })
}
