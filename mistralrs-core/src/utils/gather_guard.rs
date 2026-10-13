//! Fault-hunt diagnostic (MISTRALRS_GATHER_GUARD=1): bounds-checks u32 index
//! tensors before the unguarded index_select/copy_rows launches in the GDN
//! recurrent gather/scatter surface. Costs a D2H sync per call, so it is
//! env-gated and defaults off; a bail here names the buffer and the wild
//! index instead of leaving an HSA memory-fault VA to attribute post-hoc.
//! Remove with the census/ledger diagnostics once the fault class closes.

use candle_core::{DType, Result, Tensor};
use std::sync::OnceLock;

fn check_indices(name: &'static str, values: &[u32], source_rows: usize) -> Result<()> {
    let max = values.iter().copied().max().unwrap_or(0);
    if (max as usize) >= source_rows {
        let head: Vec<u32> = values.iter().copied().take(8).collect();
        candle_core::bail!(
            "guarded gather '{name}': index {max} >= {source_rows} source rows \
             ({} indices, head {head:?})",
            values.len(),
        );
    }
    Ok(())
}

/// No-op unless MISTRALRS_GATHER_GUARD=1 and the indices are u32.
pub(crate) fn guard_u32_gather(
    name: &'static str,
    indices: &Tensor,
    source_rows: usize,
) -> Result<()> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("MISTRALRS_GATHER_GUARD").as_deref() == Ok("1")) {
        return Ok(());
    }
    if indices.dtype() != DType::U32 {
        return Ok(());
    }
    check_indices(name, &indices.to_vec1::<u32>()?, source_rows)
}

#[cfg(test)]
mod tests {
    use super::check_indices;

    #[test]
    fn in_bounds_indices_pass() {
        assert!(check_indices("test", &[0, 2, 1], 3).is_ok());
        assert!(check_indices("test", &[], 3).is_ok());
    }

    #[test]
    fn out_of_bounds_index_bails_with_name() {
        let err = check_indices("gather_conv_state", &[0, 3], 3).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("gather_conv_state"), "msg: {msg}");
        assert!(msg.contains("index 3"), "msg: {msg}");
        assert!(msg.contains("3 source rows"), "msg: {msg}");
    }
}
