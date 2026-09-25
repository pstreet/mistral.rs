use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use candle_core::{Result, Storage, Tensor};

use mistralrs_paged_attn::BlockQuantKind;

// Block-quantized (Q8_0/Q4_0) per-layer scale sidecars, keyed by the owning
// K-cache tensor identity.
// Rationale: `PagedAttention::forward` receives resolved per-layer cache
// tensors but no layer index (45+ model call sites destructure
// `ctx.paged_layer(idx)`), so threading scale tensors through the model API
// would churn every model. Instead `CacheEngine` registers each layer's
// scales at construction and unregisters on drop; layers resolve them here
// from the K-cache tensor they already hold. Keys are (storage pointer,
// start offset), so tensor clones resolve identically while distinct layers
// never collide. Entries are removed on engine drop, so reloaded servers
// cannot pin stale multi-GB caches.

#[derive(Clone)]
pub(crate) struct BlockQuantScales {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
    pub k_kind: Option<BlockQuantKind>,
    pub v_kind: Option<BlockQuantKind>,
    // QJL 1-bit residual sidecars (Q4_0 only): U8 bit packs, 4 bytes per
    // 32-elem group. `Some` only when `MISTRALRS_Q4_QJL=1`; otherwise the
    // kernels fall back to the plain LM grid. `None` for Q8_0.
    pub k_res: Option<Tensor>,
    pub v_res: Option<Tensor>,
}

// QJL 1-bit residual for Q4_0 KV: opt-in via `MISTRALRS_Q4_QJL=1` (or
// `true`), off by default. Measured flat-to-negative on 391 exact-match
// and logprob distance vs LM-only while costing +25% KV memory, so it
// stays a configurable experiment, not the default path. Read once;
// restart the server to change.
pub(crate) fn qjl_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("MISTRALRS_Q4_QJL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn registry() -> &'static Mutex<HashMap<(usize, usize), BlockQuantScales>> {
    static REGISTRY: OnceLock<Mutex<HashMap<(usize, usize), BlockQuantScales>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_key(t: &Tensor) -> Result<(usize, usize)> {
    // Address of the storage inside its lock: shared by all clones of the
    // tensor, distinct across allocations. Paired with the start offset so
    // views resolve independently.
    let (storage, layout) = t.storage_and_layout();
    Ok((&*storage as *const Storage as usize, layout.start_offset()))
}

pub(crate) fn register_block_scales(key_cache: &Tensor, scales: BlockQuantScales) -> Result<()> {
    let key = cache_key(key_cache)?;
    registry()
        .lock()
        .expect("Q8 scale registry mutex was poisoned")
        .insert(key, scales);
    Ok(())
}

pub(crate) fn lookup_block_scales(key_cache: &Tensor) -> Result<Option<BlockQuantScales>> {
    let key = cache_key(key_cache)?;
    Ok(registry()
        .lock()
        .expect("Q8 scale registry mutex was poisoned")
        .get(&key)
        .cloned())
}

pub(crate) fn unregister_block_scales(key_cache: &Tensor) -> Result<()> {
    let key = cache_key(key_cache)?;
    registry()
        .lock()
        .expect("Q8 scale registry mutex was poisoned")
        .remove(&key);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn test_scales() -> BlockQuantScales {
        BlockQuantScales {
            k: None,
            v: None,
            k_kind: Some(BlockQuantKind::Q8_0),
            v_kind: Some(BlockQuantKind::Q8_0),
            k_res: None,
            v_res: None,
        }
    }

    #[test]
    fn register_lookup_unregister_roundtrip() -> Result<()> {
        let cache = Tensor::zeros((2, 2), DType::U8, &Device::Cpu)?;
        assert!(lookup_block_scales(&cache)?.is_none());
        register_block_scales(&cache, test_scales())?;
        let found = lookup_block_scales(&cache)?.expect("registered scales missing");
        assert_eq!(found.k_kind, Some(BlockQuantKind::Q8_0));
        assert_eq!(found.v_kind, Some(BlockQuantKind::Q8_0));
        unregister_block_scales(&cache)?;
        assert!(lookup_block_scales(&cache)?.is_none());
        Ok(())
    }

    #[test]
    fn clones_resolve_but_offset_views_do_not() -> Result<()> {
        let cache = Tensor::zeros((4, 4), DType::U8, &Device::Cpu)?;
        register_block_scales(&cache, test_scales())?;
        assert!(lookup_block_scales(&cache.clone())?.is_some());
        assert!(lookup_block_scales(&cache.narrow(0, 0, 2)?)?.is_some());
        assert!(lookup_block_scales(&cache.narrow(0, 1, 2)?)?.is_none());
        unregister_block_scales(&cache)?;
        Ok(())
    }

    #[test]
    fn distinct_tensors_never_collide() -> Result<()> {
        let a = Tensor::zeros((2, 2), DType::U8, &Device::Cpu)?;
        let b = Tensor::zeros((2, 2), DType::U8, &Device::Cpu)?;
        register_block_scales(&a, test_scales())?;
        assert!(lookup_block_scales(&b)?.is_none());
        unregister_block_scales(&a)?;
        Ok(())
    }
}
