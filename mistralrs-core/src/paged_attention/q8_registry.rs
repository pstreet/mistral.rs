use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use candle_core::{Result, Storage, Tensor};

// Q8_0 per-layer scale sidecars, keyed by the owning K-cache tensor identity.
//
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
pub(crate) struct Q8LayerScales {
    pub k: Tensor,
    pub v: Tensor,
}

fn registry() -> &'static Mutex<HashMap<(usize, usize), Q8LayerScales>> {
    static REGISTRY: OnceLock<Mutex<HashMap<(usize, usize), Q8LayerScales>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_key(t: &Tensor) -> Result<(usize, usize)> {
    // Address of the storage inside its lock: shared by all clones of the
    // tensor, distinct across allocations. Paired with the start offset so
    // views resolve independently.
    let (storage, layout) = t.storage_and_layout();
    Ok((&*storage as *const Storage as usize, layout.start_offset()))
}

pub(crate) fn register_q8_scales(key_cache: &Tensor, scales: Q8LayerScales) -> Result<()> {
    let key = cache_key(key_cache)?;
    registry()
        .lock()
        .expect("Q8 scale registry mutex was poisoned")
        .insert(key, scales);
    Ok(())
}

pub(crate) fn lookup_q8_scales(key_cache: &Tensor) -> Result<Option<Q8LayerScales>> {
    let key = cache_key(key_cache)?;
    Ok(registry()
        .lock()
        .expect("Q8 scale registry mutex was poisoned")
        .get(&key)
        .cloned())
}

pub(crate) fn unregister_q8_scales(key_cache: &Tensor) -> Result<()> {
    let key = cache_key(key_cache)?;
    registry()
        .lock()
        .expect("Q8 scale registry mutex was poisoned")
        .remove(&key);
    Ok(())
}
