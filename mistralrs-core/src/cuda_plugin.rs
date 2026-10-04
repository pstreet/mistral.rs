//! Hot-pluggable NVIDIA kernels for dual (`cuda+rocm`) builds.
//!
//! The nvcc-compiled kernels (the GDN family) live in a companion shared
//! library (`libmistralrscuda_cuda.so`, built by build.rs) instead of the
//! static archive: nothing links at build time, so the exe keeps zero GPU
//! DT_NEEDED. Same shape as the hip companion loader and the quant /
//! paged-attn CUDA plugins: each entry resolves once via dlsym and is
//! cached. GPU callers are role-gated, so a missing plugin or symbol is a
//! named panic, never a segfault.

use std::path::PathBuf;
use std::sync::OnceLock;

pub(crate) struct Plugin {
    pub(crate) lib: libloading::Library,
}

static PLUGIN: OnceLock<Result<Plugin, String>> = OnceLock::new();

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("MISTRALRS_CORE_CUDA_PLUGIN") {
        out.push(p.into());
    }
    out.push(PathBuf::from(env!("MISTRALRS_CORE_CUDA_PLUGIN")));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("libmistralrscuda_cuda.so"));
        }
    }
    out
}

/// The loaded companion library. Process-lifetime handle by construction
/// (stored in a static, never unloaded - unloading a CUDA runtime whose
/// static init ran can segfault).
pub(crate) fn load() -> Result<&'static Plugin, &'static str> {
    PLUGIN
        .get_or_init(|| {
            for path in candidates() {
                // SAFETY: loading a companion whose symbols we dlsym by
                // exact name with caller-verified signatures.
                if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
                    return Ok(Plugin { lib });
                }
            }
            Err("libmistralrscuda_cuda.so not found (set MISTRALRS_CORE_CUDA_PLUGIN)".to_string())
        })
        .as_ref()
        .map_err(String::as_str)
}
