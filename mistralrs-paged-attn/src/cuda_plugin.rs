//! Hot-pluggable NVIDIA kernels for dual (`cuda,rocm`) builds.
//!
//! The nvcc-compiled paged-attention kernels live in a companion shared
//! library (`libmistralrspagedattention_cuda.so`, linked by build.rs)
//! instead of the static archive: static linking would pin a load-time
//! cudart dependency into the exe and refuse startup on NVIDIA-less
//! machines. Same shape as `hip_plugin`: each entry resolves once via
//! dlsym and is cached. Callers are probe-gated on the NVIDIA role, so
//! the loader is unreachable without a runtime.

use std::path::PathBuf;
use std::sync::OnceLock;

pub(crate) struct Plugin {
    pub(crate) lib: libloading::Library,
}

static PLUGIN: OnceLock<Result<Plugin, String>> = OnceLock::new();

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("MISTRALRS_PAGED_ATTN_CUDA_PLUGIN") {
        out.push(p.into());
    }
    out.push(PathBuf::from(env!("MISTRALRS_PAGED_ATTN_CUDA_PLUGIN")));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("libmistralrspagedattention_cuda.so"));
        }
    }
    out
}

/// The loaded companion library. Process-lifetime handle by construction
/// (stored in a static, never unloaded).
pub(crate) fn load() -> Result<&'static Plugin, &'static str> {
    PLUGIN
        .get_or_init(|| {
            for path in candidates() {
                // SAFETY: loading a companion whose symbols we dlsym by
                // exact name with caller-verified signatures (see below).
                if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
                    return Ok(Plugin { lib });
                }
            }
            Err(
                "libmistralrspagedattention_cuda.so not found (set MISTRALRS_PAGED_ATTN_CUDA_PLUGIN)"
                    .to_string(),
            )
        })
        .as_ref()
        .map_err(String::as_str)
}
