//! Hot-pluggable NVIDIA FA3 (SM90) kernels, any cuda build shape.
//!
//! The fa3 paged-attention kernels are built into their own companion
//! shared library only when the companion targets SM90 (`has_fa3_fp8_paged`),
//! so unlike `cuda_plugin` the bake-time path env is unset on every other
//! machine and the loader resolves purely at runtime: env override first,
//! then exe-relative. Same shape as `cuda_plugin`: process-lifetime handle,
//! entries dlsym'd once by the FFI wrappers.

use std::path::PathBuf;
use std::sync::OnceLock;

pub(crate) struct Plugin {
    pub(crate) lib: libloading::Library,
}

static PLUGIN: OnceLock<Result<Plugin, String>> = OnceLock::new();

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("MISTRALRS_FA3_CUDA_PLUGIN") {
        out.push(p.into());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("libmistralrsfa3paged_cuda.so"));
        }
    }
    out
}

/// The loaded FA3 companion library. Process-lifetime handle by
/// construction (stored in a static, never unloaded).
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
                "libmistralrsfa3paged_cuda.so not found (set MISTRALRS_FA3_CUDA_PLUGIN)"
                    .to_string(),
            )
        })
        .as_ref()
        .map_err(String::as_str)
}
