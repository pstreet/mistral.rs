//! NVIDIA-role dlsym entries for the GEMV kernels.
//!
//! Same C signatures as the `ffi` decls, which in dual builds resolve
//! through the hip companion; the NVIDIA role needs the nvcc-built
//! entries loaded from `crate::cuda_plugin` at first use.

#![allow(dead_code)]
#![allow(improper_ctypes)]

use half::{bf16, f16};
use std::ffi::c_void;

macro_rules! cuda_ffi {
    ($($( #[$attr:meta] )* $fname:ident = $sym:literal ( $($arg:ident : $ty:ty),* $(,)? ) $( -> $ret:ty )?; )*) => {$(
        $( #[$attr] )*
        #[allow(dead_code)]
        #[allow(non_snake_case)]
        pub(crate) unsafe extern "C" fn $fname($($arg : $ty),*) $( -> $ret )? {
            type F = unsafe extern "C" fn($($ty),*) $( -> $ret )?;
            static CACHED: std::sync::OnceLock<libloading::Symbol<'static, F>> =
                std::sync::OnceLock::new();
            let f = CACHED.get_or_init(|| {
                let plugin = $crate::cuda_plugin::load().unwrap_or_else(|e| {
                    panic!("cuda kernel {} called with no plugin loaded: {e}", $sym)
                });
                // SAFETY: the nvcc companion builds gemv.cu from the same
                // source as the hip companion, so the signature matches.
                unsafe { plugin.lib.get($sym.as_bytes()) }
                    .unwrap_or_else(|_| panic!("cuda kernel {} missing from plugin", $sym))
            });
            unsafe { f($($arg),*) }
        }
    )*};
}

cuda_ffi! {
    launch_gemv_bf16 = "launch_gemv_bf16"(
        a: *const bf16,
        x: *const bf16,
        bias: *const bf16,
        y: *mut bf16,
        m: i32,
        k: i32,
        batch_size: i32,
        has_bias: bool,
        stream: *mut c_void,
    );
    launch_gemv_f16 = "launch_gemv_f16"(
        a: *const f16,
        x: *const f16,
        bias: *const f16,
        y: *mut f16,
        m: i32,
        k: i32,
        batch_size: i32,
        has_bias: bool,
        stream: *mut c_void,
    );
    launch_gemv_f32 = "launch_gemv_f32"(
        a: *const f32,
        x: *const f32,
        bias: *const f32,
        y: *mut f32,
        m: i32,
        k: i32,
        batch_size: i32,
        has_bias: bool,
        stream: *mut c_void,
    );
}
