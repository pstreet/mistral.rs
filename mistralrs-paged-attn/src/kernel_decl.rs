//! One declaration site for GPU kernel entries, both roles.
//!
//! Every role resolves entries at runtime from its companion plugin
//! (`cuda_plugin` for the NVIDIA role, `hip_plugin` for the AMD role):
//! same names and signatures, so call sites are untouched in every
//! shape, and no build links GPU kernels.

macro_rules! declare_kernel {
    ($($( #[$attr:meta] )* $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)? ; )*) => {$(
        $( #[$attr] )*
        #[cfg(all(feature = "cuda", not(feature = "rocm"), target_family = "unix"))]
        #[allow(dead_code)]
        #[allow(non_snake_case)]
        pub(crate) unsafe extern "C" fn $name($($arg : $ty),*) $(-> $ret)? {
            type F = unsafe extern "C" fn($($ty),*) $(-> $ret)?;
            static CACHED: std::sync::OnceLock<libloading::Symbol<'static, F>> =
                std::sync::OnceLock::new();
            let f = CACHED.get_or_init(|| {
                let plugin = $crate::cuda_plugin::load().unwrap_or_else(|e| {
                    panic!("cuda kernel {} called with no plugin loaded: {e}", stringify!($name))
                });
                // SAFETY: the companion is built from the same sources as the
                // retired static archive, so each entry has the declared
                // signature.
                unsafe { plugin.lib.get(stringify!($name).as_bytes()) }
                    .unwrap_or_else(|_| panic!("cuda kernel {} missing from plugin", stringify!($name)))
            });
            unsafe { f($($arg),*) }
        }
        $( #[$attr] )*
        #[cfg(feature = "rocm")]
        #[allow(dead_code)]
        #[allow(non_snake_case)]
        pub(crate) unsafe extern "C" fn $name($($arg : $ty),*) $(-> $ret)? {
            type F = unsafe extern "C" fn($($ty),*) $(-> $ret)?;
            static CACHED: std::sync::OnceLock<libloading::Symbol<'static, F>> =
                std::sync::OnceLock::new();
            let f = CACHED.get_or_init(|| {
                let plugin = $crate::hip_plugin::load().unwrap_or_else(|e| {
                    panic!("hip kernel {} called with no plugin loaded: {e}", stringify!($name))
                });
                // SAFETY: the plugin is built from the same sources as the
                // static archive, so each entry has the declared signature.
                unsafe { plugin.lib.get(stringify!($name).as_bytes()) }
                    .unwrap_or_else(|_| panic!("hip kernel {} missing from plugin", stringify!($name)))
            });
            // SAFETY: same as above; arguments cross unchanged from the
            // probe-gated caller.
            unsafe { f($($arg),*) }
        }
    )*};
}

pub(crate) use declare_kernel;
