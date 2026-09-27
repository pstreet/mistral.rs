//! One declaration site for GPU kernel entries, both build shapes.
//!
//! Single-vendor rocm links the static archive: entries are plain
//! `extern "C"` declarations. Dual (`cuda+rocm`) builds resolve them at
//! runtime from the companion plugin (see `hip_plugin`): same name and
//! signature, so call sites are untouched in every shape.

macro_rules! declare_kernel {
    ($($( #[$attr:meta] )* $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)? ; )*) => {$(
        $( #[$attr] )*
        #[cfg(not(all(feature = "cuda", feature = "rocm")))]
        #[allow(dead_code)]
        extern "C" {
            pub(crate) fn $name($($arg : $ty),*) $(-> $ret)?;
        }
        $( #[$attr] )*
        #[cfg(all(feature = "cuda", feature = "rocm"))]
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
