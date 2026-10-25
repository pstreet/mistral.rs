#![allow(dead_code)]

#[cfg(all(feature = "cuda", feature = "rocm"))]
use crate::kernel_decl::declare_cuda_role_kernel;
#[cfg(all(feature = "cuda", feature = "rocm"))]
use candle_core::cuda::cudarc::driver::sys::CUstream as RoleCUstream;
#[cfg(all(feature = "cuda", feature = "rocm"))]
use std::ffi::c_void;

#[cfg(all(feature = "cuda", feature = "rocm"))]
declare_cuda_role_kernel! {
    softcap_f32(input: *const c_void, output: *mut c_void, N: u32, cap: f32, stream: RoleCUstream,);
    softcap_f16_to_f32(input: *const c_void, output: *mut c_void, N: u32, cap: f32, stream: RoleCUstream,);
    softcap_bf16_to_f32(input: *const c_void, output: *mut c_void, N: u32, cap: f32, stream: RoleCUstream,);
}
