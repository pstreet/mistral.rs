use crate::kernel_decl::declare_kernel;
use candle_core::cuda::cudarc::driver::sys::CUstream;
use half::{bf16, f16};

declare_kernel! {
    dequantize_blockwise_f32_int8(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f32, blocksize: i32, n: i32, stream: CUstream,);
    dequantize_blockwise_f32_fp4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f32, blocksize: i32, n: i32, stream: CUstream,);
    dequantize_blockwise_f32_nf4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f32, blocksize: i32, n: i32, stream: CUstream,);
    dequantize_blockwise_f16_int8(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f16, blocksize: i32, n: i32, stream: CUstream,);
    dequantize_blockwise_f16_fp4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f16, blocksize: i32, n: i32, stream: CUstream,);
    dequantize_blockwise_f16_nf4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut f16, blocksize: i32, n: i32, stream: CUstream,);
    dequantize_blockwise_bf16_int8(code: *const f32, a: *const u8, absmax: *const f32, out: *mut bf16, blocksize: i32, n: i32, stream: CUstream,);
    dequantize_blockwise_bf16_fp4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut bf16, blocksize: i32, n: i32, stream: CUstream,);
    dequantize_blockwise_bf16_nf4(code: *const f32, a: *const u8, absmax: *const f32, out: *mut bf16, blocksize: i32, n: i32, stream: CUstream,);
}
