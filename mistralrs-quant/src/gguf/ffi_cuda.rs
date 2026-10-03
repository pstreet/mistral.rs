//! NVIDIA-role dlsym entries for the fused MoE decode kernels.
//!
//! Same C signatures as the hip companion (`ffi`), resolved from the nvcc
//! companion (`crate::cuda_plugin`) at first use. Only the decode-path
//! symbols: Q8_1 quantize (bf16/f16/f32 inputs), fused gate_up, and the
//! down-aggregate reduce, for every GGUF weight dtype.

macro_rules! cuda_ffi {
    ($($( #[$attr:meta] )* $fname:ident = $sym:literal ( $($arg:ident : $ty:ty),* $(,)? ); )*) => {$(
        $( #[$attr] )*
        #[allow(dead_code)]
        #[allow(non_snake_case)]
        pub(crate) unsafe extern "C" fn $fname($($arg : $ty),*) {
            type F = unsafe extern "C" fn($($ty),*);
            static CACHED: std::sync::OnceLock<libloading::Symbol<'static, F>> =
                std::sync::OnceLock::new();
            let f = CACHED.get_or_init(|| {
                let plugin = $crate::cuda_plugin::load().unwrap_or_else(|e| {
                    panic!("cuda kernel {} called with no plugin loaded: {e}", $sym)
                });
                // SAFETY: the plugin is built from the same sources as the
                // hip companion, so each entry has the declared signature.
                unsafe { plugin.lib.get($sym.as_bytes()) }
                    .unwrap_or_else(|_| panic!("cuda kernel {} missing from plugin", $sym))
            });
            unsafe { f($($arg),*) }
        }
    )*};
}

cuda_ffi! {
    cuda_launch_quantize_q8_1 = "launch_quantize_q8_1"(
        x: *const f32,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_blocks_x: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_quantize_q8_1_bf16 = "launch_quantize_q8_1_bf16"(
        x: *const std::ffi::c_void,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_quantize_q8_1_f16 = "launch_quantize_q8_1_f16"(
        x: *const std::ffi::c_void,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q8_0_q8_1 = "launch_moe_gemv_fused_gate_up_q8_0_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q4_0_q8_1 = "launch_moe_gemv_fused_gate_up_q4_0_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q4_1_q8_1 = "launch_moe_gemv_fused_gate_up_q4_1_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q5_0_q8_1 = "launch_moe_gemv_fused_gate_up_q5_0_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q5_1_q8_1 = "launch_moe_gemv_fused_gate_up_q5_1_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q8_1_q8_1 = "launch_moe_gemv_fused_gate_up_q8_1_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q2k_q8_1 = "launch_moe_gemv_fused_gate_up_q2k_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q3k_q8_1 = "launch_moe_gemv_fused_gate_up_q3k_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q4k_q8_1 = "launch_moe_gemv_fused_gate_up_q4k_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q5k_q8_1 = "launch_moe_gemv_fused_gate_up_q5k_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_fused_gate_up_q6k_q8_1 = "launch_moe_gemv_fused_gate_up_q6k_q8_1"(
        gate_weights: *const std::ffi::c_void,
        up_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q8_0_q8_1 = "launch_moe_gemv_down_aggregate_q8_0_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q4_0_q8_1 = "launch_moe_gemv_down_aggregate_q4_0_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q4_1_q8_1 = "launch_moe_gemv_down_aggregate_q4_1_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q5_0_q8_1 = "launch_moe_gemv_down_aggregate_q5_0_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q5_1_q8_1 = "launch_moe_gemv_down_aggregate_q5_1_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q8_1_q8_1 = "launch_moe_gemv_down_aggregate_q8_1_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q2k_q8_1 = "launch_moe_gemv_down_aggregate_q2k_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q3k_q8_1 = "launch_moe_gemv_down_aggregate_q3k_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q4k_q8_1 = "launch_moe_gemv_down_aggregate_q4k_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q5k_q8_1 = "launch_moe_gemv_down_aggregate_q5k_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_gemv_down_aggregate_q6k_q8_1 = "launch_moe_gemv_down_aggregate_q6k_q8_1"(
        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut std::ffi::c_void,
    );
}
