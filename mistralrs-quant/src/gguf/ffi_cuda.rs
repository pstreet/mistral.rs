//! NVIDIA-role dlsym entries for the fused MoE decode kernels.
//!
//! Same C signatures as the hip companion (`ffi`), resolved from the nvcc
//! companion (`crate::cuda_plugin`) at first use. Only the decode-path
//! symbols: Q8_1 quantize (bf16/f16/f32 inputs), fused gate_up, and the
//! down-aggregate reduce, for every GGUF weight dtype.

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

// ---- grouped prefill + dense/grouped MMQ entries (moe_grouped.cu,
// mmq_quantize.cu, mmq_instance_*.cu) ----
cuda_ffi! {

    cuda_launch_moe_dispatch = "launch_moe_dispatch"(

        topk_ids: *const i32,
        expert_bounds: *mut i32,
        sorted_token_ids: *mut i32,
        sorted_source_ids: *mut i32,
        total_assignments: i32,
        num_experts: i32,
        topk: i32,
        expert_counts: *mut i32,
        expert_cursors: *mut i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q8_0 = "launch_moe_grouped_gemm_q8_0"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q4_0 = "launch_moe_grouped_gemm_q4_0"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q4_1 = "launch_moe_grouped_gemm_q4_1"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q5_0 = "launch_moe_grouped_gemm_q5_0"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q5_1 = "launch_moe_grouped_gemm_q5_1"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q8_1 = "launch_moe_grouped_gemm_q8_1"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q2k = "launch_moe_grouped_gemm_q2k"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q3k = "launch_moe_grouped_gemm_q3k"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q4k = "launch_moe_grouped_gemm_q4k"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q5k = "launch_moe_grouped_gemm_q5k"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_grouped_gemm_q6k = "launch_moe_grouped_gemm_q6k"(

        all_weights: *const std::ffi::c_void,
        all_inputs: *const std::ffi::c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_moe_weighted_reduce_flat = "launch_moe_weighted_reduce_flat"(

        inputs: *const std::ffi::c_void,
        topk_weights: *const f32,
        outputs: *mut std::ffi::c_void,
        num_tokens: i32,
        hidden: i32,
        topk: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;


    cuda_launch_moe_weighted_reduce_flat_bf16 = "launch_moe_weighted_reduce_flat_bf16"(

        inputs: *const std::ffi::c_void,
        topk_weights: *const f32,
        outputs: *mut std::ffi::c_void,
        num_tokens: i32,
        hidden: i32,
        topk: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;


    cuda_launch_moe_weighted_reduce_flat_f16_input = "launch_moe_weighted_reduce_flat_f16_input"(

        inputs: *const std::ffi::c_void,
        topk_weights: *const f32,
        outputs: *mut std::ffi::c_void,
        num_tokens: i32,
        hidden: i32,
        topk: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;


    cuda_launch_moe_weighted_reduce_flat_bf16_input = "launch_moe_weighted_reduce_flat_bf16_input"(

        inputs: *const std::ffi::c_void,
        topk_weights: *const f32,
        outputs: *mut std::ffi::c_void,
        num_tokens: i32,
        hidden: i32,
        topk: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;


    cuda_launch_mmq_quantize_q8_1_D4 = "launch_mmq_quantize_q8_1_D4"(

        x: *const std::ffi::c_void,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_quantize_q8_1_DS4 = "launch_mmq_quantize_q8_1_DS4"(

        x: *const std::ffi::c_void,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_quantize_q8_1_D2S6 = "launch_mmq_quantize_q8_1_D2S6"(

        x: *const std::ffi::c_void,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_quantize_glu_q8_1_D4 = "launch_mmq_quantize_glu_q8_1_D4"(

        gate: *const std::ffi::c_void,
        up: *const std::ffi::c_void,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        ne0: i64,
        ne1: i64,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_quantize_glu_q8_1_D4_f32 = "launch_mmq_quantize_glu_q8_1_D4_f32"(

        gate: *const f32,
        up: *const f32,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        ne00: i64,
        s01: i64,
        ne0: i64,
        ne1: i64,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_quantize_glu_q8_1_DS4 = "launch_mmq_quantize_glu_q8_1_DS4"(

        gate: *const std::ffi::c_void,
        up: *const std::ffi::c_void,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        ne0: i64,
        ne1: i64,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_quantize_glu_q8_1_DS4_f32 = "launch_mmq_quantize_glu_q8_1_DS4_f32"(

        gate: *const f32,
        up: *const f32,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        ne00: i64,
        s01: i64,
        ne0: i64,
        ne1: i64,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_quantize_glu_q8_1_D2S6 = "launch_mmq_quantize_glu_q8_1_D2S6"(

        gate: *const std::ffi::c_void,
        up: *const std::ffi::c_void,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        ne0: i64,
        ne1: i64,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_quantize_glu_q8_1_D2S6_f32 = "launch_mmq_quantize_glu_q8_1_D2S6_f32"(

        gate: *const f32,
        up: *const f32,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        ne00: i64,
        s01: i64,
        ne0: i64,
        ne1: i64,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q2_k = "launch_mmq_gguf_q2_k"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q2_k_moe = "launch_mmq_gguf_q2_k_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q3_k = "launch_mmq_gguf_q3_k"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q3_k_moe = "launch_mmq_gguf_q3_k_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q4_0 = "launch_mmq_gguf_q4_0"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q4_0_moe = "launch_mmq_gguf_q4_0_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q4_1 = "launch_mmq_gguf_q4_1"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q4_1_moe = "launch_mmq_gguf_q4_1_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q4_k = "launch_mmq_gguf_q4_k"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q4_k_moe = "launch_mmq_gguf_q4_k_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q5_0 = "launch_mmq_gguf_q5_0"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q5_0_moe = "launch_mmq_gguf_q5_0_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q5_1 = "launch_mmq_gguf_q5_1"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q5_1_moe = "launch_mmq_gguf_q5_1_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q5_k = "launch_mmq_gguf_q5_k"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q5_k_moe = "launch_mmq_gguf_q5_k_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q6_k = "launch_mmq_gguf_q6_k"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q6_k_moe = "launch_mmq_gguf_q6_k_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q8_0 = "launch_mmq_gguf_q8_0"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );

    cuda_launch_mmq_gguf_q8_0_moe = "launch_mmq_gguf_q8_0_moe"(

        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        ids_dst: *const i32,
        expert_bounds: *const i32,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_dst: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        num_experts: i64,
        ncols_max: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut std::ffi::c_void,
    );

}
