//! NVIDIA-role dlsym entries for the GDN companion kernels.
//!
//! Same names and C signatures as the hip companion (`ffi`), resolved from
//! the nvcc companion (`crate::cuda_plugin`) at first use. The GDN twin
//! (`gdn_cuda`) is a path-swap of the hip module, so plain names keep the
//! launcher tables uniform.

macro_rules! cuda_kernel {
    ($($fname:ident = $sym:literal ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty )?;)*) => {$(
        #[allow(dead_code)]
        #[allow(non_snake_case)]
        pub(crate) unsafe extern "C" fn $fname($($arg : $ty),*) $(-> $ret )? {
            type F = unsafe extern "C" fn($($ty),*) $(-> $ret )?;
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

cuda_kernel! {

    causal_conv1d_full = "causal_conv1d_full"(

        x: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        conv_state_in: *const std::ffi::c_void,
        conv_state_out: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch_size: i32,
        conv_dim: i32,
        seq_len: i32,
        kernel_size: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        slot_indices: *const i32,
        dtype: i32,
        stream: i64,
    );

    causal_conv1d_update = "causal_conv1d_update"(

        x: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        conv_state: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch_size: i32,
        conv_dim: i32,
        kernel_size: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        slot_indices: *const i32,
        dtype: i32,
        stream: i64,
    );

    chunked_gated_delta_rule_recurrence = "chunked_gated_delta_rule_recurrence"(

        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut std::ffi::c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    fused_gdn_gating = "fused_gdn_gating"(

        b: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        beta_out: *mut std::ffi::c_void,
        g_out: *mut std::ffi::c_void,
        total_elements: i32,
        num_heads: i32,
        dtype: i32,
        stream: i64,
    );

    gated_delta_rule_recurrence = "gated_delta_rule_recurrence"(

        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut std::ffi::c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    gdn_decode_recurrence = "gdn_decode_recurrence"(

        mixed_qkv: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        state: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
        batch_size: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        tiled_v_heads: i32,
        b_batch_stride: i64,
        b_head_stride: i64,
        a_batch_stride: i64,
        a_head_stride: i64,
        slot_indices: *const i32,
        kernel_kind: i32,
        dtype: i32,
        state_dtype: i32,
        stream: i64,
    );

    gdn_deferred_recurrence_rmsnorm_gate_value_major_128 = "gdn_deferred_recurrence_rmsnorm_gate_value_major_128"(

        mixed_qkv: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        state_pool: *mut f32,
        output: *mut std::ffi::c_void,
        quantized_output: *mut std::ffi::c_void,
        output_scales: *mut f32,
        scale_stride_m: i32,
        scale_layout: i32,
        active_slots: *const u32,
        deferred_key: *mut f32,
        deferred_delta: *mut f32,
        deferred_decay: *mut f32,
        deferred_cursor: *mut u32,
        gate: *const std::ffi::c_void,
        norm_weight: *const std::ffi::c_void,
        b_stride_b: i64,
        b_stride_h: i64,
        a_stride_b: i64,
        a_stride_h: i64,
        gate_stride_b: i64,
        gate_stride_h: i64,
        gate_stride_v: i64,
        batch_size: i32,
        capacity: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        tiled_v_heads: i32,
        norm_eps: f32,
        stream: i64,
    );

    gdn_extract_ragged_conv_state = "gdn_extract_ragged_conv_state"(

        padded_input: *const std::ffi::c_void,
        initial_state: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        cu_seqlens: *const u32,
        batch_size: i32,
        padded_len: i32,
        channels: i32,
        state_width: i32,
        input_batch_stride: i64,
        input_token_stride: i64,
        dtype: i32,
        stream: i64,
    );

    gdn_flush_deferred_state_value_major_128 = "gdn_flush_deferred_state_value_major_128"(

        state_pool: *mut f32,
        active_slots: *const u32,
        deferred_key: *const f32,
        deferred_delta: *const f32,
        deferred_decay: *const f32,
        deferred_cursor: *mut u32,
        batch_size: i32,
        capacity: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        tiled_v_heads: i32,
        stream: i64,
    );

    gdn_packed_to_padded = "gdn_packed_to_padded"(

        source: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        cu_seqlens: *const u32,
        batch_size: i32,
        padded_len: i32,
        width: i32,
        source_token_stride: i64,
        padding_value: f32,
        dtype: i32,
        stream: i64,
    );

    gdn_padded_to_packed = "gdn_padded_to_packed"(

        source: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        cu_seqlens: *const u32,
        batch_size: i32,
        padded_len: i32,
        width: i32,
        source_batch_stride: i64,
        source_token_stride: i64,
        source_feature_stride: i64,
        feature_inner_width: i32,
        dtype: i32,
        stream: i64,
    );

    gdn_pending_transition_apply_batched = "gdn_pending_transition_apply_batched"(

        pointer_table: *const u64,
        active_slots: *const u32,
        layer_count: i32,
        batch_size: i32,
        max_rows: i32,
        capacity: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        conv_dim: i32,
        conv_width: i32,
        tiled_v_heads: i32,
        activation_dtype: i32,
        state_dtype: i32,
        stream: i64,
    );

    gdn_pending_transition_publish_batched = "gdn_pending_transition_publish_batched"(

        pointer_table: *const u64,
        keep_rows: *const u32,
        destination_slots: *const u32,
        layer_count: i32,
        batch_size: i32,
        max_rows: i32,
        destination_capacity: i32,
        stream: i64,
    );

    gdn_prepare_recurrence = "gdn_prepare_recurrence"(

        mixed_qkv: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        v_out: *mut f32,
        g_out: *mut f32,
        beta_out: *mut f32,
        batch_size: i32,
        seq_len: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        tiled_v_heads: i32,
        dtype: i32,
        stream: i64,
    );

    gdn_rmsnorm_gated = "gdn_rmsnorm_gated"(

        x: *const std::ffi::c_void,
        gate: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        rows: i32,
        hidden_dim: i32,
        outer_dim_1: i32,
        outer_dim_2: i32,
        x_stride_0: i64,
        x_stride_1: i64,
        x_stride_2: i64,
        x_stride_3: i64,
        gate_stride_0: i64,
        gate_stride_1: i64,
        gate_stride_2: i64,
        gate_stride_3: i64,
        eps: f32,
        dtype: i32,
        stream: i64,
    );

    gdn_rmsnorm_gated_quantized_bf16 = "gdn_rmsnorm_gated_quantized_bf16"(

        x: *const std::ffi::c_void,
        gate: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        scales: *mut f32,
        rows: i32,
        groups: i32,
        scale_stride_m: i32,
        scale_layout: i32,
        outer_dim_1: i32,
        outer_dim_2: i32,
        x_stride_0: i64,
        x_stride_1: i64,
        x_stride_2: i64,
        x_stride_3: i64,
        gate_stride_0: i64,
        gate_stride_1: i64,
        gate_stride_2: i64,
        gate_stride_3: i64,
        eps: f32,
        stream: i64,
    );

    gdn_speculative_conv_checkpoints = "gdn_speculative_conv_checkpoints"(

        x: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        state_pool: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
        active_slots: *const u32,
        batch_size: i32,
        seq_len: i32,
        conv_dim: i32,
        kernel_size: i32,
        checkpoint_lanes: i32,
        write_checkpoints: i32,
        pending_conv_input: *mut std::ffi::c_void,
        pending_keep_rows: *const u32,
        pending_epochs: *const u32,
        conv_applied_epochs: *mut u32,
        max_pending_rows: i32,
        pending_capacity: i32,
        conv_blocks: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        dtype: i32,
        stream: i64,
    );

    gdn_speculative_recurrence_checkpoints = "gdn_speculative_recurrence_checkpoints"(

        mixed_qkv: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        state_pool: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
        quantized_output: *mut std::ffi::c_void,
        output_scales: *mut f32,
        scale_stride_m: i32,
        scale_layout: i32,
        active_slots: *const u32,
        gate: *const std::ffi::c_void,
        norm_weight: *const std::ffi::c_void,
        transition_key: *mut f32,
        transition_delta: *mut f32,
        transition_decay: *mut f32,
        pending_key_banks: *mut f32,
        pending_key_bank: *const u32,
        pending_delta: *mut f32,
        pending_decay: *mut f32,
        pending_keep_rows: *const u32,
        pending_epochs: *const u32,
        recurrent_applied_epochs: *mut u32,
        max_pending_rows: i32,
        pending_capacity: i32,
        slot_indexed_transitions: i32,
        b_stride_b: i64,
        b_stride_s: i64,
        b_stride_h: i64,
        a_stride_b: i64,
        a_stride_s: i64,
        a_stride_h: i64,
        gate_stride_b: i64,
        gate_stride_s: i64,
        gate_stride_h: i64,
        gate_stride_v: i64,
        batch_size: i32,
        seq_len: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        checkpoint_lanes: i32,
        tiled_v_heads: i32,
        value_major: i32,
        norm_eps: f32,
        dtype: i32,
        state_dtype: i32,
        stream: i64,
    );

    gdn_speculative_state_commit = "gdn_speculative_state_commit"(

        mixed_qkv: *const std::ffi::c_void,
        convolved_qkv: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        initial_conv_state: *const std::ffi::c_void,
        initial_recurrent_state: *const std::ffi::c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        conv_state_pool: *mut std::ffi::c_void,
        recurrent_state_pool: *mut std::ffi::c_void,
        keep_rows: *const u32,
        slot_indices: *const u32,
        batch_size: i32,
        seq_len: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        kernel_size: i32,
        tiled_v_heads: i32,
        value_major: i32,
        dtype: i32,
        state_dtype: i32,
        stream: i64,
    );

    gdn_speculative_transition_commit_batched = "gdn_speculative_transition_commit_batched"(

        pointer_table: *const u64,
        keep_rows: *const u32,
        active_slots: *const u32,
        layer_count: i32,
        batch_size: i32,
        seq_len: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        conv_dim: i32,
        conv_width: i32,
        tiled_v_heads: i32,
        value_major: i32,
        activation_dtype: i32,
        state_dtype: i32,
        stream: i64,
    );

    gdn_speculative_transition_stage_batched = "gdn_speculative_transition_stage_batched"(

        pointer_table: *const u64,
        keep_rows: *const u32,
        destination_slots: *const u32,
        layer_count: i32,
        batch_size: i32,
        seq_len: i32,
        max_rows: i32,
        destination_capacity: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        conv_dim: i32,
        activation_dtype: i32,
        stream: i64,
    );

    vmajor_chunked_gated_delta_rule_recurrence = "vmajor_chunked_gated_delta_rule_recurrence"(

        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut std::ffi::c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    ) -> i32;


    vmajor_grouped_warp_gated_delta_rule_recurrence = "vmajor_grouped_warp_gated_delta_rule_recurrence"(

        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut std::ffi::c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        values_per_warp: i32,
        state_dtype: i32,
        stream: i64,
    ) -> i32;


    vmajor_warp_gated_delta_rule_recurrence = "vmajor_warp_gated_delta_rule_recurrence"(

        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut std::ffi::c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

    warp_gated_delta_rule_recurrence = "warp_gated_delta_rule_recurrence"(

        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut std::ffi::c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );

}
