//! Dual-only NVIDIA-role entries for the 9 dense paged-attention kernels.
//!
//! Same C symbols as the hip companion, resolved from the nvcc companion
//! (`cuda_plugin`) instead. Rust fn names carry a `cuda_` prefix so both
//! roles coexist in one binary; the CUstream type is the `cuda_backend`
//! one (distinct from the hip role's). Only the dense-model kernels are
//! covered (v1/v2, reshape_and_cache{,_q8,_q4}, copy_blocks, gather_kv);
//! FlashInfer/MLA/FA3/sinks stay hip-only until their own slices.

use core::ffi::{c_int, c_long, c_void};

use candle_core::cuda_backend::cudarc::driver::sys::CUstream;

macro_rules! cuda_kernel {
    ($($( #[$attr:meta] )* $fname:ident = $sym:literal ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)? ; )*) => {$(
        $( #[$attr] )*
        #[allow(dead_code)]
        #[allow(non_snake_case)]
        pub(crate) unsafe extern "C" fn $fname($($arg : $ty),*) $(-> $ret)? {
            type F = unsafe extern "C" fn($($ty),*) $(-> $ret)?;
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
            // SAFETY: same as above; arguments cross unchanged from the
            // probe-gated caller.
            unsafe { f($($arg),*) }
        }
    )*};
}

cuda_kernel! {
    cuda_reshape_and_cache = "reshape_and_cache"(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const c_long,
        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        stream: CUstream,
        dtype: u32,
        k_cache_dtype: u32,
        v_cache_dtype: u32,
        write_k: bool,
        write_v: bool,
        k_scale: *const f32,
        v_scale: *const f32,
    );

    cuda_reshape_and_cache_q8 = "reshape_and_cache_q8"(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scales: *mut f32,
        v_scales: *mut f32,
        slot_mapping: *const c_long,
        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        stream: CUstream,
        dtype: u32,
        write_k: bool,
        write_v: bool,
    );

    cuda_reshape_and_cache_q4 = "reshape_and_cache_q4"(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scales: *mut f32,
        v_scales: *mut f32,
        k_res: *mut u8,
        v_res: *mut u8,
        slot_mapping: *const c_long,
        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        stream: CUstream,
        dtype: u32,
        write_k: bool,
        write_v: bool,
    );

    cuda_gather_kv_cache = "gather_kv_cache"(
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_out: *const c_void,
        v_out: *const c_void,
        k_scale: *const f32,
        v_scale: *const f32,
        k_res: *const u8,
        v_res: *const u8,
        block_table: *const c_int,
        cu_seq_lens: *const c_int,
        num_tokens: c_int,
        num_seqs: c_int,
        block_size: c_int,
        block_table_stride: c_int,
        num_kv_heads: c_int,
        head_size: c_int,
        x: c_int,
        stream: CUstream,
        out_dtype: u32,
        k_cache_dtype: u32,
        v_cache_dtype: u32,
    );

    cuda_paged_attention_v1_f16 = "paged_attention_v1_f16"(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        v_block_stride: c_int,
        v_head_stride: c_int,
        stream: CUstream,
        k_cache_dtype: u32,
        v_cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        k_res: *const u8,
        v_res: *const u8,
        sinks: *const f32,
    );

    cuda_paged_attention_v1_bf16 = "paged_attention_v1_bf16"(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        v_block_stride: c_int,
        v_head_stride: c_int,
        stream: CUstream,
        k_cache_dtype: u32,
        v_cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        k_res: *const u8,
        v_res: *const u8,
        sinks: *const f32,
    );

    cuda_paged_attention_v1_f32 = "paged_attention_v1_f32"(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        v_block_stride: c_int,
        v_head_stride: c_int,
        stream: CUstream,
        k_cache_dtype: u32,
        v_cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        k_res: *const u8,
        v_res: *const u8,
        sinks: *const f32,
    );

    cuda_paged_attention_v2_f16 = "paged_attention_v2_f16"(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        v_block_stride: c_int,
        v_head_stride: c_int,
        stream: CUstream,
        k_cache_dtype: u32,
        v_cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        k_res: *const u8,
        v_res: *const u8,
        sinks: *const f32,
    );

    cuda_paged_attention_v2_bf16 = "paged_attention_v2_bf16"(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        v_block_stride: c_int,
        v_head_stride: c_int,
        stream: CUstream,
        k_cache_dtype: u32,
        v_cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        k_res: *const u8,
        v_res: *const u8,
        sinks: *const f32,
    );

    cuda_paged_attention_v2_f32 = "paged_attention_v2_f32"(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        v_block_stride: c_int,
        v_head_stride: c_int,
        stream: CUstream,
        k_cache_dtype: u32,
        v_cache_dtype: u32,
        k_scale: *const f32,
        v_scale: *const f32,
        k_res: *const u8,
        v_res: *const u8,
        sinks: *const f32,
    );

    cuda_copy_blocks_bf16 = "copy_blocks_bf16"(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block_key: i32,
        numel_per_block_value: i32,
        stream: i64,
    );

    cuda_copy_blocks_f16 = "copy_blocks_f16"(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block_key: i32,
        numel_per_block_value: i32,
        stream: i64,
    );

    cuda_copy_blocks_f32 = "copy_blocks_f32"(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block_key: i32,
        numel_per_block_value: i32,
        stream: i64,
    );

    cuda_copy_blocks_u8 = "copy_blocks_u8"(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block_key: i32,
        numel_per_block_value: i32,
        stream: i64,
    );
}
