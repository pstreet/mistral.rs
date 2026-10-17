use candle_core::{DType, Result, Tensor};

pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    _k_scale: Option<&Tensor>,
    _v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    write_k: bool,
    write_v: bool,
) -> Result<()> {
    let num_tokens = key.dims()[0];
    if num_tokens == 0 {
        return Ok(());
    }

    // Only support native (non-quantized) dtypes on CPU.
    if key_cache.dtype() != key.dtype() {
        candle_core::bail!(
            "reshape_and_cache on CPU requires matching dtypes: key {:?} vs cache {:?}",
            key.dtype(),
            key_cache.dtype()
        );
    }

    let slots: Vec<i64> = slot_mapping.to_vec1()?;

    if write_k {
        let num_kv_heads = key.dims()[1];
        let head_size = key.dims()[2];
        let block_size = key_cache.dims()[3];
        let x = key_cache.dims()[4];
        let head_size_over_x = key_cache.dims()[2];

        // Flatten key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        // -> [num_blocks * block_size, num_kv_heads, head_size_over_x, x]
        // -> [num_blocks * block_size, num_kv_heads * head_size]
        let num_blocks = key_cache.dims()[0];
        let total_slots = num_blocks * block_size;

        // Compute flat slot indices for each token.
        let mut flat_indices: Vec<u32> = Vec::with_capacity(num_tokens);
        for i in 0..num_tokens {
            let slot = slots[i] as usize;
            let block_idx = slot / block_size;
            let slot_idx = slot % block_size;
            flat_indices.push((block_idx * block_size + slot_idx) as u32);
        }
        let indices = Tensor::from_vec(flat_indices, num_tokens, key.device())?;

        // key: [num_tokens, num_kv_heads, head_size] -> [num_tokens, num_kv_heads * head_size]
        let key_flat = key.reshape((num_tokens, num_kv_heads * head_size))?;
        // key_cache flattened: [total_slots, num_kv_heads * head_size]
        let mut cache_flat = key_cache.reshape((total_slots, num_kv_heads * head_size))?;

        // scatter_set along dim 0
        cache_flat.scatter_set(&indices, &key_flat, 0)?;
    }

    if write_v {
        let num_kv_heads = value.dims()[1];
        let head_size = value.dims()[2];
        let block_size = value_cache.dims()[3];
        let num_blocks = value_cache.dims()[0];
        let total_slots = num_blocks * block_size;

        let mut flat_indices: Vec<u32> = Vec::with_capacity(num_tokens);
        for i in 0..num_tokens {
            let slot = slots[i] as usize;
            let block_idx = slot / block_size;
            let slot_idx = slot % block_size;
            flat_indices.push((block_idx * block_size + slot_idx) as u32);
        }
        let indices = Tensor::from_vec(flat_indices, num_tokens, value.device())?;

        let value_flat = value.reshape((num_tokens, num_kv_heads * head_size))?;
        let mut cache_flat = value_cache.reshape((total_slots, num_kv_heads * head_size))?;

        cache_flat.scatter_set(&indices, &value_flat, 0)?;
    }

    Ok(())
}

pub fn paged_attention(
    _query: &Tensor,
    _k_scale: Option<&Tensor>,
    _v_scale: Option<&Tensor>,
    _k_res: Option<&Tensor>,
    _v_res: Option<&Tensor>,
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _block_tables: &Tensor,
    _context_lens: &Tensor,
    _alibi_slopes: Option<&Tensor>,
    _max_context_len: usize,
    _softmax_scale: f32,
    _softcap: f32,
    _sinks: Option<&Tensor>,
    _k_quant: Option<crate::BlockQuantKind>,
    _v_quant: Option<crate::BlockQuantKind>,
) -> Result<Tensor> {
    // On CPU, paged_attention is never called directly because
    // run_decode falls back to run_decode_gather_sdpa.
    candle_core::bail!("paged_attention GPU kernel is not available on CPU")
}

pub fn reshape_and_cache_q4(
    _key: &Tensor,
    _value: &Tensor,
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _k_scale: Option<&Tensor>,
    _v_scale: Option<&Tensor>,
    _k_res: Option<&Tensor>,
    _v_res: Option<&Tensor>,
    _slot_mapping: &Tensor,
    _write_k: bool,
    _write_v: bool,
) -> Result<()> {
    candle_core::bail!("reshape_and_cache_q4 is not supported on CPU")
}

pub fn reshape_and_cache_q8(
    _key: &Tensor,
    _value: &Tensor,
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _k_scale: Option<&Tensor>,
    _v_scale: Option<&Tensor>,
    _slot_mapping: &Tensor,
    _write_k: bool,
    _write_v: bool,
) -> Result<()> {
    candle_core::bail!("reshape_and_cache_q8 is not supported on CPU")
}
