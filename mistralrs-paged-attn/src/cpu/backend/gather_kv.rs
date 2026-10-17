use candle_core::{DType, Result, Tensor};

pub fn gather_kv_cache(
    key_cache: &Tensor,
    value_cache: &Tensor,
    _k_scale: Option<&Tensor>,
    _v_scale: Option<&Tensor>,
    _k_res: Option<&Tensor>,
    _v_res: Option<&Tensor>,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    num_tokens: usize,
    out_dtype: DType,
    k_quant: Option<crate::BlockQuantKind>,
    v_quant: Option<crate::BlockQuantKind>,
) -> Result<(Tensor, Tensor)> {
    if k_quant.is_some() || v_quant.is_some() {
        candle_core::bail!("block-quantized KV cache is only supported on CUDA/ROCm");
    }

    let k_dims = key_cache.dims5()?;
    let num_kv_heads = k_dims.1;
    let head_size_over_x = k_dims.2;
    let block_size = k_dims.3;
    let x = k_dims.4;
    let head_size = head_size_over_x * x;

    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;

    if num_tokens == 0 {
        let k_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, key_cache.device())?;
        let v_out = Tensor::zeros(
            (0, num_kv_heads, head_size),
            out_dtype,
            value_cache.device(),
        )?;
        return Ok((k_out, v_out));
    }

    // Flatten key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    // -> [num_blocks * block_size, num_kv_heads * head_size]
    let num_blocks = key_cache.dims()[0];
    let total_slots = num_blocks * block_size;
    let key_cache_flat = key_cache.reshape((total_slots, num_kv_heads * head_size))?;
    let value_cache_flat = value_cache.reshape((total_slots, num_kv_heads * head_size))?;

    // block_table: [num_seqs, max_blocks]
    let block_table_dims = block_table.dims2()?;
    let max_blocks = block_table_dims.1;
    let block_table_flat: Vec<i32> = block_table.flatten_all()?.to_vec1()?;

    // cu_seq_lens: [num_seqs + 1]
    let cu_seq_lens_vec: Vec<i32> = cu_seq_lens.to_vec1()?;
    let num_seqs = cu_seq_lens_vec.len() - 1;

    // For each token, compute the flat slot index in the cache.
    let mut flat_indices: Vec<u32> = Vec::with_capacity(num_tokens);
    let mut token_idx = 0;
    for s in 0..num_seqs {
        let seq_start = cu_seq_lens_vec[s] as usize;
        let seq_end = cu_seq_lens_vec[s + 1] as usize;
        for t in seq_start..seq_end {
            let token_in_seq = t - seq_start;
            let block_idx_in_table = token_in_seq / block_size;
            let slot_idx = token_in_seq % block_size;
            let block_id = block_table_flat[s * max_blocks + block_idx_in_table] as usize;
            let flat_slot = block_id * block_size + slot_idx;
            flat_indices.push(flat_slot as u32);
            token_idx += 1;
        }
    }

    let indices = Tensor::from_vec(flat_indices, num_tokens, key_cache.device())?;

    // Gather from cache.
    let k_gathered = key_cache_flat.index_select(&indices, 0)?;
    let v_gathered = value_cache_flat.index_select(&indices, 0)?;

    // Reshape to [num_tokens, num_kv_heads, head_size] and convert dtype.
    let k_out = k_gathered
        .reshape((num_tokens, num_kv_heads, head_size))?
        .to_dtype(out_dtype)?;
    let v_out = v_gathered
        .reshape((num_tokens, num_kv_heads, head_size))?
        .to_dtype(out_dtype)?;

    Ok((k_out, v_out))
}
