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
        let num_blocks = key_cache.dims()[0];
        let total_slots = num_blocks * block_size;
        let features = num_kv_heads * head_size;

        // Flatten cache to [total_slots * features]
        let mut cache_flat = key_cache.reshape((total_slots * features,))?;

        for i in 0..num_tokens {
            let slot = slots[i] as usize;
            let block_idx = slot / block_size;
            let slot_idx = slot % block_size;
            let flat_slot = block_idx * block_size + slot_idx;

            let key_i: Vec<f32> = key
                .get(i)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let base = flat_slot * features;

            let indices: Vec<u32> = (0..features).map(|f| (base + f) as u32).collect();
            let idx_tensor = Tensor::from_vec(indices, features, key.device())?;
            let val_tensor = Tensor::from_vec(key_i, features, key.device())?;

            cache_flat.scatter_set(&idx_tensor, &val_tensor, 0)?;
        }
    }

    if write_v {
        let num_kv_heads = value.dims()[1];
        let head_size = value.dims()[2];
        let block_size = value_cache.dims()[3];
        let num_blocks = value_cache.dims()[0];
        let total_slots = num_blocks * block_size;
        let features = num_kv_heads * head_size;

        let mut cache_flat = value_cache.reshape((total_slots * features,))?;

        for i in 0..num_tokens {
            let slot = slots[i] as usize;
            let block_idx = slot / block_size;
            let slot_idx = slot % block_size;
            let flat_slot = block_idx * block_size + slot_idx;

            let value_i: Vec<f32> = value
                .get(i)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let base = flat_slot * features;

            let indices: Vec<u32> = (0..features).map(|f| (base + f) as u32).collect();
            let idx_tensor = Tensor::from_vec(indices, features, value.device())?;
            let val_tensor = Tensor::from_vec(value_i, features, value.device())?;

            cache_flat.scatter_set(&idx_tensor, &val_tensor, 0)?;
        }
    }

    Ok(())
}

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

    let num_blocks = key_cache.dims()[0];
    let total_slots = num_blocks * block_size;
    let features = num_kv_heads * head_size;

    let key_cache_flat: Vec<f32> = key_cache.reshape((total_slots * features,))?.to_vec1()?;
    let value_cache_flat: Vec<f32> = value_cache.reshape((total_slots * features,))?.to_vec1()?;

    let block_table_dims = block_table.dims2()?;
    let max_blocks = block_table_dims.1;
    let block_table_flat: Vec<i32> = if block_table.dtype() == DType::U32 {
        block_table.flatten_all()?.to_vec1::<u32>()?.iter().map(|&x| x as i32).collect()
    } else {
        block_table.flatten_all()?.to_vec1::<i32>()?
    };

    let cu_seq_lens_vec: Vec<i32> = cu_seq_lens.to_vec1()?;
    let num_seqs = cu_seq_lens_vec.len() - 1;

    let mut k_out_data: Vec<f32> = Vec::with_capacity(num_tokens * features);
    let mut v_out_data: Vec<f32> = Vec::with_capacity(num_tokens * features);

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

            for f in 0..features {
                k_out_data.push(key_cache_flat[flat_slot * features + f]);
                v_out_data.push(value_cache_flat[flat_slot * features + f]);
            }
            token_idx += 1;
        }
    }
    assert_eq!(token_idx, num_tokens);

    let k_out = Tensor::from_vec(
        k_out_data,
        (num_tokens, num_kv_heads, head_size),
        key_cache.device(),
    )?
    .to_dtype(out_dtype)?;
    let v_out = Tensor::from_vec(
        v_out_data,
        (num_tokens, num_kv_heads, head_size),
        value_cache.device(),
    )?
    .to_dtype(out_dtype)?;

    Ok((k_out, v_out))
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 2;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_SIZE: usize = 4;
    const X: usize = 2;
    const HEAD_SIZE_OVER_X: usize = HEAD_SIZE / X;

    fn make_cache(device: &Device) -> (Tensor, Tensor) {
        let key_cache = Tensor::zeros(
            (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE_OVER_X, BLOCK_SIZE, X),
            DType::F32,
            device,
        )
        .unwrap();
        let value_cache = Tensor::zeros(
            (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE),
            DType::F32,
            device,
        )
        .unwrap();
        (key_cache, value_cache)
    }

    #[test]
    fn reshape_and_cache_writes_to_correct_slots() {
        let device = Device::Cpu;
        let (key_cache, value_cache) = make_cache(&device);

        let key = Tensor::from_vec(
            (0..3 * NUM_KV_HEADS * HEAD_SIZE)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
            (3, NUM_KV_HEADS, HEAD_SIZE),
            &device,
        )
        .unwrap();
        let value = Tensor::from_vec(
            (0..3 * NUM_KV_HEADS * HEAD_SIZE)
                .map(|i| (i + 100) as f32)
                .collect::<Vec<_>>(),
            (3, NUM_KV_HEADS, HEAD_SIZE),
            &device,
        )
        .unwrap();

        let slot_mapping = Tensor::from_vec(vec![0i64, 1, 2], 3, &device).unwrap();

        reshape_and_cache(
            &key,
            &value,
            None,
            None,
            &key_cache,
            &value_cache,
            &slot_mapping,
            true,
            true,
        )
        .unwrap();

        let block_table = Tensor::from_vec(vec![0u32, 1u32], (1, 2), &device).unwrap();
        let cu_seq_lens = Tensor::from_vec(vec![0i32, 3], 2, &device).unwrap();

        let (k_out, v_out) = gather_kv_cache(
            &key_cache,
            &value_cache,
            None,
            None,
            None,
            None,
            &block_table,
            &cu_seq_lens,
            3,
            DType::F32,
            None,
            None,
        )
        .unwrap();

        let k_out_v: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        let v_out_v: Vec<f32> = v_out.flatten_all().unwrap().to_vec1().unwrap();
        let key_v: Vec<f32> = key.flatten_all().unwrap().to_vec1().unwrap();
        let value_v: Vec<f32> = value.flatten_all().unwrap().to_vec1().unwrap();

        for tok in 0..3usize {
            for h in 0..NUM_KV_HEADS {
                for d in 0..HEAD_SIZE {
                    assert_eq!(
                        k_out_v[tok * NUM_KV_HEADS * HEAD_SIZE + h * HEAD_SIZE + d],
                        key_v[tok * NUM_KV_HEADS * HEAD_SIZE + h * HEAD_SIZE + d]
                    );
                    assert_eq!(
                        v_out_v[tok * NUM_KV_HEADS * HEAD_SIZE + h * HEAD_SIZE + d],
                        value_v[tok * NUM_KV_HEADS * HEAD_SIZE + h * HEAD_SIZE + d]
                    );
                }
            }
        }
    }

    #[test]
    fn gather_kv_cache_empty_tokens() {
        let device = Device::Cpu;
        let (key_cache, value_cache) = make_cache(&device);
        let block_table = Tensor::from_vec(vec![0u32], (1, 1), &device).unwrap();
        let cu_seq_lens = Tensor::from_vec(vec![0i32, 0], 2, &device).unwrap();

        let (k_out, v_out) = gather_kv_cache(
            &key_cache,
            &value_cache,
            None,
            None,
            None,
            None,
            &block_table,
            &cu_seq_lens,
            0,
            DType::F32,
            None,
            None,
        )
        .unwrap();

        assert_eq!(k_out.dims(), &[0, NUM_KV_HEADS, HEAD_SIZE]);
        assert_eq!(v_out.dims(), &[0, NUM_KV_HEADS, HEAD_SIZE]);
    }

    #[test]
    fn reshape_and_cache_write_k_only() {
        let device = Device::Cpu;
        let (key_cache, value_cache) = make_cache(&device);

        let key = Tensor::from_vec(
            (0..1 * NUM_KV_HEADS * HEAD_SIZE)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
            (1, NUM_KV_HEADS, HEAD_SIZE),
            &device,
        )
        .unwrap();
        let value = Tensor::from_vec(
            (0..1 * NUM_KV_HEADS * HEAD_SIZE)
                .map(|i| (i + 100) as f32)
                .collect::<Vec<_>>(),
            (1, NUM_KV_HEADS, HEAD_SIZE),
            &device,
        )
        .unwrap();

        let slot_mapping = Tensor::from_vec(vec![0i64], 1, &device).unwrap();

        reshape_and_cache(
            &key,
            &value,
            None,
            None,
            &key_cache,
            &value_cache,
            &slot_mapping,
            true,
            false,
        )
        .unwrap();

        let v_sum: f32 = value_cache
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum();
        assert_eq!(v_sum, 0.0);
    }
}
