use candle_core::{DType, Result, Tensor};

fn read_int_vec(t: &Tensor) -> Result<Vec<i64>> {
    match t.dtype() {
        DType::I64 => t.flatten_all()?.to_vec1::<i64>(),
        DType::U32 => Ok(t
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|x| x as i64)
            .collect()),
        DType::I32 => Ok(t
            .flatten_all()?
            .to_vec1::<i32>()?
            .into_iter()
            .map(|x| x as i64)
            .collect()),
        other => candle_core::bail!("unsupported integer dtype {other:?} for CPU paged attention"),
    }
}

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

    let slots = read_int_vec(slot_mapping)?;
    let dev = key.device();

    if write_k {
        let num_kv_heads = key.dims()[1];
        let head_size = key.dims()[2];
        let block_size = key_cache.dims()[3];
        let num_blocks = key_cache.dims()[0];
        let total_slots = num_blocks * block_size;
        let features = num_kv_heads * head_size;

        let key = key.to_dtype(key_cache.dtype())?;
        let mut cache_flat = key_cache.reshape((total_slots * features,))?;
        for i in 0..num_tokens {
            let slot = slots[i] as usize;
            let flat_slot = slot / block_size * block_size + slot % block_size;
            let key_i = key.get(i)?.flatten_all()?;
            let base = flat_slot * features;
            let idx: Vec<u32> = (0..features).map(|f| (base + f) as u32).collect();
            let idx_t = Tensor::from_vec(idx, features, dev)?;
            cache_flat.scatter_set(&idx_t, &key_i, 0)?;
        }
    }

    if write_v {
        let num_kv_heads = value.dims()[1];
        let head_size = value.dims()[2];
        let block_size = value_cache.dims()[3];
        let num_blocks = value_cache.dims()[0];
        let total_slots = num_blocks * block_size;
        let features = num_kv_heads * head_size;

        let value = value.to_dtype(value_cache.dtype())?;
        let mut cache_flat = value_cache.reshape((total_slots * features,))?;
        for i in 0..num_tokens {
            let slot = slots[i] as usize;
            let flat_slot = slot / block_size * block_size + slot % block_size;
            let value_i = value.get(i)?.flatten_all()?;
            let base = flat_slot * features;
            let idx: Vec<u32> = (0..features).map(|f| (base + f) as u32).collect();
            let idx_t = Tensor::from_vec(idx, features, dev)?;
            cache_flat.scatter_set(&idx_t, &value_i, 0)?;
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

    let block_table_flat = read_int_vec(&block_table)?;
    let cu_vec = read_int_vec(&cu_seq_lens)?;
    let num_seqs = cu_vec.len() - 1;

    let max_blocks = block_table.dims2()?.1;

    // Element-major gather indices: token t of seq s lives at
    // block_table[s][t/block_size]*block_size + t%block_size in slot space,
    // i.e. flat_slot*features + f in the flattened cache.
    let mut elem_idx: Vec<u32> = Vec::with_capacity(num_tokens * features);
    for s in 0..num_seqs {
        let seq_start = cu_vec[s] as usize;
        let seq_end = cu_vec[s + 1] as usize;
        for t in seq_start..seq_end {
            let token_in_seq = t - seq_start;
            let block_id = block_table_flat[s * max_blocks + token_in_seq / block_size] as usize;
            let flat_slot = block_id * block_size + token_in_seq % block_size;
            let base = flat_slot * features;
            for f in 0..features {
                elem_idx.push((base + f) as u32);
            }
        }
    }
    assert_eq!(elem_idx.len(), num_tokens * features);

    let dev = key_cache.device();
    let idx_t = Tensor::from_vec(elem_idx, num_tokens * features, dev)?;
    let key_cache_flat = key_cache.reshape((total_slots * features,))?;
    let value_cache_flat = value_cache.reshape((total_slots * features,))?;

    let k_out = key_cache_flat
        .index_select(&idx_t, 0)?
        .reshape((num_tokens, num_kv_heads, head_size))?
        .to_dtype(out_dtype)?;
    let v_out = value_cache_flat
        .index_select(&idx_t, 0)?
        .reshape((num_tokens, num_kv_heads, head_size))?
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
    // run_decode routes CPU devices to run_decode_gather_sdpa before this
    // kernel can be reached.
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

    fn make_cache(device: &Device, dtype: DType) -> (Tensor, Tensor) {
        let key_cache = Tensor::zeros(
            (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE_OVER_X, BLOCK_SIZE, X),
            dtype,
            device,
        )
        .unwrap();
        let value_cache = Tensor::zeros(
            (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE),
            dtype,
            device,
        )
        .unwrap();
        (key_cache, value_cache)
    }

    #[test]
    fn reshape_and_cache_writes_to_correct_slots() {
        let device = Device::Cpu;
        let (key_cache, value_cache) = make_cache(&device, DType::F32);

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

        // contiguous slot mapping; u32 exercises the dtype dispatch
        let slot_mapping = Tensor::from_vec(vec![0u32, 1, 2], 3, &device).unwrap();

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

        // u32 block table + u32 cu_seq_lens: the dtypes prod builds them with
        let block_table = Tensor::from_vec(vec![0u32, 1u32], (1, 2), &device).unwrap();
        let cu_seq_lens = Tensor::from_vec(vec![0u32, 3], 2, &device).unwrap();

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
                    let off = tok * NUM_KV_HEADS * HEAD_SIZE + h * HEAD_SIZE + d;
                    assert_eq!(k_out_v[off], key_v[off]);
                    assert_eq!(v_out_v[off], value_v[off]);
                }
            }
        }
    }

    #[test]
    fn gather_kv_cache_empty_tokens() {
        let device = Device::Cpu;
        let (key_cache, value_cache) = make_cache(&device, DType::F32);
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
        let (key_cache, value_cache) = make_cache(&device, DType::F32);

        let key = Tensor::from_vec(
            (0..NUM_KV_HEADS * HEAD_SIZE)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
            (1, NUM_KV_HEADS, HEAD_SIZE),
            &device,
        )
        .unwrap();
        let value = Tensor::from_vec(
            (0..NUM_KV_HEADS * HEAD_SIZE)
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

    #[test]
    fn reshape_and_cache_cross_dtype_input() {
        // f32 activations into a bf16 cache must convert, not bail
        let device = Device::Cpu;
        let (key_cache, value_cache) = make_cache(&device, DType::BF16);

        let key = Tensor::from_vec(
            (0..NUM_KV_HEADS * HEAD_SIZE)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
            (1, NUM_KV_HEADS, HEAD_SIZE),
            &device,
        )
        .unwrap();
        let value = key.clone();

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
            true,
        )
        .unwrap();

        let block_table = Tensor::from_vec(vec![0u32], (1, 1), &device).unwrap();
        let cu_seq_lens = Tensor::from_vec(vec![0u32, 1], 2, &device).unwrap();

        let (k_out, _) = gather_kv_cache(
            &key_cache,
            &value_cache,
            None,
            None,
            None,
            None,
            &block_table,
            &cu_seq_lens,
            1,
            DType::F32,
            None,
            None,
        )
        .unwrap();

        let k_out_v: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        let key_v: Vec<f32> = key.flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..NUM_KV_HEADS * HEAD_SIZE {
            assert!(
                (k_out_v[i] - key_v[i]).abs() < 0.05,
                "bf16 roundtrip: {} vs {}",
                k_out_v[i],
                key_v[i]
            );
        }
    }
}
