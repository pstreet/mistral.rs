#[cfg(any(feature = "cuda", feature = "rocm"))]
mod ffi;

use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp3, Layout, Result, Shape, Storage, StorageRef,
    Tensor, WithDType,
};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct RotaryEmb {
    is_neox: bool,
}

impl RotaryEmb {
    fn cache_dims(&self, l_src: &Layout, l_cos: &Layout, l_sin: &Layout) -> Result<(usize, usize)> {
        let (batch, _, seq_len, head_dim) = l_src.shape().dims4()?;
        let (cos_rows, rot_dim) = match l_cos.shape().dims() {
            [rows, dim] => (*rows, *dim),
            [cos_batch, cos_seq, dim] if *cos_batch == batch && *cos_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => candle_core::bail!("invalid RoPE cos shape {:?}", l_cos.shape()),
        };
        let (sin_rows, sin_dim) = match l_sin.shape().dims() {
            [rows, dim] => (*rows, *dim),
            [sin_batch, sin_seq, dim] if *sin_batch == batch && *sin_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => candle_core::bail!("invalid RoPE sin shape {:?}", l_sin.shape()),
        };
        if (cos_rows, rot_dim) != (sin_rows, sin_dim) {
            candle_core::bail!(
                "RoPE cos/sin shape mismatch {:?} {:?}",
                l_cos.shape(),
                l_sin.shape()
            );
        }
        if cos_rows != seq_len && cos_rows != batch * seq_len {
            candle_core::bail!(
                "RoPE cache rows {cos_rows} are incompatible with batch {batch} and seq {seq_len}"
            );
        }
        if rot_dim == 0 || rot_dim * 2 > head_dim {
            candle_core::bail!(
                "RoPE rot dim {} is incompatible with head dim {head_dim}",
                rot_dim * 2
            );
        }
        Ok((cos_rows, rot_dim))
    }
}

impl CustomOp3 for RotaryEmb {
    fn name(&self) -> &'static str {
        "mistralrs-rotary"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
            is_neox: bool,
        ) -> Result<(CpuStorage, Shape)>
        where
            T: WithDType
                + Copy
                + Send
                + Sync
                + std::ops::Add<Output = T>
                + std::ops::Sub<Output = T>
                + std::ops::Mul<Output = T>,
        {
            let src = match l_src.contiguous_offsets() {
                Some((o1, o2)) => &src[o1..o2],
                None => candle_core::bail!("RoPE input must be contiguous"),
            };
            let cos = match l_cos.contiguous_offsets() {
                Some((o1, o2)) => &cos[o1..o2],
                None => candle_core::bail!("RoPE cos must be contiguous"),
            };
            let sin = match l_sin.contiguous_offsets() {
                Some((o1, o2)) => &sin[o1..o2],
                None => candle_core::bail!("RoPE sin must be contiguous"),
            };
            let (batch, heads, seq_len, head_dim) = l_src.shape().dims4()?;
            let (cache_rows, rot_dim) = RotaryEmb { is_neox }.cache_dims(l_src, l_cos, l_sin)?;
            let mut dst = src.to_vec();
            dst.par_chunks_mut(head_dim)
                .enumerate()
                .for_each(|(row, dst)| {
                    let batch_idx = row / (heads * seq_len);
                    let seq_idx = row % seq_len;
                    let cache_row = if cache_rows == batch * seq_len {
                        batch_idx * seq_len + seq_idx
                    } else {
                        seq_idx
                    };
                    let cache_offset = cache_row * rot_dim;
                    for pair_idx in 0..rot_dim {
                        let (x_idx, y_idx) = if is_neox {
                            (pair_idx, pair_idx + rot_dim)
                        } else {
                            (pair_idx * 2, pair_idx * 2 + 1)
                        };
                        let x = dst[x_idx];
                        let y = dst[y_idx];
                        let cos = cos[cache_offset + pair_idx];
                        let sin = sin[cache_offset + pair_idx];
                        dst[x_idx] = x * cos - y * sin;
                        dst[y_idx] = y * cos + x * sin;
                    }
                });
            Ok((T::to_cpu_storage_owned(dst), l_src.shape().clone()))
        }

        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3, self.is_neox),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3, self.is_neox),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3, self.is_neox),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3, self.is_neox),
            _ => candle_core::bail!(
                "unsupported RoPE dtype {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let (batch, heads, seq_len, head_dim) = l1.shape().dims4()?;
        let (cache_rows, rot_dim) = self.cache_dims(l1, l2, l3)?;
        let dtype = s1.dtype();
        if s2.dtype() != dtype || s3.dtype() != dtype {
            candle_core::bail!(
                "RoPE dtype mismatch {:?} {:?} {:?}",
                dtype,
                s2.dtype(),
                s3.dtype()
            );
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rotary");
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, dtype, "rotary-output")?;

        crate::metal_kernels::call_rotary(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            dtype,
            s1.buffer(),
            s2.buffer(),
            s3.buffer(),
            l1.start_offset() * dtype.size_in_bytes(),
            l2.start_offset() * dtype.size_in_bytes(),
            l3.start_offset() * dtype.size_in_bytes(),
            batch,
            heads,
            seq_len,
            head_dim,
            rot_dim,
            cache_rows,
            self.is_neox,
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        let storage = candle_core::MetalStorage::new(output, device.clone(), elem_count, dtype);
        Ok((storage, l1.shape().clone()))
    }
}

pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor, is_neox: bool) -> Result<Tensor> {
    let x = x.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    x.apply_op3_no_bwd(&cos, &sin, &RotaryEmb { is_neox })
}

#[cfg(feature = "metal")]
#[derive(Clone, Copy)]
struct RotaryDims {
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    rot_dim: usize,
    cache_rows: usize,
}

#[cfg(feature = "metal")]
fn rotary_dims(x: &Tensor, cos: &Tensor, sin: &Tensor, positioned: bool) -> Result<RotaryDims> {
    let (batch, heads, seq_len, head_dim) = x.dims4()?;
    let (cache_rows, rot_dim) = if positioned {
        cos.shape().dims2()?
    } else {
        match cos.dims() {
            [rows, dim] => (*rows, *dim),
            [cos_batch, cos_seq, dim] if *cos_batch == batch && *cos_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => candle_core::bail!("invalid RoPE cos shape {:?}", cos.shape()),
        }
    };
    let (sin_rows, sin_dim) = if positioned {
        sin.shape().dims2()?
    } else {
        match sin.dims() {
            [rows, dim] => (*rows, *dim),
            [sin_batch, sin_seq, dim] if *sin_batch == batch && *sin_seq == seq_len => {
                (batch * seq_len, *dim)
            }
            _ => candle_core::bail!("invalid RoPE sin shape {:?}", sin.shape()),
        }
    };
    if (cache_rows, rot_dim) != (sin_rows, sin_dim) {
        candle_core::bail!(
            "RoPE cos/sin shape mismatch {:?} {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if !positioned && cache_rows != seq_len && cache_rows != batch * seq_len {
        candle_core::bail!(
            "RoPE cache rows {cache_rows} are incompatible with batch {batch} and seq {seq_len}"
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        candle_core::bail!(
            "RoPE rot dim {} is incompatible with head dim {head_dim}",
            rot_dim * 2
        );
    }
    Ok(RotaryDims {
        batch,
        heads,
        seq_len,
        head_dim,
        rot_dim,
        cache_rows,
    })
}

fn check_qk_shape(q: &Tensor, k: &Tensor) -> Result<usize> {
    let (batch, _, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
        candle_core::bail!("q/k RoPE shape mismatch {:?} {:?}", q.shape(), k.shape());
    }
    Ok(k_heads)
}

fn typed_slice<'a, T>(xs: &'a [T], layout: &Layout, name: &'static str) -> Result<&'a [T]> {
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&xs[start..end]),
        None => candle_core::bail!("{name} must be contiguous for RoPE"),
    }
}

fn cpu_positions<'a>(
    storage_and_layout: &'a Option<(StorageRef<'a>, &'a Layout)>,
) -> Result<Option<&'a [u32]>> {
    let Some((storage, layout)) = storage_and_layout else {
        return Ok(None);
    };
    let Storage::Cpu(CpuStorage::U32(positions)) = &**storage else {
        candle_core::bail!("RoPE positions must be CPU u32");
    };
    Ok(Some(typed_slice(positions, layout, "positions")?))
}

struct CpuRotaryInput<'a, T> {
    src: &'a [T],
    src_l: &'a Layout,
    cos: &'a [T],
    cos_l: &'a Layout,
    sin: &'a [T],
    sin_l: &'a Layout,
    positions: Option<&'a [u32]>,
    is_neox: bool,
}

fn apply_rotary_cpu_inner<T>(input: CpuRotaryInput<'_, T>) -> Result<Tensor>
where
    T: WithDType
        + Copy
        + Send
        + Sync
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>,
{
    let CpuRotaryInput {
        src,
        src_l,
        cos,
        cos_l,
        sin,
        sin_l,
        positions,
        is_neox,
    } = input;
    let src = typed_slice(src, src_l, "RoPE input")?;
    let cos = typed_slice(cos, cos_l, "RoPE cos")?;
    let sin = typed_slice(sin, sin_l, "RoPE sin")?;
    let (batch, heads, seq_len, head_dim) = src_l.shape().dims4()?;
    let positioned = positions.is_some();
    let (cache_rows, rot_dim) = {
        let (cache_rows, rot_dim) = if positioned {
            cos_l.shape().dims2()?
        } else {
            RotaryEmb { is_neox }.cache_dims(src_l, cos_l, sin_l)?
        };
        if positioned && sin_l.shape().dims2()? != (cache_rows, rot_dim) {
            candle_core::bail!(
                "RoPE cos/sin shape mismatch {:?} {:?}",
                cos_l.shape(),
                sin_l.shape()
            );
        }
        if rot_dim == 0 || rot_dim * 2 > head_dim {
            candle_core::bail!(
                "RoPE rot dim {} is incompatible with head dim {head_dim}",
                rot_dim * 2
            );
        }
        (cache_rows, rot_dim)
    };
    if let Some(positions) = positions {
        let expected = batch * seq_len;
        if positions.len() != expected {
            candle_core::bail!(
                "RoPE positions length {} does not match token count {expected}",
                positions.len()
            );
        }
        for position in positions {
            if *position as usize >= cache_rows {
                candle_core::bail!(
                    "RoPE position {} exceeds cache rows {}",
                    position,
                    cache_rows
                );
            }
        }
    }
    let mut dst = src.to_vec();
    dst.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(row, dst)| {
            let batch_idx = row / (heads * seq_len);
            let seq_idx = row % seq_len;
            let token_idx = batch_idx * seq_len + seq_idx;
            let cache_row = if let Some(positions) = positions {
                positions[token_idx] as usize
            } else if cache_rows == batch * seq_len {
                token_idx
            } else {
                seq_idx
            };
            let cache_offset = cache_row * rot_dim;
            for pair_idx in 0..rot_dim {
                let (x_idx, y_idx) = if is_neox {
                    (pair_idx, pair_idx + rot_dim)
                } else {
                    (pair_idx * 2, pair_idx * 2 + 1)
                };
                let x = dst[x_idx];
                let y = dst[y_idx];
                let cos = cos[cache_offset + pair_idx];
                let sin = sin[cache_offset + pair_idx];
                dst[x_idx] = x * cos - y * sin;
                dst[y_idx] = y * cos + x * sin;
            }
        });
    Tensor::from_vec(dst, src_l.shape().clone(), &candle_core::Device::Cpu)
}

fn cpu_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let position_storage_and_layout = positions.as_ref().map(Tensor::storage_and_layout);
    let positions = cpu_positions(&position_storage_and_layout)?;

    let (q_s, q_l) = q.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    match (&*q_s, &*cos_s, &*sin_s) {
        (
            Storage::Cpu(CpuStorage::BF16(q)),
            Storage::Cpu(CpuStorage::BF16(cos)),
            Storage::Cpu(CpuStorage::BF16(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F16(q)),
            Storage::Cpu(CpuStorage::F16(cos)),
            Storage::Cpu(CpuStorage::F16(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F32(q)),
            Storage::Cpu(CpuStorage::F32(cos)),
            Storage::Cpu(CpuStorage::F32(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        (
            Storage::Cpu(CpuStorage::F64(q)),
            Storage::Cpu(CpuStorage::F64(cos)),
            Storage::Cpu(CpuStorage::F64(sin)),
        ) => apply_rotary_cpu_inner(CpuRotaryInput {
            src: q,
            src_l: q_l,
            cos,
            cos_l,
            sin,
            sin_l,
            positions,
            is_neox,
        }),
        _ => candle_core::bail!(
            "unsupported CPU RoPE dtype {:?} {:?} {:?}",
            q.dtype(),
            cos.dtype(),
            sin.dtype()
        ),
    }
}

fn cpu_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != k.dtype() {
        candle_core::bail!("q/k dtype mismatch {:?} {:?}", q.dtype(), k.dtype());
    }
    check_qk_shape(q, k)?;
    let (q_out, k_out) = rayon::join(
        || cpu_apply_rotary_q(q, cos, sin, positions, is_neox),
        || cpu_apply_rotary_q(k, cos, sin, positions, is_neox),
    );
    Ok((q_out?, k_out?))
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn restore_cuda_rope_layout(
    x: Tensor,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    x.reshape((batch, seq_len, heads, head_dim))?
        .transpose(1, 2)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn cuda_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    let (batch, heads, seq_len, head_dim) = q.dims4()?;
    let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
    if let Some(positions) = positions {
        apply_rotary_inplace_q_positions(&q_embed, cos, sin, positions, is_neox)?;
    } else {
        apply_rotary_inplace_q(&q_embed, cos, sin, is_neox)?;
    }
    restore_cuda_rope_layout(q_embed, batch, heads, seq_len, head_dim)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn cuda_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
        candle_core::bail!("q/k RoPE shape mismatch {:?} {:?}", q.shape(), k.shape());
    }
    let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
    let k_embed = k.transpose(1, 2)?.flatten(0, 1)?;
    if let Some(positions) = positions {
        apply_rotary_inplace_positions(&q_embed, &k_embed, cos, sin, positions, is_neox)?;
    } else {
        apply_rotary_inplace(&q_embed, &k_embed, cos, sin, is_neox)?;
    }
    Ok((
        restore_cuda_rope_layout(q_embed, batch, q_heads, seq_len, head_dim)?,
        restore_cuda_rope_layout(k_embed, batch, k_heads, seq_len, head_dim)?,
    ))
}

#[cfg(feature = "metal")]
fn metal_tensor(storage: Storage, shape: Shape) -> Tensor {
    Tensor::from((storage, shape))
}

#[cfg(feature = "metal")]
fn metal_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    use candle_core::MetalStorage;

    let q = q.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let dims = rotary_dims(&q, &cos, &sin, positions.is_some())?;
    if let Some(positions) = positions.as_ref() {
        let expected = dims.batch * dims.seq_len;
        if positions.dtype() != candle_core::DType::U32 || positions.dims1()? != expected {
            candle_core::bail!("RoPE positions must be u32 with length {expected}");
        }
    }

    let (q_s, q_l) = q.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("q must be a Metal tensor"),
    };
    let cos_s = match &*cos_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("cos must be a Metal tensor"),
    };
    let sin_s = match &*sin_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("sin must be a Metal tensor"),
    };
    let device = q_s.device();
    let output = device.new_buffer(q_l.shape().elem_count(), q_s.dtype(), "rotary-q")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("rotary-q");

    if let Some(positions) = positions.as_ref() {
        let (positions_s, positions_l) = positions.storage_and_layout();
        let positions_s = match &*positions_s {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("positions must be a Metal tensor"),
        };
        crate::metal_kernels::call_rotary_q_positions(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            q_s.dtype(),
            q_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            positions_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            positions_l.start_offset() * positions_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            is_neox,
            &output,
        )
    } else {
        crate::metal_kernels::call_rotary_q(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            q_s.dtype(),
            q_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            dims.cache_rows,
            is_neox,
            &output,
        )
    }
    .map_err(candle_core::Error::wrap)?;

    Ok(metal_tensor(
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            q_l.shape().elem_count(),
            q_s.dtype(),
        )),
        q_l.shape().clone(),
    ))
}

#[cfg(feature = "metal")]
fn metal_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    use candle_core::MetalStorage;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let dims = rotary_dims(&q, &cos, &sin, positions.is_some())?;
    let k_heads = check_qk_shape(&q, &k)?;
    if let Some(positions) = positions.as_ref() {
        let expected = dims.batch * dims.seq_len;
        if positions.dtype() != candle_core::DType::U32 || positions.dims1()? != expected {
            candle_core::bail!("RoPE positions must be u32 with length {expected}");
        }
    }

    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("q must be a Metal tensor"),
    };
    let k_s = match &*k_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("k must be a Metal tensor"),
    };
    let cos_s = match &*cos_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("cos must be a Metal tensor"),
    };
    let sin_s = match &*sin_s {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("sin must be a Metal tensor"),
    };
    let device = q_s.device();
    let q_out = device.new_buffer(q_l.shape().elem_count(), q_s.dtype(), "rotary-q")?;
    let k_out = device.new_buffer(k_l.shape().elem_count(), k_s.dtype(), "rotary-k")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("rotary-qk");

    if let Some(positions) = positions.as_ref() {
        let (positions_s, positions_l) = positions.storage_and_layout();
        let positions_s = match &*positions_s {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("positions must be a Metal tensor"),
        };
        crate::metal_kernels::call_rotary_qk_positions(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            q_s.dtype(),
            q_s.buffer(),
            k_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            positions_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            k_l.start_offset() * k_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            positions_l.start_offset() * positions_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            k_heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            is_neox,
            &q_out,
            &k_out,
        )
    } else {
        crate::metal_kernels::call_rotary_qk(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
            q_s.dtype(),
            q_s.buffer(),
            k_s.buffer(),
            cos_s.buffer(),
            sin_s.buffer(),
            q_l.start_offset() * q_s.dtype().size_in_bytes(),
            k_l.start_offset() * k_s.dtype().size_in_bytes(),
            cos_l.start_offset() * cos_s.dtype().size_in_bytes(),
            sin_l.start_offset() * sin_s.dtype().size_in_bytes(),
            dims.batch,
            dims.heads,
            k_heads,
            dims.seq_len,
            dims.head_dim,
            dims.rot_dim,
            dims.cache_rows,
            is_neox,
            &q_out,
            &k_out,
        )
    }
    .map_err(candle_core::Error::wrap)?;

    Ok((
        metal_tensor(
            Storage::Metal(MetalStorage::new(
                q_out,
                device.clone(),
                q_l.shape().elem_count(),
                q_s.dtype(),
            )),
            q_l.shape().clone(),
        ),
        metal_tensor(
            Storage::Metal(MetalStorage::new(
                k_out,
                device.clone(),
                k_l.shape().elem_count(),
                k_s.dtype(),
            )),
            k_l.shape().clone(),
        ),
    ))
}

pub fn apply_rotary_q_preselected(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
) -> Result<Tensor> {
    apply_rotary_q_inner(q, cos, sin, None, is_neox)
}

pub fn apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Tensor> {
    apply_rotary_q_inner(q, cos, sin, Some(positions), is_neox)
}

/// Normalize the cos/sin cache to per-token rows `(batch * seq, rot_dim)`,
/// mirroring the host path's `cache_dims` contract: with positions the
/// cache is 2D and rows are gathered by the flat position ids; without
/// positions, `(batch, seq, w)` flattens, and `(seq, w)` broadcasts over
/// the batch.
fn tensor_rope_rows(
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    batch: usize,
    seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = match positions {
        Some(positions) => {
            let idx = positions
                .contiguous()?
                .flatten_all()?
                .to_dtype(candle_core::DType::I64)?;
            (
                cos.contiguous()?.index_select(&idx, 0)?,
                sin.contiguous()?.index_select(&idx, 0)?,
            )
        }
        None => {
            let (mut cos, mut sin) = (cos.contiguous()?, sin.contiguous()?);
            if cos.rank() == 3 {
                cos = cos.flatten(0, 1)?;
                sin = sin.flatten(0, 1)?;
            }
            let rows = cos.dim(0)?;
            if rows == seq_len && batch > 1 {
                let width = cos.dim(1)?;
                cos = cos.broadcast_as((batch, seq_len, width))?.flatten(0, 1)?;
                sin = sin.broadcast_as((batch, seq_len, width))?.flatten(0, 1)?;
            }
            (cos, sin)
        }
    };
    if cos.dims() != sin.dims() {
        candle_core::bail!(
            "RoPE cos/sin shape mismatch {:?} {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    Ok((cos, sin))
}

/// Apply rope to one `(batch, seq, heads, head_dim)` tensor with pure
/// candle ops, so any device storage can serve it. Handles both the
/// neox (half-split) and interleaved styles and partial rotation
/// (`rot * 2 < head_dim` passes the tail through unrotated).
fn tensor_rope_one(
    x: &Tensor,
    cos_rows: &Tensor,
    sin_rows: &Tensor,
    is_neox: bool,
) -> Result<Tensor> {
    use candle_core::D;
    let (tokens, heads, head_dim) = x.dims3()?;
    let rot = cos_rows.dim(1)?;
    if rot == 0 || rot * 2 > head_dim {
        candle_core::bail!(
            "RoPE rot dim {} is incompatible with head dim {head_dim}",
            rot * 2
        );
    }
    let dtype = x.dtype();
    // (tokens, 1, rot) broadcasting over heads.
    let cos = cos_rows.to_dtype(dtype)?.unsqueeze(1)?;
    let sin = sin_rows.to_dtype(dtype)?.unsqueeze(1)?;

    let (rotated, tail) = if is_neox {
        let x1 = x.narrow(D::Minus1, 0, rot)?;
        let x2 = x.narrow(D::Minus1, rot, rot)?;
        let out = Tensor::cat(
            &[
                &(x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?,
                &(x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?,
            ],
            D::Minus1,
        )?;
        (out, head_dim - rot * 2)
    } else {
        let pairs = x
            .narrow(D::Minus1, 0, rot * 2)?
            .reshape((tokens, heads, rot, 2))?;
        let cos = cos.unsqueeze(3)?;
        let sin = sin.unsqueeze(3)?;
        let even = pairs.narrow(D::Minus1, 0, 1)?;
        let odd = pairs.narrow(D::Minus1, 1, 1)?;
        let out = Tensor::cat(
            &[
                &(even.broadcast_mul(&cos)? - odd.broadcast_mul(&sin)?)?,
                &(odd.broadcast_mul(&cos)? + even.broadcast_mul(&sin)?)?,
            ],
            3,
        )?
        .reshape((tokens, heads, rot * 2))?;
        (out, head_dim - rot * 2)
    };
    if tail > 0 {
        Ok(Tensor::cat(
            &[&rotated, &x.narrow(D::Minus1, rot * 2, tail)?],
            D::Minus1,
        )?)
    } else {
        Ok(rotated)
    }
}

/// Device-agnostic rope for GPU devices without the fused kernels (the
/// NVIDIA role in dual builds); the host-slice `cpu_apply_*` path can only
/// serve CPU storage.
fn tensor_apply_rotary_q(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    let (batch, heads, seq_len, head_dim) = q.dims4()?;
    let (cos_rows, sin_rows) = tensor_rope_rows(cos, sin, positions, batch, seq_len)?;
    // (batch, heads, seq, dim) -> (batch * seq, heads, dim) for the op path.
    let x = q.transpose(1, 2)?.flatten(0, 1)?;
    let out = tensor_rope_one(&x, &cos_rows, &sin_rows, is_neox)?;
    out.reshape((batch, seq_len, heads, head_dim))?
        .transpose(1, 2)
}

fn tensor_apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != k.dtype() {
        candle_core::bail!("q/k dtype mismatch {:?} {:?}", q.dtype(), k.dtype());
    }
    check_qk_shape(q, k)?;
    // Graph capture is thread-bound: rayon workers launch kernels from threads
    // the capture never saw, so replay silently drops a branch. Run in order.
    let q_out = tensor_apply_rotary_q(q, cos, sin, positions, is_neox);
    let k_out = tensor_apply_rotary_q(k, cos, sin, positions, is_neox);
    Ok((q_out?, k_out?))
}

fn apply_rotary_q_inner(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<Tensor> {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if crate::utils::device_has_fused_ops(q.device()) {
        return cuda_apply_rotary_q(q, cos, sin, positions, is_neox);
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return metal_apply_rotary_q(q, cos, sin, positions, is_neox);
    }
    if !q.device().is_cpu() {
        return tensor_apply_rotary_q(q, cos, sin, positions, is_neox);
    }
    cpu_apply_rotary_q(q, cos, sin, positions, is_neox)
}

pub fn apply_rotary_qk_preselected(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    apply_rotary_qk_inner(q, k, cos, sin, None, is_neox)
}

pub fn apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    apply_rotary_qk_inner(q, k, cos, sin, Some(positions), is_neox)
}

fn apply_rotary_qk_inner(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: Option<&Tensor>,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != k.dtype() {
        candle_core::bail!("q/k dtype mismatch {:?} {:?}", q.dtype(), k.dtype());
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if crate::utils::device_has_fused_ops(q.device()) {
        return cuda_apply_rotary_qk(q, k, cos, sin, positions, is_neox);
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return metal_apply_rotary_qk(q, k, cos, sin, positions, is_neox);
    }
    if !q.device().is_cpu() {
        return tensor_apply_rotary_qk(q, k, cos, sin, positions, is_neox);
    }
    cpu_apply_rotary_qk(q, k, cos, sin, positions, is_neox)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
mod cuda {
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    use candle_core::HipStorage;
    use candle_core::{
        backend::{BackendDevice, BackendStorage},
        cuda_backend::{CudaDType, CudaStorage, CudaStorageSlice},
        CpuStorage, DType, InplaceOp3, Layout, MetalStorage, Result, Storage, Tensor,
    };
    use half::{bf16, f16};
    use std::ffi::{c_int, c_long};

    #[cfg(all(feature = "cuda", feature = "rocm"))]
    use crate::utils::{hip_slice_ptr_mut_on_stream, hip_slice_ptr_on_stream};
    use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

    fn rotary_dtype(dtype: DType) -> Result<u32> {
        Ok(match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        })
    }

    struct RotaryLaunch<'a> {
        query: &'a mut CudaStorage,
        query_l: &'a Layout,
        cos_cache: &'a CudaStorage,
        cos_l: &'a Layout,
        sin_cache: &'a CudaStorage,
        sin_l: &'a Layout,
        positions: Option<(&'a CudaStorage, &'a Layout)>,
        is_neox: bool,
    }

    #[cfg(all(feature = "cuda", feature = "rocm"))]
    struct RotaryLaunchHip<'a> {
        query: &'a mut HipStorage,
        query_l: &'a Layout,
        cos_cache: &'a HipStorage,
        cos_l: &'a Layout,
        sin_cache: &'a HipStorage,
        sin_l: &'a Layout,
        positions: Option<(&'a HipStorage, &'a Layout)>,
        is_neox: bool,
    }

    fn launch_rotary<T>(args: RotaryLaunch<'_>) -> Result<()>
    where
        T: CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    {
        let RotaryLaunch {
            query,
            query_l,
            cos_cache,
            cos_l,
            sin_cache,
            sin_l,
            positions,
            is_neox,
        } = args;

        if cos_cache.dtype() != query.dtype() || sin_cache.dtype() != query.dtype() {
            candle_core::bail!("apply-rotary expects all tensors to have the same dtype");
        }

        let dev = query.device().clone();
        if !cos_cache.device().same_device(&dev) || !sin_cache.device().same_device(&dev) {
            candle_core::bail!("apply-rotary tensors must be on the same cuda device");
        }

        if query_l.stride().len() != 3 {
            candle_core::bail!("apply-rotary expects query rank 3 ({query_l:?})")
        }
        if cos_l.stride().len() != 2 || sin_l.stride().len() != 2 {
            candle_core::bail!("apply-rotary expects rank 2 caches")
        }

        let (num_tokens, num_heads, head_size) = query_l.shape().dims3()?;
        let rot_dim = cos_l.dims()[1];
        if sin_l.shape().dims2()? != (cos_l.dims()[0], rot_dim) {
            candle_core::bail!(
                "shape mismatch cos_cache {:?} and sin_cache {:?}",
                cos_l.shape(),
                sin_l.shape()
            )
        }
        if positions.is_none() && (num_tokens, rot_dim) != cos_l.shape().dims2()? {
            candle_core::bail!(
                "shape mismatch cos_cache {:?}, expected {:?}",
                cos_l.shape(),
                (num_tokens, rot_dim)
            )
        }
        if rot_dim == 0 || rot_dim * 2 > head_size {
            candle_core::bail!(
                "rotary dimension {rot_dim} is incompatible with head size {head_size}"
            )
        }

        let query_dtype = query.dtype();
        let stream = dev.cuda_stream();
        let query = query.as_cuda_slice_mut::<T>()?;
        let cos_cache = cos_cache.as_cuda_slice::<T>()?;
        let sin_cache = sin_cache.as_cuda_slice::<T>()?;
        let (query, _query_guard) = slice_ptr_mut_on_stream(query, query_l.start_offset(), &stream);
        let (cos_cache, _cos_guard) = slice_ptr_on_stream(cos_cache, cos_l.start_offset(), &stream);
        let (sin_cache, _sin_guard) = slice_ptr_on_stream(sin_cache, sin_l.start_offset(), &stream);

        let positions = if let Some((positions, positions_l)) = positions {
            if positions.dtype() != DType::U32 {
                candle_core::bail!("apply-rotary-positions expects positions to be u32");
            }
            if !positions.device().same_device(&dev) {
                candle_core::bail!("positions must be on the same cuda device as query");
            }
            if positions_l.stride().len() != 1 {
                candle_core::bail!("apply-rotary-positions expects rank 1 positions")
            }
            let positions_len = positions_l.shape().dims1()?;
            if positions_len != num_tokens {
                candle_core::bail!(
                    "positions length {positions_len} does not match token count {num_tokens}"
                );
            }
            let positions = match &positions.slice {
                CudaStorageSlice::U32(positions) => positions,
                _ => candle_core::bail!("positions dtype mismatch"),
            };
            let (positions, guard) =
                slice_ptr_on_stream(positions, positions_l.start_offset(), &stream);
            Some((positions, 1, guard))
        } else {
            None
        };

        let neox = if is_neox { 1 } else { 0 };
        let stream = stream.cu_stream() as c_long;
        let internal_type = rotary_dtype(query_dtype)?;
        match positions {
            None => unsafe {
                super::ffi::rotary_embedding(
                    query as *const core::ffi::c_void,
                    std::ptr::null(),
                    cos_cache as *const core::ffi::c_void,
                    sin_cache as *const core::ffi::c_void,
                    neox,
                    head_size as c_int,
                    num_tokens as c_long,
                    rot_dim as c_int,
                    num_heads as c_int,
                    0,
                    query_l.stride()[0] as c_long,
                    0,
                    internal_type,
                    stream,
                )
            },
            Some((positions, seq_len, _positions_guard)) => unsafe {
                super::ffi::rotary_embedding_positions(
                    query as *const core::ffi::c_void,
                    std::ptr::null(),
                    cos_cache as *const core::ffi::c_void,
                    sin_cache as *const core::ffi::c_void,
                    positions as *const core::ffi::c_void,
                    neox,
                    head_size as c_int,
                    num_tokens as c_long,
                    rot_dim as c_int,
                    seq_len as c_int,
                    num_heads as c_int,
                    0,
                    query_l.stride()[0] as c_long,
                    0,
                    internal_type,
                    stream,
                )
            },
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "rocm"))]
    fn launch_rotary_hip<T>(args: RotaryLaunchHip<'_>) -> Result<()>
    where
        T: candle_core::hip_backend::CudaDType
            + candle_core::hip_backend::cudarc::driver::DeviceRepr,
    {
        let RotaryLaunchHip {
            query,
            query_l,
            cos_cache,
            cos_l,
            sin_cache,
            sin_l,
            positions,
            is_neox,
        } = args;

        if cos_cache.dtype() != query.dtype() || sin_cache.dtype() != query.dtype() {
            candle_core::bail!("apply-rotary expects all tensors to have the same dtype");
        }

        let dev = query.device().clone();
        if !cos_cache.device().same_device(&dev) || !sin_cache.device().same_device(&dev) {
            candle_core::bail!("apply-rotary tensors must be on the same cuda device");
        }

        if query_l.stride().len() != 3 {
            candle_core::bail!("apply-rotary expects query rank 3 ({query_l:?})")
        }
        if cos_l.stride().len() != 2 || sin_l.stride().len() != 2 {
            candle_core::bail!("apply-rotary expects rank 2 caches")
        }

        let (num_tokens, num_heads, head_size) = query_l.shape().dims3()?;
        let rot_dim = cos_l.dims()[1];
        if sin_l.shape().dims2()? != (cos_l.dims()[0], rot_dim) {
            candle_core::bail!(
                "shape mismatch cos_cache {:?} and sin_cache {:?}",
                cos_l.shape(),
                sin_l.shape()
            )
        }
        if positions.is_none() && (num_tokens, rot_dim) != cos_l.shape().dims2()? {
            candle_core::bail!(
                "shape mismatch cos_cache {:?}, expected {:?}",
                cos_l.shape(),
                (num_tokens, rot_dim)
            )
        }
        if rot_dim == 0 || rot_dim * 2 > head_size {
            candle_core::bail!(
                "rotary dimension {rot_dim} is incompatible with head size {head_size}"
            )
        }

        let query_dtype = query.dtype();
        let stream = dev.cuda_stream();
        let query = query.as_cuda_slice_mut::<T>()?;
        let cos_cache = cos_cache.as_cuda_slice::<T>()?;
        let sin_cache = sin_cache.as_cuda_slice::<T>()?;
        let (query, _query_guard) =
            hip_slice_ptr_mut_on_stream(query, query_l.start_offset(), &stream);
        let (cos_cache, _cos_guard) =
            hip_slice_ptr_on_stream(cos_cache, cos_l.start_offset(), &stream);
        let (sin_cache, _sin_guard) =
            hip_slice_ptr_on_stream(sin_cache, sin_l.start_offset(), &stream);

        let positions = if let Some((positions, positions_l)) = positions {
            if positions.dtype() != DType::U32 {
                candle_core::bail!("apply-rotary-positions expects positions to be u32");
            }
            if !positions.device().same_device(&dev) {
                candle_core::bail!("positions must be on the same cuda device as query");
            }
            if positions_l.stride().len() != 1 {
                candle_core::bail!("apply-rotary-positions expects rank 1 positions")
            }
            let positions_len = positions_l.shape().dims1()?;
            if positions_len != num_tokens {
                candle_core::bail!(
                    "positions length {positions_len} does not match token count {num_tokens}"
                );
            }
            let positions = match &positions.slice {
                candle_core::hip_backend::CudaStorageSlice::U32(positions) => positions,
                _ => candle_core::bail!("positions dtype mismatch"),
            };
            let (positions, guard) =
                hip_slice_ptr_on_stream(positions, positions_l.start_offset(), &stream);
            Some((positions, 1, guard))
        } else {
            None
        };

        let neox = if is_neox { 1 } else { 0 };
        let stream = stream.cu_stream() as c_long;
        let internal_type = rotary_dtype(query_dtype)?;
        match positions {
            None => unsafe {
                super::ffi::rotary_embedding(
                    query as *const core::ffi::c_void,
                    std::ptr::null(),
                    cos_cache as *const core::ffi::c_void,
                    sin_cache as *const core::ffi::c_void,
                    neox,
                    head_size as c_int,
                    num_tokens as c_long,
                    rot_dim as c_int,
                    num_heads as c_int,
                    0,
                    query_l.stride()[0] as c_long,
                    0,
                    internal_type,
                    stream,
                )
            },
            Some((positions, seq_len, _positions_guard)) => unsafe {
                super::ffi::rotary_embedding_positions(
                    query as *const core::ffi::c_void,
                    std::ptr::null(),
                    cos_cache as *const core::ffi::c_void,
                    sin_cache as *const core::ffi::c_void,
                    positions as *const core::ffi::c_void,
                    neox,
                    head_size as c_int,
                    num_tokens as c_long,
                    rot_dim as c_int,
                    seq_len as c_int,
                    num_heads as c_int,
                    0,
                    query_l.stride()[0] as c_long,
                    0,
                    internal_type,
                    stream,
                )
            },
        }
        Ok(())
    }

    struct RotaryInplace {
        is_neox: bool,
    }

    impl InplaceOp3 for RotaryInplace {
        fn name(&self) -> &'static str {
            "mistralrs-rotary-inplace"
        }

        fn cpu_fwd(
            &self,
            _: &mut CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("apply-rotary-inplace is only supported for cuda")
        }

        fn cuda_fwd(
            &self,
            query: &mut CudaStorage,
            query_l: &Layout,
            cos_cache: &CudaStorage,
            cos_l: &Layout,
            sin_cache: &CudaStorage,
            sin_l: &Layout,
        ) -> Result<()> {
            match query.dtype() {
                DType::F16 => launch_rotary::<f16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                DType::BF16 => launch_rotary::<bf16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                DType::F32 => launch_rotary::<f32>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                dt => {
                    candle_core::bail!(
                        "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                    )
                }
            }
        }

        #[cfg(all(feature = "cuda", feature = "rocm"))]
        fn hip_fwd(
            &self,
            query: &mut HipStorage,
            query_l: &Layout,
            cos_cache: &HipStorage,
            cos_l: &Layout,
            sin_cache: &HipStorage,
            sin_l: &Layout,
        ) -> Result<()> {
            match query.dtype() {
                DType::F16 => launch_rotary_hip::<f16>(RotaryLaunchHip {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                DType::BF16 => launch_rotary_hip::<bf16>(RotaryLaunchHip {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                DType::F32 => launch_rotary_hip::<f32>(RotaryLaunchHip {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: None,
                    is_neox: self.is_neox,
                }),
                dt => {
                    candle_core::bail!(
                        "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                    )
                }
            }
        }

        fn metal_fwd(
            &self,
            _: &mut MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("apply-rotary-inplace is only supported for cuda")
        }
    }

    struct RotaryPositionsInplace<'a> {
        is_neox: bool,
        positions: &'a Tensor,
    }

    impl InplaceOp3 for RotaryPositionsInplace<'_> {
        fn name(&self) -> &'static str {
            "mistralrs-rotary-positions-inplace"
        }

        fn cpu_fwd(
            &self,
            _: &mut CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("apply-rotary-positions-inplace is only supported for cuda")
        }

        fn cuda_fwd(
            &self,
            query: &mut CudaStorage,
            query_l: &Layout,
            cos_cache: &CudaStorage,
            cos_l: &Layout,
            sin_cache: &CudaStorage,
            sin_l: &Layout,
        ) -> Result<()> {
            let (positions_storage, positions_l) = self.positions.storage_and_layout();
            let positions = match &*positions_storage {
                Storage::Cuda(positions) => positions,
                _ => candle_core::bail!("positions must be a cuda tensor"),
            };
            match query.dtype() {
                DType::F16 => launch_rotary::<f16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                DType::BF16 => launch_rotary::<bf16>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                DType::F32 => launch_rotary::<f32>(RotaryLaunch {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                dt => {
                    candle_core::bail!(
                        "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                    )
                }
            }
        }

        #[cfg(all(feature = "cuda", feature = "rocm"))]
        fn hip_fwd(
            &self,
            query: &mut HipStorage,
            query_l: &Layout,
            cos_cache: &HipStorage,
            cos_l: &Layout,
            sin_cache: &HipStorage,
            sin_l: &Layout,
        ) -> Result<()> {
            let (positions_storage, positions_l) = self.positions.storage_and_layout();
            let positions = match &*positions_storage {
                Storage::Hip(positions) => positions,
                _ => candle_core::bail!("positions must be a hip tensor"),
            };
            match query.dtype() {
                DType::F16 => launch_rotary_hip::<f16>(RotaryLaunchHip {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                DType::BF16 => launch_rotary_hip::<bf16>(RotaryLaunchHip {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                DType::F32 => launch_rotary_hip::<f32>(RotaryLaunchHip {
                    query,
                    query_l,
                    cos_cache,
                    cos_l,
                    sin_cache,
                    sin_l,
                    positions: Some((positions, positions_l)),
                    is_neox: self.is_neox,
                }),
                dt => {
                    candle_core::bail!(
                        "apply_rotary is only supported for f32, f16 and bf16 ({dt:?})"
                    )
                }
            }
        }

        fn metal_fwd(
            &self,
            _: &mut MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
            _: &MetalStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("apply-rotary-positions-inplace is only supported for cuda")
        }
    }

    fn apply_rotary_(
        query: &Tensor,
        key: Option<&Tensor>,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        let dtype = query.dtype();
        if key.is_some_and(|key| key.dtype() != dtype)
            || cos_cache.dtype() != dtype
            || sin_cache.dtype() != dtype
        {
            candle_core::bail!("apply-rotary expects all tensors to have the same dtype");
        }
        let op = RotaryInplace { is_neox };
        query.inplace_op3(cos_cache, sin_cache, &op)?;
        if let Some(key) = key {
            key.inplace_op3(cos_cache, sin_cache, &op)?;
        }
        Ok(())
    }

    fn apply_rotary_positions_(
        query: &Tensor,
        key: Option<&Tensor>,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        let dtype = query.dtype();
        if key.is_some_and(|key| key.dtype() != dtype)
            || cos_cache.dtype() != dtype
            || sin_cache.dtype() != dtype
            || positions.dtype() != DType::U32
        {
            candle_core::bail!(
                "apply-rotary-positions expects q/k/caches to share dtype and positions to be u32"
            );
        }

        let cos_cache = cos_cache.contiguous()?;
        let sin_cache = sin_cache.contiguous()?;
        let positions = positions.contiguous()?;
        let op = RotaryPositionsInplace {
            is_neox,
            positions: &positions,
        };
        query.inplace_op3(&cos_cache, &sin_cache, &op)?;
        if let Some(key) = key {
            key.inplace_op3(&cos_cache, &sin_cache, &op)?;
        }
        Ok(())
    }

    /// Apply Rotary position encoding inplace
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor of shape `(num_tokens, num_heads, head_size)`.
    /// * `key` - Key tensor of shape `(num_tokens, num_kv_heads, head_size)`.
    /// * `cos_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
    /// * `sin_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
    /// * `is_neox` - Use neox encoding instead of gpt-j style rotary
    pub fn apply_rotary_inplace(
        query: &Tensor,
        key: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match key.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_(query, Some(key), cos_cache, sin_cache, is_neox)
            }
            dt => {
                candle_core::bail!("apply_rotary is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }

    pub fn apply_rotary_inplace_q(
        query: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match query.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_(query, None, cos_cache, sin_cache, is_neox)
            }
            dt => {
                candle_core::bail!("apply_rotary is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }

    pub fn apply_rotary_inplace_positions(
        query: &Tensor,
        key: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match key.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_positions_(query, Some(key), cos_cache, sin_cache, positions, is_neox)
            }
            dt => {
                candle_core::bail!("apply_rotary is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }

    pub fn apply_rotary_inplace_q_positions(
        query: &Tensor,
        cos_cache: &Tensor,
        sin_cache: &Tensor,
        positions: &Tensor,
        is_neox: bool,
    ) -> Result<()> {
        match query.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => {
                apply_rotary_positions_(query, None, cos_cache, sin_cache, positions, is_neox)
            }
            dt => {
                candle_core::bail!("apply_rotary is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use cuda::*;

/// Apply Rotary position encoding inplace
///
/// # Arguments
///
/// * `query` - Query tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `key` - Key tensor of shape `(num_tokens, num_kv_heads, head_size)`.
/// * `cos_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
/// * `sin_cache` - Aligned cache of shape `(num_tokens, rot_dim)`
/// * `is_neox` - Use neox encoding instead of gpt-j style rotary
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub fn apply_rotary_inplace(
    _query: &candle_core::Tensor,
    _key: &candle_core::Tensor,
    _cos_cache: &candle_core::Tensor,
    _sin_cache: &candle_core::Tensor,
    _is_neox: bool,
) -> candle_core::Result<()> {
    candle_core::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub fn apply_rotary_inplace_q(
    _query: &candle_core::Tensor,
    _cos_cache: &candle_core::Tensor,
    _sin_cache: &candle_core::Tensor,
    _is_neox: bool,
) -> candle_core::Result<()> {
    candle_core::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub fn apply_rotary_inplace_positions(
    _query: &candle_core::Tensor,
    _key: &candle_core::Tensor,
    _cos_cache: &candle_core::Tensor,
    _sin_cache: &candle_core::Tensor,
    _positions: &candle_core::Tensor,
    _is_neox: bool,
) -> candle_core::Result<()> {
    candle_core::bail!("apply_rotary is only supported for cuda");
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub fn apply_rotary_inplace_q_positions(
    _query: &candle_core::Tensor,
    _cos_cache: &candle_core::Tensor,
    _sin_cache: &candle_core::Tensor,
    _positions: &candle_core::Tensor,
    _is_neox: bool,
) -> candle_core::Result<()> {
    candle_core::bail!("apply_rotary is only supported for cuda");
}

/// The tensor-op rope must match the validated host-slice path exactly on
/// CPU tensors: every layout (per-token 3D/2D cos, position-indexed cache),
/// both styles, and partial rotation.
#[cfg(test)]
mod tensor_rope_tests {
    use super::*;

    fn rope_case(
        b: usize,
        h: usize,
        s: usize,
        d: usize,
        rot: usize,
        is_neox: bool,
        cos_rank3: bool,
        positions: bool,
    ) -> Result<()> {
        let dev = &candle_core::Device::Cpu;
        let mk = |seed: usize| -> Vec<f32> {
            (0..b * h * s * d)
                .map(|i| (((i + seed * 7) % 23) as f32 * 0.37 - 1.1) * 0.5)
                .collect()
        };
        let q = candle_core::Tensor::from_vec(mk(1), (b, h, s, d), dev)?;
        let k = candle_core::Tensor::from_vec(mk(2), (b, h, s, d), dev)?;
        let cache_rows = if positions { s * 4 } else { b * s };
        let inv_freq: Vec<f32> = (0..rot)
            .map(|i| 1f32 / (10_000f32.powf(2.0 * (i as f32 / rot as f32))))
            .collect();
        let cos: Vec<f32> = (0..cache_rows)
            .flat_map(|r| inv_freq.iter().map(move |&f| (r as f32 * f).cos()))
            .collect();
        let sin: Vec<f32> = (0..cache_rows)
            .flat_map(|r| inv_freq.iter().map(move |&f| (r as f32 * f).sin()))
            .collect();
        let (cos, sin) = if cos_rank3 && !positions {
            (
                candle_core::Tensor::from_vec(cos, (b, s, rot), dev)?,
                candle_core::Tensor::from_vec(sin, (b, s, rot), dev)?,
            )
        } else {
            (
                candle_core::Tensor::from_vec(cos, (cache_rows, rot), dev)?,
                candle_core::Tensor::from_vec(sin, (cache_rows, rot), dev)?,
            )
        };
        let positions = if positions {
            let pos: Vec<u32> = (0..b * s).map(|t| ((t % s) * 3) as u32).collect();
            Some(candle_core::Tensor::from_vec(pos, (b, s), dev)?)
        } else {
            None
        };

        let (q_cpu, k_cpu) = cpu_apply_rotary_qk(&q, &k, &cos, &sin, positions.as_ref(), is_neox)?;
        let (q_t, k_t) = tensor_apply_rotary_qk(&q, &k, &cos, &sin, positions.as_ref(), is_neox)?;
        for (label, host, tensor) in [("q", &q_cpu, &q_t), ("k", &k_cpu, &k_t)] {
            let host = host.flatten_all()?.to_vec1::<f32>()?;
            let tensor = tensor.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(host.len(), tensor.len(), "{label} length");
            for (i, (a, bb)) in host.iter().zip(tensor.iter()).enumerate() {
                let diff = (a - bb).abs();
                assert!(diff < 1e-4, "{label}[{i}]: host {a} vs tensor {bb}");
            }
        }
        Ok(())
    }

    #[test]
    fn tensor_rope_neox_3d() -> Result<()> {
        rope_case(2, 4, 8, 32, 16, true, true, false)
    }
    #[test]
    fn tensor_rope_neox_2d() -> Result<()> {
        rope_case(2, 4, 8, 32, 16, true, false, false)
    }
    #[test]
    fn tensor_rope_interleaved_3d() -> Result<()> {
        rope_case(2, 4, 8, 32, 16, false, true, false)
    }
    #[test]
    fn tensor_rope_neox_positions() -> Result<()> {
        rope_case(2, 4, 8, 32, 16, true, false, true)
    }
    #[test]
    fn tensor_rope_interleaved_positions() -> Result<()> {
        rope_case(2, 4, 8, 32, 16, false, false, true)
    }
    #[test]
    fn tensor_rope_partial_rot() -> Result<()> {
        rope_case(2, 4, 8, 32, 12, true, true, false)
    }
    #[test]
    fn tensor_rope_interleaved_partial() -> Result<()> {
        rope_case(2, 4, 8, 32, 12, false, true, false)
    }
}

#[cfg(test)]
mod dual_cuda_capture_tests {
    use super::*;

    fn tensor_va(
        t: &candle_core::Tensor,
        stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(u64, usize)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let (storage, layout) = t.storage_and_layout();
        let bytes = t.elem_count() * t.dtype().size_in_bytes();
        let base: u64 = match &*storage {
            candle_core::Storage::Cuda(s) => match t.dtype() {
                candle_core::DType::U8 => {
                    let (ptr, _guard) = s.as_cuda_slice::<u8>()?.device_ptr(stream);
                    ptr
                }
                candle_core::DType::U32 => {
                    let (ptr, _guard) = s.as_cuda_slice::<u32>()?.device_ptr(stream);
                    ptr
                }
                candle_core::DType::F16 => {
                    let (ptr, _guard) = s.as_cuda_slice::<half::f16>()?.device_ptr(stream);
                    ptr
                }
                candle_core::DType::BF16 => {
                    let (ptr, _guard) = s.as_cuda_slice::<half::bf16>()?.device_ptr(stream);
                    ptr
                }
                candle_core::DType::F32 => {
                    let (ptr, _guard) = s.as_cuda_slice::<f32>()?.device_ptr(stream);
                    ptr
                }
                candle_core::DType::I64 => {
                    let (ptr, _guard) = s.as_cuda_slice::<i64>()?.device_ptr(stream);
                    ptr
                }
                dt => candle_core::bail!("probe census lacks dtype {dt:?}"),
            },
            _ => candle_core::bail!("probe tensor must be CUDA"),
        };
        Ok((base + layout.start_offset() as u64, bytes))
    }

    /// Capture+relaunch matches eager for the preselected q/k rope path.
    /// Guards the capture-thread rule: both branches must launch on the
    /// capturing thread (a rayon join once dropped a branch at replay).
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn rope_qk_capture_launch_matches_eager() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        use candle_core::{DType, Device, Tensor};

        let dev = Device::new_cuda(0)?;
        let k = Tensor::from_vec(
            (0..512).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            (1, 4, 1, 128),
            &Device::Cpu,
        )?
        .to_dtype(DType::F32)?
        .to_device(&dev)?;
        let cos_rows = Tensor::from_vec(
            (0..64).map(|i| (i as f32 * 0.02).cos()).collect::<Vec<_>>(),
            (1, 64),
            &Device::Cpu,
        )?
        .to_device(&dev)?;
        let sin_rows = Tensor::from_vec(
            (0..64).map(|i| (i as f32 * 0.02).sin()).collect::<Vec<_>>(),
            (1, 64),
            &Device::Cpu,
        )?
        .to_device(&dev)?;
        let run = || -> Result<Tensor> {
            Ok(super::tensor_apply_rotary_qk(&k, &k, &cos_rows, &sin_rows, None, true)?.1)
        };
        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run()?.flatten_all()?.to_dtype(DType::F32)?;
        let eager_vals = eager.to_vec1::<f32>()?;
        dev.synchronize()?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture_with_mode(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let out = run()?;
        let replayed = out.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
            let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "launch");
            stream.context().enable_event_tracking();
        }
        dev.synchronize()?;

        let got = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let max_diff = eager_vals
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-6, "rope q/k replay diverged: {max_diff}");
        Ok(())
    }

    /// Capture probe for the NVIDIA-role rope path (eager index_select +
    /// to_dtype + narrow + broadcast bmul + cat). Green exonerates the op
    /// family; red would bisect within the chain.
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    #[test]
    #[ignore = "requires the dual CUDA role (3090)"]
    fn rope_eager_chain_capture_launch_matches_eager() -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        use candle_core::{DType, Device, Tensor};

        let dev = Device::new_cuda(0)?;
        // k-side decode shapes: (batch, kv_heads, seq, head_dim) with
        // partial rotary (rot < head_dim), matching the 256-thread bmul
        // (1 token x 4 KV heads x rot 64).
        let (batch, kv_heads, seq, head_dim, rot) = (1usize, 4, 1, 128, 64);
        let k_host: Vec<f32> = (0..batch * kv_heads * seq * head_dim)
            .map(|i| ((i * 37) % 19) as f32 * 0.043 - 0.4)
            .collect();
        let k = Tensor::from_vec(k_host, (batch, kv_heads, seq, head_dim), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&dev)?;
        // cos/sin caches (f32, as built by the loader) over 512 positions.
        let cache_len = 512usize;
        let cos_host: Vec<f32> = (0..cache_len * rot)
            .map(|i| ((i % cache_len) as f32 * 0.02).cos())
            .collect();
        let sin_host: Vec<f32> = (0..cache_len * rot)
            .map(|i| ((i % cache_len) as f32 * 0.02).sin())
            .collect();
        let cos = Tensor::from_vec(cos_host, (cache_len, rot), &Device::Cpu)?.to_device(&dev)?;
        let sin = Tensor::from_vec(sin_host, (cache_len, rot), &Device::Cpu)?.to_device(&dev)?;
        let positions = Tensor::from_vec(vec![7u32], 1, &dev)?;

        let cuda_dev = match &dev {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("expected cuda device"),
        };
        let stream = cuda_dev.cuda_stream();
        for (label, t) in [
            ("k", &k),
            ("cos", &cos),
            ("sin", &sin),
            ("positions", &positions),
        ] {
            let (ptr, bytes) = tensor_va(t, &stream)?;
            eprintln!("[rope-probe] input {label}: ptr=0x{ptr:x} bytes={bytes}");
        }

        let run = |k: &Tensor| -> Result<Tensor> {
            tensor_apply_rotary_q(k, &cos, &sin, Some(&positions), true)
        };
        // Warmup under the htod guard populates the param cache, matching
        // the engine's warmup-then-capture order.
        let _htod_guard = match &dev {
            Device::Cuda(d) => Some(d.enable_cuda_graph_htod_cache()),
            _ => None,
        };
        let eager = run(&k)?.flatten_all()?.to_dtype(DType::F32)?;
        let eager_vals = eager.to_vec1::<f32>()?;
        dev.synchronize()?;

        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::wrap)?;
        unsafe { stream.context().disable_event_tracking() };
        stream
            .begin_capture_with_mode(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(candle_core::Error::wrap)?;
        let replayed = run(&k)?.flatten_all()?;
        let mut graph: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut exec: *mut core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let r =
                sys::cuStreamEndCapture(stream.cu_stream(), core::ptr::from_mut(&mut graph).cast());
            assert_eq!(r as i32, 0, "end capture");
            let r = sys::cuGraphInstantiateWithFlags(
                core::ptr::from_mut(&mut exec).cast(),
                graph.cast(),
                1,
            );
            assert_eq!(r as i32, 0, "instantiate");
            let r = sys::cuGraphUpload(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "upload");
            let r = sys::cuGraphLaunch(exec.cast(), stream.cu_stream());
            assert_eq!(r as i32, 0, "launch");
            stream.context().enable_event_tracking();
        }
        dev.synchronize()?;

        let (out_ptr, out_bytes) = tensor_va(&replayed, &stream)?;
        eprintln!("[rope-probe] output: ptr=0x{out_ptr:x} bytes={out_bytes}");
        let got = replayed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let bad: Vec<(usize, f32, f32)> = eager_vals
            .iter()
            .zip(got.iter())
            .enumerate()
            .filter(|(_, (x, y))| (**x - **y).abs() > 1e-2)
            .map(|(i, (x, y))| (i, *x, *y))
            .collect();
        eprintln!(
            "[rope-probe] mismatching: {} of {}; largest: {:?}",
            bad.len(),
            eager_vals.len(),
            bad.iter()
                .max_by(|x, y| (x.1 - x.2).abs().partial_cmp(&(y.1 - y.2).abs()).unwrap())
        );
        assert!(bad.is_empty(), "rope-eager-chain replay corrupt");
        Ok(())
    }
}

#[cfg(test)]
mod dual_hip_tests {
    use super::*;

    #[cfg(all(feature = "cuda", feature = "rocm"))]
    #[test]
    fn hip_rope_matches_cpu() -> Result<()> {
        let dev = crate::utils::test_gpu_device();
        let (b, h, s, d) = (2usize, 4, 8, 32);
        let q: Vec<f32> = (0..b * h * s * d)
            .map(|i| (i % 17) as f32 * 0.31 - 1.0)
            .collect();
        let q_cpu =
            candle_core::Tensor::from_vec(q.clone(), (b, h, s, d), &candle_core::Device::Cpu)?;
        let q_hip = q_cpu.to_device(&dev)?;
        let rot_d = d / 2;
        let inv_freq: Vec<f32> = (0..rot_d)
            .map(|i| 1f32 / (10_000f32.powf(2.0 * (i as f32 / rot_d as f32))))
            .collect();
        let cos: Vec<f32> = (0..b * s)
            .flat_map(|t| inv_freq.iter().map(move |&f| ((t % s) as f32 * f).cos()))
            .collect();
        let sin: Vec<f32> = (0..b * s)
            .flat_map(|t| inv_freq.iter().map(move |&f| ((t % s) as f32 * f).sin()))
            .collect();
        let dims = (b * s, rot_d);
        let cos_cpu = candle_core::Tensor::from_vec(cos.clone(), dims, &candle_core::Device::Cpu)?;
        let sin_cpu = candle_core::Tensor::from_vec(sin.clone(), dims, &candle_core::Device::Cpu)?;
        let cos_hip = cos_cpu.to_device(&dev)?;
        let sin_hip = sin_cpu.to_device(&dev)?;

        let out_cpu = apply_rotary_q_preselected(&q_cpu, &cos_cpu, &sin_cpu, false)?;
        let out_hip = apply_rotary_q_preselected(&q_hip, &cos_hip, &sin_hip, false)?;
        let hip_vals = out_hip
            .flatten_all()?
            .to_device(&candle_core::Device::Cpu)?
            .to_vec1::<f32>()?;
        let cpu_vals = out_cpu.flatten_all()?.to_vec1::<f32>()?;
        for (a, bb) in hip_vals.iter().zip(cpu_vals.iter()) {
            let diff = (a - bb).abs();
            assert!(diff < 1e-3, "{a} vs {bb}");
        }
        Ok(())
    }
}
