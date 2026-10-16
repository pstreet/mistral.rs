//! NVIDIA-role twin of the top-1 submit machinery and the ranked top-k
//! ops (dual builds only; the hip-role originals live in `ops.rs`).
//!
//! Same contracts and names, bound to `cuda_backend`, with kernels resolved
//! through the nvcc companion (`ffi_cuda`). Dual dispatch happens in `ops.rs`.
//! Also twinned: the categorical packed sampler (default-temperature path),
//! and the topk sampling submit (`sample_ranked_topk`).
//! Still NOT twinned - not on the builtin-MTP path; they refuse
//! loudly if reached on this role: the dflash selectors,
//! and the caller-less `cuda_topk_logits_packed_batched` family.
#[allow(dead_code)]
use crate::cuda::ffi_cuda as ffi;
use crate::ops::{
    final_logits_row, CategoricalLogitsPackedOutput, CudaTop1Completion, RankedTopKPackedOutput,
    TopKLogitsPackedOutput, CUDA_ASYNC_TOKEN_RING_SLOTS, CUDA_CATEGORICAL_PACKED_WIDTH,
    CUDA_TOP1_INVALID_TOKEN, CUDA_TOP1_PACKED_WIDTH, CUDA_TOPK_CHUNK_SIZE,
    CUDA_TOPK_MAX_EXACT_PACKED_VOCAB, CUDA_TOPK_MAX_GRID_Y, CUDA_TOPK_MAX_K,
    CUDA_TOPK_MAX_STAGE2_CANDIDATES, CUDA_TOPK_SAMPLING_PARAM_WIDTH, CUDA_WORKSPACE_MIN_ROWS,
};
use candle_core::Shape;

/// Wrap cuda-backend storage into the `Storage` enum (NVIDIA-role twin of
/// the ops.rs helper; dual builds only).
#[allow(dead_code)]
fn role_storage(s: role_backend::CudaStorage) -> candle_core::Storage {
    candle_core::Storage::Cuda(s)
}
#[allow(dead_code)]
use candle_core::cuda_backend as role_backend;
use candle_core::{DType, Result, Tensor};
use std::sync::Arc;

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct CudaRankedTopKPackedWorkspace {
    location: candle_core::DeviceLocation,
    stream: Arc<role_backend::cudarc::driver::CudaStream>,
    capacity_rows: usize,
    vocab: usize,
    capacity_k: usize,
    nblocks: usize,
    block_values: Tensor,
    block_indices: Tensor,
    radix_state: Tensor,
    packed: Tensor,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl CudaRankedTopKPackedWorkspace {
    fn new(
        dev: &role_backend::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> Result<Self> {
        use candle_core::backend::BackendDevice;

        let capacity_rows = rows
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k row capacity overflow"))?;
        let capacity_k = k
            .checked_next_power_of_two()
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k width capacity overflow"))?;
        let candidate_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k workspace overflow"))?;
        let radix_state_elems = capacity_rows
            .checked_mul(unsafe { ffi::topk_large_ranked_state_words_per_row() })
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k radix overflow"))?;
        let index_elems = capacity_rows
            .checked_mul(nblocks)
            .and_then(|elems| elems.checked_mul(capacity_k))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k index overflow"))?;
        let packed_elems = capacity_rows
            .checked_mul(capacity_k)
            .and_then(|elems| elems.checked_mul(2))
            .ok_or_else(|| candle_core::Error::msg("CUDA ranked top-k packed overflow"))?;
        let device = candle_core::Device::Cuda(dev.clone());
        Ok(Self {
            location: dev.location(),
            stream: dev.cuda_stream(),
            capacity_rows,
            vocab,
            capacity_k,
            nblocks,
            block_values: Tensor::zeros(candidate_elems, DType::F32, &device)?,
            block_indices: Tensor::zeros(index_elems, DType::U32, &device)?,
            radix_state: Tensor::zeros(radix_state_elems, DType::U32, &device)?,
            packed: Tensor::zeros(packed_elems, DType::F32, &device)?,
        })
    }

    fn can_hold(
        &self,
        dev: &role_backend::CudaDevice,
        rows: usize,
        vocab: usize,
        k: usize,
        nblocks: usize,
    ) -> bool {
        use candle_core::backend::BackendDevice;

        let stream = dev.cuda_stream();
        self.location == dev.location()
            && Arc::ptr_eq(self.stream.context(), stream.context())
            && self.stream.cu_stream() == stream.cu_stream()
            && self.capacity_rows >= rows
            && self.vocab == vocab
            && self.capacity_k >= k
            && self.nblocks == nblocks
    }
}
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_ranked_packed_batched(
    input: &Tensor,
    k: usize,
) -> Result<RankedTopKPackedOutput> {
    let mut workspace = None;
    cuda_topk_ranked_packed_batched_with_workspace(input, k, &mut workspace)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_ranked_packed_batched_with_workspace(
    input: &Tensor,
    k: usize,
    cache: &mut Option<CudaRankedTopKPackedWorkspace>,
) -> Result<RankedTopKPackedOutput> {
    use candle_core::backend::BackendStorage;
    use role_backend::cudarc::driver::DevicePtr;
    use role_backend::CudaStorageSlice;
    use std::ffi::c_void;

    const OP: &str = "cuda_topk_ranked_packed_batched";

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{OP} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    let vocab =
        input.dims().last().copied().ok_or_else(|| {
            candle_core::Error::Msg(format!("{OP} requires logits with rank >= 1"))
        })?;
    if vocab == 0 {
        candle_core::bail!("{OP} got an empty vocabulary");
    }
    let batch = input.elem_count() / vocab;
    if batch == 0 {
        candle_core::bail!("{OP} got an empty batch");
    }
    let k = k.min(vocab);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {}]", CUDA_TOPK_MAX_K.min(vocab));
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if vocab > i32::MAX as usize {
        candle_core::bail!("{OP} vocabulary is too large: {vocab}");
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let candidates_per_row = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate count overflow")))?;
    if candidates_per_row > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!("{OP} workspace too large: {candidates_per_row} candidates per row");
    }
    let workspace_elems = batch
        .checked_mul(candidates_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} candidate workspace overflow")))?;
    let radix_state_words_per_row = unsafe { ffi::topk_large_ranked_state_words_per_row() };
    let radix_state_elems = batch
        .checked_mul(radix_state_words_per_row)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} radix workspace overflow")))?;
    let packed_width = k
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    let packed_elems = batch
        .checked_mul(packed_width)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed output overflow")))?;

    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let k_i32 = i32::try_from(k).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{OP} requires CUDA logits"),
    };
    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let needs_alloc = cache
        .as_ref()
        .is_none_or(|workspace| !workspace.can_hold(dev, batch, vocab, k, nblocks));
    if needs_alloc {
        *cache = Some(CudaRankedTopKPackedWorkspace::new(
            dev, batch, vocab, k, nblocks,
        )?);
    }
    let workspace = cache
        .as_ref()
        .expect("CUDA ranked top-k workspace was allocated above");
    let block_values = workspace.block_values.narrow(0, 0, workspace_elems)?;
    let block_indices = workspace.block_indices.narrow(0, 0, workspace_elems)?;
    let radix_state = workspace.radix_state.narrow(0, 0, radix_state_elems)?;
    let packed_dst = workspace.packed.narrow(0, 0, packed_elems)?;

    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr =
                unsafe { (ptr as *const $ty).add(input_layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &input_storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => candle_core::bail!("{OP} logits dtype mismatch"),
    };
    let (block_values_storage_guard, block_values_layout) = block_values.storage_and_layout();
    let candle_core::Storage::Cuda(block_values_storage) = &*block_values_storage_guard else {
        unreachable!("CUDA ranked top-k workspace values are CUDA")
    };
    let CudaStorageSlice::F32(block_values_slice) = &block_values_storage.slice else {
        unreachable!("CUDA ranked top-k workspace values are F32")
    };
    let (block_values_ptr, block_values_guard) = block_values_slice.device_ptr(&stream);
    let block_values_ptr =
        unsafe { (block_values_ptr as *mut f32).add(block_values_layout.start_offset()) };
    let (block_indices_storage_guard, block_indices_layout) = block_indices.storage_and_layout();
    let candle_core::Storage::Cuda(block_indices_storage) = &*block_indices_storage_guard else {
        unreachable!("CUDA ranked top-k workspace indices are CUDA")
    };
    let CudaStorageSlice::U32(block_indices_slice) = &block_indices_storage.slice else {
        unreachable!("CUDA ranked top-k workspace indices are U32")
    };
    let (block_indices_ptr, block_indices_guard) = block_indices_slice.device_ptr(&stream);
    let block_indices_ptr =
        unsafe { (block_indices_ptr as *mut u32).add(block_indices_layout.start_offset()) };
    let (packed_storage_guard, packed_layout) = packed_dst.storage_and_layout();
    let candle_core::Storage::Cuda(packed_storage) = &*packed_storage_guard else {
        unreachable!("CUDA ranked top-k packed workspace is CUDA")
    };
    let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
        unreachable!("CUDA ranked top-k packed workspace is F32")
    };
    let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
    let packed_ptr = unsafe { (packed_ptr as *mut f32).add(packed_layout.start_offset()) };
    let (radix_storage_guard, radix_layout) = radix_state.storage_and_layout();
    let candle_core::Storage::Cuda(radix_storage) = &*radix_storage_guard else {
        unreachable!("CUDA ranked top-k radix workspace is CUDA")
    };
    let CudaStorageSlice::U32(radix_slice) = &radix_storage.slice else {
        unreachable!("CUDA ranked top-k radix workspace is U32")
    };
    let (radix_ptr, radix_guard) = radix_slice.device_ptr(&stream);
    let radix_ptr = unsafe { (radix_ptr as *mut u32).add(radix_layout.start_offset()) };

    macro_rules! launch {
        ($kernel:path, $input:expr) => {{
            unsafe {
                $kernel(
                    $input,
                    block_values_ptr,
                    block_indices_ptr,
                    packed_ptr,
                    radix_ptr.cast::<c_void>(),
                    nrows_i32,
                    ncols_i32,
                    k_i32,
                    chunk_size_i32,
                    nblocks_i32,
                    stream.cu_stream() as i64,
                )
            }
        }};
    }
    let status = match input.dtype() {
        DType::F32 => launch!(
            ffi::topk_large_ranked_f32_packed_batched,
            input_ptr.cast::<f32>()
        ),
        DType::BF16 => launch!(ffi::topk_large_ranked_bf16_packed_batched, input_ptr),
        DType::F16 => launch!(ffi::topk_large_ranked_f16_packed_batched, input_ptr),
        _ => unreachable!(),
    };

    drop(input_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(packed_guard);
    drop(radix_guard);
    drop(block_values_storage_guard);
    drop(block_indices_storage_guard);
    drop(packed_storage_guard);
    drop(radix_storage_guard);
    if status != 0 {
        candle_core::bail!("{OP} CUDA launch failed with status {status}");
    }

    Ok(RankedTopKPackedOutput {
        packed: packed_dst.reshape((batch, packed_width))?,
        k,
        _workspace: vec![block_values, block_indices, radix_state],
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub struct CudaTop1LogitsWorkspace {
    capacity_rows: usize,
    ncols: usize,
    nblocks: usize,
    location: candle_core::DeviceLocation,
    token_ring: CudaAsyncTokenRing,
    slots: Vec<CudaTop1LogitsSlot>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaTop1LogitsSlot {
    block_values: role_backend::cudarc::driver::CudaSlice<f32>,
    block_indices: role_backend::cudarc::driver::CudaSlice<u32>,
    packed: role_backend::cudarc::driver::CudaSlice<f32>,
    packed_host: role_backend::cudarc::driver::PinnedHostSlice<f32>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaAsyncTokenRing {
    id: u64,
    next_slot: usize,
    next_generation: u64,
    slots: Vec<CudaAsyncTokenSlot>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaAsyncTokenSlot {
    owned_token_ids: Tensor,
    token_ids_host: role_backend::cudarc::driver::PinnedHostSlice<u32>,
    device_ready: std::sync::Arc<role_backend::cudarc::driver::CudaEvent>,
    host_complete: std::sync::Arc<role_backend::cudarc::driver::CudaEvent>,
    consumer_complete: role_backend::cudarc::driver::CudaEvent,
    reuse_ready: role_backend::cudarc::driver::CudaEvent,
    reuse_pending: bool,
    pending: Option<CudaAsyncTokenPending>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaAsyncTokenPending {
    generation: u64,
    nrows: usize,
    producer_stream: std::sync::Arc<role_backend::cudarc::driver::CudaStream>,
    consumer_stream: Option<std::sync::Arc<role_backend::cudarc::driver::CudaStream>>,
    token_ptr: u64,
    token_end_ptr: u64,
    token_released: bool,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaAsyncTokenReservation {
    workspace_id: u64,
    slot: usize,
    generation: u64,
    nrows: usize,
    device_tokens: Tensor,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaAsyncTokenSubmission {
    reservation: CudaAsyncTokenReservation,
    device_ready: std::sync::Arc<role_backend::cudarc::driver::CudaEvent>,
    host_complete: std::sync::Arc<role_backend::cudarc::driver::CudaEvent>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaPinnedHostPrefix<'a, T> {
    inner: &'a mut role_backend::cudarc::driver::PinnedHostSlice<T>,
    len: usize,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl<T> role_backend::cudarc::driver::HostSlice<T> for CudaPinnedHostPrefix<'_, T>
where
    T: role_backend::cudarc::driver::ValidAsZeroBits,
{
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a role_backend::cudarc::driver::CudaStream,
    ) -> (&'a [T], role_backend::cudarc::driver::SyncOnDrop<'a>) {
        let (slice, guard) = unsafe { self.inner.stream_synced_slice(stream) };
        (&slice[..self.len], guard)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a role_backend::cudarc::driver::CudaStream,
    ) -> (&'a mut [T], role_backend::cudarc::driver::SyncOnDrop<'a>) {
        let (slice, guard) = unsafe { self.inner.stream_synced_mut_slice(stream) };
        (&mut slice[..self.len], guard)
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl<T> AsRef<[T]> for CudaPinnedHostPrefix<'_, T>
where
    T: role_backend::cudarc::driver::ValidAsZeroBits,
{
    fn as_ref(&self) -> &[T] {
        let full = self.inner.as_slice().expect("pinned host slice");
        &full[..self.len]
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl<T> AsMut<[T]> for CudaPinnedHostPrefix<'_, T>
where
    T: role_backend::cudarc::driver::ValidAsZeroBits,
{
    fn as_mut(&mut self) -> &mut [T] {
        let full = self.inner.as_mut_slice().expect("pinned host slice");
        &mut full[..self.len]
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl CudaAsyncTokenSubmission {
    fn batch_size(&self) -> usize {
        self.reservation.nrows
    }

    fn wait(&self) -> Result<()> {
        self.host_complete
            .synchronize()
            .map_err(candle_core::Error::wrap)
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct CudaTop1Submission {
    token: CudaAsyncTokenSubmission,
    copy_packed: bool,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl CudaTop1Submission {
    #[cfg(test)]
    pub(crate) fn device_tokens(&self) -> &Tensor {
        &self.token.reservation.device_tokens
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.token.batch_size()
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.token.wait()
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaTop1SubmitOptions<'a> {
    nrows: usize,
    ncols: usize,
    token_ids_dst: Option<&'a Tensor>,
    copy_packed: bool,
    op: &'static str,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn cuda_async_token_ring_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn same_cuda_stream(
    left: &role_backend::cudarc::driver::CudaStream,
    right: &role_backend::cudarc::driver::CudaStream,
) -> bool {
    std::sync::Arc::ptr_eq(left.context(), right.context()) && left.cu_stream() == right.cu_stream()
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn new_cuda_async_token_slot(
    dev: &role_backend::CudaDevice,
    capacity_rows: usize,
) -> Result<CudaAsyncTokenSlot> {
    use role_backend::cudarc::driver::{sys, DevicePtrMut};
    use role_backend::{CudaStorage, CudaStorageSlice};

    let stream = dev.cuda_stream();
    let context = stream.context();
    let mut token_ids = unsafe { dev.alloc::<u32>(capacity_rows) }?;
    let (_, token_ids_guard) = token_ids.device_ptr_mut(&stream);
    drop(token_ids_guard);
    let owned_token_ids = Tensor::from((
        role_storage(CudaStorage {
            slice: CudaStorageSlice::U32(token_ids),
            device: dev.clone(),
        }),
        Shape::from_dims(&[capacity_rows, 1]),
    ));
    let event_flags = Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC);
    Ok(CudaAsyncTokenSlot {
        owned_token_ids,
        token_ids_host: unsafe { context.alloc_pinned::<u32>(capacity_rows) }
            .map_err(candle_core::Error::wrap)?,
        device_ready: std::sync::Arc::new(
            context
                .new_event(event_flags)
                .map_err(candle_core::Error::wrap)?,
        ),
        host_complete: std::sync::Arc::new(
            context
                .new_event(event_flags)
                .map_err(candle_core::Error::wrap)?,
        ),
        consumer_complete: context
            .new_event(event_flags)
            .map_err(candle_core::Error::wrap)?,
        reuse_ready: context
            .new_event(event_flags)
            .map_err(candle_core::Error::wrap)?,
        reuse_pending: false,
        pending: None,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn new_cuda_async_token_ring(
    dev: &role_backend::CudaDevice,
    capacity_rows: usize,
) -> Result<CudaAsyncTokenRing> {
    let mut slots = Vec::with_capacity(CUDA_ASYNC_TOKEN_RING_SLOTS);
    for _ in 0..CUDA_ASYNC_TOKEN_RING_SLOTS {
        slots.push(new_cuda_async_token_slot(dev, capacity_rows)?);
    }
    Ok(CudaAsyncTokenRing {
        id: cuda_async_token_ring_id(),
        next_slot: 0,
        next_generation: 1,
        slots,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl CudaAsyncTokenRing {
    fn has_pending(&self) -> bool {
        self.slots.iter().any(|slot| slot.pending.is_some())
    }

    fn validate(&self, submission: &CudaAsyncTokenSubmission, op: &'static str) -> Result<()> {
        let reservation = &submission.reservation;
        if reservation.workspace_id != self.id {
            candle_core::bail!("{op} received a submission from a different workspace");
        }
        let slot = self.slots.get(reservation.slot).ok_or_else(|| {
            candle_core::Error::msg(format!("{op} received an invalid ring slot"))
        })?;
        let Some(pending) = &slot.pending else {
            candle_core::bail!("{op} received an inactive submission");
        };
        if pending.generation != reservation.generation {
            candle_core::bail!("{op} received a stale submission");
        }
        Ok(())
    }

    fn reserve(
        &mut self,
        input: &Tensor,
        token_ids_dst: Option<&Tensor>,
        nrows: usize,
        op: &'static str,
    ) -> Result<CudaAsyncTokenReservation> {
        use role_backend::cudarc::driver::DevicePtr;
        use role_backend::CudaStorageSlice;

        let candle_core::Device::Cuda(dev) = input.device() else {
            candle_core::bail!("{op} requires CUDA tensors");
        };
        let stream = dev.cuda_stream();
        let slot_index = (0..CUDA_ASYNC_TOKEN_RING_SLOTS)
            .map(|offset| (self.next_slot + offset) % CUDA_ASYNC_TOKEN_RING_SLOTS)
            .find(|&index| self.slots[index].pending.is_none())
            .ok_or_else(|| candle_core::Error::msg(format!("{op} submission ring is full")))?;
        self.next_slot = (slot_index + 1) % CUDA_ASYNC_TOKEN_RING_SLOTS;
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        let device_tokens = token_ids_dst
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| self.slots[slot_index].owned_token_ids.narrow(0, 0, nrows))?;
        let destination_capacity = match device_tokens.dims() {
            [capacity, 1] => *capacity,
            _ => 0,
        };
        if device_tokens.dtype() != DType::U32
            || destination_capacity < nrows
            || !device_tokens.is_contiguous()
            || !device_tokens.device().same_device(input.device())
        {
            candle_core::bail!(
                "{op} token destination must be contiguous CUDA U32 [capacity, 1] with capacity >= {nrows}"
            );
        }

        let (token_ptr, token_end_ptr) = {
            let (token_storage, token_layout) = device_tokens.storage_and_layout();
            let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
                unreachable!("token destination device was checked above")
            };
            let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
                unreachable!("token destination dtype was checked above")
            };
            let (token_ptr, token_guard) = token_slice.device_ptr(&stream);
            let token_ptr = unsafe { (token_ptr as *mut u32).add(token_layout.start_offset()) };
            let token_end_ptr = token_ptr as u64 + (nrows * std::mem::size_of::<u32>()) as u64;
            drop(token_guard);
            (token_ptr, token_end_ptr)
        };
        for other in &self.slots {
            let Some(pending) = &other.pending else {
                continue;
            };
            if pending.token_ptr >= token_end_ptr || token_ptr as u64 >= pending.token_end_ptr {
                continue;
            }
            if !pending.token_released {
                candle_core::bail!(
                    "{op} token destination is already leased by another submission"
                );
            }
            if other.reuse_pending {
                stream
                    .wait(&other.reuse_ready)
                    .map_err(candle_core::Error::wrap)?;
            }
        }
        let slot = &mut self.slots[slot_index];
        if slot.reuse_pending {
            stream
                .wait(&slot.reuse_ready)
                .map_err(candle_core::Error::wrap)?;
            slot.reuse_pending = false;
        }
        slot.pending = Some(CudaAsyncTokenPending {
            generation,
            nrows,
            producer_stream: stream,
            consumer_stream: None,
            token_ptr: token_ptr as u64,
            token_end_ptr,
            token_released: false,
        });
        Ok(CudaAsyncTokenReservation {
            workspace_id: self.id,
            slot: slot_index,
            generation,
            nrows,
            device_tokens,
        })
    }

    fn abort(&mut self, reservation: &CudaAsyncTokenReservation) {
        if reservation.workspace_id == self.id {
            if let Some(slot) = self.slots.get_mut(reservation.slot) {
                if slot
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.generation == reservation.generation)
                {
                    slot.pending = None;
                }
            }
        }
    }

    fn finish_launch(
        &mut self,
        reservation: CudaAsyncTokenReservation,
    ) -> Result<CudaAsyncTokenSubmission> {
        use role_backend::CudaStorageSlice;

        let launch = (|| {
            let slot = &mut self.slots[reservation.slot];
            let pending = slot
                .pending
                .as_ref()
                .filter(|pending| pending.generation == reservation.generation)
                .expect("token reservation remains active through its producer launch");
            let stream = pending.producer_stream.clone();
            let candle_core::Device::Cuda(dev) = reservation.device_tokens.device() else {
                candle_core::bail!("token reservation device must be CUDA");
            };
            slot.device_ready
                .record(&stream)
                .map_err(candle_core::Error::wrap)?;
            let (token_storage, token_layout) = reservation.device_tokens.storage_and_layout();
            let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
                unreachable!("reserved token destination is CUDA")
            };
            let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
                unreachable!("reserved token destination is U32")
            };
            let token_copy = token_slice.slice(
                token_layout.start_offset()
                    ..token_layout
                        .start_offset()
                        .saturating_add(reservation.nrows),
            );
            let mut host_prefix = CudaPinnedHostPrefix {
                inner: &mut slot.token_ids_host,
                len: reservation.nrows,
            };
            dev.memcpy_dtoh(&token_copy, &mut host_prefix)?;
            slot.host_complete
                .record(&stream)
                .map_err(candle_core::Error::wrap)?;
            Result::<()>::Ok(())
        })();
        if let Err(error) = launch {
            self.abort(&reservation);
            return Err(error);
        }
        let slot = &self.slots[reservation.slot];
        Ok(CudaAsyncTokenSubmission {
            reservation,
            device_ready: slot.device_ready.clone(),
            host_complete: slot.host_complete.clone(),
        })
    }

    fn wait_on(
        &mut self,
        submission: &CudaAsyncTokenSubmission,
        consumer_stream: &std::sync::Arc<role_backend::cudarc::driver::CudaStream>,
        op: &'static str,
    ) -> Result<()> {
        self.validate(submission, op)?;
        let slot = &mut self.slots[submission.reservation.slot];
        let pending = slot.pending.as_mut().expect("submission validated above");
        if same_cuda_stream(&pending.producer_stream, consumer_stream) {
            slot.reuse_ready
                .record(&pending.producer_stream)
                .map_err(candle_core::Error::wrap)?;
            slot.reuse_pending = true;
            pending.token_released = true;
            return Ok(());
        }
        if let Some(current) = &pending.consumer_stream {
            if same_cuda_stream(current, consumer_stream) {
                return Ok(());
            }
            candle_core::bail!("{op} only supports one cross-stream consumer per submission");
        }
        consumer_stream
            .wait(&submission.device_ready)
            .map_err(candle_core::Error::wrap)?;
        pending.consumer_stream = Some(consumer_stream.clone());
        Ok(())
    }

    fn release_after(
        &mut self,
        submission: &CudaAsyncTokenSubmission,
        consumer_stream: &std::sync::Arc<role_backend::cudarc::driver::CudaStream>,
        op: &'static str,
    ) -> Result<()> {
        self.validate(submission, op)?;
        let slot = &mut self.slots[submission.reservation.slot];
        let pending = slot.pending.as_mut().expect("submission validated above");
        if same_cuda_stream(&pending.producer_stream, consumer_stream) {
            return Ok(());
        }
        let Some(current) = &pending.consumer_stream else {
            candle_core::bail!("{op} requires wait_on before release_after");
        };
        if !same_cuda_stream(current, consumer_stream) {
            candle_core::bail!("{op} consumer stream does not match wait_on");
        }
        slot.consumer_complete
            .record(consumer_stream)
            .map_err(candle_core::Error::wrap)?;
        pending
            .producer_stream
            .wait(&slot.consumer_complete)
            .map_err(candle_core::Error::wrap)?;
        slot.reuse_ready
            .record(&pending.producer_stream)
            .map_err(candle_core::Error::wrap)?;
        slot.reuse_pending = true;
        pending.consumer_stream = None;
        pending.token_released = true;
        Ok(())
    }

    fn complete<'a>(
        &'a mut self,
        submission: &CudaAsyncTokenSubmission,
        op: &'static str,
    ) -> Result<&'a [u32]> {
        self.validate(submission, op)?;
        let slot = &mut self.slots[submission.reservation.slot];
        let pending = slot.pending.as_ref().expect("submission validated above");
        if pending.consumer_stream.is_some() {
            candle_core::bail!("{op} requires release_after for the cross-stream consumer");
        }
        slot.host_complete
            .synchronize()
            .map_err(candle_core::Error::wrap)?;
        let nrows = pending.nrows;
        slot.pending = None;
        Ok(&slot
            .token_ids_host
            .as_slice()
            .map_err(candle_core::Error::wrap)?[..nrows])
    }

    fn cancel(&mut self, submission: &CudaAsyncTokenSubmission, op: &'static str) -> Result<()> {
        self.validate(submission, op)?;
        let slot = &mut self.slots[submission.reservation.slot];
        let pending = slot.pending.as_mut().expect("submission validated above");
        if let Some(consumer_stream) = pending.consumer_stream.take() {
            slot.consumer_complete
                .record(&consumer_stream)
                .map_err(candle_core::Error::wrap)?;
            pending
                .producer_stream
                .wait(&slot.consumer_complete)
                .map_err(candle_core::Error::wrap)?;
            slot.reuse_ready
                .record(&pending.producer_stream)
                .map_err(candle_core::Error::wrap)?;
            slot.reuse_pending = true;
        }
        slot.host_complete
            .synchronize()
            .map_err(candle_core::Error::wrap)?;
        slot.pending = None;
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn new_cuda_top1_slot(
    dev: &role_backend::CudaDevice,
    workspace_elems: usize,
    packed_elems: usize,
) -> Result<CudaTop1LogitsSlot> {
    let stream = dev.cuda_stream();
    let context = stream.context();

    Ok(CudaTop1LogitsSlot {
        block_values: unsafe { dev.alloc::<f32>(workspace_elems) }?,
        block_indices: unsafe { dev.alloc::<u32>(workspace_elems) }?,
        packed: unsafe { dev.alloc::<f32>(packed_elems) }?,
        packed_host: unsafe { context.alloc_pinned::<f32>(packed_elems) }
            .map_err(candle_core::Error::wrap)?,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn new_cuda_top1_workspace(
    dev: &role_backend::CudaDevice,
    nrows: usize,
    ncols: usize,
    nblocks: usize,
) -> Result<CudaTop1LogitsWorkspace> {
    use candle_core::backend::BackendDevice;

    let nrows = nrows.max(CUDA_WORKSPACE_MIN_ROWS);
    let workspace_elems = nrows
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg("CUDA top-1 workspace overflow".to_string()))?;
    let packed_elems = nrows
        .checked_mul(CUDA_TOP1_PACKED_WIDTH)
        .ok_or_else(|| candle_core::Error::Msg("CUDA top-1 output overflow".to_string()))?;
    let mut slots = Vec::with_capacity(CUDA_ASYNC_TOKEN_RING_SLOTS);
    for _ in 0..CUDA_ASYNC_TOKEN_RING_SLOTS {
        slots.push(new_cuda_top1_slot(dev, workspace_elems, packed_elems)?);
    }
    Ok(CudaTop1LogitsWorkspace {
        capacity_rows: nrows,
        ncols,
        nblocks,
        location: dev.location(),
        token_ring: new_cuda_async_token_ring(dev, nrows)?,
        slots,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn validate_cuda_top1_submission<'a>(
    workspace: &'a CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    op: &'static str,
) -> Result<&'a CudaTop1LogitsSlot> {
    workspace.token_ring.validate(&submission.token, op)?;
    let slot_index = submission.token.reservation.slot;
    let slot = workspace
        .slots
        .get(slot_index)
        .ok_or_else(|| candle_core::Error::Msg(format!("{op} received an invalid ring slot")))?;
    Ok(slot)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn cuda_top1_logits_submit_inner(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
    options: CudaTop1SubmitOptions<'_>,
) -> Result<CudaTop1Submission> {
    use candle_core::backend::{BackendDevice, BackendStorage};
    use role_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use role_backend::CudaStorageSlice;
    use std::ffi::c_void;

    let CudaTop1SubmitOptions {
        nrows,
        ncols,
        token_ids_dst,
        copy_packed,
        op,
    } = options;

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{op} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op });
    }
    if nrows == 0 || ncols == 0 {
        candle_core::bail!("{op} requires non-empty logits");
    }
    if ncols > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{op} vocabulary size {ncols} cannot be represented exactly by packed F32 indices"
        );
    }
    if nrows > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{op} batch is too large for a 2D CUDA launch: {nrows}");
    }
    let expected_elems = nrows
        .checked_mul(ncols)
        .ok_or_else(|| candle_core::Error::Msg(format!("{op} input size overflow")))?;
    if input.elem_count() != expected_elems {
        candle_core::bail!(
            "{op} expected {nrows} rows of {ncols} logits, got {} elements",
            input.elem_count()
        );
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let (storage, layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("{op} requires CUDA logits"),
    };
    let dev = storage.device();
    let location = dev.location();
    let needs_alloc = cache.as_ref().is_none_or(|workspace| {
        workspace.capacity_rows < nrows
            || workspace.ncols != ncols
            || workspace.nblocks != nblocks
            || workspace.location != location
    });
    if needs_alloc {
        if cache
            .as_ref()
            .is_some_and(|workspace| workspace.token_ring.has_pending())
        {
            candle_core::bail!("{op} cannot resize while submissions are pending");
        }
        if cache.is_some() {
            // quiesce before the old workspace's pinned buffers free below
            dev.cuda_stream()
                .synchronize()
                .map_err(candle_core::Error::wrap)?;
        }
        *cache = Some(new_cuda_top1_workspace(dev, nrows, ncols, nblocks)?);
    }

    let stream = dev.cuda_stream();
    macro_rules! input_ptr {
        ($slice:expr, $ty:ty) => {{
            let (ptr, guard) = $slice.device_ptr(&stream);
            let ptr = unsafe { (ptr as *const $ty).add(layout.start_offset()) as *const c_void };
            (ptr, guard)
        }};
    }
    let (input_ptr, input_guard) = match &storage.slice {
        CudaStorageSlice::F32(slice) => input_ptr!(slice, f32),
        CudaStorageSlice::BF16(slice) => input_ptr!(slice, half::bf16),
        CudaStorageSlice::F16(slice) => input_ptr!(slice, half::f16),
        _ => unreachable!("logits dtype was validated above"),
    };
    let workspace = cache
        .as_mut()
        .expect("CUDA top-1 workspace was allocated above");
    let reservation = workspace
        .token_ring
        .reserve(input, token_ids_dst, nrows, op)?;
    let slot_index = reservation.slot;
    let device_tokens_storage = reservation.device_tokens.clone();
    let (token_storage, token_layout) = device_tokens_storage.storage_and_layout();
    let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
        unreachable!("reserved token destination is CUDA")
    };
    let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
        unreachable!("reserved token destination is U32")
    };
    let (token_ptr, token_guard) = token_slice.device_ptr(&stream);
    let token_ptr = unsafe { (token_ptr as *mut u32).add(token_layout.start_offset()) };
    let slot = &mut workspace.slots[slot_index];

    let result = (|| {
        let (block_values_ptr, block_values_guard) = slot.block_values.device_ptr_mut(&stream);
        let (block_indices_ptr, block_indices_guard) = slot.block_indices.device_ptr_mut(&stream);
        let (packed_ptr, packed_guard) = if copy_packed {
            let (packed_ptr, packed_guard) = slot.packed.device_ptr_mut(&stream);
            (packed_ptr as *mut f32, Some(packed_guard))
        } else {
            (std::ptr::null_mut(), None)
        };

        let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
        let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;
        let chunk_size_i32 =
            i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
        let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;
        macro_rules! launch {
            ($single:path, $batched:path, $ptr:expr) => {{
                unsafe {
                    if nrows == 1 {
                        $single(
                            $ptr,
                            block_values_ptr as *mut f32,
                            block_indices_ptr as *mut u32,
                            packed_ptr,
                            token_ptr,
                            ncols_i32,
                            chunk_size_i32,
                            nblocks_i32,
                            stream.cu_stream() as i64,
                        );
                    } else {
                        $batched(
                            $ptr,
                            block_values_ptr as *mut f32,
                            block_indices_ptr as *mut u32,
                            packed_ptr,
                            token_ptr,
                            nrows_i32,
                            ncols_i32,
                            chunk_size_i32,
                            nblocks_i32,
                            stream.cu_stream() as i64,
                        );
                    }
                }
            }};
        }
        match input.dtype() {
            DType::F32 => launch!(
                ffi::top1_large_f32_packed,
                ffi::top1_large_f32_packed_batched,
                input_ptr.cast::<f32>()
            ),
            DType::BF16 => launch!(
                ffi::top1_large_bf16_packed,
                ffi::top1_large_bf16_packed_batched,
                input_ptr
            ),
            DType::F16 => launch!(
                ffi::top1_large_f16_packed,
                ffi::top1_large_f16_packed_batched,
                input_ptr
            ),
            _ => unreachable!("logits dtype was validated above"),
        }

        drop(input_guard);
        drop(block_values_guard);
        drop(block_indices_guard);
        drop(packed_guard);
        drop(token_guard);
        if copy_packed {
            dev.memcpy_dtoh(&slot.packed, &mut slot.packed_host)?;
        }
        Result::<()>::Ok(())
    })();
    if let Err(error) = result {
        let _ = stream.synchronize();
        workspace.token_ring.abort(&reservation);
        return Err(error);
    }
    let token = match workspace.token_ring.finish_launch(reservation) {
        Ok(token) => token,
        Err(error) => {
            let _ = stream.synchronize();
            return Err(error);
        }
    };
    Ok(CudaTop1Submission { token, copy_packed })
}
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_top1_logits_submit_batched(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: None,
            copy_packed: false,
            op: OP,
        },
    )
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_top1_logits_submit_batched_packed(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched_packed";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: None,
            copy_packed: true,
            op: OP,
        },
    )
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_top1_logits_submit_batched_into(
    input: &Tensor,
    token_ids_dst: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<CudaTop1Submission> {
    const OP: &str = "cuda_top1_logits_submit_batched_into";
    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows: *batch,
            ncols: *vocab,
            token_ids_dst: Some(token_ids_dst),
            copy_packed: false,
            op: OP,
        },
    )
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_top1_submission_complete<'a>(
    workspace: &'a mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
) -> Result<CudaTop1Completion<'a>> {
    const OP: &str = "cuda_top1_submission_complete";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    let slot_index = submission.token.reservation.slot;
    let nrows = submission.token.reservation.nrows;
    let token_ids = workspace.token_ring.complete(&submission.token, OP)?;
    let packed = if submission.copy_packed {
        Some(
            &workspace.slots[slot_index]
                .packed_host
                .as_slice()
                .map_err(candle_core::Error::wrap)?[..nrows * CUDA_TOP1_PACKED_WIDTH],
        )
    } else {
        None
    };
    Ok(CudaTop1Completion { token_ids, packed })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_top1_submission_cancel(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
) -> Result<()> {
    const OP: &str = "cuda_top1_submission_cancel";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    workspace.token_ring.cancel(&submission.token, OP)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_top1_device_tokens_wait_on(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    consumer_stream: &Arc<role_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    const OP: &str = "cuda_top1_device_tokens_wait_on";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    workspace
        .token_ring
        .wait_on(&submission.token, consumer_stream, OP)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_top1_device_tokens_release_after(
    workspace: &mut CudaTop1LogitsWorkspace,
    submission: &CudaTop1Submission,
    consumer_stream: &Arc<role_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    const OP: &str = "cuda_top1_device_tokens_release_after";
    validate_cuda_top1_submission(workspace, submission, OP)?;
    workspace
        .token_ring
        .release_after(&submission.token, consumer_stream, OP)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn cuda_top1_logits_f32_packed_cached_inner<'a>(
    input: &Tensor,
    nrows: usize,
    ncols: usize,
    cache: &'a mut Option<CudaTop1LogitsWorkspace>,
    op: &'static str,
) -> Result<&'a [f32]> {
    let submission = cuda_top1_logits_submit_inner(
        input,
        cache,
        CudaTop1SubmitOptions {
            nrows,
            ncols,
            token_ids_dst: None,
            copy_packed: true,
            op,
        },
    )?;
    let completion = cuda_top1_submission_complete(
        cache
            .as_mut()
            .expect("CUDA top-1 workspace was allocated during submission"),
        &submission,
    )?;
    Ok(completion
        .packed()
        .expect("packed output was requested during submission"))
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn cuda_top1_logits_f32_cached(
    input: &Tensor,
    cache: &mut Option<CudaTop1LogitsWorkspace>,
) -> Result<[f32; CUDA_TOP1_PACKED_WIDTH]> {
    const OP: &str = "cuda_top1_logits_f32_cached";
    let input = final_logits_row(input)?;
    let ncols = input.elem_count();
    let packed = cuda_top1_logits_f32_packed_cached_inner(&input, 1, ncols, cache, OP)?;
    Ok([packed[0], packed[1]])
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_categorical_logits_f32_packed_batched(
    input: &Tensor,
    inverse_temperatures: &Tensor,
    uniforms: &Tensor,
) -> Result<CategoricalLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use role_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use role_backend::CudaStorageSlice;

    const OP: &str = "cuda_categorical_logits_f32_packed_batched";
    if input.dtype() != DType::F32
        || inverse_temperatures.dtype() != DType::F32
        || uniforms.dtype() != DType::F32
    {
        candle_core::bail!("{OP} requires F32 tensors");
    }
    if !input.is_contiguous() || !inverse_temperatures.is_contiguous() || !uniforms.is_contiguous()
    {
        return Err(candle_core::Error::RequiresContiguous { op: OP });
    }
    if !input.device().same_device(inverse_temperatures.device())
        || !input.device().same_device(uniforms.device())
    {
        candle_core::bail!("{OP} tensors must be on the same CUDA device");
    }

    let [batch, vocab] = input.dims() else {
        candle_core::bail!("{OP} requires logits with shape [batch, vocab]");
    };
    let (batch, vocab) = (*batch, *vocab);
    if batch == 0 || vocab == 0 {
        candle_core::bail!("{OP} requires a non-empty batch and vocabulary");
    }
    if inverse_temperatures.dims() != [batch] || uniforms.dims() != [batch] {
        candle_core::bail!(
            "{OP} expected sampling tensors with shape [{batch}], got {:?} and {:?}",
            inverse_temperatures.dims(),
            uniforms.dims()
        );
    }
    if vocab > CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        candle_core::bail!(
            "{OP} vocabulary size {vocab} cannot be represented exactly by packed F32 indices"
        );
    }
    if batch > CUDA_TOPK_MAX_GRID_Y {
        candle_core::bail!("{OP} batch is too large for a 2D CUDA launch: {batch}");
    }

    let nblocks = vocab.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let workspace_elems = batch
        .checked_mul(nblocks)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} workspace overflow")))?;
    let packed_elems = batch
        .checked_mul(CUDA_CATEGORICAL_PACKED_WIDTH)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} output overflow")))?;
    let nrows_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let chunk_size_i32 = i32::try_from(CUDA_TOPK_CHUNK_SIZE).map_err(candle_core::Error::wrap)?;
    let nblocks_i32 = i32::try_from(nblocks).map_err(candle_core::Error::wrap)?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let candle_core::Storage::Cuda(input_storage) = &*input_storage else {
        candle_core::bail!("{OP} requires CUDA logits");
    };
    let CudaStorageSlice::F32(input_slice) = &input_storage.slice else {
        candle_core::bail!("{OP} only supports F32 logits");
    };
    let (temperature_storage, temperature_layout) = inverse_temperatures.storage_and_layout();
    let candle_core::Storage::Cuda(temperature_storage) = &*temperature_storage else {
        candle_core::bail!("{OP} requires CUDA inverse temperatures");
    };
    let CudaStorageSlice::F32(temperature_slice) = &temperature_storage.slice else {
        candle_core::bail!("{OP} only supports F32 inverse temperatures");
    };
    let (uniform_storage, uniform_layout) = uniforms.storage_and_layout();
    let candle_core::Storage::Cuda(uniform_storage) = &*uniform_storage else {
        candle_core::bail!("{OP} requires CUDA uniforms");
    };
    let CudaStorageSlice::F32(uniform_slice) = &uniform_storage.slice else {
        candle_core::bail!("{OP} only supports F32 uniforms");
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(packed_elems) }?;

    let (input_ptr, input_guard) = input_slice.device_ptr(&stream);
    let (temperature_ptr, temperature_guard) = temperature_slice.device_ptr(&stream);
    let (uniform_ptr, uniform_guard) = uniform_slice.device_ptr(&stream);
    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);
    let input_ptr = unsafe { (input_ptr as *const f32).add(input_layout.start_offset()) };
    let temperature_ptr =
        unsafe { (temperature_ptr as *const f32).add(temperature_layout.start_offset()) };
    let uniform_ptr = unsafe { (uniform_ptr as *const f32).add(uniform_layout.start_offset()) };

    unsafe {
        ffi::categorical_large_f32_packed_batched(
            input_ptr,
            temperature_ptr,
            uniform_ptr,
            block_values_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            packed_ptr as *mut f32,
            nrows_i32,
            ncols_i32,
            chunk_size_i32,
            nblocks_i32,
            stream.cu_stream() as i64,
        );
    }

    drop(input_guard);
    drop(temperature_guard);
    drop(uniform_guard);
    drop(block_values_guard);
    drop(block_sums_guard);
    drop(packed_guard);

    let workspace = vec![
        Tensor::from((
            role_storage(role_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
        Tensor::from((
            role_storage(role_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[batch, nblocks]),
        )),
    ];
    let packed_storage = role_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };

    Ok(CategoricalLogitsPackedOutput {
        packed: Tensor::from((
            role_storage(packed_storage),
            Shape::from_dims(&[batch, CUDA_CATEGORICAL_PACKED_WIDTH]),
        )),
        _workspace: workspace,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_logits_f32_packed(
    input: &Tensor,
    k: usize,
    temperature: f64,
) -> Result<TopKLogitsPackedOutput> {
    use candle_core::backend::BackendStorage;
    use role_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use role_backend::CudaStorageSlice;

    const OP: &str = "cuda_topk_logits_f32_packed";

    if temperature <= 0.0 || !temperature.is_finite() {
        candle_core::bail!("{OP} requires a positive finite temperature");
    }

    let input = input.contiguous()?;
    if input.dtype() != DType::F32 {
        candle_core::bail!("{OP} requires F32 logits");
    }

    let ncols = input.elem_count();
    if ncols == 0 {
        candle_core::bail!("{OP} got empty logits");
    }
    let k = k.min(ncols);
    if k == 0 || k > CUDA_TOPK_MAX_K {
        candle_core::bail!("{OP} k={k} must be in [1, {}]", CUDA_TOPK_MAX_K.min(ncols));
    }

    let nblocks = ncols.div_ceil(CUDA_TOPK_CHUNK_SIZE);
    let stage2_candidates = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} workspace overflow")))?;
    if stage2_candidates > CUDA_TOPK_MAX_STAGE2_CANDIDATES {
        candle_core::bail!("{OP} workspace too large: {stage2_candidates} candidates");
    }

    let (storage, layout) = input.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("{OP} requires CUDA logits");
    };
    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let CudaStorageSlice::F32(inp) = &storage.slice else {
        candle_core::bail!("{OP} only supports F32");
    };

    let workspace_elems = nblocks
        .checked_mul(k)
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} workspace overflow")))?;
    let block_elems = nblocks;
    let packed_elems = 2usize
        .checked_mul(k)
        .and_then(|width| width.checked_add(2))
        .ok_or_else(|| candle_core::Error::Msg(format!("{OP} packed width overflow")))?;
    let mut block_values = unsafe { dev.alloc::<f32>(workspace_elems) }?;
    let mut block_indices = unsafe { dev.alloc::<u32>(workspace_elems) }?;
    let mut block_maxes = unsafe { dev.alloc::<f32>(block_elems) }?;
    let mut block_sums = unsafe { dev.alloc::<f32>(block_elems) }?;
    let mut packed_dst = unsafe { dev.alloc::<f32>(packed_elems) }?;

    let (src_ptr, src_guard) = inp.device_ptr(&stream);
    let (block_values_ptr, block_values_guard) = block_values.device_ptr_mut(&stream);
    let (block_indices_ptr, block_indices_guard) = block_indices.device_ptr_mut(&stream);
    let (block_maxes_ptr, block_maxes_guard) = block_maxes.device_ptr_mut(&stream);
    let (block_sums_ptr, block_sums_guard) = block_sums.device_ptr_mut(&stream);
    let (packed_ptr, packed_guard) = packed_dst.device_ptr_mut(&stream);
    let src_ptr = unsafe { (src_ptr as *const f32).add(layout.start_offset()) };

    unsafe {
        ffi::topk_large_f32_packed(
            src_ptr,
            block_values_ptr as *mut f32,
            block_indices_ptr as *mut u32,
            block_maxes_ptr as *mut f32,
            block_sums_ptr as *mut f32,
            packed_ptr as *mut f32,
            ncols as i32,
            k as i32,
            CUDA_TOPK_CHUNK_SIZE as i32,
            nblocks as i32,
            (1.0 / temperature) as f32,
            stream_raw,
        );
    }

    drop(src_guard);
    drop(block_values_guard);
    drop(block_indices_guard);
    drop(block_maxes_guard);
    drop(block_sums_guard);
    drop(packed_guard);

    let packed_storage = role_backend::CudaStorage {
        slice: CudaStorageSlice::F32(packed_dst),
        device: dev.clone(),
    };
    let workspace = vec![
        Tensor::from((
            role_storage(role_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_values),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            role_storage(role_backend::CudaStorage {
                slice: CudaStorageSlice::U32(block_indices),
                device: dev.clone(),
            }),
            Shape::from_dims(&[workspace_elems]),
        )),
        Tensor::from((
            role_storage(role_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_maxes),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
        Tensor::from((
            role_storage(role_backend::CudaStorage {
                slice: CudaStorageSlice::F32(block_sums),
                device: dev.clone(),
            }),
            Shape::from_dims(&[nblocks]),
        )),
    ];

    Ok(TopKLogitsPackedOutput {
        packed: Tensor::from((role_storage(packed_storage), Shape::from_dims(&[2 * k + 2]))),
        k,
        _workspace: workspace,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct CudaTopKSamplingWorkspace {
    capacity_rows: usize,
    capacity_k: usize,
    vocab: usize,
    location: candle_core::DeviceLocation,
    ranked: Option<CudaRankedTopKPackedWorkspace>,
    token_ring: CudaAsyncTokenRing,
    slots: Vec<CudaTopKSamplingSlot>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct CudaTopKSamplingSlot {
    params: role_backend::cudarc::driver::CudaSlice<f32>,
    params_host: role_backend::cudarc::driver::PinnedHostSlice<f32>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct CudaTopKSamplingSubmission {
    token: CudaAsyncTokenSubmission,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl CudaTopKSamplingSubmission {
    pub(crate) fn batch_size(&self) -> usize {
        self.token.batch_size()
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.token.wait()
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn new_cuda_topk_sampling_slot(
    dev: &role_backend::CudaDevice,
    capacity_rows: usize,
) -> Result<CudaTopKSamplingSlot> {
    let stream = dev.cuda_stream();
    let context = stream.context();
    let param_elems = capacity_rows
        .checked_mul(CUDA_TOPK_SAMPLING_PARAM_WIDTH)
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling parameter overflow"))?;
    let mut params_host =
        unsafe { context.alloc_pinned::<f32>(param_elems) }.map_err(candle_core::Error::wrap)?;
    params_host
        .as_mut_slice()
        .map_err(candle_core::Error::wrap)?
        .fill(0.0);

    Ok(CudaTopKSamplingSlot {
        params: unsafe { dev.alloc::<f32>(param_elems) }?,
        params_host,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn new_cuda_topk_sampling_workspace(
    dev: &role_backend::CudaDevice,
    rows: usize,
    vocab: usize,
    k: usize,
) -> Result<CudaTopKSamplingWorkspace> {
    use candle_core::backend::BackendDevice;

    let rows = rows.max(CUDA_WORKSPACE_MIN_ROWS);
    let capacity_rows = rows
        .checked_next_power_of_two()
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling row capacity overflow"))?;
    let capacity_k = k
        .checked_next_power_of_two()
        .ok_or_else(|| candle_core::Error::msg("CUDA top-k sampling width capacity overflow"))?;
    let mut slots = Vec::with_capacity(CUDA_ASYNC_TOKEN_RING_SLOTS);
    for _ in 0..CUDA_ASYNC_TOKEN_RING_SLOTS {
        slots.push(new_cuda_topk_sampling_slot(dev, capacity_rows)?);
    }
    Ok(CudaTopKSamplingWorkspace {
        capacity_rows,
        capacity_k,
        vocab,
        location: dev.location(),
        ranked: None,
        token_ring: new_cuda_async_token_ring(dev, capacity_rows)?,
        slots,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn validate_cuda_topk_sampling_params(
    params: &[crate::ops::CudaTopKSamplingParams],
    rows: usize,
    vocab: usize,
    op: &'static str,
) -> Result<usize> {
    if params.len() != rows {
        candle_core::bail!("{op} expected {rows} sampling parameter rows");
    }
    let mut max_k = 0usize;
    for params in params {
        if !params.inverse_temperature.is_finite() || params.inverse_temperature <= 0.0 {
            candle_core::bail!("{op} requires positive finite inverse temperatures");
        }
        if params.top_k == 0 || params.top_k > CUDA_TOPK_MAX_K {
            candle_core::bail!("{op} top-k must be in [1, {}]", CUDA_TOPK_MAX_K);
        }
        if !params.top_p.is_finite() || !params.min_p.is_finite() {
            candle_core::bail!("{op} requires finite top-p and min-p values");
        }
        if !(0.0..1.0).contains(&params.uniform) {
            candle_core::bail!("{op} requires uniforms in [0, 1)");
        }
        max_k = max_k.max(params.top_k.min(vocab));
    }
    Ok(max_k)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn copy_cuda_topk_sampling_params(
    dev: &role_backend::CudaDevice,
    slot: &mut CudaTopKSamplingSlot,
    params: &[crate::ops::CudaTopKSamplingParams],
) -> Result<()> {
    let host = slot
        .params_host
        .as_mut_slice()
        .map_err(candle_core::Error::wrap)?;
    for (row, params) in params.iter().enumerate() {
        let start = row * CUDA_TOPK_SAMPLING_PARAM_WIDTH;
        let top_k = u16::try_from(params.top_k).map_err(candle_core::Error::wrap)?;
        host[start] = params.inverse_temperature;
        host[start + 1] = f32::from(top_k);
        host[start + 2] = params.top_p;
        host[start + 3] = params.min_p;
        host[start + 4] = params.uniform;
    }
    dev.memcpy_htod(&slot.params_host, &mut slot.params)?;
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn cuda_topk_sampling_submit_inner(
    input: &Tensor,
    token_ids_dst: Option<&Tensor>,
    params: &[crate::ops::CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
    op: &'static str,
) -> Result<CudaTopKSamplingSubmission> {
    use candle_core::backend::{BackendDevice, BackendStorage};
    use role_backend::cudarc::driver::DevicePtr;
    use role_backend::CudaStorageSlice;

    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("{op} requires BF16, F16, or F32 logits");
    }
    if !input.is_contiguous() {
        return Err(candle_core::Error::RequiresContiguous { op });
    }
    let [rows, vocab] = input.dims() else {
        candle_core::bail!("{op} requires logits with shape [batch, vocab]");
    };
    if *rows == 0 || *vocab == 0 {
        candle_core::bail!("{op} requires non-empty logits");
    }
    let max_k = validate_cuda_topk_sampling_params(params, *rows, *vocab, op)?;
    let (storage, _) = input.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("{op} requires CUDA logits");
    };

    let dev = storage.device();
    let needs_alloc = cache.as_ref().is_none_or(|workspace| {
        workspace.capacity_rows < *rows
            || workspace.capacity_k < max_k
            || workspace.vocab != *vocab
            || workspace.location != dev.location()
    });
    if needs_alloc {
        if cache
            .as_ref()
            .is_some_and(|workspace| workspace.token_ring.has_pending())
        {
            candle_core::bail!("{op} cannot resize while submissions are pending");
        }
        if cache.is_some() {
            dev.cuda_stream()
                .synchronize()
                .map_err(candle_core::Error::wrap)?;
        }
        *cache = Some(new_cuda_topk_sampling_workspace(dev, *rows, *vocab, max_k)?);
    }

    let stream = dev.cuda_stream();
    let workspace = cache
        .as_mut()
        .expect("CUDA top-k sampling workspace was allocated above");
    let reservation = workspace
        .token_ring
        .reserve(input, token_ids_dst, *rows, op)?;
    let slot_index = reservation.slot;
    let result = (|| {
        let ranked =
            cuda_topk_ranked_packed_batched_with_workspace(input, max_k, &mut workspace.ranked)?;
        let slot = &mut workspace.slots[slot_index];
        copy_cuda_topk_sampling_params(dev, slot, params)?;

        let (packed_storage, packed_layout) = ranked.packed.storage_and_layout();
        let candle_core::Storage::Cuda(packed_storage) = &*packed_storage else {
            candle_core::bail!("{op} ranked top-k output is not CUDA");
        };
        let CudaStorageSlice::F32(packed_slice) = &packed_storage.slice else {
            candle_core::bail!("{op} ranked top-k output is not F32");
        };
        let (token_storage, token_layout) = reservation.device_tokens.storage_and_layout();
        let candle_core::Storage::Cuda(token_storage) = &*token_storage else {
            candle_core::bail!("{op} reserved token destination is not CUDA");
        };
        let CudaStorageSlice::U32(token_slice) = &token_storage.slice else {
            candle_core::bail!("{op} reserved token destination is not U32");
        };
        let (packed_ptr, packed_guard) = packed_slice.device_ptr(&stream);
        let (params_ptr, params_guard) = slot.params.device_ptr(&stream);
        let (tokens_ptr, tokens_guard) = token_slice.device_ptr(&stream);
        let packed_ptr = unsafe { (packed_ptr as *const f32).add(packed_layout.start_offset()) };
        let params_ptr = params_ptr as *const f32;
        let tokens_ptr = unsafe { (tokens_ptr as *mut u32).add(token_layout.start_offset()) };
        unsafe {
            ffi::sample_ranked_topk(
                packed_ptr,
                params_ptr,
                tokens_ptr,
                i32::try_from(*rows).map_err(candle_core::Error::wrap)?,
                i32::try_from(ranked.k).map_err(candle_core::Error::wrap)?,
                stream.cu_stream() as i64,
            );
        }
        drop(packed_guard);
        drop(params_guard);
        drop(tokens_guard);
        Result::<()>::Ok(())
    })();
    if let Err(error) = result {
        let _ = stream.synchronize();
        workspace.token_ring.abort(&reservation);
        return Err(error);
    }
    let token = match workspace.token_ring.finish_launch(reservation) {
        Ok(token) => token,
        Err(error) => {
            let _ = stream.synchronize();
            return Err(error);
        }
    };
    Ok(CudaTopKSamplingSubmission { token })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_sampling_submit_batched(
    input: &Tensor,
    params: &[crate::ops::CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
) -> Result<CudaTopKSamplingSubmission> {
    cuda_topk_sampling_submit_inner(
        input,
        None,
        params,
        cache,
        "cuda_topk_sampling_submit_batched",
    )
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_sampling_submit_batched_into(
    input: &Tensor,
    token_ids_dst: &Tensor,
    params: &[crate::ops::CudaTopKSamplingParams],
    cache: &mut Option<CudaTopKSamplingWorkspace>,
) -> Result<CudaTopKSamplingSubmission> {
    cuda_topk_sampling_submit_inner(
        input,
        Some(token_ids_dst),
        params,
        cache,
        "cuda_topk_sampling_submit_batched_into",
    )
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_sampling_device_tokens_wait_on(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
    consumer_stream: &Arc<role_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    const OP: &str = "cuda_topk_sampling_device_tokens_wait_on";
    workspace.token_ring.validate(&submission.token, OP)?;
    workspace.token_ring.wait_on(
        &submission.token,
        consumer_stream,
        "cuda_topk_sampling_device_tokens_wait_on",
    )
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_sampling_device_tokens_release_after(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
    consumer_stream: &Arc<role_backend::cudarc::driver::CudaStream>,
) -> Result<()> {
    const OP: &str = "cuda_topk_sampling_device_tokens_release_after";
    workspace.token_ring.validate(&submission.token, OP)?;
    workspace.token_ring.release_after(
        &submission.token,
        consumer_stream,
        "cuda_topk_sampling_device_tokens_release_after",
    )
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_sampling_submission_complete<'a>(
    workspace: &'a mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
) -> Result<CudaTopKSamplingCompletion<'a>> {
    const OP: &str = "cuda_topk_sampling_submission_complete";
    workspace.token_ring.validate(&submission.token, OP)?;
    let token_ids = workspace.token_ring.complete(&submission.token, OP)?;
    if token_ids.contains(&CUDA_TOP1_INVALID_TOKEN) {
        candle_core::bail!("invalid CUDA top-k sampling output");
    }
    Ok(CudaTopKSamplingCompletion { token_ids })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn cuda_topk_sampling_submission_cancel(
    workspace: &mut CudaTopKSamplingWorkspace,
    submission: &CudaTopKSamplingSubmission,
) -> Result<()> {
    const OP: &str = "cuda_topk_sampling_submission_cancel";
    workspace.token_ring.validate(&submission.token, OP)?;
    workspace.token_ring.cancel(&submission.token, OP)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct CudaTopKSamplingCompletion<'a> {
    token_ids: &'a [u32],
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl<'a> CudaTopKSamplingCompletion<'a> {
    pub(crate) fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }
}
