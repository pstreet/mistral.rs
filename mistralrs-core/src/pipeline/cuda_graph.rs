use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    fmt,
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

// GPU-driver bindings for the decode-graph stack. This file is the fork's
// serving stack: the AMD role in a dual build (all C kernels are hipcc-built)
// and the compiled vendor in single-vendor builds. The NVIDIA-original
// cu*-graph paths below stay for cuda-only builds.
use crate::pipeline::graph_role;
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
use crate::pipeline::graph_role::GraphStreamExt;
use crate::pipeline::graph_role::{
    graph_alloc_pinned, graph_destroy, graph_end_capture, graph_exec_destroy, graph_instantiate,
    graph_launch, graph_memcpy_htod_async, graph_stage_ptr, graph_upload, GraphEvent,
    GraphPinnedHost, GraphStream,
};
#[cfg(feature = "rocm")]
use candle_core::role::backend::cudarc::driver::CudaSlice as ArenaSlice;
use candle_core::{DType, Device, DeviceLocation, Storage, Tensor, Var};

#[cfg(target_family = "unix")]
use crate::paged_attention::plan::DecodePlan;
use crate::{
    flashinfer::{
        make_fa3_decode_state, Fa3DecodeState, FlashInferMetadata, FlashInferPagedAttentionView,
        FlashInferPagedAttentionViews, FlashInferPagedKv, FlashInferTilePlan,
    },
    paged_attention::{
        block_table_rows::BlockTableSnapshot, AttentionBackendKind, ModelConfigLike,
    },
};
#[cfg(all(feature = "cuda", not(feature = "rocm"), target_family = "unix"))]
use mistralrs_paged_attn::fa3_prepare_decode_metadata;
#[cfg(all(feature = "cuda", feature = "rocm", target_family = "unix"))]
use mistralrs_paged_attn::fa3_prepare_decode_metadata_auto as fa3_prepare_decode_metadata;

use crate::device_map::DeviceMapper;
use crate::kv_cache::HybridCache;
use crate::paged_attention::_PAD_SLOT_ID;
use crate::pipeline::{
    decode_positions_tensor,
    text_models_inputs_processor::{
        make_flash_params, DecodePagedRows, DecodePagedRowsGraphKey, FlashParams,
        PagedAttentionInputMetadata, PagedDecodeMetadataRequirements,
    },
    DecodeGraphPrecaptureCtx, RecurrentBatchKind,
};
use crate::speculative::SpeculativeGraphState;

// ---------------------------------------------------------------------------
// Cross-engine CUDA graph capture window
// ---------------------------------------------------------------------------
//
// Stream captures are only valid while no other thread issues
// capture-unsafe device work. Even under THREAD_LOCAL capture isolation,
// cross-thread ops of the sync-alloc/free class (hipFree, hipHostFree,
// hipMallocManaged, and the graph-exec destroys inside a memory reclaim)
// trigger implicit device synchronization and invalidate an in-flight
// capture. Observed on prod: an idle engine's graph reclaim stretched
// another engine's capture from 8s to 118s through retry/fallback with a
// hip 900/901 storm, and a decoding engine beside a capture failed both
// engines. The window serializes every capture against every engine's
// registered device-work segments: captures take the write side, engine
// segments take the read side (engines still decode concurrently with
// each other). parking_lot is write-preferring, so a busy box cannot
// starve a load's captures.
//
// Lazy captures run INSIDE a read segment (a step's forward). To avoid
// self-deadlock, the capture first drops this thread's registered read
// guards, takes the write, and restores them afterwards. Cross-thread
// readers keep their guards, so a capture only ever waits for the
// bounded remainder of another engine's segment, and readers never hold
// the guard across an await by construction (every registration site is
// a synchronous segment).

static CUDA_GRAPH_CAPTURE_WINDOW: parking_lot::RwLock<()> = parking_lot::RwLock::new(());

thread_local! {
    static CUDA_GRAPH_STEP_GUARDS:
        RefCell<Vec<(u64, parking_lot::RwLockReadGuard<'static, ()>)>> =
        const { RefCell::new(Vec::new()) };
    static CUDA_GRAPH_STEP_GUARD_NEXT_ID: Cell<u64> = const { Cell::new(0) };
    static CUDA_GRAPH_CAPTURE_ACTIVE_ON_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// RAII registration of an engine device-work segment on the read side of
/// the cross-engine capture window. Guards never span an `.await`; every
/// registration site is a synchronous segment, so a pending capture waits
/// at most the remainder of the current segment.
pub(crate) struct CudaGraphStepWindow {
    id: u64,
}

#[must_use]
pub(crate) fn enter_cuda_graph_step_window() -> CudaGraphStepWindow {
    let guard = CUDA_GRAPH_CAPTURE_WINDOW.read();
    let id = CUDA_GRAPH_STEP_GUARD_NEXT_ID.with(|next| {
        let id = next.get();
        next.set(id.wrapping_add(1));
        id
    });
    CUDA_GRAPH_STEP_GUARDS.with(|guards| guards.borrow_mut().push((id, guard)));
    CudaGraphStepWindow { id }
}

impl Drop for CudaGraphStepWindow {
    fn drop(&mut self) {
        CUDA_GRAPH_STEP_GUARDS.with(|guards| {
            guards.borrow_mut().retain(|(id, _)| *id != self.id);
        });
    }
}

/// The write side of the capture window, taken by every capture (both
/// pipelines, load-time precapture and lazy). Dropping it releases the
/// write and re-registers the read guards this thread held before the
/// capture (the lazy-capture dance).
pub(crate) struct CudaGraphCaptureWindow {
    // Ids of this thread's step segments that were active when the capture
    // started. Their read guards are DROPPED for the capture's duration
    // (a capture inside a step must not wait on its own thread's reads)
    // and fresh reads are re-acquired for the same ids afterwards so the
    // enclosing segments stay covered. Step segments never nest, so this
    // holds at most one id in practice.
    saved_ids: Vec<u64>,
    write: Option<parking_lot::RwLockWriteGuard<'static, ()>>,
}

pub(crate) fn enter_cuda_graph_capture_window() -> CudaGraphCaptureWindow {
    debug_assert!(
        !CUDA_GRAPH_CAPTURE_ACTIVE_ON_THREAD.with(|active| active.get()),
        "same-thread nested graph captures are not supported"
    );
    CUDA_GRAPH_CAPTURE_ACTIVE_ON_THREAD.with(|active| active.set(true));
    // Lazy-capture dance: this thread is inside a read segment (a step's
    // forward). DROP its registered guards - remembering only their ids -
    // so taking the write cannot self-deadlock. (Relocating the guards
    // would keep the read alive and deadlock the write on ourselves, which
    // is exactly what wedged the first lazy capture in testing.) Cross-
    // thread readers are unaffected; they bound the write's wait.
    let saved_ids = CUDA_GRAPH_STEP_GUARDS.with(|guards| {
        guards
            .borrow_mut()
            .drain(..)
            .map(|(id, _guard)| id)
            .collect::<Vec<u64>>()
    });
    let write = match CUDA_GRAPH_CAPTURE_WINDOW.try_write() {
        Some(write) => write,
        None => {
            // Another engine's segment is still draining; this is the
            // contention the window exists to serialize, so make it visible.
            metrics::counter!("mistralrs_cuda_graph_capture_window_waits_total").increment(1);
            CUDA_GRAPH_CAPTURE_WINDOW.write()
        }
    };
    CudaGraphCaptureWindow {
        saved_ids,
        write: Some(write),
    }
}

impl Drop for CudaGraphCaptureWindow {
    fn drop(&mut self) {
        // Release the write before re-registering this thread's read
        // guards so the re-acquisition cannot queue behind our own write.
        self.write = None;
        if !self.saved_ids.is_empty() {
            let saved_ids = std::mem::take(&mut self.saved_ids);
            CUDA_GRAPH_STEP_GUARDS.with(|guards| {
                let mut guards_mut = guards.borrow_mut();
                for id in saved_ids {
                    guards_mut.push((id, CUDA_GRAPH_CAPTURE_WINDOW.read()));
                }
            });
        }
        CUDA_GRAPH_CAPTURE_ACTIVE_ON_THREAD.with(|active| active.set(false));
    }
}

fn dbg_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// AUTO_FREE_ON_LAUNCH=1 is REQUIRED: the captured graphs contain memfree
// nodes (Rust temporaries free_async during capture), and persisting those
// allocations across launches segfaults the driver at launch (verified:
// flags=0 with MMVQ-era graphs = SIGSEGV in libcuda; flags=1 = the
// documented pairing for graphs with free nodes).
const CUDA_GRAPH_INSTANTIATE_FLAGS: u64 = 1;
// Matches the standard CUDA paged-attention V2 partition size.
const PAGED_ATTENTION_PARTITION_SIZE: usize = 512;
const TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY: usize = 64;
const TARGET_CUDA_DECODE_GRAPH_CACHE_MAX_CAPACITY: usize = 96;
const TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM: usize = 16;
const CUDA_GRAPH_BATCH_BUCKET_GRANULARITY: usize = 8;
const CUDA_GRAPH_COARSE_EXACT_BATCH_BUCKETS: usize = 8;
const TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS: usize = 16;
pub(crate) const CUDA_GRAPH_MAX_BATCH_BUCKET: usize = 128;
pub(crate) const CUDA_GRAPH_PRECAPTURE_MAX_BATCH: usize = CUDA_GRAPH_MAX_BATCH_BUCKET;
const CUDA_GRAPH_SPEC_STATE_BUDGET_FLOOR_PERCENT: usize = 4;
const CUDA_GRAPH_SPEC_STATE_BUDGET_CEILING_PERCENT: usize = 8;
// C128 buckets plus one extra max-batch context graph require just under ten largest entries.
const CUDA_GRAPH_SPEC_STATE_WORKING_SET_MULTIPLIER: usize = 10;
const CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES_ENV: &str = "MISTRALRS_CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES";
const CUDA_PHASE_TIMINGS_ENV: &str = "MISTRALRS_CUDA_PHASE_TIMINGS";
const CUDA_GRAPH_EVENTS_METRIC: &str = "mistralrs_cuda_graph_events_total";
const CUDA_GRAPH_DISPATCH_METRIC: &str = "mistralrs_cuda_graph_dispatch_total";
const CUDA_GRAPH_EVICTIONS_METRIC: &str = "mistralrs_cuda_graph_evictions_total";
const CUDA_GRAPH_RESIDENT_ENTRIES_METRIC: &str = "mistralrs_cuda_graph_resident_entries";
// A few capture-time allocations (metadata graph vars) the warmup did not make.
const DECODE_GRAPH_MIN_ARENA_BYTES: usize = 32 * 1024 * 1024;

/// Capture attempts per graph (1 initial + growth retries). Each overflow
/// doubles the arena, so the budget bounds growth at 8x the first estimate.
/// Six attempts (32x) because the cuda-role capture forward needs ~200MB+
/// against a ~30MB warmup-seeded start: the bump allocator never reuses
/// freed transients, so 62 layers of twin workspaces accumulate (measured:
/// bucket-A converges at 204MB on attempt 3). Only triggers on genuine
/// overflow; converged roles never notice the extra budget.
const DECODE_GRAPH_CAPTURE_ATTEMPTS: usize = 6;

/// Precapture stops taking new buckets once free device memory (as the
/// device-memory query reports it) falls below this floor. Resident graphs
/// pin their arenas for the process lifetime, so an unbounded precapture
/// can walk the device straight into a fatal out-of-memory inside a kernel
/// launch; the floor keeps the remaining shapes on the lazy path instead.
pub(crate) const DECODE_GRAPH_PRECAPTURE_MIN_FREE_BYTES: usize = 16 * 1024 * 1024 * 1024;
static NEXT_CUDA_DECODE_GRAPH_GENERATION: AtomicU64 = AtomicU64::new(1);
static CUDA_GRAPH_MEMORY_POOL_SCOPES: OnceLock<Mutex<HashMap<usize, MemoryPoolScopeState>>> =
    OnceLock::new();
static CUDA_PHASE_TIMINGS_ENABLED: OnceLock<bool> = OnceLock::new();

pub(crate) struct CudaPhaseTimer {
    start: GraphEvent,
    stream: GraphStream,
}

impl CudaPhaseTimer {
    pub(crate) fn start(stream: &GraphStream) -> candle_core::Result<Option<Self>> {
        let enabled = *CUDA_PHASE_TIMINGS_ENABLED.get_or_init(|| {
            std::env::var(CUDA_PHASE_TIMINGS_ENV)
                .ok()
                .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        });
        if !enabled {
            return Ok(None);
        }
        if !stream.capture_status_none()? {
            return Ok(None);
        }
        let start = stream.record_timing_event()?;
        Ok(Some(Self {
            start,
            stream: stream.clone(),
        }))
    }

    pub(crate) fn finish(
        self,
        component: &'static str,
        batch: usize,
        rows: usize,
    ) -> candle_core::Result<()> {
        let end = self.stream.record_timing_event()?;
        end.synchronize().map_err(candle_core::Error::wrap)?;
        let latency_ms = self
            .start
            .elapsed_ms(&end)
            .map_err(candle_core::Error::wrap)?;
        tracing::info!(component, batch, rows, latency_ms, "CUDA phase timing");
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphComponent {
    Target,
    #[cfg(any(test, all(feature = "flash-attn", target_family = "unix")))]
    DFlash,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphEvictionReason {
    Capacity,
    MemoryPressure,
    SpecStateBudget,
}

impl CudaGraphEvictionReason {
    const fn label(self) -> &'static str {
        match self {
            Self::Capacity => "capacity",
            Self::MemoryPressure => "memory_pressure",
            Self::SpecStateBudget => "spec_state_budget",
        }
    }
}

impl CudaGraphComponent {
    const fn label(self) -> &'static str {
        match self {
            Self::Target => "target",
            #[cfg(any(test, all(feature = "flash-attn", target_family = "unix")))]
            Self::DFlash => "dflash",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphEvent {
    Capture,
    Replay,
    EagerFallback,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphDispatchMode {
    Replay,
    Eager,
    Skipped,
}

impl CudaGraphDispatchMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Replay => "replay",
            Self::Eager => "eager",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaGraphDispatchReason {
    CacheHit,
    Disabled,
    ModelUnsupported,
    SpeculativeConflict,
    PagedAttentionUnavailable,
    Prefill,
    IncompatibleShape,
    BatchUnsupported,
    CacheConfigUnavailable,
    RuntimeDisabled,
    PaddingUnavailable,
    CachePopulation,
    Fallback,
}

impl CudaGraphDispatchReason {
    const fn label(self) -> &'static str {
        match self {
            Self::CacheHit => "cache_hit",
            Self::Disabled => "disabled",
            Self::ModelUnsupported => "model_unsupported",
            Self::SpeculativeConflict => "speculative_conflict",
            Self::PagedAttentionUnavailable => "paged_attention_unavailable",
            Self::Prefill => "prefill",
            Self::IncompatibleShape => "incompatible_shape",
            Self::BatchUnsupported => "batch_unsupported",
            Self::CacheConfigUnavailable => "cache_config_unavailable",
            Self::RuntimeDisabled => "runtime_disabled",
            Self::PaddingUnavailable => "padding_unavailable",
            Self::CachePopulation => "cache_population",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CudaGraphDispatchLabels {
    component: &'static str,
    mode: &'static str,
    reason: &'static str,
}

const fn cuda_graph_dispatch_labels(
    component: CudaGraphComponent,
    mode: CudaGraphDispatchMode,
    reason: CudaGraphDispatchReason,
) -> CudaGraphDispatchLabels {
    CudaGraphDispatchLabels {
        component: component.label(),
        mode: mode.label(),
        reason: reason.label(),
    }
}

pub(crate) fn record_cuda_graph_dispatch(
    component: CudaGraphComponent,
    mode: CudaGraphDispatchMode,
    reason: CudaGraphDispatchReason,
) {
    let labels = cuda_graph_dispatch_labels(component, mode, reason);
    metrics::counter!(
        CUDA_GRAPH_DISPATCH_METRIC,
        "component" => labels.component,
        "mode" => labels.mode,
        "reason" => labels.reason,
    )
    .increment(1);
}

impl CudaGraphEvent {
    const fn label(self) -> &'static str {
        match self {
            Self::Capture => "capture",
            Self::Replay => "replay",
            Self::EagerFallback => "eager_fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CudaGraphOutcome {
    Success,
    Failure,
}

impl CudaGraphOutcome {
    const fn label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CudaGraphEventLabels {
    component: &'static str,
    event: &'static str,
    outcome: &'static str,
}

const fn cuda_graph_event_labels(
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
) -> CudaGraphEventLabels {
    CudaGraphEventLabels {
        component: component.label(),
        event: event.label(),
        outcome: outcome.label(),
    }
}

fn record_cuda_graph_event(
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
) {
    let labels = cuda_graph_event_labels(component, event, outcome);
    metrics::counter!(
        CUDA_GRAPH_EVENTS_METRIC,
        "component" => labels.component,
        "event" => labels.event,
        "outcome" => labels.outcome,
    )
    .increment(1);
}

pub(crate) fn record_cuda_graph_evictions(
    component: CudaGraphComponent,
    reason: CudaGraphEvictionReason,
    count: usize,
) {
    metrics::counter!(
        CUDA_GRAPH_EVICTIONS_METRIC,
        "component" => component.label(),
        "reason" => reason.label()
    )
    .increment(u64::try_from(count).unwrap_or(u64::MAX));
}

pub(crate) fn record_cuda_graph_resident_entries(component: CudaGraphComponent, count: usize) {
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    metrics::gauge!(
        CUDA_GRAPH_RESIDENT_ENTRIES_METRIC,
        "component" => component.label()
    )
    .set(f64::from(count));
}

pub(crate) fn take_cuda_graph_capacity_eviction<T>(
    entries: &mut Vec<T>,
    capacity: usize,
) -> Option<T> {
    assert!(capacity > 0, "CUDA graph cache capacity must be nonzero");
    (entries.len() >= capacity).then(|| entries.remove(0))
}

pub(crate) fn reclaim_cuda_graph_entries(
    max_entries: usize,
    reclaim_target: impl FnOnce(usize) -> usize,
    reclaim_speculative: impl FnOnce(usize) -> usize,
) -> usize {
    if max_entries == 0 {
        return 0;
    }
    // Cross-engine capture window: reclaims destroy graph execs and free
    // pinned/managed staging - the sync-free class whose implicit device
    // synchronization invalidates another engine's in-flight capture.
    let _step_window = enter_cuda_graph_step_window();
    let target = reclaim_target(max_entries);
    debug_assert!(target <= max_entries);
    let remaining = max_entries - target;
    if remaining == 0 {
        return target;
    }
    let speculative = reclaim_speculative(remaining);
    debug_assert!(speculative <= remaining);
    target + speculative
}

#[must_use]
pub(crate) struct CudaGraphEventGuard {
    component: CudaGraphComponent,
    event: CudaGraphEvent,
    outcome: CudaGraphOutcome,
}

impl CudaGraphEventGuard {
    pub(crate) const fn new(component: CudaGraphComponent, event: CudaGraphEvent) -> Self {
        Self {
            component,
            event,
            outcome: CudaGraphOutcome::Failure,
        }
    }

    pub(crate) fn success(mut self) {
        self.outcome = CudaGraphOutcome::Success;
    }
}

impl Drop for CudaGraphEventGuard {
    fn drop(&mut self) {
        record_cuda_graph_event(self.component, self.event, self.outcome);
        if self.outcome == CudaGraphOutcome::Success && self.event == CudaGraphEvent::EagerFallback
        {
            record_cuda_graph_dispatch(
                self.component,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::Fallback,
            );
        }
    }
}

struct MemoryPoolScopeState {
    guards: usize,
    release_threshold: u64,
}

const fn cuda_graph_exact_batch_buckets(component: CudaGraphComponent, q_len: usize) -> usize {
    if matches!(component, CudaGraphComponent::Target) && q_len == 1 {
        TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS
    } else {
        CUDA_GRAPH_COARSE_EXACT_BATCH_BUCKETS
    }
}

/// Graph batch bucket a decode batch pads up to, or None when it is too large to graph.
pub(crate) fn cuda_graph_batch_bucket(
    component: CudaGraphComponent,
    q_len: usize,
    batch: usize,
) -> Option<usize> {
    if batch == 0 {
        None
    } else if batch <= cuda_graph_exact_batch_buckets(component, q_len) {
        Some(batch)
    } else {
        let bucket = batch
            .div_ceil(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY)
            .saturating_mul(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY);
        (bucket <= CUDA_GRAPH_MAX_BATCH_BUCKET).then_some(bucket)
    }
}

pub(crate) fn cuda_graph_precapture_max_batch(
    component: CudaGraphComponent,
    q_len: usize,
    configured_max: usize,
) -> usize {
    if configured_max == 0 {
        0
    } else {
        cuda_graph_batch_bucket(component, q_len, configured_max)
            .unwrap_or(CUDA_GRAPH_MAX_BATCH_BUCKET)
    }
}

pub(crate) fn target_cuda_graph_cache_capacity(
    startup_shapes: usize,
    runtime_batch_shapes: usize,
) -> usize {
    let working_set = startup_shapes.saturating_add(runtime_batch_shapes);
    let rounded = working_set
        .div_ceil(TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM)
        .saturating_mul(TARGET_CUDA_DECODE_GRAPH_CACHE_CAPACITY_QUANTUM);
    rounded.clamp(
        TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY,
        TARGET_CUDA_DECODE_GRAPH_CACHE_MAX_CAPACITY,
    )
}

/// The batch buckets captured ahead of time at load.
pub(crate) fn cuda_graph_precapture_batches(
    component: CudaGraphComponent,
    q_len: usize,
) -> impl Iterator<Item = usize> {
    let exact = cuda_graph_exact_batch_buckets(component, q_len);
    (1..=exact).chain(
        (exact + CUDA_GRAPH_BATCH_BUCKET_GRANULARITY..=CUDA_GRAPH_PRECAPTURE_MAX_BATCH)
            .step_by(CUDA_GRAPH_BATCH_BUCKET_GRANULARITY),
    )
}

pub(crate) fn cuda_graph_startup_capture_allowed(q_len: usize) -> bool {
    q_len > 0
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn prepare_fa3_decode_schedules(
    metadata: &PagedAttentionInputMetadata,
) -> candle_core::Result<()> {
    let Some(flashinfer) = metadata.flashinfer.as_ref() else {
        return Ok(());
    };
    flashinfer.for_each_fa3_decode_schedule(|prepare| {
        fa3_prepare_decode_metadata(
            mistralrs_paged_attn::Fa3DecodeMetadata {
                paged_kv_indptr: prepare.paged_kv_indptr,
                paged_kv_indices: prepare.paged_kv_indices,
                paged_kv_last_page_len: prepare.paged_kv_last_page_len,
                page_table: &prepare.buffers.page_table,
                seqused_k: &prepare.buffers.seqused_k,
                cu_seqlens_q: &prepare.buffers.cu_seqlens_q,
                scheduler_metadata: &prepare.buffers.scheduler_metadata,
            },
            prepare.buffers.schedule(prepare.key)?,
        )
    })
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
pub(crate) fn prepare_fa3_decode_schedules(
    _metadata: &PagedAttentionInputMetadata,
) -> candle_core::Result<()> {
    Ok(())
}

/// One decode step, padded up to its graph batch bucket. Pad rows alias row 0 for reads and skip
/// their KV and recurrent writes, so the model can run them and drop their outputs.
#[derive(Clone)]
pub(crate) struct CudaGraphDecodeStep {
    pub(crate) input_ids: Tensor,
    pub(crate) seqlen_offsets: Vec<usize>,
    pub(crate) context_lens: Vec<(usize, usize)>,
    pub(crate) position_ids: Vec<usize>,
    pub(crate) metadata: PagedAttentionInputMetadata,
    pub(crate) state_indices: Option<Vec<u32>>,
    pub(crate) real_batch: usize,
}

pub(crate) struct CudaGraphDecodeStepInputs<'a> {
    pub(crate) input_ids: &'a Tensor,
    pub(crate) seqlen_offsets: &'a [usize],
    pub(crate) context_lens: &'a [(usize, usize)],
    pub(crate) position_ids: &'a [usize],
    pub(crate) metadata: &'a PagedAttentionInputMetadata,
    pub(crate) state_indices: Option<&'a [u32]>,
    pub(crate) pad_slot: Option<u32>,
}

impl CudaGraphDecodeStep {
    /// Returns None when the step can't be padded (no host rows to rebuild the metadata from, or a
    /// hybrid batch without a pad slot).
    pub(crate) fn padded(
        inputs: CudaGraphDecodeStepInputs<'_>,
        batch: usize,
    ) -> candle_core::Result<Option<Self>> {
        let CudaGraphDecodeStepInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices,
            pad_slot,
        } = inputs;
        let real_batch = input_ids.dim(0)?;
        if real_batch == batch {
            return Ok(Some(Self {
                input_ids: input_ids.clone(),
                seqlen_offsets: seqlen_offsets.to_vec(),
                context_lens: context_lens.to_vec(),
                position_ids: position_ids.to_vec(),
                metadata: metadata.clone(),
                state_indices: state_indices.map(<[u32]>::to_vec),
                real_batch,
            }));
        }
        let Some(rows) = metadata.decode_rows.as_ref() else {
            return Ok(None);
        };
        let state_indices = match (state_indices, pad_slot) {
            (Some(slots), Some(pad_slot)) => {
                let mut padded = slots.to_vec();
                padded.resize(batch, pad_slot);
                Some(padded)
            }
            (Some(_), None) => return Ok(None),
            (None, _) => None,
        };
        let pad = batch - real_batch;
        let (_, q_len) = input_ids.dims2()?;
        let input_ids = if input_ids.dtype() == DType::U32 && input_ids.device().is_cuda() {
            crate::cuda::input_packing::pad_decode_input(input_ids, batch)?
        } else {
            let pad_ids = input_ids.narrow(0, 0, 1)?.repeat((pad, 1))?;
            Tensor::cat(&[input_ids, &pad_ids], 0)?
        };
        let mut seqlen_offsets = seqlen_offsets.to_vec();
        seqlen_offsets.resize(batch, seqlen_offsets[0]);
        let mut context_lens = context_lens.to_vec();
        context_lens.resize(batch, context_lens[0]);
        let mut position_ids = position_ids.to_vec();
        position_ids.resize(batch, position_ids[0]);
        let rows = Arc::new(rows.padded(batch));
        if rows.query_len != q_len {
            candle_core::bail!(
                "CUDA graph decode rows cover {} query tokens but the input has {q_len}",
                rows.query_len
            );
        }
        let metadata = rows.build().map_err(candle_core::Error::msg)?;
        Ok(Some(Self {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices,
            real_batch,
        }))
    }

    pub(crate) fn batch(&self) -> usize {
        self.seqlen_offsets.len()
    }

    /// Drops the pad rows from a `[batch, ...]` or `[batch * q, ...]` output.
    pub(crate) fn narrow_rows(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        let batch = self.batch();
        if batch == self.real_batch {
            return Ok(tensor.clone());
        }
        let rows = tensor.dim(0)? / batch * self.real_batch;
        tensor.narrow(0, 0, rows)
    }

    fn one_token_continuation(&self, input_ids: Tensor) -> candle_core::Result<Option<Self>> {
        let (batch, q_len) = input_ids.dims2()?;
        let Some(rows) = self.metadata.decode_rows.as_ref() else {
            return Ok(None);
        };
        if q_len != 1
            || rows.query_len != 1
            || rows.decode_window != 1
            || batch != self.batch()
            || rows.batch_size() != batch
            || self.real_batch == 0
            || self.real_batch > batch
            || self.seqlen_offsets.len() != batch
            || self.context_lens.len() != batch
            || self.position_ids.len() != batch
        {
            return Ok(None);
        }

        let mut slot_mappings = Vec::with_capacity(self.real_batch);
        let mut block_tables = Vec::with_capacity(self.real_batch);
        let mut context_lens = Vec::with_capacity(self.real_batch);
        let mut full_context_lens = Vec::with_capacity(self.real_batch);
        for row in 0..self.real_batch {
            let Some(&current_slot) = rows.slot_mappings[row].first() else {
                return Ok(None);
            };
            let Ok(current_slot) = usize::try_from(current_slot) else {
                return Ok(None);
            };
            let full_table = rows.full_block_table(row);
            let current_block = current_slot / rows.block_size;
            let Some(current_block_idx) =
                full_table.iter().position(|&block| block == current_block)
            else {
                return Ok(None);
            };
            let current_block_offset = current_slot % rows.block_size;
            let next_slot = if current_block_offset + 1 < rows.block_size {
                current_slot + 1
            } else {
                let next_block_idx = current_block_idx + 1;
                let Some(&next_block) = full_table.get(next_block_idx) else {
                    return Ok(None);
                };
                let Some(next_slot) = next_block.checked_mul(rows.block_size) else {
                    return Ok(None);
                };
                next_slot
            };
            let Ok(next_slot) = i64::try_from(next_slot) else {
                return Ok(None);
            };
            let Some(next_full_context_len) = rows.full_context_lens[row].checked_add(1) else {
                return Ok(None);
            };

            let paged_context_len = match rows.sliding_window {
                Some(window) => {
                    let window_start = next_full_context_len.saturating_sub(window);
                    let block_aligned_start = window_start / rows.block_size * rows.block_size;
                    next_full_context_len - block_aligned_start
                }
                None => next_full_context_len,
            };
            slot_mappings.push(vec![next_slot]);
            block_tables.push(
                rows.block_tables
                    .table_arc(rows.block_tables.row_table_index(row)),
            );
            context_lens.push(paged_context_len);
            full_context_lens.push(next_full_context_len);
        }

        let rows = Arc::new(
            DecodePagedRows {
                slot_mappings,
                block_tables: BlockTableSnapshot::from_sequence_tables(block_tables, 1),
                context_lens,
                full_context_lens,
                query_len: 1,
                block_size: rows.block_size,
                use_standard_metadata: rows.use_standard_metadata,
                max_paged_context_len: rows.max_paged_context_len,
                sliding_window: rows.sliding_window,
                decode_window: rows.decode_window,
                devices: rows.devices.clone(),
                num_kv_heads: rows.num_kv_heads,
            }
            .padded(batch),
        );
        let metadata = rows.build_graph_staged().map_err(candle_core::Error::msg)?;
        let Some(mut seqlen_offsets) = self.seqlen_offsets[..self.real_batch]
            .iter()
            .map(|offset| offset.checked_add(1))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        seqlen_offsets.resize(batch, seqlen_offsets[0]);
        let mut context_lens = self.context_lens[..self.real_batch].to_vec();
        context_lens.resize(batch, context_lens[0]);
        let Some(mut position_ids) = self.position_ids[..self.real_batch]
            .iter()
            .map(|position| position.checked_add(1))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        position_ids.resize(batch, position_ids[0]);
        Ok(Some(Self {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            state_indices: self.state_indices.clone(),
            real_batch: self.real_batch,
        }))
    }
}

/// A fabricated batch-1 decode step (token 0 at position 0 over one block, no KV writes) that the
/// precapture pads up to every bucket.
pub(crate) struct CudaGraphPrecaptureInputs {
    pub(crate) input_ids: Tensor,
    pub(crate) seqlen_offsets: Vec<usize>,
    pub(crate) context_lens: Vec<(usize, usize)>,
    pub(crate) position_ids: Vec<usize>,
    pub(crate) metadata: PagedAttentionInputMetadata,
    pub(crate) flash_meta: FlashParams,
}

impl CudaGraphPrecaptureInputs {
    pub(crate) fn new(
        ctx: &DecodeGraphPrecaptureCtx,
        q_len: usize,
        device: &Device,
        mapper: Option<&dyn DeviceMapper>,
    ) -> candle_core::Result<Self> {
        let devices = mapper
            .map(|mapper| mapper.get_unique_devices())
            .unwrap_or_else(|| vec![device.clone()]);
        let rows = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![_PAD_SLOT_ID; q_len]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![0]], q_len),
            context_lens: vec![1; q_len],
            full_context_lens: vec![1; q_len],
            query_len: q_len,
            block_size: ctx.block_size,
            use_standard_metadata: ctx.attention_backend == AttentionBackendKind::Standard,
            max_paged_context_len: ctx.max_paged_context_len,
            sliding_window: ctx.sliding_window,
            decode_window: 1,
            devices,
            num_kv_heads: ctx.num_kv_heads,
        });
        let metadata = rows.build_materialized().map_err(candle_core::Error::msg)?;
        let q_len_u32 = u32::try_from(q_len).map_err(candle_core::Error::wrap)?;
        let flash_meta = if crate::using_flash_attn() {
            make_flash_params(
                device,
                mapper,
                &[0, q_len_u32],
                &[0, q_len_u32],
                ctx.sliding_window,
                true,
                false,
            )
            .map_err(candle_core::Error::msg)?
        } else {
            FlashParams::empty(true)
        };
        Ok(Self {
            input_ids: Tensor::zeros((1, q_len), DType::U32, device)?,
            seqlen_offsets: vec![0],
            context_lens: vec![(0, q_len)],
            position_ids: vec![q_len],
            metadata,
            flash_meta,
        })
    }

    pub(crate) fn step_inputs<'a>(
        &'a self,
        state_indices: Option<&'a [u32]>,
        pad_slot: Option<u32>,
    ) -> CudaGraphDecodeStepInputs<'a> {
        CudaGraphDecodeStepInputs {
            input_ids: &self.input_ids,
            seqlen_offsets: &self.seqlen_offsets,
            context_lens: &self.context_lens,
            position_ids: &self.position_ids,
            metadata: &self.metadata,
            state_indices,
            pad_slot,
        }
    }
}

pub(crate) struct HybridGraphSlots {
    pub(crate) real: Vec<u32>,
    pub(crate) storage_generation: u64,
    pub(crate) pad_row: Option<u32>,
}

/// The batch's live recurrent slots after reserving graph capacity.
pub(crate) fn hybrid_graph_slots(
    cache: &mut HybridCache,
) -> candle_core::Result<Option<HybridGraphSlots>> {
    let Some(real) = cache.state_indices_host().map(<[u32]>::to_vec) else {
        return Ok(None);
    };
    let pad_row = cache
        .graph_pad_slot()?
        .and_then(|row| u32::try_from(row).ok());
    Ok(Some(HybridGraphSlots {
        real,
        storage_generation: cache.recurrent_storage_generation(),
        pad_row,
    }))
}

/// Points the hybrid cache's state indices at fresh `Var` buffers holding `host`, one per recurrent
/// device, so a captured forward reads slots the replay can overwrite.
pub(crate) fn install_hybrid_graph_state_indices(
    cache: &mut HybridCache,
    host: &[u32],
) -> candle_core::Result<CudaGraphVarMap> {
    let mut vars = CudaGraphVarMap::new();
    let mut tensors = Vec::new();
    for device in cache.recurrent_devices() {
        let var = Var::from_tensor(&Tensor::from_vec(host.to_vec(), (host.len(),), &device)?)?;
        tensors.push((device.clone(), var.as_detached_tensor()));
        vars.insert(device.location(), var);
    }
    cache.set_state_indices_tensors(host.to_vec(), tensors);
    Ok(vars)
}

fn copy_state_indices(
    dst: &CudaGraphVarMap,
    host: &[u32],
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    for (location, var) in dst {
        host_staging.copy_from_u32_slice("state_indices", *location, host, var)?;
    }
    Ok(())
}

// Pointer-correlation census for the NVIDIA-role graphs crash (all opt-in
// via MISTRALRS_GRAPH_PTR_CENSUS, zero behavior change when unset): record
// every graph-managed buffer at capture as (name, device, ptr, bytes), and
// re-verify entry-held buffers before first launch (a moved VA bails loudly
// naming the buffer instead of segfaulting). Driver node inventory segfaults
// on this driver, so the correlation stays engine-level; the SIGSEGV handler
// below names the census allocation containing any fault VA, then re-raises.

/// One census row: the capture-time VA range of a graph-managed buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphPtrRecord {
    pub(crate) name: &'static str,
    pub(crate) device: DeviceLocation,
    pub(crate) ptr: u64,
    pub(crate) bytes: u64,
}

pub(crate) fn graph_ptr_census_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MISTRALRS_GRAPH_PTR_CENSUS").is_ok())
}

/// Best-effort device VA of a tensor's first byte (None for CPU, empty, or
/// non-tracked dtypes). Dual-aware: the CUDA arm serves the NVIDIA role,
/// the hip arm the AMD role. Read-only: only the VA is observed.
pub(crate) fn graph_tensor_va(t: &Tensor) -> candle_core::Result<Option<(u64, u64)>> {
    use candle_core::{DType, Device, Storage};
    let bytes = t.elem_count() * t.dtype().size_in_bytes();
    if bytes == 0 {
        return Ok(None);
    }
    let (storage, layout) = t.storage_and_layout();
    let start = layout.start_offset() as u64;
    match (&*storage, t.device()) {
        #[cfg(feature = "cuda")]
        (Storage::Cuda(s), Device::Cuda(d)) => {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            let stream = d.cuda_stream();
            macro_rules! va_of {
                ($ty:ty) => {{
                    let (ptr, _guard) = s.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                    ptr + start
                }};
            }
            let base = match t.dtype() {
                DType::U8 => va_of!(u8),
                DType::U32 => va_of!(u32),
                DType::I64 => va_of!(i64),
                DType::F16 => va_of!(half::f16),
                DType::BF16 => va_of!(half::bf16),
                DType::F32 => va_of!(f32),
                _ => return Ok(None),
            };
            Ok(Some((base, bytes as u64)))
        }
        #[cfg(all(feature = "cuda", feature = "rocm"))]
        (Storage::Hip(s), Device::Hip(d)) => {
            use candle_core::hip_backend::cudarc::driver::DevicePtr;
            let stream = d.cuda_stream();
            macro_rules! va_of {
                ($ty:ty) => {{
                    let (ptr, _guard) = s.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                    ptr + start
                }};
            }
            let base = match t.dtype() {
                DType::U8 => va_of!(u8),
                DType::U32 => va_of!(u32),
                DType::I64 => va_of!(i64),
                DType::F16 => va_of!(half::f16),
                DType::BF16 => va_of!(half::bf16),
                DType::F32 => va_of!(f32),
                _ => return Ok(None),
            };
            Ok(Some((base, bytes as u64)))
        }
        _ => Ok(None),
    }
}

const GRAPH_PTR_CENSUS_CAP: usize = 4096;

struct GraphPtrCensusSlot {
    ptr: std::sync::atomic::AtomicU64,
    bytes: std::sync::atomic::AtomicU64,
    // (&'static str addr, len) as u64s: read-only static memory, safe to
    // dereference in the signal handler. The ptr store lands last, so the
    // handler skips half-written slots (ptr == 0).
    name_ptr: std::sync::atomic::AtomicU64,
    name_len: std::sync::atomic::AtomicU64,
}

impl GraphPtrCensusSlot {
    const fn empty() -> Self {
        Self {
            ptr: std::sync::atomic::AtomicU64::new(0),
            bytes: std::sync::atomic::AtomicU64::new(0),
            name_ptr: std::sync::atomic::AtomicU64::new(0),
            name_len: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

static GRAPH_PTR_CENSUS: std::sync::OnceLock<[GraphPtrCensusSlot; GRAPH_PTR_CENSUS_CAP]> =
    std::sync::OnceLock::new();
static GRAPH_PTR_CENSUS_LEN: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Record one buffer in the process-wide census (env-gated; also installs
/// the segfault auto-namer on first use).
pub(crate) fn graph_ptr_census_record(
    name: &'static str,
    device: DeviceLocation,
    ptr: u64,
    bytes: u64,
) {
    use std::sync::atomic::Ordering;
    if !graph_ptr_census_enabled() {
        return;
    }
    install_graph_ptr_census_handler();
    let slots =
        GRAPH_PTR_CENSUS.get_or_init(|| std::array::from_fn(|_| GraphPtrCensusSlot::empty()));
    let idx = GRAPH_PTR_CENSUS_LEN.fetch_add(1, Ordering::Relaxed);
    if idx >= GRAPH_PTR_CENSUS_CAP {
        return;
    }
    slots[idx]
        .name_ptr
        .store(name.as_ptr() as u64, Ordering::Relaxed);
    slots[idx]
        .name_len
        .store(name.len() as u64, Ordering::Relaxed);
    slots[idx].bytes.store(bytes, Ordering::Relaxed);
    slots[idx].ptr.store(ptr, Ordering::Relaxed);
    eprintln!("[graph-ptr-census] {name} {device:?}: ptr=0x{ptr:x} bytes={bytes}");
}

/// Record every tensor of a var map (`None` maps are skipped).
pub(crate) fn graph_ptr_census_record_var_map(name: &'static str, map: Option<&CudaGraphVarMap>) {
    let Some(map) = map else { return };
    for (location, var) in map {
        match graph_tensor_va(var.as_tensor()) {
            Ok(Some((ptr, bytes))) => graph_ptr_census_record(name, *location, ptr, bytes),
            Ok(None) => eprintln!("[graph-ptr-census] {name} {location:?}: untracked storage"),
            Err(err) => eprintln!("[graph-ptr-census] {name} {location:?}: VA read failed: {err}"),
        }
    }
}

/// Record every tensor of a per-device tensor map.
pub(crate) fn graph_ptr_census_record_tensor_map(
    name: &'static str,
    map: Option<&HashMap<DeviceLocation, Tensor>>,
) {
    let Some(map) = map else { return };
    for (location, tensor) in map {
        match graph_tensor_va(tensor) {
            Ok(Some((ptr, bytes))) => graph_ptr_census_record(name, *location, ptr, bytes),
            Ok(None) => eprintln!("[graph-ptr-census] {name} {location:?}: untracked storage"),
            Err(err) => eprintln!("[graph-ptr-census] {name} {location:?}: VA read failed: {err}"),
        }
    }
}

/// Re-verify census rows against live tensors: the pre-first-launch sweep.
/// A moved VA or a resized buffer is a loud bail naming the buffer (the
/// segfault it would have caused), so the entry is dropped and the step
/// falls back to eager.
pub(crate) fn graph_ptr_census_verify(
    records: &[GraphPtrRecord],
    live: &[(&'static str, DeviceLocation, &Tensor)],
) -> candle_core::Result<()> {
    for (name, location, tensor) in live {
        let current = graph_tensor_va(tensor)?;
        let expected = records
            .iter()
            .find(|r| r.name == *name && r.device == *location);
        match (expected, current) {
            (Some(exp), Some((now, now_bytes))) if exp.ptr != now || exp.bytes != now_bytes => {
                candle_core::bail!(
                    "graph ptr census: {name} {location:?} changed since capture \
                     (capture=0x{:x}+{} replay=0x{:x}+{}) - stale VA, skipping launch",
                    exp.ptr,
                    exp.bytes,
                    now,
                    now_bytes,
                );
            }
            (Some(_), None) => {
                candle_core::bail!(
                    "graph ptr census: {name} {location:?} left device storage since capture - skipping launch"
                );
            }
            _ => {}
        }
    }
    Ok(())
}

/// Content dump for the correlation: the U32 metadata actually staged for
/// a capture vs a replay (block ids drive every pool-indexing kernel; a
/// wild id here is the smoking gun). Diagnostic gate only; syncs.
pub(crate) fn graph_ptr_census_dump_content(
    tag: &'static str,
    live: &[(&'static str, DeviceLocation, &Tensor)],
) {
    if !graph_ptr_census_enabled() {
        return;
    }
    for (name, location, tensor) in live {
        let content = match tensor.dtype() {
            DType::U32 => tensor
                .flatten_all()
                .and_then(|flat| flat.to_vec1::<u32>())
                .map(|v| {
                    let sum = v.iter().fold(0u64, |acc, x| acc.wrapping_add(*x as u64));
                    (
                        v.len(),
                        sum,
                        format!("{v:?}").chars().take(120).collect::<String>(),
                    )
                }),
            DType::I64 => tensor
                .flatten_all()
                .and_then(|flat| flat.to_vec1::<i64>())
                .map(|v| {
                    let sum = v.iter().fold(0i64, |acc, x| acc.wrapping_add(*x));
                    (
                        v.len(),
                        sum as u64,
                        format!("{v:?}").chars().take(120).collect::<String>(),
                    )
                }),
            _ => continue,
        };
        match content {
            Ok((len, sum, head)) => {
                eprintln!(
                    "[graph-ptr-census] content {tag} {name} {location:?} {:?}: len={len} sum={sum} head={head}",
                    tensor.dtype(),
                );
            }
            Err(err) => {
                eprintln!(
                    "[graph-ptr-census] content {tag} {name} {location:?}: read failed: {err}"
                )
            }
        }
    }
}

/// Collect the sweepable (name, device, tensor) triples held by a decode
/// entry. Single construction site shared by capture (record) and replay
/// (verify) so the two can never drift apart.
pub(crate) fn decode_entry_census_tensors<'a>(
    input_ids: &'a Var,
    metadata_buffers: &'a CudaDecodeGraphMetadataBuffers,
    state_indices: &'a Option<CudaGraphVarMap>,
    logits: &'a Tensor,
) -> Vec<(&'static str, DeviceLocation, &'a Tensor)> {
    fn push_var_map<'a>(
        out: &mut Vec<(&'static str, DeviceLocation, &'a Tensor)>,
        name: &'static str,
        map: Option<&'a CudaGraphVarMap>,
    ) {
        if let Some(map) = map {
            out.extend(
                map.iter()
                    .map(|(location, var)| (name, *location, var.as_tensor())),
            );
        }
    }
    let mut out = Vec::new();
    out.push((
        "input_ids",
        input_ids.as_tensor().device().location(),
        input_ids.as_tensor(),
    ));
    push_var_map(
        &mut out,
        "slot_mappings",
        Some(&metadata_buffers.slot_mappings),
    );
    push_var_map(
        &mut out,
        "block_tables",
        metadata_buffers.block_tables.as_ref(),
    );
    push_var_map(
        &mut out,
        "context_lens",
        metadata_buffers.context_lens.as_ref(),
    );
    push_var_map(
        &mut out,
        "rope_positions",
        Some(&metadata_buffers.rope_positions),
    );
    push_var_map(
        &mut out,
        "paged_kv_indptr",
        metadata_buffers.paged_kv_indptr.as_ref(),
    );
    push_var_map(
        &mut out,
        "paged_kv_indices",
        metadata_buffers.paged_kv_indices.as_ref(),
    );
    push_var_map(
        &mut out,
        "paged_kv_last_page_len",
        metadata_buffers.paged_kv_last_page_len.as_ref(),
    );
    push_var_map(&mut out, "state_indices", state_indices.as_ref());
    out.push(("logits", logits.device().location(), logits));
    out
}

#[cfg(all(any(feature = "cuda", feature = "rocm"), target_family = "unix"))]
static GRAPH_PTR_CENSUS_HANDLER_INSTALLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Install the segfault auto-namer (unix only; elsewhere the census and
/// sweep still work, without the fault-VA report). Idempotent.
pub(crate) fn install_graph_ptr_census_handler() {
    #[cfg(all(any(feature = "cuda", feature = "rocm"), target_family = "unix"))]
    {
        use std::sync::atomic::Ordering;
        if GRAPH_PTR_CENSUS_HANDLER_INSTALLED.swap(true, Ordering::SeqCst) {
            return;
        }
        // SAFETY: sigaction with SA_SIGINFO; the handler only reads the
        // append-only census (atomics + static strings), formats into a
        // stack buffer, write()s, and re-raises.
        unsafe {
            let mut act: libc::sigaction = std::mem::zeroed();
            act.sa_sigaction = census_segfault_handler as usize;
            act.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigemptyset(&mut act.sa_mask);
            libc::sigaction(libc::SIGSEGV, &act, std::ptr::null_mut());
        }
    }
}

/// SIGSEGV handler: name the census allocation containing the fault VA
/// (or the nearest one), then re-raise with the default disposition for
/// the core dump. Async-signal-safe by construction: atomics, static
/// strings, a stack buffer, and raw write() only.
#[cfg(all(any(feature = "cuda", feature = "rocm"), target_family = "unix"))]
extern "C" fn census_segfault_handler(
    _sig: libc::c_int,
    info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    use std::sync::atomic::Ordering;
    let fault = unsafe { (*info).si_addr() as u64 };

    let mut contained_name: u64 = 0;
    let mut contained_len: u64 = 0;
    let mut contained_ptr: u64 = 0;
    let mut contained_bytes: u64 = 0;
    let mut best_below_name: u64 = 0;
    let mut best_below_len: u64 = 0;
    let mut best_below_gap: u64 = u64::MAX;
    let mut best_above_name: u64 = 0;
    let mut best_above_len: u64 = 0;
    let mut best_above_gap: u64 = u64::MAX;
    if let Some(slots) = GRAPH_PTR_CENSUS.get() {
        let len = GRAPH_PTR_CENSUS_LEN
            .load(Ordering::Relaxed)
            .min(GRAPH_PTR_CENSUS_CAP);
        let mut i = 0;
        while i < len {
            let ptr = slots[i].ptr.load(Ordering::Relaxed);
            let bytes = slots[i].bytes.load(Ordering::Relaxed);
            let name_ptr = slots[i].name_ptr.load(Ordering::Relaxed);
            let name_len = slots[i].name_len.load(Ordering::Relaxed);
            if ptr != 0 && bytes != 0 && name_ptr != 0 {
                if fault >= ptr && fault < ptr + bytes {
                    contained_name = name_ptr;
                    contained_len = name_len;
                    contained_ptr = ptr;
                    contained_bytes = bytes;
                    break;
                } else if fault < ptr {
                    let gap = ptr - fault;
                    if gap < best_above_gap {
                        best_above_gap = gap;
                        best_above_name = name_ptr;
                        best_above_len = name_len;
                    }
                } else {
                    let end = ptr + bytes;
                    if fault >= end {
                        let gap = fault - end;
                        if gap < best_below_gap {
                            best_below_gap = gap;
                            best_below_name = name_ptr;
                            best_below_len = name_len;
                        }
                    }
                }
            }
            i += 1;
        }
    }

    // Stack buffer report: prefix + fault VA + verdict. No allocation,
    // no formatting machinery.
    let mut buf = [0u8; 512];
    let mut n = 0usize;
    let mut push = |bytes: &[u8]| {
        let take = bytes.len().min(buf.len() - n);
        buf[n..n + take].copy_from_slice(&bytes[..take]);
        n += take;
    };
    push(b"[graph-ptr-census] SIGSEGV fault_va=0x");
    let mut hex = [b'0'; 16];
    let mut v = fault;
    let mut k = 16;
    while k > 0 {
        k -= 1;
        hex[k] = b"0123456789abcdef"[(v & 0xf) as usize];
        v >>= 4;
    }
    push(&hex);
    if contained_name != 0 {
        push(b" INSIDE ");
        push(unsafe {
            core::slice::from_raw_parts(
                contained_name as *const u8,
                (contained_len as usize).min(64),
            )
        });
        push(b" +0x");
        let mut off = [b'0'; 16];
        let mut w = fault - contained_ptr;
        let mut k = 16;
        while k > 0 {
            k -= 1;
            off[k] = b"0123456789abcdef"[(w & 0xf) as usize];
            w >>= 4;
        }
        // Trim leading zeros to 1..8 digits for readability.
        let mut start = 0;
        while start < 12 && off[start] == b'0' {
            start += 1;
        }
        push(&off[start..]);
        push(b" (bytes=");
        let mut dec = [b'0'; 20];
        let mut w = contained_bytes;
        let mut k = 20;
        while k > 0 {
            k -= 1;
            dec[k] = b'0' + (w % 10) as u8;
            w /= 10;
            if w == 0 && k < 20 {
                // keep one leading digit slot trimming below
            }
        }
        let mut start = 0;
        while start < 19 && dec[start] == b'0' {
            start += 1;
        }
        push(&dec[start..]);
        push(b")");
    } else {
        push(b" UNTRACKED");
        if best_below_name != 0 {
            push(b" below=");
            push(unsafe {
                core::slice::from_raw_parts(
                    best_below_name as *const u8,
                    (best_below_len as usize).min(64),
                )
            });
        }
        if best_above_name != 0 {
            push(b" above=");
            push(unsafe {
                core::slice::from_raw_parts(
                    best_above_name as *const u8,
                    (best_above_len as usize).min(64),
                )
            });
        }
    }
    push(b"\n");
    unsafe {
        libc::write(2, buf.as_ptr().cast(), n);
        libc::signal(libc::SIGSEGV, libc::SIG_DFL);
        libc::raise(libc::SIGSEGV);
    }
}

pub(crate) struct CudaGraphHandle {
    graph: *mut core::ffi::c_void,
    exec: *mut core::ffi::c_void,
    stream: GraphStream,
    // ROCm: capture-time allocations are carved from this arena and it must
    // outlive the exec.
    #[cfg(feature = "rocm")]
    arena_keepalive: Option<Arc<ArenaSlice<u8>>>,
    // Capture arenas on every role: allocations carved during capture
    // must outlive the exec. Solo-cuda used to drop this (dual-only
    // keepalive) and fault the first replay with a poisoned context.
    #[cfg(feature = "cuda")]
    cuda_arena_keepalive: Option<Arc<crate::pipeline::graph_role::CuStreamSlice>>,
}

#[cfg(feature = "rocm")]
impl CudaGraphHandle {
    pub(crate) fn set_arena(&mut self, arena: Option<Arc<ArenaSlice<u8>>>) {
        self.arena_keepalive = arena;
    }

    pub(crate) fn arena_bytes(&self) -> usize {
        self.arena_keepalive.as_ref().map_or(0, |arena| arena.len())
    }
}

#[cfg(feature = "cuda")]
impl CudaGraphHandle {
    pub(crate) fn set_cuda_arena(
        &mut self,
        arena: Option<Arc<crate::pipeline::graph_role::CuStreamSlice>>,
    ) {
        self.cuda_arena_keepalive = arena;
    }
}

unsafe impl Send for CudaGraphHandle {}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        let _ = self.stream.bind_context_to_thread();
        if !self.exec.is_null() {
            let _ = graph_exec_destroy(&self.stream, self.exec);
            self.exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            let _ = graph_destroy(&self.stream, self.graph);
            self.graph = std::ptr::null_mut();
        }
    }
}

impl CudaGraphHandle {
    pub(crate) fn end_capture(stream: &GraphStream) -> candle_core::Result<Option<Self>> {
        let mut graph = std::ptr::null_mut();
        graph_end_capture(stream, &mut graph)
            .map_err(|err| err.context("CUDA graph stream end capture failed"))?;
        // Raw end-capture leaves the fork's capturing flag / arena pointer
        // set, which would route later eager allocs into the stale arena.
        stream.reset_capture_bookkeeping();
        if graph.is_null() {
            return Ok(None);
        }

        let mut exec = std::ptr::null_mut();
        if let Err(err) = graph_instantiate(stream, graph, &mut exec, CUDA_GRAPH_INSTANTIATE_FLAGS)
        {
            let _ = graph_destroy(stream, graph);
            return Err(err.context("CUDA graph instantiate failed"));
        }

        Ok(Some(Self {
            graph,
            exec,
            stream: stream.clone(),
            #[cfg(feature = "rocm")]
            arena_keepalive: None,
            #[cfg(feature = "cuda")]
            cuda_arena_keepalive: None,
        }))
    }

    pub(crate) fn upload(&self) -> candle_core::Result<()> {
        graph_upload(&self.stream, self.exec)
            .map_err(|err| err.context("CUDA graph upload failed"))?;
        let _ = self.stream.graph_check_err();
        Ok(())
    }

    pub(crate) fn launch(&self) -> candle_core::Result<()> {
        graph_launch(&self.stream, self.exec)
            .map_err(|err| err.context("CUDA graph launch failed"))?;
        let _ = self.stream.graph_check_err();
        Ok(())
    }

    pub(crate) fn stream(&self) -> &GraphStream {
        &self.stream
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CudaDecodeGraphKey {
    device: DeviceLocation,
    input_shape: Vec<usize>,
    input_dtype: DType,
    recurrent_batch_kind: RecurrentBatchKind,
    max_context_len: Option<usize>,
    full_max_context_len: Option<usize>,
    tensors: Vec<CudaGraphTensorKey>,
    decode_rows: Option<DecodePagedRowsGraphKey>,
}

// Stable capture-shape identity for arena sizing: batch width, query length,
// dtype size, and batch kind drive capture-time allocations, while context
// lengths and block tables do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DecodeArenaBucketKey {
    batch: usize,
    seq_len: usize,
    dtype_bytes: usize,
    kind: RecurrentBatchKind,
}

impl CudaDecodeGraphKey {
    fn arena_bucket(&self) -> DecodeArenaBucketKey {
        DecodeArenaBucketKey {
            batch: self.input_shape.first().copied().unwrap_or(0),
            seq_len: self.input_shape.get(1).copied().unwrap_or(0),
            dtype_bytes: self.input_dtype.size_in_bytes(),
            kind: self.recurrent_batch_kind,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CudaGraphTensorKey {
    name: &'static str,
    location: DeviceLocation,
    shape: Vec<usize>,
    dtype: DType,
}

type CudaGraphVarMap = HashMap<DeviceLocation, Var>;
type FlashInferDecodeScratchMaps = (
    Option<HashMap<DeviceLocation, Tensor>>,
    Option<HashMap<DeviceLocation, Tensor>>,
);

struct CudaGraphPinnedAllocation<T> {
    allocation: GraphPinnedHost<T>,
    ptr: NonNull<T>,
}

// The allocation owns the pointer and all accesses require an exclusive borrow.
unsafe impl<T: Send> Send for CudaGraphPinnedAllocation<T> {}

impl<T> CudaGraphPinnedAllocation<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.allocation.len()) }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

enum CudaGraphPinnedData {
    U8(CudaGraphPinnedAllocation<u8>),
    U32(CudaGraphPinnedAllocation<u32>),
    I32(CudaGraphPinnedAllocation<i32>),
    I64(CudaGraphPinnedAllocation<i64>),
    F32(CudaGraphPinnedAllocation<f32>),
}

struct CudaGraphPinnedBuffer {
    data: CudaGraphPinnedData,
    initialized: bool,
}

struct CudaGraphCopyCompletion {
    event: GraphEvent,
    stream: GraphStream,
    pending: bool,
    active: bool,
    ordered_after_graph: bool,
}

pub(crate) struct CudaGraphHostStaging {
    buffers: HashMap<(&'static str, DeviceLocation), CudaGraphPinnedBuffer>,
    completions: HashMap<DeviceLocation, CudaGraphCopyCompletion>,
    graph_complete: GraphEvent,
    graph_stream: GraphStream,
    graph_pending: bool,
}

fn same_cuda_stream(left: &GraphStream, right: &GraphStream) -> bool {
    left.same_as(right)
}

impl Drop for CudaGraphHostStaging {
    fn drop(&mut self) {
        for completion in self.completions.values() {
            if completion.pending {
                let _ = completion.event.synchronize();
            } else if completion.active {
                let _ = completion.stream.synchronize();
            }
        }
    }
}

pub(crate) struct CudaDecodeGraphCaptureCtx<'a> {
    pub(crate) key: CudaDecodeGraphKey,
    pub(crate) input_ids: &'a Tensor,
    pub(crate) seqlen_offsets: &'a [usize],
    pub(crate) position_ids: &'a [usize],
    pub(crate) block_size: usize,
    pub(crate) kv_cache: &'a [(Tensor, Tensor)],
    pub(crate) metadata: &'a PagedAttentionInputMetadata,
    pub(crate) model_metadata: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
    pub(crate) activation_dtype: DType,
    pub(crate) warmup_logits: &'a Tensor,
    pub(crate) state_indices: Option<CudaGraphVarMap>,
    pub(crate) real_batch: usize,
    pub(crate) arena_bytes: usize,
}

struct CudaDecodeGraphMetadataInput<'a> {
    metadata: &'a PagedAttentionInputMetadata,
    seqlen_offsets: &'a [usize],
    position_ids: &'a [usize],
    seq_len: usize,
    block_size: usize,
    kv_cache: &'a [(Tensor, Tensor)],
    model_metadata: Option<&'a (dyn ModelConfigLike + Send + Sync)>,
    activation_dtype: DType,
}

pub(crate) struct CudaDecodeGraphMetadataBuffers {
    requirements: PagedDecodeMetadataRequirements,
    flashinfer_views_alias: bool,
    slot_mappings: CudaGraphVarMap,
    block_tables: Option<CudaGraphVarMap>,
    context_lens: Option<CudaGraphVarMap>,
    full_block_tables: Option<CudaGraphVarMap>,
    full_context_lens: Option<CudaGraphVarMap>,
    paged_kv_indptr: Option<CudaGraphVarMap>,
    paged_kv_indices: Option<CudaGraphVarMap>,
    paged_kv_last_page_len: Option<CudaGraphVarMap>,
    full_paged_kv_indptr: Option<CudaGraphVarMap>,
    full_paged_kv_indices: Option<CudaGraphVarMap>,
    full_paged_kv_last_page_len: Option<CudaGraphVarMap>,
    paged_kv_request_indices: Option<CudaGraphVarMap>,
    paged_kv_tile_indices: Option<CudaGraphVarMap>,
    paged_kv_o_indptr: Option<CudaGraphVarMap>,
    paged_kv_chunk_size: Option<CudaGraphVarMap>,
    paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    full_paged_kv_request_indices: Option<CudaGraphVarMap>,
    full_paged_kv_tile_indices: Option<CudaGraphVarMap>,
    full_paged_kv_o_indptr: Option<CudaGraphVarMap>,
    full_paged_kv_chunk_size: Option<CudaGraphVarMap>,
    full_paged_kv_block_valid_mask: Option<CudaGraphVarMap>,
    flashinfer_decode_tmp_v: Option<HashMap<DeviceLocation, Tensor>>,
    flashinfer_decode_tmp_s: Option<HashMap<DeviceLocation, Tensor>>,
    fa3_decode: Option<Fa3DecodeState>,
    rope_positions: CudaGraphVarMap,
}

impl CudaDecodeGraphKey {
    fn has_same_spec_state_shape(&self, other: &Self) -> bool {
        self.device == other.device
            && self.input_shape == other.input_shape
            && self.input_dtype == other.input_dtype
            && self.recurrent_batch_kind == other.recurrent_batch_kind
    }

    pub(crate) fn new(
        input_ids: &Tensor,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
        recurrent_batch_kind: RecurrentBatchKind,
    ) -> candle_core::Result<Self> {
        let decode_rows = metadata.decode_rows.as_ref().map(|rows| rows.graph_key());
        let mut tensors = Vec::new();
        if decode_rows.is_none() {
            push_graph_tensor_keys("slot_mappings", Some(&metadata.slot_mappings), &mut tensors);
            push_graph_tensor_keys("block_tables", metadata.block_tables.as_ref(), &mut tensors);
            push_graph_tensor_keys("context_lens", metadata.context_lens.as_ref(), &mut tensors);
            push_graph_tensor_keys(
                "full_block_tables",
                metadata.full_block_tables.as_ref(),
                &mut tensors,
            );
            push_graph_tensor_keys(
                "full_context_lens",
                metadata.full_context_lens.as_ref(),
                &mut tensors,
            );
            push_flashinfer_graph_tensor_keys(metadata, &mut tensors);
            if flashinfer_views_alias(metadata) {
                tensors.retain(|tensor| !tensor.name.starts_with("full_"));
            }
        }
        tensors.sort_by(|a, b| {
            a.name.cmp(b.name).then_with(|| {
                device_location_sort_key(&a.location).cmp(&device_location_sort_key(&b.location))
            })
        });

        Ok(Self {
            device: input_ids.device().location(),
            input_shape: input_ids.dims().to_vec(),
            input_dtype: input_ids.dtype(),
            recurrent_batch_kind,
            max_context_len: decode_rows
                .is_none()
                .then(|| {
                    graph_context_len(
                        metadata.max_context_len,
                        bucket_context_len(metadata.block_tables.as_ref(), block_size),
                    )
                })
                .flatten(),
            full_max_context_len: decode_rows
                .is_none()
                .then(|| {
                    graph_context_len(
                        metadata.full_max_context_len,
                        bucket_context_len(metadata.full_block_tables.as_ref(), block_size),
                    )
                })
                .flatten(),
            tensors,
            decode_rows,
        })
    }
}

impl CudaDecodeGraphMetadataBuffers {
    fn new(
        input: CudaDecodeGraphMetadataInput<'_>,
    ) -> candle_core::Result<(Self, PagedAttentionInputMetadata)> {
        let CudaDecodeGraphMetadataInput {
            metadata,
            seqlen_offsets,
            position_ids,
            seq_len,
            block_size,
            kv_cache,
            model_metadata,
            activation_dtype,
        } = input;
        let slot_mappings = var_map_from_tensor_map(&metadata.slot_mappings)?;
        if seqlen_offsets.len() != position_ids.len() {
            candle_core::bail!(
                "CUDA graph decode has {} KV offsets but {} position ends",
                seqlen_offsets.len(),
                position_ids.len()
            );
        }
        let rope_positions =
            rope_positions_var_map(&metadata.slot_mappings, position_ids, seq_len)?;
        let (flashinfer_decode_tmp_v, flashinfer_decode_tmp_s) = flashinfer_decode_scratch_maps(
            metadata,
            seqlen_offsets.len(),
            kv_cache,
            model_metadata,
            activation_dtype,
        )?;
        let fa3_decode = metadata
            .flashinfer
            .as_ref()
            .map(|flashinfer| {
                make_fa3_decode_state(
                    flashinfer,
                    seqlen_offsets.len(),
                    seq_len,
                    kv_cache,
                    model_metadata,
                    activation_dtype,
                )
            })
            .transpose()?
            .flatten();
        let flashinfer_views_alias = flashinfer_views_alias(metadata);
        let requirements = PagedDecodeMetadataRequirements::graph(
            metadata.block_tables.is_some(),
            metadata.context_lens.is_some(),
            metadata.flashinfer.is_some(),
            metadata.flashinfer.is_some(),
        );
        let mut buffers = Self {
            requirements,
            flashinfer_views_alias,
            slot_mappings,
            block_tables: option_var_map_from_tensor_map(metadata.block_tables.as_ref())?,
            context_lens: option_var_map_from_tensor_map(metadata.context_lens.as_ref())?,
            full_block_tables: option_var_map_from_tensor_map_if_distinct(
                metadata.full_block_tables.as_ref(),
                flashinfer_views_alias,
            )?,
            full_context_lens: option_var_map_from_tensor_map_if_distinct(
                metadata.full_context_lens.as_ref(),
                flashinfer_views_alias,
            )?,
            paged_kv_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
            )?,
            paged_kv_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
            )?,
            paged_kv_last_page_len: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
            )?,
            full_paged_kv_indptr: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                flashinfer_views_alias,
            )?,
            full_paged_kv_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_last_page_len: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
                flashinfer_views_alias,
            )?,
            paged_kv_request_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
            )?,
            paged_kv_tile_indices: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
            )?,
            paged_kv_o_indptr: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.o_indptr),
            )?,
            paged_kv_chunk_size: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
            )?,
            paged_kv_block_valid_mask: option_var_map_from_tensor_map(
                flashinfer_paged_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
            )?,
            full_paged_kv_request_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_tile_indices: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_tile_indices),
                flashinfer_views_alias,
            )?,
            full_paged_kv_o_indptr: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.o_indptr),
                flashinfer_views_alias,
            )?,
            full_paged_kv_chunk_size: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.kv_chunk_size),
                flashinfer_views_alias,
            )?,
            full_paged_kv_block_valid_mask: option_var_map_from_tensor_map_if_distinct(
                flashinfer_full_view(metadata).map(|view| &view.tile_plan.block_valid_mask),
                flashinfer_views_alias,
            )?,
            flashinfer_decode_tmp_v,
            flashinfer_decode_tmp_s,
            fa3_decode,
            rope_positions,
        };
        if flashinfer_views_alias {
            buffers.full_block_tables = buffers.block_tables.clone();
            buffers.full_context_lens = buffers.context_lens.clone();
            buffers.full_paged_kv_indptr = buffers.paged_kv_indptr.clone();
            buffers.full_paged_kv_indices = buffers.paged_kv_indices.clone();
            buffers.full_paged_kv_last_page_len = buffers.paged_kv_last_page_len.clone();
            buffers.full_paged_kv_request_indices = buffers.paged_kv_request_indices.clone();
            buffers.full_paged_kv_tile_indices = buffers.paged_kv_tile_indices.clone();
            buffers.full_paged_kv_o_indptr = buffers.paged_kv_o_indptr.clone();
            buffers.full_paged_kv_chunk_size = buffers.paged_kv_chunk_size.clone();
            buffers.full_paged_kv_block_valid_mask = buffers.paged_kv_block_valid_mask.clone();
        }
        let metadata = buffers.metadata_from(metadata, block_size);
        Ok((buffers, metadata))
    }

    fn finish_capture(&mut self, metadata: &PagedAttentionInputMetadata) {
        let tile_plan_used = metadata
            .flashinfer
            .as_ref()
            .is_some_and(FlashInferMetadata::decode_tile_plan_was_used);
        self.requirements = PagedDecodeMetadataRequirements::graph(
            self.block_tables.is_some(),
            self.context_lens.is_some(),
            self.fa3_decode.is_some() || tile_plan_used,
            tile_plan_used,
        );
    }

    fn copy_from(
        &mut self,
        metadata: &PagedAttentionInputMetadata,
        position_ids: &[usize],
        seq_len: usize,
        host_staging: &mut CudaGraphHostStaging,
    ) -> candle_core::Result<()> {
        let graph_update = if metadata.has_host_staged_decode_tensors() {
            Some(
                metadata
                    .decode_rows
                    .as_ref()
                    .expect("host-staged decode metadata requires source rows")
                    .build_graph_update(self.requirements)
                    .map_err(candle_core::Error::msg)?,
            )
        } else {
            None
        };
        let metadata = graph_update.as_ref().unwrap_or(metadata);
        copy_var_map(
            &self.slot_mappings,
            &metadata.slot_mappings,
            "slot_mappings",
            host_staging,
        )?;
        if self.requirements.context_lens {
            copy_option_var_map(
                &self.context_lens,
                metadata.context_lens.as_ref(),
                "context_lens",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_context_lens,
                    metadata.full_context_lens.as_ref(),
                    "full_context_lens",
                    host_staging,
                )?;
            }
        }
        if self.requirements.block_tables {
            copy_option_var_map(
                &self.block_tables,
                metadata.block_tables.as_ref(),
                "block_tables",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_block_tables,
                    metadata.full_block_tables.as_ref(),
                    "full_block_tables",
                    host_staging,
                )?;
            }
        }
        if self.requirements.flashinfer_paged_kv {
            copy_option_var_map(
                &self.paged_kv_last_page_len,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.last_page_len),
                "paged_kv_last_page_len",
                host_staging,
            )?;
            copy_option_var_map(
                &self.paged_kv_indptr,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indptr),
                "paged_kv_indptr",
                host_staging,
            )?;
            copy_option_var_map(
                &self.paged_kv_indices,
                flashinfer_paged_view(metadata).map(|view| &view.paged_kv.indices),
                "paged_kv_indices",
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_option_var_map(
                    &self.full_paged_kv_last_page_len,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.last_page_len),
                    "full_paged_kv_last_page_len",
                    host_staging,
                )?;
                copy_option_var_map(
                    &self.full_paged_kv_indptr,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.indptr),
                    "full_paged_kv_indptr",
                    host_staging,
                )?;
                copy_option_var_map(
                    &self.full_paged_kv_indices,
                    flashinfer_full_view(metadata).map(|view| &view.paged_kv.indices),
                    "full_paged_kv_indices",
                    host_staging,
                )?;
            }
        }
        if self.requirements.flashinfer_tile_plan {
            copy_flashinfer_tile_plan(
                metadata,
                false,
                FlashInferTilePlanVars {
                    request_indices: &self.paged_kv_request_indices,
                    kv_tile_indices: &self.paged_kv_tile_indices,
                    o_indptr: &self.paged_kv_o_indptr,
                    kv_chunk_size: &self.paged_kv_chunk_size,
                    block_valid_mask: &self.paged_kv_block_valid_mask,
                },
                host_staging,
            )?;
            if !self.flashinfer_views_alias {
                copy_flashinfer_tile_plan(
                    metadata,
                    true,
                    FlashInferTilePlanVars {
                        request_indices: &self.full_paged_kv_request_indices,
                        kv_tile_indices: &self.full_paged_kv_tile_indices,
                        o_indptr: &self.full_paged_kv_o_indptr,
                        kv_chunk_size: &self.full_paged_kv_chunk_size,
                        block_valid_mask: &self.full_paged_kv_block_valid_mask,
                    },
                    host_staging,
                )?;
            }
        }
        copy_rope_positions(&self.rope_positions, position_ids, seq_len, host_staging)?;
        Ok(())
    }

    fn flashinfer_metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> Option<FlashInferMetadata> {
        let original = metadata.flashinfer.as_ref()?;
        let logical = FlashInferPagedAttentionView {
            block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            max_context_len: original
                .views
                .logical
                .max_context_len
                .or_else(|| bucket_context_len_from_vars(&self.full_block_tables, block_size)),
            paged_kv: flashinfer_paged_kv_from_vars(
                &self.full_paged_kv_indptr,
                &self.full_paged_kv_indices,
                &self.full_paged_kv_last_page_len,
            )?,
            tile_plan: flashinfer_tile_plan_from_vars(
                &self.full_paged_kv_request_indices,
                &self.full_paged_kv_tile_indices,
                &self.full_paged_kv_o_indptr,
                &self.full_paged_kv_chunk_size,
                &self.full_paged_kv_block_valid_mask,
            )?,
        };
        let sliding = if let Some(view) = original.views.sliding.as_ref() {
            Some(FlashInferPagedAttentionView {
                block_tables: option_tensor_map_from_var_map(&self.block_tables),
                context_lens: option_tensor_map_from_var_map(&self.context_lens),
                max_context_len: view
                    .max_context_len
                    .or_else(|| bucket_context_len_from_vars(&self.block_tables, block_size)),
                paged_kv: flashinfer_paged_kv_from_vars(
                    &self.paged_kv_indptr,
                    &self.paged_kv_indices,
                    &self.paged_kv_last_page_len,
                )?,
                tile_plan: flashinfer_tile_plan_from_vars(
                    &self.paged_kv_request_indices,
                    &self.paged_kv_tile_indices,
                    &self.paged_kv_o_indptr,
                    &self.paged_kv_chunk_size,
                    &self.paged_kv_block_valid_mask,
                )?,
            })
        } else {
            None
        };

        Some(
            FlashInferMetadata {
                views: FlashInferPagedAttentionViews { logical, sliding },
                decode_tmp_v: self.flashinfer_decode_tmp_v.clone(),
                decode_tmp_s: self.flashinfer_decode_tmp_s.clone(),
                fa3_decode: self.fa3_decode.clone(),
                decode_tile_plan_used: None,
            }
            .track_decode_tile_plan(),
        )
    }

    fn metadata_from(
        &self,
        metadata: &PagedAttentionInputMetadata,
        block_size: usize,
    ) -> PagedAttentionInputMetadata {
        PagedAttentionInputMetadata {
            block_tables: option_tensor_map_from_var_map(&self.block_tables),
            context_lens: option_tensor_map_from_var_map(&self.context_lens),
            block_size: metadata.block_size,
            paged_context_lens_cpu: metadata.paged_context_lens_cpu.clone(),
            full_paged_context_lens_cpu: metadata.full_paged_context_lens_cpu.clone(),
            slot_mappings: tensor_map_from_var_map(&self.slot_mappings),
            max_context_len: graph_context_len(
                metadata.max_context_len,
                bucket_context_len_from_vars(&self.block_tables, block_size),
            ),
            full_block_tables: option_tensor_map_from_var_map(&self.full_block_tables),
            full_context_lens: option_tensor_map_from_var_map(&self.full_context_lens),
            full_max_context_len: graph_context_len(
                metadata.full_max_context_len,
                bucket_context_len_from_vars(&self.full_block_tables, block_size),
            ),
            is_first_prompt_chunk: metadata.is_first_prompt_chunk,
            is_final_prompt_chunk: metadata.is_final_prompt_chunk,
            prompt_chunk_attention_policy: metadata.prompt_chunk_attention_policy,
            has_noncausal_mm_context: metadata.has_noncausal_mm_context,
            prefix_gather_workspace_limit: metadata.prefix_gather_workspace_limit,
            mm_prefix_ranges: metadata.mm_prefix_ranges.clone(),
            full_mm_prefix_ranges: metadata.full_mm_prefix_ranges.clone(),
            prefill_attention_heads: metadata.prefill_attention_heads,
            prefill_key_value_heads: metadata.prefill_key_value_heads,
            prefill_head_dim: metadata.prefill_head_dim,
            flashinfer: self.flashinfer_metadata_from(metadata, block_size),
            rope_positions: Some(tensor_map_from_var_map(&self.rope_positions)),
            num_cached_tokens: metadata.num_cached_tokens.clone(),
            query_lens: metadata.query_lens.clone(),
            cu_seqlens_q: metadata.cu_seqlens_q.clone(),
            cu_seqlens_kv: metadata.cu_seqlens_kv.clone(),
            decode_rows: metadata.decode_rows.clone(),
        }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SpeculativeGraphTensorMetadata {
    shape: Vec<usize>,
    strides: Vec<usize>,
    contiguous: bool,
    dtype: DType,
    device: candle_core::DeviceLocation,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn speculative_graph_tensor_metadata(
    state: &dyn crate::speculative::SpeculativeGraphState,
) -> Vec<SpeculativeGraphTensorMetadata> {
    state
        .tensors()
        .into_iter()
        .map(|tensor| {
            let (_storage, layout) = tensor.storage_and_layout();
            SpeculativeGraphTensorMetadata {
                shape: tensor.dims().to_vec(),
                strides: layout.stride().to_vec(),
                contiguous: tensor.is_contiguous(),
                dtype: tensor.dtype(),
                device: tensor.device().location(),
            }
        })
        .collect()
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn validate_speculative_graph_tensor_metadata(
    expected: &[SpeculativeGraphTensorMetadata],
    actual: &[SpeculativeGraphTensorMetadata],
) -> candle_core::Result<()> {
    if actual.len() != expected.len() {
        candle_core::bail!(
            "speculative graph state changed tensor count between warmup and capture"
        );
    }
    if actual.iter().zip(expected).any(|(actual, expected)| {
        actual.shape != expected.shape
            || actual.strides != expected.strides
            || actual.contiguous != expected.contiguous
            || actual.dtype != expected.dtype
            || actual.device != expected.device
    }) {
        candle_core::bail!(
            "speculative graph state changed tensor metadata between warmup and capture"
        );
    }
    Ok(())
}

#[cfg(test)]
mod speculative_graph_tensor_metadata_tests {
    use super::{validate_speculative_graph_tensor_metadata, SpeculativeGraphTensorMetadata};
    use candle_core::{DType, DeviceLocation};

    fn metadata(
        shape: &[usize],
        strides: &[usize],
        contiguous: bool,
        dtype: DType,
    ) -> SpeculativeGraphTensorMetadata {
        SpeculativeGraphTensorMetadata {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            contiguous,
            dtype,
            device: DeviceLocation::Cpu,
        }
    }

    #[test]
    fn validates_tensor_count_and_metadata() {
        let expected = vec![metadata(&[4, 8, 16], &[128, 16, 1], true, DType::BF16)];
        assert!(validate_speculative_graph_tensor_metadata(&expected, &expected).is_ok());
        assert!(validate_speculative_graph_tensor_metadata(&expected, &[]).is_err());
        assert!(validate_speculative_graph_tensor_metadata(
            &expected,
            &[metadata(&[4, 8, 17], &[136, 17, 1], true, DType::BF16)]
        )
        .is_err());
        assert!(validate_speculative_graph_tensor_metadata(
            &expected,
            &[metadata(&[4, 8, 16], &[128, 16, 1], true, DType::F16)]
        )
        .is_err());
        assert!(validate_speculative_graph_tensor_metadata(
            &expected,
            &[metadata(&[4, 8, 16], &[1, 64, 4], false, DType::BF16)]
        )
        .is_err());
        assert!(validate_speculative_graph_tensor_metadata(
            &expected,
            &[metadata(&[4, 8, 16], &[128, 16, 1], false, DType::BF16)]
        )
        .is_err());
    }
}

#[derive(Clone, Default)]
pub(crate) struct CudaGraphSpecStateUsage {
    bytes: HashMap<DeviceLocation, usize>,
    device_totals: HashMap<DeviceLocation, usize>,
}

impl CudaGraphSpecStateUsage {
    fn from_state(state: &dyn SpeculativeGraphState) -> candle_core::Result<Self> {
        let mut usage = Self::default();
        for tensor in state.tensors() {
            let location = tensor.device().location();
            let bytes = tensor
                .elem_count()
                .saturating_mul(tensor.dtype().size_in_bytes());
            usage
                .bytes
                .entry(location)
                .and_modify(|total| *total = total.saturating_add(bytes))
                .or_insert(bytes);
            if let std::collections::hash_map::Entry::Vacant(entry) =
                usage.device_totals.entry(location)
            {
                let (info, total) = match tensor.device() {
                    // Dual builds serve both roles; the (cuda,not-rocm) gate
                    // below used to leave dual-build Cuda tensors unmatched
                    // (same class as the Oct-8 topk port gap).
                    #[cfg(feature = "cuda")]
                    Device::Cuda(device) => device
                        .cuda_stream()
                        .context()
                        .mem_get_info()
                        .map_err(candle_core::Error::wrap)?,
                    #[cfg(feature = "rocm")]
                    Device::Hip(device) => device
                        .cuda_stream()
                        .context()
                        .mem_get_info()
                        .map_err(candle_core::Error::wrap)?,
                    _ => {
                        candle_core::bail!("CUDA graph speculative state expected GPU tensors")
                    }
                };
                let _ = info;
                entry.insert(total);
            }
        }
        Ok(usage)
    }

    fn total_bytes(&self) -> usize {
        self.bytes
            .values()
            .fold(0usize, |total, bytes| total.saturating_add(*bytes))
    }
}

fn default_spec_state_budget(total: usize, largest_entry: usize) -> usize {
    let floor = total.saturating_mul(CUDA_GRAPH_SPEC_STATE_BUDGET_FLOOR_PERCENT) / 100;
    let ceiling = total.saturating_mul(CUDA_GRAPH_SPEC_STATE_BUDGET_CEILING_PERCENT) / 100;
    let working_set = largest_entry.saturating_mul(CUDA_GRAPH_SPEC_STATE_WORKING_SET_MULTIPLIER);
    floor.max(working_set.min(ceiling)).max(largest_entry)
}

fn configured_spec_state_budget(total: usize, largest_entry: usize) -> usize {
    std::env::var(CUDA_GRAPH_SPEC_STATE_BUDGET_BYTES_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| default_spec_state_budget(total, largest_entry))
}

fn spec_state_eviction_plan(
    existing: &[CudaGraphSpecStateUsage],
    incoming: &CudaGraphSpecStateUsage,
    budgets: &HashMap<DeviceLocation, usize>,
) -> Vec<usize> {
    let mut totals = incoming.bytes.clone();
    for usage in existing {
        for (location, bytes) in &usage.bytes {
            totals
                .entry(*location)
                .and_modify(|total| *total = total.saturating_add(*bytes))
                .or_insert(*bytes);
        }
    }

    let mut retained = vec![true; existing.len()];
    let mut evictions = Vec::new();
    loop {
        let over_budget = totals
            .iter()
            .filter_map(|(location, bytes)| {
                (*bytes > budgets.get(location).copied().unwrap_or(usize::MAX)).then_some(*location)
            })
            .collect::<Vec<_>>();
        if over_budget.is_empty() {
            break;
        }
        let Some((idx, usage)) = existing.iter().enumerate().find(|(idx, usage)| {
            retained[*idx]
                && over_budget
                    .iter()
                    .any(|location| usage.bytes.get(location).is_some_and(|bytes| *bytes > 0))
        }) else {
            break;
        };
        retained[idx] = false;
        evictions.push(idx);
        for (location, bytes) in &usage.bytes {
            totals
                .entry(*location)
                .and_modify(|total| *total = total.saturating_sub(*bytes));
        }
    }
    evictions
}

pub(crate) struct CudaDecodeGraphEntry {
    generation: u64,
    replay_epoch: u64,
    key: CudaDecodeGraphKey,
    host_staging: CudaGraphHostStaging,
    input_ids: Var,
    metadata_buffers: CudaDecodeGraphMetadataBuffers,
    state_indices: Option<CudaGraphVarMap>,
    _metadata: PagedAttentionInputMetadata,
    logits: Tensor,
    // Proposer-facing graph outputs refreshed by replay.
    spec_state: Option<Arc<dyn SpeculativeGraphState>>,
    spec_state_usage: CudaGraphSpecStateUsage,
    // Pointer-correlation census (MISTRALRS_GRAPH_PTR_CENSUS only):
    // capture-time VAs of entry-held buffers, consumed by the
    // pre-first-launch sweep. None after the sweep runs.
    ptr_census: Option<Vec<GraphPtrRecord>>,
    // Must stay last so graph-backed tensors enqueue their frees before the graph exec is destroyed.
    graph: CudaGraphHandle,
}

pub struct CudaDecodeGraphLaunch {
    generation: u64,
    replay_epoch: u64,
    key: CudaDecodeGraphKey,
    input_ids: Tensor,
    graph_stream: GraphStream,
    real_batch: usize,
    source: CudaGraphDecodeStep,
}

impl CudaDecodeGraphLaunch {
    pub(crate) fn resident_input(&self) -> &Tensor {
        &self.input_ids
    }

    pub(crate) fn graph_stream(&self) -> &GraphStream {
        &self.graph_stream
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn real_batch(&self) -> usize {
        self.real_batch
    }

    fn one_token_continuation(&self) -> candle_core::Result<Option<CudaGraphDecodeStep>> {
        let Some(continuation) = self.source.one_token_continuation(self.input_ids.clone())? else {
            return Ok(None);
        };
        let key = CudaDecodeGraphKey::new(
            &continuation.input_ids,
            &continuation.metadata,
            continuation
                .metadata
                .decode_rows
                .as_ref()
                .expect("continuation must retain decode rows")
                .block_size,
            self.key.recurrent_batch_kind,
        )?;
        Ok((key == self.key).then_some(continuation))
    }

    fn matches(&self, entry: &CudaDecodeGraphEntry) -> bool {
        cuda_graph_replay_version_matches(
            entry.generation,
            entry.replay_epoch,
            self.generation,
            self.replay_epoch,
        ) && self.key == entry.key
    }
}

fn cuda_graph_replay_version_matches(
    entry_generation: u64,
    entry_replay_epoch: u64,
    launch_generation: u64,
    launch_replay_epoch: u64,
) -> bool {
    entry_generation == launch_generation && entry_replay_epoch == launch_replay_epoch
}

impl fmt::Debug for CudaDecodeGraphLaunch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDecodeGraphLaunch")
            .field("generation", &self.generation)
            .field("replay_epoch", &self.replay_epoch)
            .field("input_shape", &self.input_ids.shape())
            .field("input_device", self.input_ids.device())
            .field("real_batch", &self.real_batch)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CudaDecodeGraphReplayInput<'a> {
    Host,
    Resident(&'a CudaDecodeGraphLaunch),
}

impl CudaDecodeGraphEntry {
    pub(crate) fn with_spec_state(
        mut self,
        spec_state: Option<Box<dyn SpeculativeGraphState>>,
        usage: Option<CudaGraphSpecStateUsage>,
    ) -> Self {
        assert_eq!(spec_state.is_some(), usage.is_some());
        self.spec_state = spec_state.map(Arc::from);
        self.spec_state_usage = usage.unwrap_or_default();
        self
    }

    fn launch(
        &self,
        step: &CudaGraphDecodeStep,
        replay_epoch: u64,
    ) -> candle_core::Result<Option<CudaDecodeGraphLaunch>> {
        let input_ids = self.input_ids.as_detached_tensor();
        let (batch, q_len) = input_ids.dims2()?;
        if input_ids.dtype() != DType::U32
            || q_len != 1
            || !input_ids.is_contiguous()
            || self.spec_state.is_some()
        {
            return Ok(None);
        }
        if step.real_batch > batch {
            candle_core::bail!(
                "CUDA graph resident input has batch capacity {batch}, smaller than {} live rows",
                step.real_batch
            );
        }
        Ok(Some(CudaDecodeGraphLaunch {
            generation: self.generation,
            replay_epoch,
            key: self.key.clone(),
            input_ids,
            graph_stream: self.graph.stream.clone(),
            real_batch: step.real_batch,
            source: step.clone(),
        }))
    }

    fn release(self) -> (GraphStream, candle_core::Result<()>) {
        let Self {
            generation: _,
            replay_epoch,
            key: _,
            host_staging,
            input_ids,
            metadata_buffers,
            state_indices,
            _metadata,
            logits,
            spec_state,
            spec_state_usage: _,
            ptr_census: _,
            graph,
        } = self;
        let stream = graph.stream.clone();
        let mut release_result = stream
            .synchronize()
            .map_err(candle_core::Error::wrap)
            .map_err(|err| err.context("CUDA graph entry release wait failed"));
        drop_cuda_graph_entry_resource(host_staging, &stream, "host staging", &mut release_result);
        drop_cuda_graph_entry_output(
            spec_state,
            &stream,
            replay_epoch,
            "speculative state",
            &mut release_result,
        );
        drop_cuda_graph_entry_output(logits, &stream, replay_epoch, "logits", &mut release_result);
        drop_cuda_graph_entry_resource(
            _metadata,
            &stream,
            "paged-attention metadata",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(
            state_indices,
            &stream,
            "state indices",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(
            metadata_buffers,
            &stream,
            "metadata buffers",
            &mut release_result,
        );
        drop_cuda_graph_entry_resource(input_ids, &stream, "input ids", &mut release_result);
        let storage_result = stream
            .synchronize()
            .map_err(candle_core::Error::wrap)
            .map_err(|err| err.context("CUDA graph entry storage release failed"));
        if release_result.is_ok() {
            release_result = storage_result;
        }
        drop(graph);
        (stream, release_result)
    }
}

fn drop_cuda_graph_entry_output<T>(
    output: T,
    stream: &GraphStream,
    replay_epoch: u64,
    name: &'static str,
    release_result: &mut candle_core::Result<()>,
) {
    drop(output);
    if let Err(err) = stream.graph_check_err() {
        if replay_epoch == 0 && err.is_invalid_value() {
            return;
        }
        if release_result.is_ok() {
            *release_result = Err(err
                .into_candle()
                .context(format!("CUDA graph entry {name} release failed")));
        }
    }
}

fn drop_cuda_graph_entry_resource<T>(
    resource: T,
    stream: &GraphStream,
    name: &'static str,
    release_result: &mut candle_core::Result<()>,
) {
    drop(resource);
    if let Err(err) = stream.graph_check_err() {
        if release_result.is_ok() {
            *release_result = Err(err
                .into_candle()
                .context(format!("CUDA graph entry {name} release failed")));
        }
    }
}

pub(crate) struct CudaDecodeGraphReplay {
    pub(crate) logits: Tensor,
    pub(crate) spec_state: Option<Arc<dyn SpeculativeGraphState>>,
    pub(crate) launch: Option<CudaDecodeGraphLaunch>,
}

pub(crate) struct CudaDecodeGraphState {
    entries: Vec<CudaDecodeGraphEntry>,
    spec_state_budgets: HashMap<DeviceLocation, usize>,
    capacity: usize,
    disabled: bool,
    suspended: bool,
    eager_retry_blocked: bool,
    recurrent_storage_generation: Option<u64>,
    arena_observations: HashMap<DecodeArenaBucketKey, usize>,
}

impl Default for CudaDecodeGraphState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            spec_state_budgets: HashMap::new(),
            capacity: TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY,
            disabled: false,
            suspended: false,
            eager_retry_blocked: false,
            recurrent_storage_generation: None,
            arena_observations: HashMap::new(),
        }
    }
}

impl Drop for CudaDecodeGraphState {
    fn drop(&mut self) {
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, 0);
    }
}

impl CudaDecodeGraphState {
    pub(crate) fn ensure_capacity(&mut self, capacity: usize) {
        self.capacity = self.capacity.max(capacity);
    }

    pub(crate) fn record_arena_observation(
        &mut self,
        bucket: DecodeArenaBucketKey,
        consumed_bytes: usize,
    ) {
        if consumed_bytes == 0 {
            return;
        }
        let slot = self.arena_observations.entry(bucket).or_insert(0);
        *slot = (*slot).max(consumed_bytes);
    }

    pub(crate) fn estimated_arena_bytes(
        &self,
        bucket: &DecodeArenaBucketKey,
        warmup_bytes: usize,
    ) -> usize {
        match self.arena_observations.get(bucket) {
            Some(&observed) => observed
                .saturating_add(observed / 8)
                .saturating_add(DECODE_GRAPH_MIN_ARENA_BYTES),
            // Measured on the hybrid+MTP 27B, capture consumption lands
            // within ~0.1% of the warmup's cumulative alloc bytes; the old
            // x6 pinned six-times-oversized arenas per graph, which drained
            // iGPU "available" (host-derived) and starved later buckets.
            // x1.25 plus the overflow regrow path (the regrow exists to
            // absorb undershoot) keeps the pinning close to the real need.
            None => warmup_bytes
                .saturating_mul(5)
                .saturating_div(4)
                .saturating_add(DECODE_GRAPH_MIN_ARENA_BYTES),
        }
    }

    pub(crate) fn disabled(&self) -> bool {
        self.disabled || self.suspended
    }

    pub(crate) fn disable(&mut self) {
        self.disabled = true;
        self.clear();
    }

    pub(crate) fn take_eager_retry_allowed(&mut self) -> bool {
        !std::mem::take(&mut self.eager_retry_blocked)
    }

    pub(crate) fn block_eager_retry(&mut self) {
        self.eager_retry_blocked = true;
    }

    pub(crate) fn clear(&mut self) {
        self.eager_retry_blocked = false;
        let entries = std::mem::take(&mut self.entries);
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, 0);
        release_cuda_graph_entries(entries);
    }

    pub(crate) fn evict_lru_for_memory_pressure(&mut self, max_entries: usize) -> usize {
        let entries = drain_lru_entries(&mut self.entries, max_entries);
        let evicted = entries.len();
        if evicted == 0 {
            return 0;
        }
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
        release_cuda_graph_entries(entries);
        record_cuda_graph_evictions(
            CudaGraphComponent::Target,
            CudaGraphEvictionReason::MemoryPressure,
            evicted,
        );
        evicted
    }

    pub(crate) fn observe_recurrent_storage_generation(&mut self, generation: u64) {
        let previous = self.recurrent_storage_generation.replace(generation);
        if previous.is_some_and(|previous| previous != generation) {
            self.clear();
        }
    }

    pub(crate) fn suspend(&mut self) {
        self.suspended = true;
        self.clear();
    }

    pub(crate) fn resume(&mut self) {
        self.suspended = false;
        self.clear();
    }

    pub(crate) fn contains(&self, key: &CudaDecodeGraphKey) -> bool {
        self.entries.iter().any(|entry| entry.key == *key)
    }

    pub(crate) fn replay(
        &mut self,
        key: &CudaDecodeGraphKey,
        step: &CudaGraphDecodeStep,
        input: CudaDecodeGraphReplayInput<'_>,
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        let Some(pos) = self.entries.iter().position(|entry| entry.key == *key) else {
            return Ok(None);
        };
        let mut entry = self.entries.remove(pos);
        if let CudaDecodeGraphReplayInput::Resident(launch) = input {
            if !launch.matches(&entry) || launch.real_batch != step.real_batch {
                self.entries.push(entry);
                return Ok(None);
            }
        }
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::Target, CudaGraphEvent::Replay);
        let prelaunch_fn = || -> candle_core::Result<_> {
            match input {
                CudaDecodeGraphReplayInput::Host => {
                    entry.input_ids.set(&step.input_ids).map_err(|err| {
                        err.context(format!(
                            "CUDA graph input update failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                            entry.generation, entry.replay_epoch, step.real_batch, entry.key
                        ))
                    })?
                }
                CudaDecodeGraphReplayInput::Resident(_) => {}
            }
            let replay_epoch = entry
                .replay_epoch
                .checked_add(1)
                .expect("CUDA decode graph replay epoch overflow");
            let spec_state = match &entry.spec_state {
                Some(state) if step.real_batch == step.input_ids.dim(0)? => Some(state.clone()),
                Some(state) => Some(Arc::from(state.for_real_batch(step.real_batch)?)),
                None => None,
            };
            let replay = CudaDecodeGraphReplay {
                logits: step.narrow_rows(&entry.logits)?,
                spec_state,
                launch: entry.launch(step, replay_epoch)?,
            };
            let (_, seq_len) = step.input_ids.dims2()?;
            let metadata_buffers = &mut entry.metadata_buffers;
            let state_indices = &entry.state_indices;
            entry
                .host_staging
                .update(|host_staging| {
                    metadata_buffers.copy_from(
                        &step.metadata,
                        &step.position_ids,
                        seq_len,
                        host_staging,
                    )?;
                    match (state_indices, &step.state_indices) {
                        (Some(dst), Some(host)) => copy_state_indices(dst, host, host_staging),
                        (None, None) => Ok(()),
                        _ => candle_core::bail!(
                            "hybrid state indices changed optional state during CUDA graph replay"
                        ),
                    }
                })
                .map_err(|err| {
                    err.context(format!(
                        "CUDA graph metadata update failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                        entry.generation, entry.replay_epoch, step.real_batch, entry.key
                    ))
                })?;
            entry.host_staging.order_before_graph().map_err(|err| {
                err.context(format!(
                    "CUDA graph metadata ordering failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                    entry.generation, entry.replay_epoch, step.real_batch, entry.key
                ))
            })?;
            // Pointer-correlation content dump (diagnostic gate only,
            // first launch only): the U32 metadata actually staged for
            // this replay, beside the capture baseline.
            if entry.ptr_census.is_some() {
                let live = decode_entry_census_tensors(
                    &entry.input_ids,
                    &entry.metadata_buffers,
                    &entry.state_indices,
                    &entry.logits,
                );
                graph_ptr_census_dump_content("replay", &live);
            }
            Ok(Some((replay_epoch, replay)))
        };
        // A panicking prelaunch must not lose the LRU entry or poison the graph lock; fall back to eager.
        let prelaunch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(prelaunch_fn));
        let (replay_epoch, mut replay) = match prelaunch {
            Ok(Ok(Some(prelaunch))) => prelaunch,
            Ok(Ok(None)) => {
                self.entries.push(entry);
                return Ok(None);
            }
            Ok(Err(err)) => {
                self.entries.push(entry);
                return Err(err);
            }
            Err(_) => {
                self.entries.push(entry);
                return Err(candle_core::Error::msg(
                    "CUDA decode graph replay prelaunch panicked",
                ));
            }
        };
        let (_, timing_rows) = step.input_ids.dims2()?;
        let phase_timer = CudaPhaseTimer::start(entry.graph.stream())?;
        // Pointer-correlation sweep (MISTRALRS_GRAPH_PTR_CENSUS only): on
        // the first launch of a censused entry, re-verify entry-held VAs
        // before the driver touches them. A moved VA becomes a loud bail
        // naming the buffer (the entry is dropped, the step falls back to
        // eager, the next step re-captures fresh) instead of a segfault.
        if let Some(records) = entry.ptr_census.take() {
            let live = decode_entry_census_tensors(
                &entry.input_ids,
                &entry.metadata_buffers,
                &entry.state_indices,
                &entry.logits,
            );
            if let Err(err) = graph_ptr_census_verify(&records, &live) {
                return Err(err.context(format!(
                    "CUDA graph ptr-census sweep failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                    entry.generation, entry.replay_epoch, step.real_batch, entry.key
                )));
            }
        }
        if let Err(err) = entry.graph.launch().map_err(|err| {
            err.context(format!(
                "CUDA graph replay launch failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                entry.generation, entry.replay_epoch, step.real_batch, entry.key
            ))
        }) {
            self.disabled = true;
            self.block_eager_retry();
            self.entries.push(entry);
            return Err(err);
        }
        if let Some(timer) = phase_timer {
            timer.finish("target_graph", step.real_batch, timing_rows)?;
        }
        if let Err(record_err) = entry.host_staging.record_graph_complete().map_err(|err| {
            err.context(format!(
                "CUDA graph completion recording failed for generation {}, replay epoch {}, {} live rows, key {:?}",
                entry.generation, entry.replay_epoch, step.real_batch, entry.key
            ))
        }) {
            let synchronize_result = entry
                .graph
                .stream()
                .synchronize()
                .map_err(candle_core::Error::wrap)
                .map_err(|err| err.context("CUDA graph replay recovery synchronization failed"));
            entry.replay_epoch = replay_epoch;
            replay.launch = None;
            self.disabled = true;
            self.block_eager_retry();
            self.entries.push(entry);
            return match synchronize_result {
                Ok(()) => {
                    tracing::warn!(
                        "CUDA decode graphs retired after completion recording error: {record_err:?}"
                    );
                    record_cuda_graph_dispatch(
                        CudaGraphComponent::Target,
                        CudaGraphDispatchMode::Replay,
                        CudaGraphDispatchReason::CacheHit,
                    );
                    Ok(Some(replay))
                }
                Err(synchronize_err) => {
                    tracing::warn!(
                        "CUDA decode graph completion recording and recovery synchronization failed: {record_err:?}; {synchronize_err:?}"
                    );
                    Err(candle_core::Error::msg(format!(
                        "{record_err}; CUDA graph state may have advanced and recovery failed: {synchronize_err}"
                    )))
                }
            };
        }
        entry.replay_epoch = replay_epoch;
        self.entries.push(entry);
        graph_event.success();
        record_cuda_graph_dispatch(
            CudaGraphComponent::Target,
            CudaGraphDispatchMode::Replay,
            CudaGraphDispatchReason::CacheHit,
        );
        Ok(Some(replay))
    }

    pub(crate) fn replay_one_token(
        &mut self,
        launch: CudaDecodeGraphLaunch,
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        let Some(step) = launch.one_token_continuation()? else {
            return Ok(None);
        };
        self.replay(
            &launch.key,
            &step,
            CudaDecodeGraphReplayInput::Resident(&launch),
        )
    }

    pub(crate) fn prepare_spec_state_admission(
        &mut self,
        spec_state: &dyn SpeculativeGraphState,
    ) -> candle_core::Result<CudaGraphSpecStateUsage> {
        let usage = CudaGraphSpecStateUsage::from_state(spec_state)?;
        self.evict_for_spec_state(&usage);
        Ok(usage)
    }

    pub(crate) fn prepare_spec_state_admission_for_key(&mut self, key: &CudaDecodeGraphKey) {
        let usage = self
            .entries
            .iter()
            .rev()
            .find(|entry| entry.key.has_same_spec_state_shape(key))
            .map(|entry| entry.spec_state_usage.clone());
        if let Some(usage) = usage {
            self.evict_for_spec_state(&usage);
        }
    }

    pub(crate) fn insert(&mut self, mut entry: CudaDecodeGraphEntry) {
        self.evict_for_spec_state(&entry.spec_state_usage);
        if let Some(evicted) = take_cuda_graph_capacity_eviction(&mut self.entries, self.capacity) {
            record_cuda_graph_evictions(
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::Capacity,
                1,
            );
            release_cuda_graph_entries(vec![evicted]);
        }
        entry.generation = self.allocate_generation();
        self.entries.push(entry);
        record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
    }

    fn evict_for_spec_state(&mut self, incoming: &CudaGraphSpecStateUsage) {
        for (location, total) in &incoming.device_totals {
            let incoming_bytes = incoming.bytes.get(location).copied().unwrap_or(0);
            let configured = configured_spec_state_budget(*total, incoming_bytes);
            self.spec_state_budgets
                .entry(*location)
                .and_modify(|budget| *budget = (*budget).max(configured))
                .or_insert(configured);
        }
        let usages = self
            .entries
            .iter()
            .map(|entry| entry.spec_state_usage.clone())
            .collect::<Vec<_>>();
        let mut evictions = spec_state_eviction_plan(&usages, incoming, &self.spec_state_budgets);
        if !evictions.is_empty() {
            let evicted = evictions.len();
            tracing::debug!(
                entries = evicted,
                incoming_bytes = incoming.total_bytes(),
                "Evicting CUDA graphs to stay within the speculative state budget"
            );
            evictions.sort_unstable();
            let mut entries = Vec::with_capacity(evictions.len());
            for idx in evictions.into_iter().rev() {
                entries.push(self.entries.remove(idx));
            }
            record_cuda_graph_resident_entries(CudaGraphComponent::Target, self.entries.len());
            release_cuda_graph_entries(entries);
            record_cuda_graph_evictions(
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::SpecStateBudget,
                evicted,
            );
        }
    }

    fn allocate_generation(&mut self) -> u64 {
        NEXT_CUDA_DECODE_GRAPH_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .expect("CUDA decode graph generation overflow")
    }
}

fn drain_lru_entries<T>(entries: &mut Vec<T>, max_entries: usize) -> Vec<T> {
    let count = max_entries.min(entries.len());
    entries.drain(..count).collect()
}

fn release_cuda_graph_entries(entries: Vec<CudaDecodeGraphEntry>) {
    let mut streams = Vec::new();
    for entry in entries {
        let (stream, release_result) = entry.release();
        if let Err(err) = release_result {
            tracing::warn!("Failed to release CUDA graph entry storage: {err:?}");
        }
        if !streams
            .iter()
            .any(|known: &GraphStream| known.cu_device() == stream.cu_device())
        {
            streams.push(stream);
        }
    }
    for stream in streams {
        if let Err(err) = trim_cuda_graph_memory(&stream) {
            tracing::warn!("Failed to trim released CUDA graph memory: {err:?}");
        }
    }
}

/// Free device memory as the driver reports it. On integrated GPUs the
/// MemoryUsage query derives "available" from host RAM times a fraction,
/// which collapses once a large KV pool pins GTT-backed host pages - that
/// pessimism zeroed arenas and blocked precapture entirely on a loaded
/// server. The graph-arena budget is the allocator's own view: the GTT
/// heap free for the hip role, the VRAM free for the cuda role.
pub(crate) fn graph_arena_device_free(device: &Device) -> candle_core::Result<usize> {
    match device {
        // Dual builds serve both roles; the (cuda,not-rocm) gate used to
        // leave dual-build Cuda devices on the usize::MAX fallthrough,
        // sizing arenas without any budget (same class as the topk port).
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let (free, _total) = dev
                .cuda_stream()
                .context()
                .mem_get_info()
                .map_err(candle_core::Error::wrap)?;
            Ok(free)
        }
        #[cfg(feature = "rocm")]
        Device::Hip(dev) => {
            let (free, _total) = dev
                .cuda_stream()
                .context()
                .mem_get_info()
                .map_err(candle_core::Error::wrap)?;
            Ok(free)
        }
        _ => Ok(usize::MAX),
    }
}

// Clamp a computed decode-graph arena to what the device can actually provide.
// A pathological dry-run sum must not turn into an impossible allocation;
// the overflow regrow path sizes back up if the clamp undershoots.
fn clamp_decode_arena_bytes(
    requested_bytes: usize,
    available_bytes: usize,
    device: &Device,
) -> usize {
    requested_bytes.min(
        crate::paged_attention::device_memory_cap(available_bytes, device)
            .max(DECODE_GRAPH_MIN_ARENA_BYTES),
    )
}

fn clamp_decode_arena_to_device(
    device: &Device,
    requested_bytes: usize,
) -> candle_core::Result<usize> {
    let available = graph_arena_device_free(device)?;
    let clamped = clamp_decode_arena_bytes(requested_bytes, available, device);
    if clamped < requested_bytes {
        tracing::debug!(
            requested_bytes,
            clamped_bytes = clamped,
            available_bytes = available,
            "Clamping CUDA decode graph arena to available device memory"
        );
    }
    Ok(clamped)
}

#[cfg(feature = "rocm")]
fn begin_hip_arena_capture(
    stream: &GraphStream,
    device: &candle_core::Device,
    bytes: usize,
) -> candle_core::Result<std::sync::Arc<ArenaSlice<u8>>> {
    graph_role::graph_arena_alloc(device, bytes).and_then(|arena| {
        stream
            .begin_capture_arena(arena.clone())
            .map_err(candle_core::Error::wrap)
            .map(|()| arena)
    })
}

#[cfg(feature = "cuda")]
fn begin_cuda_arena_capture(
    stream: &GraphStream,
    device: &candle_core::Device,
    bytes: usize,
) -> candle_core::Result<std::sync::Arc<crate::pipeline::graph_role::CuStreamSlice>> {
    graph_role::graph_arena_alloc_cuda(device, bytes).and_then(|arena| {
        stream
            .begin_capture_arena_cuda(arena.clone())
            .map_err(candle_core::Error::wrap)
            .map(|()| arena)
    })
}

pub(crate) fn capture_cuda_decode_graph<F>(
    ctx: CudaDecodeGraphCaptureCtx<'_>,
    mut forward: F,
    state: &mut CudaDecodeGraphState,
) -> candle_core::Result<CudaDecodeGraphEntry>
where
    F: FnMut(&Tensor, &PagedAttentionInputMetadata) -> candle_core::Result<Tensor>,
{
    // The whole capture (warmups, admission, the capture itself, and
    // instantiation) runs inside the cross-engine capture window: no other
    // thread's device work can invalidate it, so no retry/fallback storms.
    let _capture_window = enter_cuda_graph_capture_window();
    let CudaDecodeGraphCaptureCtx {
        key,
        input_ids,
        seqlen_offsets,
        position_ids,
        block_size,
        kv_cache,
        metadata,
        model_metadata,
        activation_dtype,
        warmup_logits,
        state_indices,
        real_batch,
        arena_bytes,
    } = ctx;
    let materialized_metadata = metadata
        .materialize_decode_tensors()
        .map_err(candle_core::Error::msg)?;
    let metadata = &materialized_metadata;
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids = Var::from_tensor(input_ids)?;
    let (mut metadata_buffers, metadata) =
        CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata,
            seqlen_offsets,
            position_ids,
            seq_len,
            block_size,
            kv_cache,
            model_metadata,
            activation_dtype,
        })?;
    let graph_input_ids = input_ids.as_detached_tensor();
    // Serving role: the compiled vendor in single-vendor builds, the AMD
    // role in dual builds (this file's kernels are all hipcc-built).
    let stream = GraphStream::from_graph_device(&graph_input_ids.device())
        .map_err(|_| candle_core::Error::msg("CUDA graph decode expected CUDA input ids"))?;
    graph_input_ids.device().synchronize()?;
    let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
    let restore_event_tracking = disable_event_tracking_for_capture(&stream);
    let _htod_cache_guard = graph_role::graph_enable_htod_cache(&graph_input_ids.device())?;

    // ROCm graphs that contain mempool allocation nodes can only be launched
    // once; carve capture-time allocations from a plain pre-allocated arena so
    // the graph replays cleanly. Size it from the warmup's allocation sum.
    tracing::debug!("CUDA decode graph arena is {arena_bytes} bytes");
    // Capture records everything the dry run allocates plus the recorded buffer
    // copies kept live in the graph; the dry run reuses pool memory so it
    // under-measures, and the true need is found by re-capturing on overflow.
    let warmup_bytes = arena_bytes;
    let arena_bytes = state.estimated_arena_bytes(&key.arena_bucket(), warmup_bytes);
    let arena_bytes = match std::env::var("MISTRALRS_DECODE_ARENA_OVERRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        // Explicit override is expert mode: use as-is.
        Some(explicit) => explicit,
        // The dry-run sum times the headroom multiplier can exceed the device;
        // clamp to what is actually available and let the overflow regrow path
        // size back up if the clamp undershoots.
        None => clamp_decode_arena_to_device(graph_input_ids.device(), arena_bytes)?,
    };
    tracing::debug!(
        target: "mistralrs",
        "[cudarc] CAPTURE begin warmup={warmup_bytes} arena={arena_bytes} t={}",
        dbg_nanos()
    );
    #[cfg(feature = "rocm")]
    let mut arena_keepalive: Option<Arc<ArenaSlice<u8>>> = None;
    #[cfg(feature = "cuda")]
    let mut cuda_arena_keepalive: Option<Arc<crate::pipeline::graph_role::CuStreamSlice>> = None;

    // Every role carves capture-time allocations from a pre-allocated
    // arena (graphs with mempool nodes launch once on either driver).
    // Solo shapes have exactly one role linked; dual picks per stream.
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    let use_hip_arena = stream.is_hip();
    #[cfg(not(all(feature = "cuda", feature = "rocm")))]
    let use_hip_arena = cfg!(feature = "rocm");

    let mut begin_err: Option<candle_core::Error> = None;
    #[cfg(feature = "cuda")]
    if !use_hip_arena {
        match begin_cuda_arena_capture(&stream, &graph_input_ids.device(), arena_bytes) {
            Ok(arena) => cuda_arena_keepalive = Some(arena),
            Err(err) => begin_err = Some(err),
        }
    }
    #[cfg(feature = "rocm")]
    if use_hip_arena {
        match begin_hip_arena_capture(&stream, &graph_input_ids.device(), arena_bytes) {
            Ok(arena) => arena_keepalive = Some(arena),
            Err(err) => begin_err = Some(err),
        }
    }
    if let Some(err) = begin_err {
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(
            candle_core::Error::msg(err.to_string()).context("CUDA graph begin capture failed")
        );
    }
    if let Err(err) = prepare_fa3_decode_schedules(&metadata) {
        end_cuda_capture_discard(&stream);
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(err.context("FA3 decode preparation capture failed"));
    }

    let mut logits = None;
    // An attempt that overflows aborts at the first failed allocation, so
    // `consumed` counts only the prefix of the forward that ran - the tail is
    // invisible. The fresh-bucket warmup heuristic can underestimate by 2x or
    // more (hybrid+MTP capture needs ~19x the warmup's raw-alloc bytes), so a
    // fixed 12.5% margin cannot bridge the gap. Grow by max(prefix+margin,
    // 2x current) per retry, bounded by the attempt budget.
    let mut current_arena_bytes = arena_bytes;
    for attempt in 0..DECODE_GRAPH_CAPTURE_ATTEMPTS {
        tracing::info!("[cudarc] CAPTURE forward attempt={attempt}");
        // A panicking forward must still discard the in-flight capture; the caller falls back to eager.
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            forward(&graph_input_ids, &metadata)
        })) {
            Ok(result) => result,
            Err(_) => {
                end_cuda_capture_discard(&stream);
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(
                    candle_core::Error::msg("CUDA graph captured forward panicked")
                        .context("CUDA graph captured forward failed"),
                );
            }
        };
        // The dry run under-measures what capture records (recorded buffers
        // outlive the step plus graph-only copies). Overflowed arena allocs are
        // non-fatal to the forward, so a successful forward can still leave a
        // corrupted graph; check the arena after the fact, not just on error.
        let overflowed = stream.capture_arena_overflowed();
        // info-level (not debug): attempts/retries are rare (only during
        // captures) and this is the only record of which attempt failed
        // and why when a capture dies mid-retry.
        tracing::info!(
            "[cudarc] CAPTURE forward done attempt={attempt} ok={} overflowed={} consumed={} t={}",
            result.is_ok(),
            overflowed,
            stream.capture_arena_consumed(),
            dbg_nanos()
        );

        if overflowed {
            end_cuda_capture_discard(&stream);
            let consumed = stream.capture_arena_consumed();
            let new_bytes = (consumed + consumed / 8 + DECODE_GRAPH_MIN_ARENA_BYTES)
                .max(current_arena_bytes.saturating_mul(2));
            if attempt + 1 == DECODE_GRAPH_CAPTURE_ATTEMPTS {
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(
                    candle_core::Error::msg(format!(
                        "CUDA graph capture arena overflowed on retry (needed > {new_bytes} bytes, consumed prefix {consumed})"
                    ))
                    .context("CUDA graph captured forward failed"),
                );
            }
            tracing::info!(
                "CUDA decode graph arena grew {current_arena_bytes} -> {new_bytes} bytes (attempt {attempt}, consumed prefix {consumed})"
            );
            graph_input_ids.device().synchronize()?;
            // Release the overflowed arena BEFORE allocating its 2x
            // replacement: on a tight device the regrow cannot fit
            // alongside the arena it replaces (observed: 71MB regrow
            // against ~50MB free with the 32MB overflowed arena still
            // held). The discard above detached it from the stream and
            // the synchronize drained in-flight use, so the free is safe.
            #[cfg(feature = "cuda")]
            if !use_hip_arena {
                cuda_arena_keepalive = None;
                let arena = begin_cuda_arena_capture(&stream, &graph_input_ids.device(), new_bytes)
                    .map_err(|err| err.context("CUDA graph capture arena regrow failed"))?;
                cuda_arena_keepalive = Some(arena);
            }
            #[cfg(feature = "rocm")]
            if use_hip_arena {
                arena_keepalive = None;
                let arena = begin_hip_arena_capture(&stream, &graph_input_ids.device(), new_bytes)
                    .map_err(|err| err.context("CUDA graph capture arena regrow failed"))?;
                arena_keepalive = Some(arena);
            }
            current_arena_bytes = new_bytes;
            continue;
        }
        match result {
            Ok(logits_out) => {
                logits = Some(logits_out);
                break;
            }
            Err(err) => {
                // Discard before unwinding: the pool-guard drop and the
                // caller's checkpoint rollback both upload on this stream,
                // which is illegal while a poisoned capture is still open
                // (aborted the process via a slice-clone unwrap, 2026-10-07).
                end_cuda_capture_discard(&stream);
                restore_event_tracking_after_capture(&stream, restore_event_tracking);
                return Err(err.context("CUDA graph captured forward failed"));
            }
        }
    }
    let logits = logits.expect("capture forward attempted");
    if logits.shape() != warmup_logits.shape()
        || logits.dtype() != warmup_logits.dtype()
        || logits.device().location() != warmup_logits.device().location()
        || !logits.is_contiguous()
    {
        end_cuda_capture_discard(&stream);
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        return Err(candle_core::Error::msg(
            "captured CUDA graph logits do not match the contiguous warmup output",
        ));
    }

    // Read consumption before end_capture resets the bookkeeping, and remember
    // it so the next capture of this bucket sizes from measurement.
    let captured_consumed_bytes = stream.capture_arena_consumed();
    state.record_arena_observation(key.arena_bucket(), captured_consumed_bytes);

    #[cfg_attr(feature = "rocm", allow(unused_mut))]
    let mut graph = match CudaGraphHandle::end_capture(&stream) {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(candle_core::Error::msg(
                "CUDA graph capture returned no graph",
            ));
        }
        Err(err) => {
            restore_event_tracking_after_capture(&stream, restore_event_tracking);
            return Err(err);
        }
    };
    restore_event_tracking_after_capture(&stream, restore_event_tracking);

    #[cfg(feature = "rocm")]
    graph.set_arena(arena_keepalive.take());
    #[cfg(feature = "cuda")]
    if !use_hip_arena {
        graph.set_cuda_arena(cuda_arena_keepalive.take());
    }

    graph.upload()?;
    metadata_buffers.finish_capture(&metadata);
    let host_staging = CudaGraphHostStaging::new(graph.stream.clone())?;
    tracing::debug!(
        "Captured CUDA decode graph: batch bucket {batch} ({real_batch} live rows), {seq_len} query tokens"
    );

    // Pointer-correlation census (MISTRALRS_GRAPH_PTR_CENSUS only):
    // record the capture-time VAs of every graph-managed buffer AFTER
    // finish_capture (these are the replay VAs), plus the flashinfer view
    // pre-images the captured forward read.
    let mut ptr_census: Option<Vec<GraphPtrRecord>> = None;
    if graph_ptr_census_enabled() {
        let live =
            decode_entry_census_tensors(&input_ids, &metadata_buffers, &state_indices, &logits);
        let mut records = Vec::with_capacity(live.len());
        for (name, device, tensor) in &live {
            match graph_tensor_va(tensor) {
                Ok(Some((ptr, bytes))) => {
                    graph_ptr_census_record(name, *device, ptr, bytes);
                    records.push(GraphPtrRecord {
                        name,
                        device: *device,
                        ptr,
                        bytes,
                    });
                }
                Ok(None) => {
                    eprintln!("[graph-ptr-census] {name} {device:?}: untracked storage")
                }
                Err(err) => {
                    eprintln!("[graph-ptr-census] {name} {device:?}: VA read failed: {err}")
                }
            }
        }
        // U32 content at capture time: the baseline the replay dumps compare.
        graph_ptr_census_dump_content("capture", &live);
        // Tier 2 (handler correlation only): the flashinfer view pre-images.
        let paged = flashinfer_paged_view(&metadata);
        let full = flashinfer_full_view(&metadata);
        graph_ptr_census_record_tensor_map(
            "view_paged_kv_indptr",
            paged.map(|view| &view.paged_kv.indptr),
        );
        graph_ptr_census_record_tensor_map(
            "view_paged_kv_indices",
            paged.map(|view| &view.paged_kv.indices),
        );
        graph_ptr_census_record_tensor_map(
            "view_paged_kv_last_page_len",
            paged.map(|view| &view.paged_kv.last_page_len),
        );
        graph_ptr_census_record_tensor_map(
            "view_full_paged_kv_indptr",
            full.map(|view| &view.paged_kv.indptr),
        );
        graph_ptr_census_record_tensor_map(
            "view_full_paged_kv_indices",
            full.map(|view| &view.paged_kv.indices),
        );
        graph_ptr_census_record_tensor_map(
            "view_full_paged_kv_last_page_len",
            full.map(|view| &view.paged_kv.last_page_len),
        );
        graph_ptr_census_record_tensor_map(
            "view_paged_kv_request_indices",
            paged.map(|view| &view.tile_plan.request_indices),
        );
        graph_ptr_census_record_tensor_map(
            "view_paged_kv_tile_indices",
            paged.map(|view| &view.tile_plan.kv_tile_indices),
        );
        graph_ptr_census_record_tensor_map(
            "view_paged_kv_o_indptr",
            paged.map(|view| &view.tile_plan.o_indptr),
        );
        graph_ptr_census_record_tensor_map(
            "view_paged_kv_chunk_size",
            paged.map(|view| &view.tile_plan.kv_chunk_size),
        );
        graph_ptr_census_record_tensor_map(
            "view_paged_kv_block_valid_mask",
            paged.map(|view| &view.tile_plan.block_valid_mask),
        );
        graph_ptr_census_record_tensor_map(
            "view_full_paged_kv_request_indices",
            full.map(|view| &view.tile_plan.request_indices),
        );
        graph_ptr_census_record_tensor_map(
            "view_full_paged_kv_tile_indices",
            full.map(|view| &view.tile_plan.kv_tile_indices),
        );
        graph_ptr_census_record_tensor_map(
            "view_full_paged_kv_o_indptr",
            full.map(|view| &view.tile_plan.o_indptr),
        );
        graph_ptr_census_record_tensor_map(
            "view_full_paged_kv_chunk_size",
            full.map(|view| &view.tile_plan.kv_chunk_size),
        );
        graph_ptr_census_record_tensor_map(
            "view_full_paged_kv_block_valid_mask",
            full.map(|view| &view.tile_plan.block_valid_mask),
        );
        ptr_census = Some(records);
    }

    Ok(CudaDecodeGraphEntry {
        generation: 0,
        replay_epoch: 0,
        key,
        host_staging,
        input_ids,
        metadata_buffers,
        state_indices,
        _metadata: metadata,
        logits,
        spec_state: None,
        spec_state_usage: CudaGraphSpecStateUsage::default(),
        ptr_census,
        graph,
    })
}

pub(crate) fn cuda_decode_graphs_enabled() -> bool {
    crate::perf_flags::cuda_graphs_enabled()
}

pub(crate) fn cuda_decode_graph_batch_kind_supported(kind: RecurrentBatchKind) -> bool {
    matches!(
        kind,
        RecurrentBatchKind::Decode | RecurrentBatchKind::SpeculativeDecode
    )
}

pub(crate) fn cuda_decode_graph_supported_for_model(
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
) -> bool {
    let Some(metadata) = model_metadata else {
        return false;
    };
    #[cfg(target_family = "unix")]
    {
        (0..metadata.num_layers()).all(|layer_idx| {
            !DecodePlan::requires_host_context_lengths(
                metadata.attention_backend_kind_for_layer(layer_idx),
                metadata.k_head_dim_for_layer(layer_idx),
            )
        })
    }
    #[cfg(not(target_family = "unix"))]
    {
        (0..metadata.num_layers()).all(|layer_idx| {
            !matches!(
                metadata.attention_backend_kind_for_layer(layer_idx),
                AttentionBackendKind::FlashInfer
            )
        })
    }
}

#[must_use]
pub(crate) struct CudaGraphMemoryPoolGuard {
    stream: GraphStream,
    pool: Option<usize>,
}

impl Drop for CudaGraphMemoryPoolGuard {
    fn drop(&mut self) {
        let Some(pool) = self.pool else { return };
        if let Err(err) = self.stream.bind_context_to_thread() {
            tracing::warn!("Failed to bind CUDA context while restoring graph memory pool: {err}");
        }
        let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
        let mut scopes = scopes
            .lock()
            .expect("CUDA graph memory pool scopes poisoned");
        let Some(scope) = scopes.get_mut(&pool) else {
            tracing::warn!("CUDA graph memory pool scope disappeared before restoration");
            return;
        };
        scope.guards = scope
            .guards
            .checked_sub(1)
            .expect("CUDA graph memory pool guard underflow");
        if scope.guards != 0 {
            return;
        }
        let release_threshold = scope.release_threshold;
        let pool_ptr = pool as *mut core::ffi::c_void;
        if let Err(err) =
            set_memory_pool_release_threshold(&self.stream, pool_ptr, release_threshold)
        {
            tracing::warn!("Failed to restore CUDA graph memory pool threshold: {err:?}");
        }
        let _ = self.stream.synchronize();
        // A/B probe: skip the post-capture trim. If the captured graph
        // references pool memory the Rust side freed (a lifetime bug
        // upstream of the driver), the trim unmaps it and the first
        // replay faults with ILLEGAL_ADDRESS.
        if std::env::var("MISTRALRS_NO_POST_CAPTURE_TRIM").is_ok() {
            tracing::warn!("skipping post-capture graph memory trim (A/B probe)");
        } else if let Err(err) = trim_cuda_graph_memory_bound(&self.stream) {
            tracing::warn!("Failed to trim CUDA graph memory after capture: {err:?}");
        }
        scopes.remove(&pool);
    }
}

fn memory_pool_release_threshold(
    stream: &GraphStream,
    pool: *mut core::ffi::c_void,
) -> candle_core::Result<u64> {
    graph_role::graph_mempool_release_threshold_get(stream, pool)
        .map_err(|err| err.context("CUDA graph mempool release threshold lookup failed"))
}

fn set_memory_pool_release_threshold(
    stream: &GraphStream,
    pool: *mut core::ffi::c_void,
    value: u64,
) -> candle_core::Result<()> {
    graph_role::graph_mempool_release_threshold_set(stream, pool, value)
        .map_err(|err| err.context("CUDA graph mempool release threshold setup failed"))
}

fn cuda_memory_pool(stream: &GraphStream) -> candle_core::Result<*mut core::ffi::c_void> {
    graph_role::graph_device_mempool(stream)
        .map_err(|err| err.context("CUDA graph mempool lookup failed"))
}

pub(crate) fn cuda_graph_memory_pool_scope_active(device: &Device) -> candle_core::Result<bool> {
    let Ok(stream) = GraphStream::from_graph_device(device) else {
        return Ok(false);
    };
    if !stream.has_async_alloc() {
        return Ok(false);
    }
    let pool = cuda_memory_pool(&stream)? as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    Ok(scopes.contains_key(&pool))
}

fn trim_cuda_graph_memory_bound(stream: &GraphStream) -> candle_core::Result<()> {
    graph_role::graph_device_mem_trim(stream)
        .map_err(|err| err.context("CUDA graph memory trim failed"))
}

#[cfg(all(test, feature = "cuda"))]
fn cuda_graph_memory_attribute(
    stream: &GraphStream,
    attribute: candle_core::cuda_backend::cudarc::driver::sys::CUgraphMem_attribute,
) -> candle_core::Result<usize> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    stream.bind_context_to_thread()?;
    let mut value = 0usize;
    let result = unsafe {
        sys::cuDeviceGetGraphMemAttribute(
            stream.cu_device(),
            attribute,
            (&mut value as *mut usize).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(candle_core::Error::msg(format!("{result:?}"))
            .context("CUDA graph memory attribute lookup failed"));
    }
    Ok(value)
}

pub(crate) fn trim_cuda_graph_memory(stream: &GraphStream) -> candle_core::Result<()> {
    stream
        .bind_context_to_thread()
        .map_err(|err| err.context("CUDA graph memory trim context bind failed"))?;
    let pool = cuda_memory_pool(stream)? as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    if scopes.contains_key(&pool) {
        return Ok(());
    }
    stream
        .synchronize()
        .map_err(candle_core::Error::wrap)
        .map_err(|err| err.context("CUDA graph memory trim synchronization failed"))?;
    trim_cuda_graph_memory_bound(stream)
}

pub(crate) fn prepare_cuda_graph_memory_pool(
    stream: &GraphStream,
) -> candle_core::Result<CudaGraphMemoryPoolGuard> {
    if !stream.has_async_alloc() {
        return Ok(CudaGraphMemoryPoolGuard {
            stream: stream.clone(),
            pool: None,
        });
    }

    let pool = cuda_memory_pool(stream)?;
    let pool_key = pool as usize;
    let scopes = CUDA_GRAPH_MEMORY_POOL_SCOPES.get_or_init(Default::default);
    let mut scopes = scopes
        .lock()
        .expect("CUDA graph memory pool scopes poisoned");
    if let Some(scope) = scopes.get_mut(&pool_key) {
        scope.guards = scope
            .guards
            .checked_add(1)
            .expect("CUDA graph memory pool guard overflow");
    } else {
        let release_threshold = memory_pool_release_threshold(stream, pool)?;
        set_memory_pool_release_threshold(stream, pool, u64::MAX)?;
        scopes.insert(
            pool_key,
            MemoryPoolScopeState {
                guards: 1,
                release_threshold,
            },
        );
    }
    drop(scopes);

    let guard = CudaGraphMemoryPoolGuard {
        stream: stream.clone(),
        pool: Some(pool_key),
    };
    graph_role::graph_mempool_reuse_enable(stream, pool)
        .map_err(|err| err.context("CUDA graph mempool reuse setup failed"))?;

    Ok(guard)
}

fn flashinfer_decode_scratch_maps(
    metadata: &PagedAttentionInputMetadata,
    batch: usize,
    kv_cache: &[(Tensor, Tensor)],
    model_metadata: Option<&(dyn ModelConfigLike + Send + Sync)>,
    activation_dtype: DType,
) -> candle_core::Result<FlashInferDecodeScratchMaps> {
    let Some(model_metadata) = model_metadata else {
        return Ok((None, None));
    };
    let split_rows = flashinfer_split_rows(metadata, batch)?;
    if split_rows.is_empty() {
        return Ok((None, None));
    }

    let mut specs: HashMap<DeviceLocation, (Device, DType, usize, usize)> = HashMap::new();
    let layer_count = model_metadata.num_layers().min(kv_cache.len());
    for (layer_idx, (key_cache, value_cache)) in kv_cache.iter().enumerate().take(layer_count) {
        if model_metadata.attention_backend_kind_for_layer(layer_idx)
            != AttentionBackendKind::FlashInfer
        {
            continue;
        }
        let location = key_cache.device().location();
        if !split_rows.contains_key(&location) {
            continue;
        }
        if key_cache.dtype() != value_cache.dtype() {
            candle_core::bail!("FlashInfer graph scratch expects matching KV cache dtypes");
        }
        let (_, _, _, head_dim) = key_cache.dims4()?;
        let num_qo_heads = model_metadata.num_attn_heads_for_layer(layer_idx);
        let entry = specs.entry(location).or_insert((
            key_cache.device().clone(),
            activation_dtype,
            num_qo_heads,
            head_dim,
        ));
        if entry.1 != activation_dtype {
            candle_core::bail!("FlashInfer graph scratch expects one activation dtype per device");
        }
        entry.2 = entry.2.max(num_qo_heads);
        entry.3 = entry.3.max(head_dim);
    }

    let mut tmp_v = HashMap::new();
    let mut tmp_s = HashMap::new();
    for (location, rows) in split_rows {
        let Some((device, dtype, num_qo_heads, head_dim)) = specs.get(&location) else {
            continue;
        };
        tmp_v.insert(location, unsafe {
            Tensor::empty((rows, *num_qo_heads, *head_dim), *dtype, device)?
        });
        tmp_s.insert(location, unsafe {
            Tensor::empty((rows, *num_qo_heads), DType::F32, device)?
        });
    }

    if tmp_v.is_empty() {
        Ok((None, None))
    } else {
        Ok((Some(tmp_v), Some(tmp_s)))
    }
}

fn flashinfer_split_rows(
    metadata: &PagedAttentionInputMetadata,
    batch: usize,
) -> candle_core::Result<HashMap<DeviceLocation, usize>> {
    let mut rows = HashMap::new();
    collect_flashinfer_split_rows(
        flashinfer_paged_view(metadata).map(|view| &view.tile_plan.request_indices),
        batch,
        &mut rows,
    )?;
    collect_flashinfer_split_rows(
        flashinfer_full_view(metadata).map(|view| &view.tile_plan.request_indices),
        batch,
        &mut rows,
    )?;
    Ok(rows)
}

pub(crate) fn disable_event_tracking_for_capture(stream: &GraphStream) -> bool {
    let restore = stream.is_event_tracking();
    if restore {
        unsafe { stream.disable_event_tracking() };
    }
    restore
}

pub(crate) fn restore_event_tracking_after_capture(stream: &GraphStream, restore: bool) {
    if restore {
        unsafe { stream.enable_event_tracking() };
    }
}

pub(crate) fn end_cuda_capture_discard(stream: &GraphStream) {
    if stream.capture_status_none().unwrap_or(true) {
        return;
    }
    let mut graph = std::ptr::null_mut();
    if graph_end_capture(stream, &mut graph).is_err() {
        // a failed abort leaves the capture half-open: drain in-flight
        // work, retry once, then force-reset the bookkeeping either way
        let _ = stream.synchronize();
        let _ = graph_end_capture(stream, &mut graph);
    }
    if !graph.is_null() {
        let _ = graph_destroy(stream, graph);
    }
    stream.reset_capture_bookkeeping();
}

fn device_location_sort_key(location: &DeviceLocation) -> (u8, usize) {
    match location {
        DeviceLocation::Cpu => (0, 0),
        DeviceLocation::Cuda { gpu_id } => (1, *gpu_id),
        DeviceLocation::Metal { gpu_id } => (2, *gpu_id),
        #[cfg(feature = "rocm")]
        DeviceLocation::Hip { gpu_id } => (1, *gpu_id),
    }
}

fn push_graph_tensor_keys(
    name: &'static str,
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    keys: &mut Vec<CudaGraphTensorKey>,
) {
    if let Some(map) = map {
        keys.extend(map.iter().map(|(location, tensor)| CudaGraphTensorKey {
            name,
            location: *location,
            shape: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
        }));
    }
}

fn push_flashinfer_graph_tensor_keys(
    metadata: &PagedAttentionInputMetadata,
    keys: &mut Vec<CudaGraphTensorKey>,
) {
    let paged = flashinfer_paged_view(metadata);
    let full = flashinfer_full_view(metadata);
    push_graph_tensor_keys(
        "paged_kv_indptr",
        paged.map(|view| &view.paged_kv.indptr),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_indices",
        paged.map(|view| &view.paged_kv.indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_last_page_len",
        paged.map(|view| &view.paged_kv.last_page_len),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_indptr",
        full.map(|view| &view.paged_kv.indptr),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_indices",
        full.map(|view| &view.paged_kv.indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_last_page_len",
        full.map(|view| &view.paged_kv.last_page_len),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_request_indices",
        paged.map(|view| &view.tile_plan.request_indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_tile_indices",
        paged.map(|view| &view.tile_plan.kv_tile_indices),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_o_indptr",
        paged.map(|view| &view.tile_plan.o_indptr),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_chunk_size",
        paged.map(|view| &view.tile_plan.kv_chunk_size),
        keys,
    );
    push_graph_tensor_keys(
        "paged_kv_block_valid_mask",
        paged.map(|view| &view.tile_plan.block_valid_mask),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_request_indices",
        full.map(|view| &view.tile_plan.request_indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_tile_indices",
        full.map(|view| &view.tile_plan.kv_tile_indices),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_o_indptr",
        full.map(|view| &view.tile_plan.o_indptr),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_chunk_size",
        full.map(|view| &view.tile_plan.kv_chunk_size),
        keys,
    );
    push_graph_tensor_keys(
        "full_paged_kv_block_valid_mask",
        full.map(|view| &view.tile_plan.block_valid_mask),
        keys,
    );
}

fn flashinfer_paged_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    let views = &metadata.flashinfer.as_ref()?.views;
    Some(views.sliding.as_ref().unwrap_or(&views.logical))
}

fn flashinfer_full_view(
    metadata: &PagedAttentionInputMetadata,
) -> Option<&FlashInferPagedAttentionView> {
    Some(&metadata.flashinfer.as_ref()?.views.logical)
}

fn flashinfer_views_alias(metadata: &PagedAttentionInputMetadata) -> bool {
    metadata
        .flashinfer
        .as_ref()
        .is_some_and(|flashinfer| flashinfer.views.sliding.is_none())
}

fn flashinfer_paged_kv_from_vars(
    indptr: &Option<CudaGraphVarMap>,
    indices: &Option<CudaGraphVarMap>,
    last_page_len: &Option<CudaGraphVarMap>,
) -> Option<FlashInferPagedKv> {
    Some(FlashInferPagedKv {
        indptr: option_tensor_map_from_var_map(indptr)?,
        indices: option_tensor_map_from_var_map(indices)?,
        last_page_len: option_tensor_map_from_var_map(last_page_len)?,
    })
}

fn flashinfer_tile_plan_from_vars(
    request_indices: &Option<CudaGraphVarMap>,
    kv_tile_indices: &Option<CudaGraphVarMap>,
    o_indptr: &Option<CudaGraphVarMap>,
    kv_chunk_size: &Option<CudaGraphVarMap>,
    block_valid_mask: &Option<CudaGraphVarMap>,
) -> Option<FlashInferTilePlan> {
    Some(FlashInferTilePlan {
        request_indices: option_tensor_map_from_var_map(request_indices)?,
        kv_tile_indices: option_tensor_map_from_var_map(kv_tile_indices)?,
        o_indptr: option_tensor_map_from_var_map(o_indptr)?,
        kv_chunk_size: option_tensor_map_from_var_map(kv_chunk_size)?,
        block_valid_mask: option_tensor_map_from_var_map(block_valid_mask)?,
    })
}

fn collect_flashinfer_split_rows(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    batch: usize,
    split_rows: &mut HashMap<DeviceLocation, usize>,
) -> candle_core::Result<()> {
    let Some(map) = map else {
        return Ok(());
    };
    for (location, tensor) in map {
        let rows = tensor.dims1()?;
        if rows > batch {
            split_rows
                .entry(*location)
                .and_modify(|current| *current = (*current).max(rows))
                .or_insert(rows);
        }
    }
    Ok(())
}

fn bucket_context_len_from_vars(map: &Option<CudaGraphVarMap>, block_size: usize) -> Option<usize> {
    map.as_ref()
        .and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

fn bucket_context_len(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    block_size: usize,
) -> Option<usize> {
    map.and_then(|map| map.values().next())
        .and_then(|tensor| tensor.dims().last().copied())
        .map(|blocks| blocks * block_size)
}

fn graph_context_len(actual: Option<usize>, capacity: Option<usize>) -> Option<usize> {
    match (actual, capacity) {
        (Some(actual), Some(capacity)) => Some(
            actual
                .div_ceil(PAGED_ATTENTION_PARTITION_SIZE)
                .max(1)
                .saturating_mul(PAGED_ATTENTION_PARTITION_SIZE)
                .min(capacity),
        ),
        (Some(actual), None) => Some(actual),
        (None, capacity) => capacity,
    }
}

fn var_map_from_tensor_map(
    map: &HashMap<DeviceLocation, Tensor>,
) -> candle_core::Result<CudaGraphVarMap> {
    map.iter()
        .map(|(location, tensor)| Ok((*location, Var::from_tensor(tensor)?)))
        .collect()
}

fn option_var_map_from_tensor_map(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
) -> candle_core::Result<Option<CudaGraphVarMap>> {
    map.map(var_map_from_tensor_map).transpose()
}

fn option_var_map_from_tensor_map_if_distinct(
    map: Option<&HashMap<DeviceLocation, Tensor>>,
    aliases_existing: bool,
) -> candle_core::Result<Option<CudaGraphVarMap>> {
    if aliases_existing {
        Ok(None)
    } else {
        option_var_map_from_tensor_map(map)
    }
}

fn tensor_map_from_var_map(map: &CudaGraphVarMap) -> HashMap<DeviceLocation, Tensor> {
    map.iter()
        .map(|(location, var)| (*location, var.as_detached_tensor()))
        .collect()
}

fn option_tensor_map_from_var_map(
    map: &Option<CudaGraphVarMap>,
) -> Option<HashMap<DeviceLocation, Tensor>> {
    map.as_ref().map(tensor_map_from_var_map)
}

impl CudaGraphPinnedBuffer {
    fn new(dst: &Var, stream: &GraphStream) -> candle_core::Result<Self> {
        let len = dst.elem_count();
        macro_rules! allocate {
            ($variant:ident, $ty:ty) => {{
                let mut allocation = graph_alloc_pinned::<$ty>(stream, len)?;
                let ptr = NonNull::new(unsafe {
                    allocation.as_mut_ptr().map_err(candle_core::Error::wrap)?
                })
                .ok_or_else(|| candle_core::Error::msg("CUDA returned a null pinned pointer"))?;
                CudaGraphPinnedData::$variant(CudaGraphPinnedAllocation { allocation, ptr })
            }};
        }
        let data = match dst.dtype() {
            DType::U8 => allocate!(U8, u8),
            DType::U32 => allocate!(U32, u32),
            DType::I32 => allocate!(I32, i32),
            DType::I64 => allocate!(I64, i64),
            DType::F32 => allocate!(F32, f32),
            dtype => candle_core::bail!(
                "CUDA graph host staging does not support metadata dtype {dtype:?}"
            ),
        };
        Ok(Self {
            data,
            initialized: false,
        })
    }

    fn copy_from(
        &mut self,
        src: &Tensor,
        dst: &Var,
        stream: &GraphStream,
    ) -> candle_core::Result<()> {
        if src.shape() != dst.shape() || src.dtype() != dst.dtype() {
            candle_core::bail!("CUDA graph host staging expected matching tensors");
        }
        let (src_storage, src_layout) = src.storage_and_layout();
        let Storage::Cpu(src_storage) = &*src_storage else {
            candle_core::bail!("CUDA graph host staging expected CPU source metadata");
        };
        if !src_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous source metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous destination metadata");
        }
        let len = src.elem_count();
        let src_offset = src_layout.start_offset();
        let dst_offset = dst_layout.start_offset();

        macro_rules! stage_and_copy {
            ($variant:ident, $ty:ty) => {{
                let CudaGraphPinnedData::$variant(host) = &mut self.data else {
                    candle_core::bail!("CUDA graph host staging dtype changed");
                };
                let src = src_storage.as_slice::<$ty>()?;
                let src = &src[src_offset..src_offset + len];
                let host_slice = host.as_mut_slice();
                if self.initialized && host_slice == src {
                    return Ok(());
                }
                host_slice.copy_from_slice(src);
                self.initialized = true;
                let (dst_ptr, _dst_guard) =
                    graph_stage_ptr::<$ty>(&dst_storage, dst_offset, stream)?;
                graph_memcpy_htod_async(
                    stream,
                    dst_ptr,
                    host.as_ptr().cast(),
                    len * std::mem::size_of::<$ty>(),
                )
                .map_err(|err| err.context("CUDA graph metadata H2D copy failed"))?;
            }};
        }

        match src.dtype() {
            DType::U8 => stage_and_copy!(U8, u8),
            DType::U32 => stage_and_copy!(U32, u32),
            DType::I32 => stage_and_copy!(I32, i32),
            DType::I64 => stage_and_copy!(I64, i64),
            DType::F32 => stage_and_copy!(F32, f32),
            dtype => candle_core::bail!(
                "CUDA graph host staging does not support metadata dtype {dtype:?}"
            ),
        }
        Ok(())
    }

    fn copy_from_u32_slice(
        &mut self,
        src: &[u32],
        dst: &Var,
        stream: &GraphStream,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::U32 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching u32 state indices");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous state indices");
        }
        let CudaGraphPinnedData::U32(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging state index dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let (dst_ptr, _dst_guard) =
            graph_stage_ptr::<u32>(&dst_storage, dst_layout.start_offset(), stream)?;
        graph_memcpy_htod_async(
            stream,
            dst_ptr,
            host.as_ptr().cast(),
            std::mem::size_of_val(src),
        )
        .map_err(|err| err.context("CUDA graph state index H2D copy failed"))?;
        Ok(())
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    fn copy_from_f32_slice(
        &mut self,
        src: &[f32],
        dst: &Var,
        stream: &GraphStream,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::F32 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching f32 metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous f32 metadata");
        }
        let CudaGraphPinnedData::F32(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging f32 metadata dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let (dst_ptr, _dst_guard) =
            graph_stage_ptr::<f32>(&dst_storage, dst_layout.start_offset(), stream)?;
        graph_memcpy_htod_async(
            stream,
            dst_ptr,
            host.as_ptr().cast(),
            std::mem::size_of_val(src),
        )
        .map_err(|err| err.context("CUDA graph f32 metadata H2D copy failed"))?;
        Ok(())
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    fn copy_from_i64_slice(
        &mut self,
        src: &[i64],
        dst: &Var,
        stream: &GraphStream,
    ) -> candle_core::Result<()> {
        if dst.dtype() != DType::I64 || dst.elem_count() != src.len() {
            candle_core::bail!("CUDA graph host staging expected matching i64 metadata");
        }
        let (dst_storage, dst_layout) = dst.storage_and_layout();
        if !dst_layout.is_contiguous() {
            candle_core::bail!("CUDA graph host staging expected contiguous i64 metadata");
        }
        let CudaGraphPinnedData::I64(host) = &mut self.data else {
            candle_core::bail!("CUDA graph host staging i64 metadata dtype changed");
        };
        let host_slice = host.as_mut_slice();
        if self.initialized && host_slice == src {
            return Ok(());
        }
        host_slice.copy_from_slice(src);
        self.initialized = true;
        let (dst_ptr, _dst_guard) =
            graph_stage_ptr::<i64>(&dst_storage, dst_layout.start_offset(), stream)?;
        graph_memcpy_htod_async(
            stream,
            dst_ptr,
            host.as_ptr().cast(),
            std::mem::size_of_val(src),
        )
        .map_err(|err| err.context("CUDA graph i64 metadata H2D copy failed"))?;
        Ok(())
    }
}

impl CudaGraphHostStaging {
    pub(crate) fn new(graph_stream: GraphStream) -> candle_core::Result<Self> {
        let graph_complete = graph_stream.new_graph_event()?;
        Ok(Self {
            buffers: HashMap::new(),
            completions: HashMap::new(),
            graph_complete,
            graph_stream,
            graph_pending: false,
        })
    }

    pub(crate) fn update(
        &mut self,
        copy: impl FnOnce(&mut Self) -> candle_core::Result<()>,
    ) -> candle_core::Result<()> {
        self.begin_update()?;
        let copy_result = copy(self);
        let finish_result = self.finish_update();
        copy_result.and(finish_result)
    }

    fn begin_update(&mut self) -> candle_core::Result<()> {
        for completion in self.completions.values_mut() {
            completion.ordered_after_graph = false;
            if completion.active {
                completion
                    .stream
                    .synchronize()
                    .map_err(candle_core::Error::wrap)?;
                completion.active = false;
            }
            if completion.pending {
                completion
                    .event
                    .synchronize()
                    .map_err(candle_core::Error::wrap)?;
                completion.pending = false;
            }
        }
        Ok(())
    }

    fn finish_update(&mut self) -> candle_core::Result<()> {
        let mut result = Ok(());
        for completion in self.completions.values_mut() {
            if !completion.active {
                continue;
            }
            match completion.event.record(&completion.stream) {
                Ok(()) => {
                    completion.pending = true;
                    completion.active = false;
                }
                Err(err) => {
                    completion.pending = false;
                    if completion.stream.synchronize().is_ok() {
                        completion.active = false;
                    }
                    if result.is_ok() {
                        result = Err(candle_core::Error::wrap(err));
                    }
                }
            }
        }
        result
    }

    pub(crate) fn order_before_graph(&self) -> candle_core::Result<()> {
        for completion in self.completions.values() {
            if completion.pending && !same_cuda_stream(&completion.stream, &self.graph_stream) {
                self.graph_stream.wait_on_event(&completion.event)?;
            }
        }
        Ok(())
    }

    pub(crate) fn record_graph_complete(&mut self) -> candle_core::Result<()> {
        self.graph_complete
            .record(&self.graph_stream)
            .map_err(candle_core::Error::wrap)?;
        self.graph_pending = true;
        Ok(())
    }

    fn prepare_copy(
        &mut self,
        location: DeviceLocation,
        stream: &GraphStream,
    ) -> candle_core::Result<()> {
        let completion = match self.completions.entry(location) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let event = stream.new_blocking_event()?;
                entry.insert(CudaGraphCopyCompletion {
                    event,
                    stream: stream.clone(),
                    pending: false,
                    active: false,
                    ordered_after_graph: false,
                })
            }
        };
        if !same_cuda_stream(&completion.stream, stream) {
            candle_core::bail!("CUDA graph metadata stream changed during replay");
        }
        if self.graph_pending && !completion.ordered_after_graph {
            if !same_cuda_stream(&completion.stream, &self.graph_stream) {
                completion.stream.wait_on_event(&self.graph_complete)?;
            }
            completion.ordered_after_graph = true;
        }
        completion.active = true;
        Ok(())
    }

    fn copy_from(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &Tensor,
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = GraphStream::from_graph_device(&dst.device())?;
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst, &stream)?)
            }
        };
        buffer.copy_from(src, dst, &stream)
    }

    pub(crate) fn copy_from_u32_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[u32],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = GraphStream::from_graph_device(&dst.device())?;
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst, &stream)?)
            }
        };
        buffer.copy_from_u32_slice(src, dst, &stream)
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn copy_from_f32_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[f32],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = GraphStream::from_graph_device(&dst.device())?;
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst, &stream)?)
            }
        };
        buffer.copy_from_f32_slice(src, dst, &stream)
    }

    #[cfg(all(feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn copy_from_i64_slice(
        &mut self,
        name: &'static str,
        location: DeviceLocation,
        src: &[i64],
        dst: &Var,
    ) -> candle_core::Result<()> {
        let stream = GraphStream::from_graph_device(&dst.device())?;
        self.prepare_copy(location, &stream)?;
        let buffer = match self.buffers.entry((name, location)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CudaGraphPinnedBuffer::new(dst, &stream)?)
            }
        };
        buffer.copy_from_i64_slice(src, dst, &stream)
    }
}

/// Devices whose decode graphs are capturable: the hip role through the
/// fork's arena capture API. The NVIDIA role stays parked in dual builds:
/// capture is fixed and replays cleanly in-request, but the exec's first
/// cross-request launch fails with INVALID_VALUE (launches 1..N-1 clean);
/// tracked as its own slice - see the journal.
/// Devices whose decode graphs are capturable: the hip role through the
/// fork's arena capture API, and the NVIDIA role through the cudarc-nvidia
/// companion fork (arena plus capture-aware event guards; see journal).
#[cfg(all(feature = "cuda", feature = "rocm"))]
pub(crate) fn is_cuda_graph_device(device: &candle_core::Device) -> bool {
    device.is_hip() || device.is_cuda()
}
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
pub(crate) fn is_cuda_graph_device(device: &candle_core::Device) -> bool {
    device.is_cuda()
}
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub(crate) fn is_cuda_graph_device(device: &candle_core::Device) -> bool {
    device.is_hip()
}

fn copy_var_map(
    dst: &CudaGraphVarMap,
    src: &HashMap<DeviceLocation, Tensor>,
    name: &'static str,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    if dst.len() != src.len() {
        candle_core::bail!("{name} device count changed during CUDA graph replay");
    }
    for (location, dst) in dst {
        let src = src
            .get(location)
            .ok_or_else(|| candle_core::Error::msg(format!("{name} missing {location:?}")))?;
        if src.device().is_cpu() && is_cuda_graph_device(&dst.device()) {
            host_staging.copy_from(name, *location, src, dst)?;
        } else {
            dst.set(src)?;
        }
    }
    Ok(())
}

fn copy_option_var_map(
    dst: &Option<CudaGraphVarMap>,
    src: Option<&HashMap<DeviceLocation, Tensor>>,
    name: &'static str,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    match (dst, src) {
        (Some(dst), Some(src)) => copy_var_map(dst, src, name, host_staging),
        (None, None) => Ok(()),
        _ => candle_core::bail!("{name} changed optional state during CUDA graph replay"),
    }
}

struct FlashInferTilePlanVars<'a> {
    request_indices: &'a Option<CudaGraphVarMap>,
    kv_tile_indices: &'a Option<CudaGraphVarMap>,
    o_indptr: &'a Option<CudaGraphVarMap>,
    kv_chunk_size: &'a Option<CudaGraphVarMap>,
    block_valid_mask: &'a Option<CudaGraphVarMap>,
}

fn copy_flashinfer_tile_plan(
    metadata: &PagedAttentionInputMetadata,
    full: bool,
    vars: FlashInferTilePlanVars<'_>,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    let view = if full {
        flashinfer_full_view(metadata)
    } else {
        flashinfer_paged_view(metadata)
    };
    copy_option_var_map(
        vars.request_indices,
        view.map(|view| &view.tile_plan.request_indices),
        if full {
            "full_paged_kv_request_indices"
        } else {
            "paged_kv_request_indices"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.kv_tile_indices,
        view.map(|view| &view.tile_plan.kv_tile_indices),
        if full {
            "full_paged_kv_tile_indices"
        } else {
            "paged_kv_tile_indices"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.o_indptr,
        view.map(|view| &view.tile_plan.o_indptr),
        if full {
            "full_paged_kv_o_indptr"
        } else {
            "paged_kv_o_indptr"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.kv_chunk_size,
        view.map(|view| &view.tile_plan.kv_chunk_size),
        if full {
            "full_paged_kv_chunk_size"
        } else {
            "paged_kv_chunk_size"
        },
        host_staging,
    )?;
    copy_option_var_map(
        vars.block_valid_mask,
        view.map(|view| &view.tile_plan.block_valid_mask),
        if full {
            "full_paged_kv_block_valid_mask"
        } else {
            "paged_kv_block_valid_mask"
        },
        host_staging,
    )
}

fn rope_positions_var_map(
    slot_mappings: &HashMap<DeviceLocation, Tensor>,
    position_ids: &[usize],
    seq_len: usize,
) -> candle_core::Result<CudaGraphVarMap> {
    slot_mappings
        .iter()
        .map(|(location, tensor)| {
            let positions = decode_positions_tensor(position_ids, seq_len, tensor.device())?;
            Ok((*location, Var::from_tensor(&positions)?))
        })
        .collect()
}

fn copy_rope_positions(
    dst: &CudaGraphVarMap,
    position_ids: &[usize],
    seq_len: usize,
    host_staging: &mut CudaGraphHostStaging,
) -> candle_core::Result<()> {
    let positions = decode_positions_tensor(position_ids, seq_len, &Device::Cpu)?;
    for (location, dst) in dst {
        if is_cuda_graph_device(dst.device()) {
            host_staging.copy_from("rope_positions", *location, &positions, dst)?;
        } else {
            dst.set(&positions)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_estimate_uses_recordings_then_falls_back() {
        let bucket = DecodeArenaBucketKey {
            batch: 8,
            seq_len: 1,
            dtype_bytes: 2,
            kind: RecurrentBatchKind::Decode,
        };
        let mut state = CudaDecodeGraphState::default();
        // Fallback tracks the warmup's cumulative alloc bytes at x1.25
        // (calibrated on the hybrid+MTP 27B; the old x6 pinned
        // six-times-oversized arenas per graph).
        assert_eq!(
            state.estimated_arena_bytes(&bucket, 100),
            100 * 5 / 4 + DECODE_GRAPH_MIN_ARENA_BYTES
        );
        state.record_arena_observation(bucket, 1000);
        assert_eq!(
            state.estimated_arena_bytes(&bucket, 100),
            1000 + 1000 / 8 + DECODE_GRAPH_MIN_ARENA_BYTES
        );
        state.record_arena_observation(bucket, 0);
        state.record_arena_observation(bucket, 500);
        assert_eq!(
            state.estimated_arena_bytes(&bucket, 100),
            1000 + 1000 / 8 + DECODE_GRAPH_MIN_ARENA_BYTES
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn decode_arena_clamp_floors_degenerate_caps() -> candle_core::Result<()> {
        let device = Device::Cpu;
        // A degenerate availability reading still leaves the 32MiB arena
        // floor in place: captures cannot run under it, and the floor is
        // the shipped behavior (right-size work, 2026-10-01).
        assert_eq!(
            super::clamp_decode_arena_bytes(usize::MAX, 10_000, &device),
            DECODE_GRAPH_MIN_ARENA_BYTES
        );
        assert_eq!(super::clamp_decode_arena_bytes(1024, 10_000, &device), 1024);
        assert_eq!(
            super::clamp_decode_arena_bytes(10_000, 10_000, &device),
            10_000
        );
        assert_eq!(super::clamp_decode_arena_bytes(0, 10_000, &device), 0);
        // CPU is not a graph device: the free query falls through to
        // usize::MAX, so the to-device clamp passes requests through
        // unchanged instead of clamping to a budget.
        assert_eq!(super::clamp_decode_arena_to_device(&device, 4096)?, 4096);
        Ok(())
    }

    fn spec_usage(bytes: &[(usize, usize)]) -> CudaGraphSpecStateUsage {
        CudaGraphSpecStateUsage {
            bytes: bytes
                .iter()
                .map(|(gpu_id, bytes)| (DeviceLocation::Cuda { gpu_id: *gpu_id }, *bytes))
                .collect(),
            device_totals: HashMap::new(),
        }
    }

    #[test]
    fn graph_event_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let mut labels = HashSet::new();
        for component in [CudaGraphComponent::Target, CudaGraphComponent::DFlash] {
            for event in [
                CudaGraphEvent::Capture,
                CudaGraphEvent::Replay,
                CudaGraphEvent::EagerFallback,
            ] {
                for outcome in [CudaGraphOutcome::Success, CudaGraphOutcome::Failure] {
                    assert!(labels.insert(cuda_graph_event_labels(component, event, outcome)));
                }
            }
        }

        assert_eq!(
            CUDA_GRAPH_EVENTS_METRIC,
            "mistralrs_cuda_graph_events_total"
        );
        assert_eq!(labels.len(), 12);
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.component)
                .collect::<HashSet<_>>(),
            HashSet::from(["target", "dflash"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.event)
                .collect::<HashSet<_>>(),
            HashSet::from(["capture", "replay", "eager_fallback"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.outcome)
                .collect::<HashSet<_>>(),
            HashSet::from(["success", "failure"])
        );
    }

    #[test]
    fn graph_dispatch_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let reasons = [
            CudaGraphDispatchReason::CacheHit,
            CudaGraphDispatchReason::Disabled,
            CudaGraphDispatchReason::ModelUnsupported,
            CudaGraphDispatchReason::SpeculativeConflict,
            CudaGraphDispatchReason::PagedAttentionUnavailable,
            CudaGraphDispatchReason::Prefill,
            CudaGraphDispatchReason::IncompatibleShape,
            CudaGraphDispatchReason::BatchUnsupported,
            CudaGraphDispatchReason::CacheConfigUnavailable,
            CudaGraphDispatchReason::RuntimeDisabled,
            CudaGraphDispatchReason::PaddingUnavailable,
            CudaGraphDispatchReason::CachePopulation,
            CudaGraphDispatchReason::Fallback,
        ];
        let mut labels = HashSet::new();
        for component in [CudaGraphComponent::Target, CudaGraphComponent::DFlash] {
            for mode in [
                CudaGraphDispatchMode::Replay,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchMode::Skipped,
            ] {
                for reason in reasons {
                    assert!(labels.insert(cuda_graph_dispatch_labels(component, mode, reason)));
                }
            }
        }

        assert_eq!(
            CUDA_GRAPH_DISPATCH_METRIC,
            "mistralrs_cuda_graph_dispatch_total"
        );
        assert_eq!(labels.len(), 78);
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.mode)
                .collect::<HashSet<_>>(),
            HashSet::from(["replay", "eager", "skipped"])
        );
        assert_eq!(
            labels
                .iter()
                .map(|labels| labels.reason)
                .collect::<HashSet<_>>()
                .len(),
            reasons.len()
        );
    }

    #[test]
    fn graph_eviction_labels_have_fixed_cardinality() {
        use std::collections::HashSet;

        let labels = [
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::Capacity,
            ),
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::MemoryPressure,
            ),
            (
                CudaGraphComponent::Target,
                CudaGraphEvictionReason::SpecStateBudget,
            ),
            (
                CudaGraphComponent::DFlash,
                CudaGraphEvictionReason::Capacity,
            ),
            (
                CudaGraphComponent::DFlash,
                CudaGraphEvictionReason::MemoryPressure,
            ),
        ]
        .into_iter()
        .map(|(component, reason)| (component.label(), reason.label()))
        .collect::<HashSet<_>>();

        assert_eq!(
            CUDA_GRAPH_EVICTIONS_METRIC,
            "mistralrs_cuda_graph_evictions_total"
        );
        assert_eq!(
            CUDA_GRAPH_RESIDENT_ENTRIES_METRIC,
            "mistralrs_cuda_graph_resident_entries"
        );
        assert_eq!(labels.len(), 5);
        assert!(labels.contains(&("target", "capacity")));
        assert!(labels.contains(&("dflash", "capacity")));
        assert!(labels.contains(&("target", "memory_pressure")));
        assert!(labels.contains(&("dflash", "memory_pressure")));
        assert!(labels.contains(&("target", "spec_state_budget")));
    }

    #[test]
    fn capacity_eviction_removes_one_lru_entry_and_preserves_the_bound() {
        let mut entries = vec![10, 20, 30];
        assert_eq!(take_cuda_graph_capacity_eviction(&mut entries, 4), None);
        entries.push(40);
        assert_eq!(take_cuda_graph_capacity_eviction(&mut entries, 4), Some(10));
        entries.push(50);
        assert_eq!(entries, vec![20, 30, 40, 50]);
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn target_cache_retains_64_entries_before_lru_eviction() {
        let capacity = TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY;
        assert_eq!(capacity, 64);

        let mut entries = (0..capacity - 1).collect::<Vec<_>>();
        assert_eq!(
            take_cuda_graph_capacity_eviction(&mut entries, capacity),
            None
        );
        entries.push(capacity - 1);
        assert_eq!(entries.len(), capacity);

        assert_eq!(
            take_cuda_graph_capacity_eviction(&mut entries, capacity),
            Some(0),
        );
        entries.push(capacity);

        assert_eq!(entries.len(), capacity);
        assert_eq!(entries.first(), Some(&1));
        assert_eq!(entries.last(), Some(&capacity));
    }

    #[test]
    fn target_cache_capacity_tracks_large_speculative_working_sets() {
        assert_eq!(target_cuda_graph_cache_capacity(23, 23), 64);
        assert_eq!(target_cuda_graph_cache_capacity(38, 15), 64);
        assert_eq!(target_cuda_graph_cache_capacity(54, 23), 80);
        assert_eq!(target_cuda_graph_cache_capacity(61, 30), 96);
        assert_eq!(target_cuda_graph_cache_capacity(80, 23), 96);
        assert_eq!(target_cuda_graph_cache_capacity(usize::MAX, usize::MAX), 96);
    }

    #[test]
    fn target_single_token_batch_buckets_are_exact_through_16() {
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 0),
            None
        );
        for batch in 1..=TARGET_SINGLE_TOKEN_EXACT_BATCH_BUCKETS {
            assert_eq!(
                cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, batch),
                Some(batch)
            );
        }
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 17),
            Some(24)
        );
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, 31),
            Some(32)
        );
        assert_eq!(
            cuda_graph_batch_bucket(CudaGraphComponent::Target, 1, CUDA_GRAPH_MAX_BATCH_BUCKET),
            Some(CUDA_GRAPH_MAX_BATCH_BUCKET)
        );
        assert_eq!(
            cuda_graph_batch_bucket(
                CudaGraphComponent::Target,
                1,
                CUDA_GRAPH_MAX_BATCH_BUCKET + 1
            ),
            None
        );
    }

    #[test]
    fn multi_token_and_dflash_batches_keep_coarse_buckets() {
        for (component, q_len) in [
            (CudaGraphComponent::Target, 4),
            (CudaGraphComponent::DFlash, 1),
            (CudaGraphComponent::DFlash, 8),
        ] {
            assert_eq!(cuda_graph_batch_bucket(component, q_len, 8), Some(8));
            assert_eq!(cuda_graph_batch_bucket(component, q_len, 9), Some(16));
            assert_eq!(cuda_graph_batch_bucket(component, q_len, 17), Some(24));
        }
    }

    #[test]
    fn precapture_batches_follow_component_and_query_width() {
        assert_eq!(
            cuda_graph_precapture_batches(CudaGraphComponent::Target, 1).collect::<Vec<_>>(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 40, 48, 56, 64, 72,
                80, 88, 96, 104, 112, 120, 128,
            ]
        );
        assert_eq!(
            cuda_graph_precapture_batches(CudaGraphComponent::Target, 4).collect::<Vec<_>>(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
                128,
            ]
        );
        assert_eq!(
            cuda_graph_precapture_batches(CudaGraphComponent::DFlash, 1).collect::<Vec<_>>(),
            cuda_graph_precapture_batches(CudaGraphComponent::Target, 4).collect::<Vec<_>>()
        );
    }

    #[test]
    fn precapture_max_batch_uses_the_shape_policy() {
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 0),
            0
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 9),
            9
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 4, 9),
            16
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::DFlash, 8, 9),
            16
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 17),
            24
        );
        assert_eq!(
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, 129),
            128
        );
    }

    #[test]
    fn target_single_token_buckets_fit_the_default_cache() {
        let bucket_count = cuda_graph_precapture_batches(CudaGraphComponent::Target, 1).count();
        assert_eq!(bucket_count, 30);
        assert_eq!(
            target_cuda_graph_cache_capacity(bucket_count, bucket_count),
            TARGET_CUDA_DECODE_GRAPH_CACHE_DEFAULT_CAPACITY
        );
    }

    #[test]
    fn startup_precapture_accepts_decode_and_verification_widths() {
        assert!(!cuda_graph_startup_capture_allowed(0));
        assert!(cuda_graph_startup_capture_allowed(1));
        assert!(cuda_graph_startup_capture_allowed(4));
        assert!(cuda_graph_startup_capture_allowed(8));
    }

    #[test]
    fn decode_graph_batch_kind_rejects_every_prefill_chunk() {
        assert!(!cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::Prefill
        ));
        assert!(cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::Decode
        ));
        assert!(cuda_decode_graph_batch_kind_supported(
            RecurrentBatchKind::SpeculativeDecode
        ));
    }

    #[test]
    fn graph_context_len_tracks_paged_attention_partitions() {
        assert_eq!(graph_context_len(Some(1), Some(2048)), Some(512));
        assert_eq!(graph_context_len(Some(512), Some(2048)), Some(512));
        assert_eq!(graph_context_len(Some(513), Some(2048)), Some(1024));
        assert_eq!(graph_context_len(Some(1537), Some(2048)), Some(2048));
    }

    #[test]
    fn graph_context_len_preserves_nonstandard_metadata() {
        assert_eq!(graph_context_len(Some(513), None), Some(513));
        assert_eq!(graph_context_len(None, Some(2048)), Some(2048));
        assert_eq!(graph_context_len(None, None), None);
    }

    #[test]
    fn decode_row_graph_key_is_independent_of_materialization() {
        let table = vec![1, 2, 3, 4];
        let rows = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![127]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
            context_lens: vec![128],
            full_context_lens: vec![128],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 1_604_288,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 4,
        });
        let staged = rows.build_graph_staged().unwrap();
        let materialized = rows.build_materialized().unwrap();
        let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
        let staged_key =
            CudaDecodeGraphKey::new(&input_ids, &staged, 32, RecurrentBatchKind::Decode).unwrap();
        let materialized_key =
            CudaDecodeGraphKey::new(&input_ids, &materialized, 32, RecurrentBatchKind::Decode)
                .unwrap();
        assert_eq!(staged_key, materialized_key);
        assert!(staged_key.tensors.is_empty());
        assert!(staged_key.decode_rows.is_some());
        let speculative_key = CudaDecodeGraphKey::new(
            &input_ids,
            &materialized,
            32,
            RecurrentBatchKind::SpeculativeDecode,
        )
        .unwrap();
        assert_ne!(staged_key, speculative_key);
        assert!(!staged_key.has_same_spec_state_shape(&speculative_key));

        let mut next_bucket_rows = (*rows).clone();
        next_bucket_rows.block_tables =
            BlockTableSnapshot::from_owned_sequence_tables(vec![(1..=17).collect()], 1);
        next_bucket_rows.context_lens = vec![513];
        next_bucket_rows.full_context_lens = vec![513];
        let next_bucket = Arc::new(next_bucket_rows).build_graph_staged().unwrap();
        let next_bucket_key =
            CudaDecodeGraphKey::new(&input_ids, &next_bucket, 32, RecurrentBatchKind::Decode)
                .unwrap();
        assert_ne!(staged_key, next_bucket_key);
        assert!(staged_key.has_same_spec_state_shape(&next_bucket_key));
    }

    #[test]
    fn graph_buffers_share_unsliding_flashinfer_metadata() {
        let table = vec![1, 2, 3, 4];
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![127]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
            context_lens: vec![128],
            full_context_lens: vec![128],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 1_604_288,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 4,
        })
        .build()
        .unwrap();
        let input_ids = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();
        let key =
            CudaDecodeGraphKey::new(&input_ids, &metadata, 32, RecurrentBatchKind::Decode).unwrap();
        assert!(key
            .tensors
            .iter()
            .all(|tensor| !tensor.name.starts_with("full_")));

        let (buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata: &metadata,
            seqlen_offsets: &[127],
            position_ids: &[128],
            seq_len: 1,
            block_size: 32,
            kv_cache: &[],
            model_metadata: None,
            activation_dtype: DType::F32,
        })
        .unwrap();
        assert!(buffers.flashinfer_views_alias);
        let location = Device::Cpu.location();
        assert_eq!(
            buffers.paged_kv_indices.as_ref().unwrap()[&location].id(),
            buffers.full_paged_kv_indices.as_ref().unwrap()[&location].id()
        );
        assert_eq!(
            buffers.paged_kv_request_indices.as_ref().unwrap()[&location].id(),
            buffers.full_paged_kv_request_indices.as_ref().unwrap()[&location].id()
        );
    }

    #[test]
    fn graph_rope_positions_expand_adjusted_ends_for_verification() {
        let table = vec![1, 2];
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![37, 38, 39], vec![37, 38, 39]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                vec![table.clone(), table],
                3,
            ),
            context_lens: vec![38, 39, 40, 38, 39, 40],
            full_context_lens: vec![38, 39, 40, 38, 39, 40],
            query_len: 3,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 128,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 4,
        })
        .build_materialized()
        .unwrap();

        let (buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata: &metadata,
            seqlen_offsets: &[97, 97],
            position_ids: &[100, 52],
            seq_len: 3,
            block_size: 32,
            kv_cache: &[],
            model_metadata: None,
            activation_dtype: DType::F32,
        })
        .unwrap();
        let positions = buffers.rope_positions[&Device::Cpu.location()]
            .as_detached_tensor()
            .to_vec1::<u32>()
            .unwrap();

        assert_eq!(positions, vec![97, 98, 99, 49, 50, 51]);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_metadata_replay_updates_persistent_cuda_buffers() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let rows = |table: Vec<usize>, context_len: usize| -> anyhow::Result<_> {
            Ok(Arc::new(DecodePagedRows {
                slot_mappings: vec![vec![i64::try_from(context_len - 1)?]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![table], 1),
                context_lens: vec![context_len],
                full_context_lens: vec![context_len],
                query_len: 1,
                block_size: 32,
                use_standard_metadata: false,
                max_paged_context_len: 1_604_288,
                sliding_window: None,
                decode_window: 1,
                devices: vec![device.clone()],
                num_kv_heads: 4,
            }))
        };
        let initial = rows(vec![1, 2, 3, 4], 128)?.build_materialized()?;
        let updated = rows(vec![5, 6, 7, 8, 9, 10, 11, 12], 256)?.build()?;
        assert!(updated.has_host_staged_decode_tensors());

        let (mut buffers, _) = CudaDecodeGraphMetadataBuffers::new(CudaDecodeGraphMetadataInput {
            metadata: &initial,
            seqlen_offsets: &[127],
            position_ids: &[128],
            seq_len: 1,
            block_size: 32,
            kv_cache: &[],
            model_metadata: None,
            activation_dtype: DType::F32,
        })?;
        let location = device.location();
        let state_indices = Var::from_tensor(&Tensor::zeros((3,), DType::U32, &device)?)?;
        let mut state_indices_map = HashMap::from([(location, state_indices)]);
        let mut host_staging = CudaGraphHostStaging::new(GraphStream::from_graph_device(&device)?)?;
        host_staging.update(|host_staging| {
            buffers.copy_from(&updated, &[256], 1, host_staging)?;
            copy_state_indices(&state_indices_map, &[3, 5, 7], host_staging)
        })?;
        host_staging.update(|host_staging| {
            buffers.copy_from(&updated, &[256], 1, host_staging)?;
            copy_state_indices(&state_indices_map, &[11, 13, 17], host_staging)
        })?;
        device.synchronize()?;

        let indices = buffers.paged_kv_indices.as_ref().unwrap()[&location]
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<i32>()?;
        assert_eq!(&indices[..8], &[5, 6, 7, 8, 9, 10, 11, 12]);
        let positions = buffers.rope_positions[&location]
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        assert_eq!(positions, vec![255]);
        assert!(host_staging.buffers.len() > 1);
        assert_eq!(host_staging.completions.len(), 1);

        let state_indices = state_indices_map
            .remove(&location)
            .unwrap()
            .as_detached_tensor()
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        assert_eq!(state_indices, vec![11, 13, 17]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_staging_orders_secondary_streams_both_directions() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let graph_stream = GraphStream::from_graph_device(&device)?;
        let copy_stream = graph_stream.fork()?;
        assert!(!same_cuda_stream(&graph_stream, &copy_stream));
        let location = device.location();
        let mut staging = CudaGraphHostStaging::new(graph_stream.clone())?;

        staging.record_graph_complete()?;
        staging.update(|staging| staging.prepare_copy(location, &copy_stream))?;
        assert!(staging.completions[&location].ordered_after_graph);
        staging.order_before_graph()?;

        staging.record_graph_complete()?;
        staging.update(|staging| staging.prepare_copy(location, &copy_stream))?;
        assert!(staging.completions[&location].ordered_after_graph);
        staging.order_before_graph()?;
        graph_stream.synchronize()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_memory_pool_scope_restores_release_threshold() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let stream = GraphStream::from_graph_device(&device)?;
        let pool = cuda_memory_pool(&stream)?;
        let original = memory_pool_release_threshold(&stream, pool)?;

        let first = prepare_cuda_graph_memory_pool(&stream)?;
        let second = prepare_cuda_graph_memory_pool(&stream)?;
        assert_eq!(memory_pool_release_threshold(&stream, pool)?, u64::MAX);
        drop(first);
        assert_eq!(memory_pool_release_threshold(&stream, pool)?, u64::MAX);
        drop(second);
        assert_eq!(memory_pool_release_threshold(&stream, pool)?, original);
        Ok(())
    }

    // cudarc-hip takes no capture mode argument
    #[cfg(any(
        all(feature = "cuda", not(feature = "rocm")),
        all(feature = "cuda", feature = "rocm")
    ))]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_memory_cleanup_returns_allocator_to_baseline() -> anyhow::Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let device = Device::new_cuda(0)?;
        let stream = GraphStream::from_graph_device(&device)?;
        let used_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
        let reserved_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT;
        trim_cuda_graph_memory(&stream)?;
        let used_before = cuda_graph_memory_attribute(&stream, used_attribute)?;
        let reserved_before = cuda_graph_memory_attribute(&stream, reserved_attribute)?;

        let guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_graph_capture(true)?;
        let output = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;
        graph.launch()?;
        stream.synchronize()?;
        drop(output);
        stream.synchronize()?;
        drop(graph);
        drop(guard);

        trim_cuda_graph_memory(&stream)?;
        assert_eq!(
            cuda_graph_memory_attribute(&stream, used_attribute)?,
            used_before
        );
        assert_eq!(
            cuda_graph_memory_attribute(&stream, reserved_attribute)?,
            reserved_before
        );
        Ok(())
    }

    // cudarc-hip takes no capture mode argument
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn unlaunched_graph_cleanup_returns_allocator_to_baseline() -> anyhow::Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let device = Device::new_cuda(0)?;
        let stream = GraphStream::from_graph_device(&device)?;
        let used_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
        let reserved_attribute = sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT;
        trim_cuda_graph_memory(&stream)?;
        let used_before = cuda_graph_memory_attribute(&stream, used_attribute)?;
        let reserved_before = cuda_graph_memory_attribute(&stream, reserved_attribute)?;

        let guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_graph_capture(true)?;
        let logits = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        let mut release_result = Ok(());
        drop_cuda_graph_entry_output(logits, &stream, 0, "logits", &mut release_result);
        release_result?;
        drop(graph);
        drop(guard);
        trim_cuda_graph_memory(&stream)?;

        assert_eq!(
            cuda_graph_memory_attribute(&stream, used_attribute)?,
            used_before
        );
        assert_eq!(
            cuda_graph_memory_attribute(&stream, reserved_attribute)?,
            reserved_before
        );
        Ok(())
    }

    // cudarc-hip takes no capture mode argument
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_replay_retains_captured_output_storage() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let stream = GraphStream::from_graph_device(&device)?;
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;

        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;
        let warmup = input.as_detached_tensor().affine(2.0, 1.0)?;
        device.synchronize()?;
        drop(warmup);

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_graph_capture(true)?;
        let output = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        for (values, expected) in [
            (vec![3f32, 5.0], vec![7f32, 11.0]),
            (vec![8f32, 13.0], vec![17f32, 27.0]),
        ] {
            input.set(&Tensor::from_vec(values, 2, &device)?)?;
            graph.launch()?;
            stream.synchronize()?;
            assert_eq!(output.to_vec1::<f32>()?, expected);
        }

        drop(graph);
        drop(output);
        device.synchronize()?;
        Ok(())
    }

    // cudarc-hip takes no capture mode argument
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn replayed_graph_output_releases_without_driver_error() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let stream = GraphStream::from_graph_device(&device)?;
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;
        let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_graph_capture(true)?;
        let output = input.as_detached_tensor().affine(2.0, 1.0)?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;
        graph.launch()?;
        stream.synchronize()?;

        let mut release_result = Ok(());
        drop_cuda_graph_entry_output(output, &stream, 1, "test output", &mut release_result);
        release_result?;
        drop(graph);
        Ok(())
    }

    // cudarc-hip takes no capture mode argument
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn graph_copy_supports_dense_row_source() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let stream = GraphStream::from_graph_device(&device)?;
        let _memory_pool_guard = prepare_cuda_graph_memory_pool(&stream)?;

        let input = Var::from_tensor(&Tensor::from_vec(
            (1u16..=16).map(f32::from).collect::<Vec<_>>(),
            (2, 2, 4),
            &device,
        )?)?;
        let output = Var::from_tensor(&Tensor::zeros((2, 2, 2), DType::F32, &device)?)?;
        let source = input.as_detached_tensor().narrow(2, 1, 2)?;
        assert!(!source.is_contiguous());

        let restore_event_tracking = disable_event_tracking_for_capture(&stream);
        stream.begin_graph_capture(true)?;
        crate::cuda::graph::copy_tensor(&source, &output.as_detached_tensor())?;
        let graph = CudaGraphHandle::end_capture(&stream)?
            .ok_or_else(|| anyhow::anyhow!("CUDA graph capture returned no graph"))?;
        restore_event_tracking_after_capture(&stream, restore_event_tracking);
        graph.upload()?;

        graph.launch()?;
        stream.synchronize()?;
        assert_eq!(
            output.as_detached_tensor().to_vec3::<f32>()?,
            vec![
                vec![vec![2.0, 3.0], vec![6.0, 7.0]],
                vec![vec![10.0, 11.0], vec![14.0, 15.0]],
            ]
        );

        input.set(&Tensor::from_vec(
            (1u16..=16).rev().map(f32::from).collect::<Vec<_>>(),
            (2, 2, 4),
            &device,
        )?)?;
        graph.launch()?;
        stream.synchronize()?;
        assert_eq!(
            output.as_detached_tensor().to_vec3::<f32>()?,
            vec![
                vec![vec![15.0, 14.0], vec![11.0, 10.0]],
                vec![vec![7.0, 6.0], vec![3.0, 2.0]],
            ]
        );
        Ok(())
    }

    #[test]
    fn graph_suspension_does_not_clear_permanent_disable() {
        let mut state = CudaDecodeGraphState::default();
        state.suspend();
        assert!(state.disabled());
        state.resume();
        assert!(!state.disabled());
        state.disable();
        state.suspend();
        state.resume();
        assert!(state.disabled());
    }

    #[test]
    fn eager_retry_block_applies_to_one_failed_step() {
        let mut state = CudaDecodeGraphState::default();
        assert!(state.take_eager_retry_allowed());
        state.block_eager_retry();
        assert!(!state.take_eager_retry_allowed());
        assert!(state.take_eager_retry_allowed());
        state.block_eager_retry();
        state.clear();
        assert!(state.take_eager_retry_allowed());
    }

    #[test]
    fn graph_generations_are_not_reused_after_cleanup() {
        let mut state = CudaDecodeGraphState::default();
        let first = state.allocate_generation();
        state.clear();
        let second = state.allocate_generation();
        state.suspend();
        state.resume();
        let third = state.allocate_generation();
        assert!(first < second && second < third);
        assert!(!cuda_graph_replay_version_matches(second, 0, first, 1));
        assert!(!cuda_graph_replay_version_matches(third, 0, second, 1));
        assert!(cuda_graph_replay_version_matches(third, 7, third, 7));
    }

    #[test]
    fn memory_pressure_eviction_drains_lru_entries_in_one_batch() {
        let mut entries = vec![10, 20, 30, 40];
        assert!(drain_lru_entries(&mut entries, 0).is_empty());
        assert_eq!(drain_lru_entries(&mut entries, 2), vec![10, 20]);
        assert_eq!(entries, vec![30, 40]);
        assert_eq!(drain_lru_entries(&mut entries, usize::MAX), vec![30, 40]);
        assert!(entries.is_empty());
    }

    #[test]
    fn graph_reclaim_quota_is_shared_between_target_and_dflash() {
        let mut target_entries = 2usize;
        let mut dflash_entries = 4usize;
        let reclaimed = reclaim_cuda_graph_entries(
            3,
            |limit| {
                let reclaimed = target_entries.min(limit);
                target_entries -= reclaimed;
                reclaimed
            },
            |limit| {
                let reclaimed = dflash_entries.min(limit);
                dflash_entries -= reclaimed;
                reclaimed
            },
        );
        assert_eq!(reclaimed, 3);
        assert_eq!(target_entries, 0);
        assert_eq!(dflash_entries, 3);

        let mut dflash_called = false;
        assert_eq!(
            reclaim_cuda_graph_entries(
                2,
                |_| 2,
                |_| {
                    dflash_called = true;
                    0
                }
            ),
            2
        );
        assert!(!dflash_called);

        let mut target_called = false;
        assert_eq!(
            reclaim_cuda_graph_entries(
                0,
                |_| {
                    target_called = true;
                    0
                },
                |_| 0,
            ),
            0
        );
        assert!(!target_called);
    }

    #[test]
    fn speculative_state_budget_scales_with_the_largest_entry() {
        assert_eq!(default_spec_state_budget(10_000, 0), 400);
        assert_eq!(default_spec_state_budget(10_000, 50), 500);
        assert_eq!(default_spec_state_budget(10_000, 100), 800);
        assert_eq!(default_spec_state_budget(10_000, 200), 800);
        assert_eq!(default_spec_state_budget(10_000, 300), 800);
        assert_eq!(default_spec_state_budget(10_000, 900), 900);
    }

    #[test]
    fn speculative_state_budget_evicts_lru_entries_per_device() {
        let gpu0 = DeviceLocation::Cuda { gpu_id: 0 };
        let gpu1 = DeviceLocation::Cuda { gpu_id: 1 };
        let budgets = HashMap::from([(gpu0, 100), (gpu1, 100)]);
        let existing = vec![
            spec_usage(&[]),
            spec_usage(&[(0, 40)]),
            spec_usage(&[(1, 60)]),
        ];

        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(1, 50)]), &budgets),
            vec![2]
        );
        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(0, 120)]), &budgets),
            vec![1]
        );
        let existing = vec![
            spec_usage(&[(0, 30)]),
            spec_usage(&[(0, 50)]),
            spec_usage(&[(0, 10)]),
        ];
        assert_eq!(
            spec_state_eviction_plan(&existing, &spec_usage(&[(0, 60)]), &budgets),
            vec![0, 1]
        );
    }

    #[test]
    fn one_token_continuation_advances_host_decode_state() -> anyhow::Result<()> {
        let rows = Arc::new(
            DecodePagedRows {
                slot_mappings: vec![vec![39]],
                block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![9, 17]], 1),
                context_lens: vec![4],
                full_context_lens: vec![4],
                query_len: 1,
                block_size: 4,
                use_standard_metadata: false,
                max_paged_context_len: 64,
                sliding_window: Some(4),
                decode_window: 1,
                devices: vec![Device::Cpu],
                num_kv_heads: 1,
            }
            .padded(2),
        );
        let step = CudaGraphDecodeStep {
            input_ids: Tensor::zeros((2, 1), DType::U32, &Device::Cpu)?,
            seqlen_offsets: vec![100, 100],
            context_lens: vec![(0, 1), (0, 1)],
            position_ids: vec![43, 43],
            metadata: rows.build_graph_staged()?,
            state_indices: Some(vec![2, 5]),
            real_batch: 1,
        };
        let continuation = step
            .one_token_continuation(step.input_ids.clone())?
            .expect("next allocated block should permit one token");
        let rows = continuation.metadata.decode_rows.as_ref().unwrap();
        assert_eq!(rows.slot_mappings, vec![vec![68], vec![_PAD_SLOT_ID]]);
        assert_eq!(
            rows.materialized_block_tables(),
            vec![vec![9, 17], vec![9, 17]]
        );
        assert_eq!(rows.context_lens, vec![5, 5]);
        assert_eq!(rows.full_context_lens, vec![5, 5]);
        assert_eq!(continuation.seqlen_offsets, vec![101, 101]);
        assert_eq!(continuation.context_lens, vec![(0, 1), (0, 1)]);
        assert_eq!(continuation.position_ids, vec![44, 44]);
        assert_eq!(continuation.state_indices, Some(vec![2, 5]));
        Ok(())
    }

    #[test]
    fn one_token_continuation_requires_an_allocated_boundary_slot() -> anyhow::Result<()> {
        let rows = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![39]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![9]], 1),
            context_lens: vec![4],
            full_context_lens: vec![4],
            query_len: 1,
            block_size: 4,
            use_standard_metadata: false,
            max_paged_context_len: 64,
            sliding_window: None,
            decode_window: 1,
            devices: vec![Device::Cpu],
            num_kv_heads: 1,
        });
        let step = CudaGraphDecodeStep {
            input_ids: Tensor::zeros((1, 1), DType::U32, &Device::Cpu)?,
            seqlen_offsets: vec![3],
            context_lens: vec![(0, 1)],
            position_ids: vec![4],
            metadata: rows.build_graph_staged()?,
            state_indices: None,
            real_batch: 1,
        };
        assert!(step
            .one_token_continuation(step.input_ids.clone())?
            .is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn resident_replay_skips_the_host_input_update() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: vec![vec![0]],
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(vec![vec![0]], 1),
            context_lens: vec![1],
            full_context_lens: vec![1],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 32,
            sliding_window: None,
            decode_window: 1,
            devices: vec![device.clone()],
            num_kv_heads: 1,
        })
        .build_materialized()?;
        let initial_ids = Tensor::from_vec(vec![1u32], (1, 1), &device)?;
        let key = CudaDecodeGraphKey::new(&initial_ids, &metadata, 32, RecurrentBatchKind::Decode)?;
        let warmup_logits = initial_ids.to_dtype(DType::F32)?;
        let mut state = CudaDecodeGraphState::default();
        let entry = capture_cuda_decode_graph(
            CudaDecodeGraphCaptureCtx {
                key: key.clone(),
                input_ids: &initial_ids,
                seqlen_offsets: &[0],
                position_ids: &[1],
                block_size: 32,
                kv_cache: &[],
                metadata: &metadata,
                model_metadata: None,
                activation_dtype: DType::F32,
                warmup_logits: &warmup_logits,
                state_indices: None,
                real_batch: 1,
                arena_bytes: 0,
            },
            |input_ids, _| input_ids.to_dtype(DType::F32),
            &mut state,
        )?;
        state.insert(entry);

        let step = |token: u32| -> candle_core::Result<CudaGraphDecodeStep> {
            Ok(CudaGraphDecodeStep {
                input_ids: Tensor::from_vec(vec![token], (1, 1), &device)?,
                seqlen_offsets: vec![0],
                context_lens: vec![(0, 1)],
                position_ids: vec![1],
                metadata: metadata.clone(),
                state_indices: None,
                real_batch: 1,
            })
        };
        let host_step = step(7)?;
        let host_replay = state
            .replay(&key, &host_step, CudaDecodeGraphReplayInput::Host)?
            .expect("captured graph missing");
        let launch = host_replay.launch.expect("qlen=1 launch missing");
        launch.graph_stream().synchronize()?;
        assert_eq!(host_replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);

        let resident_step = step(13)?;
        let resident_replay = state
            .replay(
                &key,
                &resident_step,
                CudaDecodeGraphReplayInput::Resident(&launch),
            )?
            .expect("resident graph entry changed");
        resident_replay
            .launch
            .as_ref()
            .unwrap()
            .graph_stream()
            .synchronize()?;
        assert_eq!(resident_replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
        let resident_launch = resident_replay.launch.unwrap();
        assert_eq!(resident_launch.generation(), launch.generation());
        assert!(resident_launch.one_token_continuation()?.is_some());
        assert!(state.replay_one_token(launch)?.is_none());
        let lookahead = state
            .replay_one_token(resident_launch)?
            .expect("one-token continuation should retain the graph key");
        lookahead
            .launch
            .as_ref()
            .unwrap()
            .graph_stream()
            .synchronize()?;
        assert_eq!(lookahead.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
        Ok(())
    }

    struct BatchFixtures {
        metadata: PagedAttentionInputMetadata,
        input_ids: Tensor,
        warmup_logits: Tensor,
        key: CudaDecodeGraphKey,
    }

    #[allow(clippy::cast_possible_truncation)]
    fn batch_fixtures(device: &Device, batch: usize) -> anyhow::Result<BatchFixtures> {
        let metadata = Arc::new(DecodePagedRows {
            slot_mappings: (0..batch as i64).map(|row| vec![row]).collect(),
            block_tables: BlockTableSnapshot::from_owned_sequence_tables(
                (0..batch).map(|row| vec![row]).collect(),
                1,
            ),
            context_lens: vec![1; batch],
            full_context_lens: vec![1; batch],
            query_len: 1,
            block_size: 32,
            use_standard_metadata: false,
            max_paged_context_len: 32,
            sliding_window: None,
            decode_window: 1,
            devices: vec![device.clone()],
            num_kv_heads: 1,
        })
        .build_materialized()?;
        let input_ids =
            Tensor::from_vec((1..=batch as u32).collect::<Vec<_>>(), (batch, 1), device)?;
        let warmup_logits = input_ids.to_dtype(DType::F32)?;
        let key = CudaDecodeGraphKey::new(&input_ids, &metadata, 32, RecurrentBatchKind::Decode)?;
        Ok(BatchFixtures {
            metadata,
            input_ids,
            warmup_logits,
            key,
        })
    }

    fn capture_identity(
        fixtures: &BatchFixtures,
        batch: usize,
        state: &mut CudaDecodeGraphState,
    ) -> candle_core::Result<CudaDecodeGraphEntry> {
        capture_cuda_decode_graph(
            CudaDecodeGraphCaptureCtx {
                key: fixtures.key.clone(),
                input_ids: &fixtures.input_ids,
                seqlen_offsets: &vec![0; batch],
                position_ids: &vec![1; batch],
                block_size: 32,
                kv_cache: &[],
                metadata: &fixtures.metadata,
                model_metadata: None,
                activation_dtype: DType::F32,
                warmup_logits: &fixtures.warmup_logits,
                state_indices: None,
                real_batch: batch,
                arena_bytes: 0,
            },
            |input_ids, _| input_ids.to_dtype(DType::F32),
            state,
        )
    }

    fn decode_step(
        device: &Device,
        metadata: &PagedAttentionInputMetadata,
        batch: usize,
        token: u32,
    ) -> candle_core::Result<CudaGraphDecodeStep> {
        Ok(CudaGraphDecodeStep {
            input_ids: Tensor::from_vec(vec![token; batch], (batch, 1), device)?,
            seqlen_offsets: vec![0; batch],
            context_lens: vec![(0, 1); batch],
            position_ids: vec![1; batch],
            metadata: metadata.clone(),
            state_indices: None,
            real_batch: batch,
        })
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn prelaunch_panic_keeps_the_lru_entry_and_reports() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let fixtures = batch_fixtures(&device, 1)?;
        let mut state = CudaDecodeGraphState::default();
        let entry = capture_identity(&fixtures, 1, &mut state)?;
        state.insert(entry);
        let pos = state
            .entries
            .iter()
            .position(|entry| entry.key == fixtures.key)
            .expect("inserted entry missing");
        state.entries[pos].replay_epoch = u64::MAX;
        let step = decode_step(&device, &fixtures.metadata, 1, 7)?;
        match state.replay(&fixtures.key, &step, CudaDecodeGraphReplayInput::Host) {
            Err(err) => assert!(
                err.to_string().contains("panicked"),
                "unexpected error: {err:?}"
            ),
            Ok(_) => anyhow::bail!("prelaunch panic escaped as success"),
        }
        assert!(state.contains(&fixtures.key));
        assert_eq!(state.entries.len(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    #[cfg(feature = "rocm")]
    fn arena_overflow_regrows_and_records_the_observation() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let fixtures = batch_fixtures(&device, 1)?;
        let mut state = CudaDecodeGraphState::default();
        drop(capture_identity(&fixtures, 1, &mut state)?);
        let bucket = fixtures.key.arena_bucket();
        let observed = state.arena_observations.get(&bucket).copied().unwrap_or(0);
        assert!(
            observed > 2,
            "capture should observe a nonzero need, got {observed}"
        );
        std::env::set_var(
            "MISTRALRS_DECODE_ARENA_OVERRIDE",
            (observed / 2).to_string(),
        );
        let regrown = capture_identity(&fixtures, 1, &mut state);
        std::env::remove_var("MISTRALRS_DECODE_ARENA_OVERRIDE");
        state.insert(regrown?);
        assert!(
            state.arena_observations.get(&bucket).copied().unwrap_or(0) >= observed,
            "regrow should observe at least as much"
        );
        let step = decode_step(&device, &fixtures.metadata, 1, 7)?;
        let replay = state
            .replay(&fixtures.key, &step, CudaDecodeGraphReplayInput::Host)?
            .expect("regrown graph missing");
        replay
            .launch
            .as_ref()
            .unwrap()
            .graph_stream()
            .synchronize()?;
        assert_eq!(replay.logits.to_vec2::<f32>()?, vec![vec![7.0]]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn memory_pressure_eviction_releases_the_oldest_graph_first() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        let mut state = CudaDecodeGraphState::default();
        let mut batches = Vec::new();
        for batch in 1..=3usize {
            let fixtures = batch_fixtures(&device, batch)?;
            let entry = capture_identity(&fixtures, batch, &mut state)?;
            state.insert(entry);
            batches.push(fixtures);
        }
        assert_eq!(state.entries.len(), 3);
        assert_eq!(state.evict_lru_for_memory_pressure(1), 1);
        assert_eq!(state.entries.len(), 2);
        assert!(!state.contains(&batches[0].key));
        assert!(state.contains(&batches[1].key));
        assert!(state.contains(&batches[2].key));
        let step = decode_step(&device, &batches[1].metadata, 2, 7)?;
        let replay = state
            .replay(&batches[1].key, &step, CudaDecodeGraphReplayInput::Host)?
            .expect("surviving graph missing");
        replay
            .launch
            .as_ref()
            .unwrap()
            .graph_stream()
            .synchronize()?;
        assert_eq!(replay.logits.to_vec2::<f32>()?, vec![vec![7.0], vec![7.0]]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    #[cfg(feature = "rocm")]
    fn arena_clamp_never_exceeds_device_memory() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        // Available memory moves between queries, so assert properties, not exact values.
        let clamped = clamp_decode_arena_to_device(&device, usize::MAX)?;
        assert!(clamped > 0);
        assert!(clamped < usize::MAX);
        let small = 1024 * 1024;
        assert!(clamp_decode_arena_to_device(&device, small)? <= small);
        Ok(())
    }
}
