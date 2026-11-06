use std::sync::OnceLock;

const CUDA_GRAPHS_ENV: &str = "MISTRALRS_CUDA_GRAPHS";
const CUDA_GRAPHS_HIP_SOLO_ENV: &str = "MISTRALRS_CUDA_GRAPHS_HIP_SOLO";
const CUDA_GRAPHS_HIP_UNGATED_ENV: &str = "MISTRALRS_CUDA_GRAPHS_HIP_UNGATED";
const FLASHINFER_DECODE_ENV: &str = "MISTRALRS_FLASHINFER_DECODE";
const PAGED_DECODE_CLAMP_ENV: &str = "MISTRALRS_PAGED_DECODE_CLAMP";

static CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
static CUDA_GRAPHS_HIP_SOLO_ENABLED: OnceLock<bool> = OnceLock::new();
static CUDA_GRAPHS_HIP_UNGATED: OnceLock<bool> = OnceLock::new();
static FLASHINFER_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();
static PAGED_DECODE_CLAMP_ENABLED: OnceLock<bool> = OnceLock::new();

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on") {
                true
            } else if matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off") {
                false
            } else {
                default
            }
        })
        .unwrap_or(default)
}

pub(crate) fn cuda_graphs_enabled() -> bool {
    // Default-off per the measured prod posture (2026-11-06): no decode win
    // at agentic depths, mixed-phase replay corrupts on this ROCm runtime.
    *CUDA_GRAPHS_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_ENV, false))
}

/// HIP decode graphs replay only on solo decode steps; on this runtime replay
/// corrupts under mixed prefill+decode batches. NVIDIA is unaffected. The gate
/// is the arming mechanism (see hip_decode_graph_replay_allowed), so default-on
/// here means HIP replays gated by default; set 0 to disable HIP replay entirely.
pub(crate) fn cuda_graphs_hip_solo_enabled() -> bool {
    *CUDA_GRAPHS_HIP_SOLO_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_HIP_SOLO_ENV, true))
}

/// Ungated HIP replay fault-hunt lever: every capturable decode step replays with no solo verdict; gate is conc2, never prod.
pub(crate) fn cuda_graphs_hip_ungated() -> bool {
    *CUDA_GRAPHS_HIP_UNGATED.get_or_init(|| env_flag(CUDA_GRAPHS_HIP_UNGATED_ENV, false))
}

const GPU_MAX_HW_QUEUES_ENV: &str = "GPU_MAX_HW_QUEUES";
const GPU_MAX_HW_QUEUES_SAFE: &str = "1";

/// GPU_MAX_HW_QUEUES is consumed by the HIP runtime at init, not by this crate;
/// multi-queue corrupts single-sequence replay on this runtime. Called from
/// initialize_logging so every entry point lands it before first HIP init.
pub fn apply_env_defaults() {
    if std::env::var_os(GPU_MAX_HW_QUEUES_ENV).is_none() {
        std::env::set_var(GPU_MAX_HW_QUEUES_ENV, GPU_MAX_HW_QUEUES_SAFE);
    }
}

pub(crate) fn flashinfer_decode_enabled() -> bool {
    *FLASHINFER_DECODE_ENABLED.get_or_init(|| env_flag(FLASHINFER_DECODE_ENV, true))
}

/// HIP paged decode tanh-caps scores at ATTENTION_SCORE_CLAMP when the model has
/// no softcap, matching the CK prefill clamp; the CUDA reference lane stays
/// unclamped. Set 0 to A/B the unclamped decode kernel.
pub(crate) fn paged_decode_clamp_enabled() -> bool {
    *PAGED_DECODE_CLAMP_ENABLED.get_or_init(|| env_flag(PAGED_DECODE_CLAMP_ENV, true))
}
