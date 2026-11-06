use std::sync::OnceLock;

const CUDA_GRAPHS_ENV: &str = "MISTRALRS_CUDA_GRAPHS";
const CUDA_GRAPHS_HIP_SOLO_ENV: &str = "MISTRALRS_CUDA_GRAPHS_HIP_SOLO";
const CUDA_GRAPHS_HIP_UNGATED_ENV: &str = "MISTRALRS_CUDA_GRAPHS_HIP_UNGATED";
const FLASHINFER_DECODE_ENV: &str = "MISTRALRS_FLASHINFER_DECODE";

static CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
static CUDA_GRAPHS_HIP_SOLO_ENABLED: OnceLock<bool> = OnceLock::new();
static CUDA_GRAPHS_HIP_UNGATED: OnceLock<bool> = OnceLock::new();
static FLASHINFER_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();

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
    *CUDA_GRAPHS_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_ENV, true))
}

/// HIP decode graphs replay only on solo decode steps; on this runtime replay
/// corrupts under mixed prefill+decode batches. NVIDIA is unaffected.
pub(crate) fn cuda_graphs_hip_solo_enabled() -> bool {
    *CUDA_GRAPHS_HIP_SOLO_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_HIP_SOLO_ENV, false))
}

/// Ungated HIP replay fault-hunt lever: every capturable decode step replays with no solo verdict; gate is conc2, never prod.
pub(crate) fn cuda_graphs_hip_ungated() -> bool {
    *CUDA_GRAPHS_HIP_UNGATED.get_or_init(|| env_flag(CUDA_GRAPHS_HIP_UNGATED_ENV, false))
}

pub(crate) fn flashinfer_decode_enabled() -> bool {
    *FLASHINFER_DECODE_ENABLED.get_or_init(|| env_flag(FLASHINFER_DECODE_ENV, true))
}
