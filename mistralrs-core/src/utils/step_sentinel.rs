//! Decode-step sentinel (default-on; MISTRALRS_STEP_SENTINEL=0 opts out). Fails a request when a
//! step lands outside the sane numeric band instead of serving it, then keeps
//! serving: the subcritical form of the NaN/garble class (TODO 2026-11-03/05)
//! reaches the sampler as finite tokens and no existing check trips - a bail
//! here converts silent session poisoning into a client retry. Clamp-call and
//! GDN-state counters give the probe lane the signal rates to attribute the
//! trigger class later.

use candle_core::{DType, Result, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// F32 decode chains this hot are already broken. ln(2e-7) as the argmax
/// floor: a greedy winner that unlikely means the head stopped
/// discriminating. 64.0 ceiling: healthy 35B greedy tops out ~33.5 (canary
/// 2026-11-05); the blowup class lands in the hundreds.
pub(crate) const MAX_GREEDY_LOGIT: f32 = 64.0;
pub(crate) const MIN_GREEDY_LOGPROB: f32 = -16.0;
/// D2H sync cost amortization: inspect GDN state every Nth flushed step.
pub(crate) const STATE_CHECK_INTERVAL: u64 = 64;
/// Ledger-only line: state scale is model-dependent, blowup announce precedes
/// any hard bail by steps here.
pub(crate) const GDN_STATE_LEDGER_MAX_ABS: f32 = 10_000.0;

const CLAMP_REPORT_EVERY: u64 = 65_536;

static ENABLED: OnceLock<bool> = OnceLock::new();
static STATE_CHECK_STEPS: AtomicU64 = AtomicU64::new(0);
pub(crate) static NAIVE_CLAMP_CALLS: AtomicU64 = AtomicU64::new(0);

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("MISTRALRS_STEP_SENTINEL").as_deref(),
            Ok("0") | Ok("false") | Ok("no") | Ok("off")
        )
    })
}

fn check_greedy_logit(max_logit: f32) -> Result<()> {
    if !max_logit.is_finite() || max_logit > MAX_GREEDY_LOGIT {
        candle_core::bail!(
            "step sentinel: greedy max_logit {max_logit} outside finite..{MAX_GREEDY_LOGIT}"
        );
    }
    Ok(())
}

fn check_greedy_logprob(logprob: f32, token: u32) -> Result<()> {
    if !logprob.is_finite() || logprob < MIN_GREEDY_LOGPROB {
        candle_core::bail!(
            "step sentinel: greedy token {token} logprob {logprob} below floor {MIN_GREEDY_LOGPROB}"
        );
    }
    Ok(())
}

fn check_sampled(logprob: f32, token: u32) -> Result<()> {
    if !logprob.is_finite() {
        candle_core::bail!("step sentinel: non-finite sample logprob {logprob} for token {token}");
    }
    Ok(())
}

pub(crate) fn greedy_logit(max_logit: f32) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    check_greedy_logit(max_logit).inspect_err(|e| eprintln!("{e}"))
}

pub(crate) fn greedy_logprob(logprob: f32, token: u32) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    check_greedy_logprob(logprob, token).inspect_err(|e| eprintln!("{e}"))
}

pub(crate) fn sampled(logprob: f32, token: u32) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    check_sampled(logprob, token).inspect_err(|e| eprintln!("{e}"))
}

pub(crate) fn note_clamp_call() {
    if !enabled() {
        return;
    }
    if NAIVE_CLAMP_CALLS.fetch_add(1, Ordering::Relaxed) % CLAMP_REPORT_EVERY == 0 {
        eprintln!(
            "step sentinel: attention clamp calls {}",
            NAIVE_CLAMP_CALLS.load(Ordering::Relaxed)
        );
    }
}

/// GDN recurrent-state health for one LinearAttention layer pool. Interval-
/// gated: one D2H scalar readback per STATE_CHECK_INTERVAL flushes, capture-
/// gated since a readback during GPU graph capture trips the capture ban.
pub(crate) fn gdn_state(state: &Tensor) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    if STATE_CHECK_STEPS.fetch_add(1, Ordering::Relaxed) % STATE_CHECK_INTERVAL != 0 {
        return Ok(());
    }
    if crate::utils::gather_guard::is_capturing(state.device()) {
        return Ok(());
    }
    let peak = state
        .abs()?
        .to_dtype(DType::F32)?
        .max_all()?
        .to_scalar::<f32>()?;
    if !peak.is_finite() {
        let e = "step sentinel: GDN state went non-finite";
        eprintln!("step sentinel: {e} (abs max {peak})");
        candle_core::bail!("{e}");
    }
    if peak > GDN_STATE_LEDGER_MAX_ABS {
        eprintln!(
            "step sentinel: GDN state abs max {peak:e} over ledger {GDN_STATE_LEDGER_MAX_ABS:e}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        check_greedy_logit, check_greedy_logprob, check_sampled, MAX_GREEDY_LOGIT,
        MIN_GREEDY_LOGPROB,
    };

    #[test]
    fn greedy_logit_bails_over_ceiling_and_an() {
        assert!(check_greedy_logit(24.0).is_ok());
        assert!(check_greedy_logit(-28.0).is_ok());
        assert!(check_greedy_logit(MAX_GREEDY_LOGIT).is_ok());
        // 35B temp-0 canary: healthy saturated step measured at 33.5.
        assert!(check_greedy_logit(33.5).is_ok());
        let e = check_greedy_logit(1.0e6).unwrap_err();
        assert!(e.to_string().contains("greedy max_logit"));
        assert!(check_greedy_logit(f32::NAN).is_err());
    }

    #[test]
    fn greedy_logprob_bails_under_floor_and_nan() {
        assert!(check_greedy_logprob(-0.5, 7).is_ok());
        assert!(check_greedy_logprob(MIN_GREEDY_LOGPROB, 7).is_ok());
        let e = check_greedy_logprob(MIN_GREEDY_LOGPROB - 0.01, 7).unwrap_err();
        assert!(e.to_string().contains("below floor"), "msg: {e}");
        assert!(check_greedy_logprob(f32::NAN, 7).is_err());
    }

    #[test]
    fn sampled_only_requires_finite() {
        assert!(check_sampled(-300.0, 5).is_ok());
        assert!(check_sampled(f32::NAN, 5).is_err());
    }
}
