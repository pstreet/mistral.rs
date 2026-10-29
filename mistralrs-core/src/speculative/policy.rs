#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeBatchPlan {
    pub proposal_len: usize,
    pub needs_target_hiddens: bool,
}

impl SpeculativeBatchPlan {
    pub const fn new(proposal_len: usize) -> Self {
        Self {
            proposal_len,
            needs_target_hiddens: true,
        }
    }

    pub const fn without_target_hiddens(mut self) -> Self {
        self.needs_target_hiddens = false;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeGraphPlan {
    pub proposal_len: usize,
    pub max_batch_size: Option<usize>,
}

impl SpeculativeGraphPlan {
    pub const fn new(proposal_len: usize, max_batch_size: Option<usize>) -> Self {
        Self {
            proposal_len,
            max_batch_size,
        }
    }
}

/// Tracks MTP draft acceptance rate for automatic fallback at long context.
/// When acceptance drops below `min_acceptance` over `window` batches, the
/// system falls back to no-speculation to avoid amplifying garbage outputs.
#[derive(Clone, Debug)]
pub struct MtpAcceptanceTracker {
    /// Sliding window of (proposed, accepted) pairs.
    window: Vec<(usize, usize)>,
    /// Minimum acceptance ratio to keep speculation enabled (0.0..1.0).
    min_acceptance: f64,
    /// Minimum number of batches to wait before triggering fallback.
    min_batches: usize,
}

impl MtpAcceptanceTracker {
    pub fn new(min_acceptance: f64, min_batches: usize) -> Self {
        Self {
            window: Vec::new(),
            min_acceptance,
            min_batches,
        }
    }

    /// Record one batch observation. Returns `Some(())` when fallback should
    /// be triggered (acceptance rate dropped below threshold and we have
    /// enough samples).
    pub fn record(&mut self, proposed: usize, accepted: usize) -> Option<()> {
        self.window.push((proposed, accepted));
        // Keep window bounded
        const MAX_WINDOW: usize = 50;
        if self.window.len() > MAX_WINDOW {
            self.window.drain(..self.window.len() - MAX_WINDOW);
        }
        if self.window.len() < self.min_batches {
            return None;
        }
        let total_proposed: usize = self.window.iter().map(|(p, _)| *p).sum();
        let total_accepted: usize = self.window.iter().map(|(_, a)| *a).sum();
        if total_proposed == 0 {
            return None;
        }
        let rate = total_accepted as f64 / total_proposed as f64;
        if rate < self.min_acceptance {
            return Some(());
        }
        None
    }

    /// Reset the tracker (e.g., after bypass is triggered).
    pub fn reset(&mut self) {
        self.window.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeBatchObservation {
    pub batch_size: usize,
    pub proposal_len: usize,
    pub sequences: usize,
    pub proposed_drafts: usize,
    pub accepted_drafts: usize,
}

#[cfg(test)]
mod tests {
    use super::SpeculativeBatchPlan;

    #[test]
    fn target_hiddens_are_required_by_default() {
        assert!(SpeculativeBatchPlan::new(7).needs_target_hiddens);
        assert!(
            !SpeculativeBatchPlan::new(7)
                .without_target_hiddens()
                .needs_target_hiddens
        );
    }
}
