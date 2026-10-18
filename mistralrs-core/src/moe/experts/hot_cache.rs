use std::sync::{mpsc, Arc, Mutex, OnceLock};

use candle_core::{DType, Device, Result, Tensor, D};

use mistralrs_quant::QuantMethod;

use super::backends::FastExpertsWeights;
use super::forward::{MoEForward, MoEForwardConfig, MoEForwardShape};

// Admission after this many routed uses; keeps one-hit experts out of the cache.
const HOT_ADMISSION_USES: u32 = 2;

pub(super) struct HotSources {
    pub(super) gate: Arc<dyn QuantMethod>,
    pub(super) up: Arc<dyn QuantMethod>,
    pub(super) down: Arc<dyn QuantMethod>,
}

#[derive(Clone)]
enum HotSlot {
    Empty,
    Filling,
    Filled,
}

struct HotState {
    slot_kind: Vec<HotSlot>,
    slot_expert: Vec<u32>,
    slot_of_expert: Vec<Option<u32>>,
    last_used: Vec<u64>,
    use_count: Vec<u32>,
    tick: u64,
    // [slots, 2 * inter, hidden] and [slots, hidden, inter], BF16; spliced per fill
    gate_up: Option<Tensor>,
    down: Option<Tensor>,
}

struct HotShared {
    state: Mutex<HotState>,
    sources: HotSources,
    device: Device,
    inter: usize,
    hidden: usize,
    slots: usize,
    num_experts: usize,
}

struct FillJob {
    shared: Arc<HotShared>,
    slot: usize,
    expert: usize,
}

// One fill worker for every layer cache: fills are ~ms each and must never
// stall a decode step.
fn fill_sender() -> &'static mpsc::Sender<FillJob> {
    static SENDER: OnceLock<mpsc::Sender<FillJob>> = OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<FillJob>();
        std::thread::spawn(move || {
            for job in rx {
                if let Err(err) = run_fill(&job) {
                    tracing::error!("expert hot cache fill failed: {err}");
                    let mut state = job.shared.state.lock().unwrap();
                    state.slot_kind[job.slot] = HotSlot::Empty;
                    state.slot_expert[job.slot] = u32::MAX;
                }
            }
        });
        tx
    })
}

fn run_fill(job: &FillJob) -> Result<()> {
    let (device, inter, hidden) = (&job.shared.device, job.shared.inter, job.shared.hidden);
    let sources = &job.shared.sources;
    let (slot, expert) = (job.slot, job.expert);
    let gate = sources
        .gate
        .dequantize_expert_rows(expert, inter, hidden, device)?
        .expect("gguf experts keep quantized storage");
    let up = sources
        .up
        .dequantize_expert_rows(expert, inter, hidden, device)?
        .expect("gguf experts keep quantized storage");
    let down = sources
        .down
        .dequantize_expert_rows(expert, hidden, inter, device)?
        .expect("gguf experts keep quantized storage");
    let gate_up = Tensor::cat(&[gate, up], 0)?.unsqueeze(0)?;
    let down = down.unsqueeze(0)?;
    let mut state = job.shared.state.lock().unwrap();
    let stacked_gate_up = match &state.gate_up {
        Some(old) => splice(old, &gate_up, slot)?,
        None => {
            let zeros = Tensor::zeros(
                (job.shared.slots, gate_up.dim(1)?, gate_up.dim(2)?),
                DType::BF16,
                device,
            )?;
            splice(&zeros, &gate_up, slot)?
        }
    };
    let stacked_down = match &state.down {
        Some(old) => splice(old, &down, slot)?,
        None => {
            let zeros = Tensor::zeros(
                (job.shared.slots, down.dim(1)?, down.dim(2)?),
                DType::BF16,
                device,
            )?;
            splice(&zeros, &down, slot)?
        }
    };
    state.gate_up = Some(stacked_gate_up);
    state.down = Some(stacked_down);
    state.slot_kind[slot] = HotSlot::Filled;
    state.slot_of_expert[expert] = Some(slot as u32);
    Ok(())
}

// dense [slots, ...] stays stable: unfilled rows keep zeros and are never routed to
fn splice(stacked: &Tensor, slot_tensor: &Tensor, slot: usize) -> Result<Tensor> {
    let slots = stacked.dim(0)?;
    let below = stacked.narrow(0, 0, slot)?;
    let above = stacked.narrow(0, slot + 1, slots - slot - 1)?;
    if slot == 0 {
        Ok(Tensor::cat(&[slot_tensor, &above], 0)?)
    } else {
        Ok(Tensor::cat(&[&below, slot_tensor, &above], 0)?)
    }
}

pub(super) struct HotExpertCache {
    shared: Arc<HotShared>,
}

pub(super) struct HotForward<'a> {
    pub xs: &'a Tensor,
    pub xs_flat: &'a Tensor,
    pub topk_weights: &'a Tensor,
    pub topk_ids: &'a Tensor,
    pub cold: &'a FastExpertsWeights,
    pub config: MoEForwardConfig,
    pub shape: MoEForwardShape,
}

impl HotExpertCache {
    pub(super) fn new(
        slots: usize,
        num_experts: usize,
        inter: usize,
        hidden: usize,
        device: &Device,
        sources: HotSources,
    ) -> Result<Self> {
        if slots == 0 || !device.is_cuda() {
            candle_core::bail!("expert hot cache requires cuda and at least one slot");
        }
        let shared = Arc::new(HotShared {
            state: Mutex::new(HotState {
                slot_kind: vec![HotSlot::Empty; slots],
                slot_expert: vec![u32::MAX; slots],
                slot_of_expert: vec![None; num_experts],
                last_used: vec![0; slots],
                use_count: vec![0; num_experts],
                tick: 0,
                gate_up: None,
                down: None,
            }),
            sources,
            device: device.clone(),
            inter,
            hidden,
            slots,
            num_experts,
        });
        Ok(Self { shared })
    }

    /// Split forward: hot pairs against VRAM slots, cold pairs through the CPU
    /// gather path, partial weighted sums added. None when no slot is filled
    /// yet (caller falls through to the unsplit path).
    pub(super) fn try_forward_split(&self, f: HotForward) -> Result<Option<Tensor>> {
        let topk = f.config.num_experts_per_tok;
        let hidden = f.shape.hidden_dim;
        let tokens = f.shape.num_tokens;
        let n_pairs = tokens * topk;
        let ids_host = f
            .topk_ids
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        let weights_host = f
            .topk_weights
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;

        let (gate_up, down, slot_of_pair, anchor) = {
            let mut state = self.shared.state.lock().unwrap();
            state.tick += 1;
            let tick = state.tick;
            let mut slot_of_pair = vec![u32::MAX; n_pairs];
            let mut anchor = None;
            for (pair, &e) in ids_host.iter().enumerate() {
                let e = e as usize;
                state.use_count[e] = state.use_count[e].saturating_add(1);
                if let Some(slot) = state.slot_of_expert[e] {
                    slot_of_pair[pair] = slot;
                    state.last_used[slot as usize] = tick;
                    if anchor.is_none() {
                        anchor = Some(slot);
                    }
                }
            }
            self.admit(&mut state, &ids_host, tick);
            if anchor.is_none() {
                return Ok(None);
            }
            (
                state
                    .gate_up
                    .clone()
                    .expect("filled slots imply stacked weights"),
                state
                    .down
                    .clone()
                    .expect("filled slots imply stacked weights"),
                slot_of_pair,
                anchor.unwrap(),
            )
        };

        let dev = f.xs.device().clone();
        let mut remapped_ids = Vec::with_capacity(n_pairs);
        let mut remapped_weights = Vec::with_capacity(n_pairs);
        let mut cold_pairs: Vec<u32> = Vec::new();
        for (pair, &slot) in slot_of_pair.iter().enumerate() {
            if slot != u32::MAX {
                remapped_ids.push(slot);
                remapped_weights.push(weights_host[pair]);
            } else {
                remapped_ids.push(anchor);
                remapped_weights.push(0.0);
                cold_pairs.push(pair as u32);
            }
        }
        let w_per_pair = remapped_weights.clone();
        let ids_t = Tensor::from_vec(remapped_ids, (tokens, topk), &dev)?;
        let weights_t = Tensor::from_vec(remapped_weights, (tokens, topk), &dev)?;
        let inter = self.shared.inter;
        let dev_t = f.xs.device().clone();
        let mut hot_partial = Tensor::zeros((tokens, hidden), f.xs.dtype(), &dev_t)?;
        let mut slot_pairs: Vec<Vec<usize>> = vec![Vec::new(); gate_up.dim(0)?];
        for (pair, &slot) in slot_of_pair.iter().enumerate() {
            if slot != u32::MAX {
                slot_pairs[slot as usize].push(pair);
            }
        }
        for (slot, pairs) in slot_pairs.iter().enumerate() {
            if pairs.is_empty() {
                continue;
            }
            let m = pairs.len();
            let token_idx: Vec<u32> = pairs.iter().map(|&p| (p / topk) as u32).collect();
            let token_idx_t = Tensor::from_vec(token_idx.clone(), m, &dev_t)?;
            let xs_pairs = f.xs_flat.index_select(&token_idx_t, 0)?;
            let gate_w = gate_up.narrow(0, slot, 1)?.squeeze(0)?.t()?;
            let gate_up_out = xs_pairs.matmul(&gate_w)?;
            let gate = gate_up_out.narrow(1, 0, inter)?.contiguous()?;
            let up = gate_up_out.narrow(1, inter, inter)?.contiguous()?;
            let down_inputs = (up * gate.apply(&f.config.act)?)?.reshape((m, inter))?;
            let down_w = down.narrow(0, slot, 1)?.squeeze(0)?.t()?;
            let down_out = down_inputs.matmul(&down_w)?;
            let w: Vec<f32> = pairs.iter().map(|&p| w_per_pair[p]).collect();
            let w_t = Tensor::from_vec(w, (m, 1), &dev_t)?.to_dtype(f.xs.dtype())?;
            let weighted = down_out.broadcast_mul(&w_t)?;
            hot_partial = hot_partial.index_add(&token_idx_t, &weighted, 0)?;
        }
        if cold_pairs.is_empty() {
            return Ok(Some(hot_partial));
        }

        if cold_pairs.is_empty() {
            return Ok(Some(hot_partial));
        }

        let t_cold = std::time::Instant::now();
        let n_cold = cold_pairs.len();
        let cold_tokens: Vec<u32> = cold_pairs.iter().map(|&p| p / topk as u32).collect();
        let cold_ids: Vec<u32> = cold_pairs.iter().map(|&p| ids_host[p as usize]).collect();
        let cold_weights: Vec<f32> = cold_pairs
            .iter()
            .map(|&p| weights_host[p as usize])
            .collect();
        let cold_ids = Tensor::from_vec(cold_ids, (n_cold, 1), &Device::Cpu)?;
        let cold_weights = Tensor::from_vec(cold_weights, (n_cold, 1), &Device::Cpu)?;
        let token_idx_cpu = Tensor::from_vec(cold_tokens.clone(), n_cold, &Device::Cpu)?;
        let xs_cpu = f.xs.to_device(&Device::Cpu)?;
        let cold_xs = xs_cpu
            .reshape((tokens, hidden))?
            .index_select(&token_idx_cpu, 0)?
            .reshape((n_cold, 1, hidden))?;
        let cold_flat = cold_xs.reshape((n_cold, hidden))?;
        let cold_forward = MoEForward {
            xs: &cold_xs,
            xs_flat: &cold_flat,
            topk_weights: &cold_weights,
            topk_ids: &cold_ids,
            original_dtype: f.xs.dtype(),
            shape: MoEForwardShape::new(n_cold, 1, hidden),
            lora: None,
        };
        let cold_out = f.cold.forward_gather(
            &cold_forward,
            MoEForwardConfig {
                num_experts: f.config.num_experts,
                num_experts_per_tok: 1,
                act: f.config.act,
            },
        )?;
        let token_idx_dev = token_idx_cpu.to_device(&dev_t)?;
        let scattered = Tensor::zeros((tokens, hidden), f.xs.dtype(), &dev_t)?.index_add(
            &token_idx_dev,
            &cold_out.to_device(&dev_t)?,
            0,
        )?;
        Ok(Some((&hot_partial + &scattered)?))
    }

    fn admit(&self, state: &mut HotState, ids: &[u32], tick: u64) {
        let mut seen = vec![false; self.shared.num_experts];
        for &e in ids {
            let e = e as usize;
            if seen[e] {
                continue;
            }
            seen[e] = true;
            let filling = (0..self.shared.slots).any(|s| {
                matches!(state.slot_kind[s], HotSlot::Filling) && state.slot_expert[s] == e as u32
            });
            if state.slot_of_expert[e].is_some()
                || state.use_count[e] < HOT_ADMISSION_USES
                || filling
            {
                continue;
            }
            let slot = (0..self.shared.slots)
                .find(|&s| matches!(state.slot_kind[s], HotSlot::Empty))
                .or_else(|| {
                    (0..self.shared.slots)
                        .filter(|&s| matches!(state.slot_kind[s], HotSlot::Filled))
                        .filter(|&s| state.last_used[s] < tick)
                        .min_by_key(|&s| state.last_used[s])
                });
            let Some(slot) = slot else {
                continue;
            };
            state.slot_kind[slot] = HotSlot::Filling;
            state.slot_expert[slot] = e as u32;
            let _ = fill_sender().send(FillJob {
                shared: self.shared.clone(),
                slot,
                expert: e,
            });
        }
    }
}
