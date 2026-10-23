use std::sync::{mpsc, Arc, Mutex, OnceLock};

use candle_core::{DType, Device, Result, Tensor};

use mistralrs_quant::QuantMethod;

use super::backends::FastExpertsWeights;
use super::forward::{MoEForward, MoEForwardConfig, MoEForwardShape};

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

#[derive(Clone)]
struct SlotWeights {
    gate_up: Arc<Tensor>,
    down: Arc<Tensor>,
}

struct HotState {
    slot_kind: Vec<HotSlot>,
    slot_expert: Vec<u32>,
    slot_of_expert: Vec<Option<u32>>,
    last_used: Vec<u64>,
    use_count: Vec<u32>,
    tick: u64,
    slot_weights: Vec<Option<SlotWeights>>,
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

// Single worker for every layer cache: fills are ~ms each and must never stall a decode step.
fn fill_sender() -> &'static mpsc::Sender<FillJob> {
    static SENDER: OnceLock<mpsc::Sender<FillJob>> = OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<FillJob>();
        std::thread::spawn(move || {
            for job in rx {
                if let Err(err) = run_fill(&job) {
                    tracing::error!("expert hot cache fill failed: {err}");
                    let mut state = job.shared.state.lock().unwrap();
                    let slot = job.slot;
                    let expert = state.slot_expert[slot];
                    if expert != u32::MAX {
                        state.slot_of_expert[expert as usize] = None;
                    }
                    state.slot_kind[slot] = HotSlot::Empty;
                    state.slot_expert[slot] = u32::MAX;
                }
            }
        });
        tx
    })
}

fn run_fill(job: &FillJob) -> Result<()> {
    let shared = &job.shared;
    let (device, inter, hidden) = (&shared.device, shared.inter, shared.hidden);
    let sources = &shared.sources;
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
    // Small per-expert tensors only; no whole-stack splice.
    let gate_up = Tensor::cat(&[gate, up], 0)?.unsqueeze(0)?.contiguous()?;
    let down = down.unsqueeze(0)?.contiguous()?;
    let mut state = shared.state.lock().unwrap();
    state.slot_weights[slot] = Some(SlotWeights {
        gate_up: Arc::new(gate_up),
        down: Arc::new(down),
    });
    state.slot_kind[slot] = HotSlot::Filled;
    state.slot_expert[slot] = expert as u32;
    state.slot_of_expert[expert] = Some(slot as u32);
    Ok(())
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
                slot_weights: vec![None; slots],
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

    pub(super) fn try_forward_split(&self, f: HotForward) -> Result<Option<Tensor>> {
        let shared = &self.shared;
        let (tokens, topk, hidden) = (f.shape.num_tokens, f.topk_ids.dims()[1], f.shape.hidden_dim);
        let inter = shared.inter;
        let dev = f.xs.device().clone();
        let dev_t = dev.clone();
        let topk_ids_host: Vec<u32> = f
            .topk_ids
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let weights_host: Vec<f32> = f.topk_weights.flatten_all()?.to_vec1::<f32>()?;
        let n_pairs = topk_ids_host.len();

        let (slot_of_pair, anchor) = {
            let mut state = shared.state.lock().unwrap();
            state.tick += 1;
            let tick = state.tick;
            let mut slot_of_pair = vec![u32::MAX; n_pairs];
            let mut anchor = None;
            for (pair, &e) in topk_ids_host.iter().enumerate() {
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
            self.admit(&mut state, &topk_ids_host);
            (slot_of_pair, anchor)
        };
        // No filled slot means nothing to serve from VRAM this step.
        if anchor.is_none() {
            return Ok(None);
        }

        let mut cold_pairs: Vec<u32> = Vec::new();
        let mut slot_pairs: Vec<Vec<usize>> = vec![Vec::new(); shared.slots];
        for (pair, &slot) in slot_of_pair.iter().enumerate() {
            if slot == u32::MAX {
                cold_pairs.push(pair as u32);
            } else {
                slot_pairs[slot as usize].push(pair);
            }
        }
        let active: Vec<usize> = (0..shared.slots)
            .filter(|s| !slot_pairs[*s].is_empty())
            .collect();

        let mut hot_partial = Tensor::zeros((tokens, hidden), f.xs.dtype(), &dev_t)?;
        if !active.is_empty() {
            let s = active.len();
            let max_m = active
                .iter()
                .map(|sl| slot_pairs[*sl].len())
                .max()
                .unwrap_or(0);
            let rows = s * max_m;
            let mut batch_idx: Vec<u32> = Vec::with_capacity(rows);
            let mut pair_w: Vec<f32> = Vec::with_capacity(rows);
            for sl in &active {
                let pairs = &slot_pairs[*sl];
                for &pair in pairs {
                    batch_idx.push((pair / topk) as u32);
                    pair_w.push(weights_host[pair]);
                }
                for _ in pairs.len()..max_m {
                    // Pad slots map to token 0 with zero weight (no contribution).
                    batch_idx.push(0);
                    pair_w.push(0.0);
                }
            }
            // Pull active slot weights under the mutex; cat outside the lock.
            let (gu_list, dn_list): (Vec<Arc<Tensor>>, Vec<Arc<Tensor>>) = {
                let state = shared.state.lock().unwrap();
                let mut gus = Vec::with_capacity(s);
                let mut dns = Vec::with_capacity(s);
                for sl in &active {
                    if let Some(w) = &state.slot_weights[*sl] {
                        gus.push(w.gate_up.clone());
                        dns.push(w.down.clone());
                    }
                }
                (gus, dns)
            };
            if gu_list.len() == active.len() {
                let idx_t = Tensor::from_vec(batch_idx, rows, &dev)?;
                let xs_rows = f
                    .xs_flat
                    .to_dtype(DType::BF16)?
                    .index_select(&idx_t, 0)?
                    .reshape((s, max_m, hidden))?;
                let gu_cat =
                    Tensor::cat(&gu_list.iter().map(|a| a.as_ref()).collect::<Vec<_>>(), 0)?;
                let dn_cat =
                    Tensor::cat(&dn_list.iter().map(|a| a.as_ref()).collect::<Vec<_>>(), 0)?;
                let gate_up_t = gu_cat.transpose(1, 2)?; // [s, h, 2i] view
                let down_t = dn_cat.transpose(1, 2)?; // [s, inter, h] view
                let gu_out = xs_rows.matmul(&gate_up_t)?; // [s, max_m, 2i]
                let gate = gu_out.narrow(2, 0, inter)?;
                let up = gu_out.narrow(2, inter, inter)?;
                let down_in = (up * gate.apply(&f.config.act)?)?; // [s, max_m, inter]
                let hot_out = down_in.matmul(&down_t)?; // [s, max_m, hidden]
                let w_t = Tensor::from_vec(pair_w, (rows, 1), &dev_t)?.to_dtype(f.xs.dtype())?;
                let flat_out = hot_out.reshape((rows, hidden))?;
                let weighted = flat_out.broadcast_mul(&w_t)?;
                hot_partial = hot_partial.index_add(&idx_t, &weighted, 0)?;
            }
        }
        if cold_pairs.is_empty() {
            return Ok(Some(hot_partial));
        }
        // Cold path: CPU gather for the pairs whose experts are not cached.
        let n_cold = cold_pairs.len();
        let cold_tokens: Vec<u32> = cold_pairs.iter().map(|&p| p / topk as u32).collect();
        let cold_ids: Vec<u32> = cold_pairs
            .iter()
            .map(|&p| topk_ids_host[p as usize])
            .collect();
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
        // Fast path: the shared direct indexed pass (gate+up fused, down via
        // indexed_gemv) skips forward_gather's per-call tensor-op overhead.
        // Weighting runs in F32 to match forward_gather's precision; the
        // expert axis is size 1 here, so no sum is needed.
        let cold_forward_config = MoEForwardConfig {
            num_experts: f.config.num_experts,
            num_experts_per_tok: 1,
            act: f.config.act,
        };
        let cold_out = match f.cold.forward_indexed(&cold_forward, cold_forward_config)? {
            Some(down) => {
                let input_dtype = f.xs.dtype();
                let down = down.squeeze(1)?.to_dtype(DType::F32)?; // [n, hidden]
                let w = cold_weights.to_dtype(DType::F32)?; // [n, 1]
                down.broadcast_mul(&w)?.to_dtype(input_dtype)?
            }
            None => f.cold.forward_gather(&cold_forward, cold_forward_config)?,
        };
        let token_idx_dev = token_idx_cpu.to_device(&dev_t)?;
        let scattered = Tensor::zeros((tokens, hidden), f.xs.dtype(), &dev_t)?.index_add(
            &token_idx_dev,
            &cold_out.to_device(&dev_t)?,
            0,
        )?;
        Ok(Some((&hot_partial + &scattered)?))
    }

    fn admit(&self, state: &mut HotState, ids: &[u32]) {
        // One admission attempt per distinct expert per step: routing ids
        // contain duplicates (several pairs often select the same expert)
        // and the slot is only bound on fill completion, so a repeat id
        // would double-admit and evict an extra victim.
        let mut unique = ids.to_vec();
        unique.sort_unstable();
        unique.dedup();
        for &e in &unique {
            let e = e as usize;
            if state.slot_of_expert[e].is_some() {
                continue;
            }
            // Prefetch: the router already selected this expert, so start
            // the fill (dequant + H2D) now and overlap it with this step's
            // compute. Empty slots are free; evicting a warm slot for a
            // possibly one-off expert needs the stronger use-count gate.
            if let Some(slot) =
                (0..self.shared.slots).find(|&s| matches!(state.slot_kind[s], HotSlot::Empty))
            {
                state.slot_kind[slot] = HotSlot::Filling;
                state.slot_expert[slot] = e as u32;
                let _ = fill_sender().send(FillJob {
                    shared: self.shared.clone(),
                    slot,
                    expert: e,
                });
                continue;
            }
            if state.use_count[e] < HOT_ADMISSION_USES {
                continue;
            }
            // All slots busy: evict the LRU filled slot.
            if let Some(slot) = (0..self.shared.slots)
                .min_by_key(|&s| {
                    if matches!(state.slot_kind[s], HotSlot::Filled) {
                        state.last_used[s]
                    } else {
                        u64::MAX
                    }
                })
                .filter(|&s| matches!(state.slot_kind[s], HotSlot::Filled))
            {
                let old_expert = state.slot_expert[slot];
                if old_expert != u32::MAX {
                    state.slot_of_expert[old_expert as usize] = None;
                }
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
}
