#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use crate::gdn::{
    try_forward_grouped_packed_gdn, GatedDeltaNet, GdnConfig, GdnForwardStash,
    GdnInputProjectionKind, GdnLayerCache, GdnSpeculativeStash, GdnStateDType,
    GdnTransitionCommitConfig, GdnTransitionStash, GdnVHeadLayout, PackedGdnLayout,
};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    get_mut_arcmutex,
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
        RecurrentStateLayout,
    },
    layers::{
        contains_tensor_or_weight_source, embedding_with_legacy_tied_uqff, linear_no_bias,
        CausalMasker, GemmaRmsNorm, RotaryEmbedding, Sdpa,
    },
    layers_masker::PastKvLenCache,
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, ForwardMaskCache, IsqModel, KvCache, ModelForwardContext,
        NormalLoadingMetadata, NormalModel, RecurrentBatchKind,
    },
    serde_default_fn,
    speculative::{
        paged_rows::make_paged_rows_metadata, proposer::sample_draft_rows, MtpRuntimeConfig,
        SpeculativeAttachInfo, SpeculativeBatchPlan, SpeculativeCommitRow, SpeculativeConfig,
        SpeculativeGraphPlan, SpeculativeGraphState, SpeculativeKvCache, SpeculativePrefillCtx,
        SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
        SpeculativeTargetMixin, TargetAttentionInputs,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, default_tie, true);
serde_default_fn!(f64, default_rope_theta, 10_000.0);
serde_default_fn!(f64, default_rms_norm_eps, 1e-6);
serde_default_fn!(usize, default_full_attn_interval, 4);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(usize, default_decoder_sparse_step, 1);
serde_default_fn!(f64, default_partial_rotary_factor, 0.25);
serde_default_fn!(bool, default_norm_topk_prob, true);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: crate::layers::Activation,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub head_dim: usize,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    // GDN (Gated Delta Net) config
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub mamba_ssm_dtype: GdnStateDType,
    // MoE config
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default, rename = "_mistralrs_gdn_v_head_layout")]
    gdn_v_head_layout: GdnVHeadLayout,
    /// Injected by the loader when the built-in MTP head should be loaded (see `MTP_CONFIG_KEY`).
    #[serde(default, rename = "_mistralrs_mtp")]
    pub mtp: bool,
}

impl Config {
    /// Whether to load the built-in MTP head and report its layer in the paged KV mask.
    pub fn mtp_layers(&self) -> usize {
        usize::from(self.mtp)
    }

    /// Paged-KV layers index of the MTP block (right after the main stack).
    pub fn mtp_kv_layer_idx(&self) -> usize {
        self.num_hidden_layers
    }

    /// Paged-KV mask over the main stack plus any MTP block appended after it.
    #[allow(dead_code)]
    pub fn paged_kv_layers(&self) -> Vec<bool> {
        let mut layers = self
            .layer_types()
            .into_iter()
            .map(|ty| matches!(ty, LayerType::FullAttention))
            .collect::<Vec<_>>();
        layers.extend(std::iter::repeat_n(true, self.mtp_layers()));
        layers
    }
}

#[derive(Debug, Clone)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

impl Config {
    pub fn layer_types(&self) -> Vec<LayerType> {
        (0..self.num_hidden_layers)
            .map(|i| {
                // full_attention_interval=4 means layers 3,7,11,... are full attention
                if (i + 1) % self.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    /// Total key dimension = linear_num_key_heads * linear_key_head_dim
    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Total value dimension = linear_num_value_heads * linear_value_head_dim
    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Conv dim for GDN = key_dim * 2 + value_dim (q, k, v before split)
    pub fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

impl GdnConfig for Config {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }
    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }
    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }
    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.quantization_config
    }
    fn v_head_layout(&self) -> GdnVHeadLayout {
        self.gdn_v_head_layout
    }
}

// ====================== Full Attention layer ======================

#[allow(dead_code)]
struct FullAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl FullAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rotary_emb: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb_sa = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // q_proj outputs num_heads * head_dim * 2 (doubled for gate)
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim * 2, // q + gate
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("o_proj"),
        )?;

        // QK norms use (1+weight) formulation; pass loading_isq=false to ensure device placement
        let vb_sa_norms = mapper.set_device(layer_idx, vb.pp("self_attn"), false);
        let q_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("q_norm"))?;
        let k_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("k_norm"))?;

        let sliding_window = None;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(num_kv_heads, num_heads, comm)?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let (q_gate, k, v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        // Split q_gate into q and gate: first reshape to per-head (head_dim*2), then chunk
        // Reference: view(*input_shape, -1, head_dim*2), chunk(2, dim=-1)
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        // gate: (batch, seq, num_heads, head_dim) -> (batch, seq, num_heads * head_dim)
        let gate = gate.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        // Reshape to (batch, heads, seq, head_dim)
        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let rope_positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
        (q, k) = self.rotary_emb.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            rope_positions,
        )?;
        let metadata = ctx.paged_layer(layer_idx);

        // Standard attention
        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(ctx.flash_params()),
                    &self.sdpa_params,
                )?
            }
        };

        y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        // Apply output gate: y = y * sigmoid(gate)
        if let Some(res) = crate::ops::try_fused_gated_projection(
            &gate,
            &y,
            crate::layers::Activation::Sigmoid,
            &*self.o_proj,
        )? {
            return Ok(res);
        }
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        y = y.broadcast_mul(&gate)?;

        let res = self.o_proj.forward(&y)?;
        Ok(res)
    }
}

// ====================== MoE ======================

/// Sparse MoE block with shared expert and shared expert gate
struct SparseMoeBlock {
    gate: Linear,
    gate_lora: Option<Arc<mistralrs_quant::LoraSiteHandle>>,
    experts: MoEExperts,
    shared_expert: crate::layers::Mlp,
    shared_expert_gate: Linear,
    shared_expert_gate_lora: Option<Arc<mistralrs_quant::LoraSiteHandle>>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        let gate_vb = vb.pp("gate").set_device(layer_device.clone());
        let gate = linear_no_bias(cfg.hidden_size, cfg.num_experts, gate_vb.clone())?;
        let gate_lora = mistralrs_quant::register_dynamic_lora_site(
            &gate_vb,
            mistralrs_quant::LoraLinearSpec::replicated(cfg.hidden_size, cfg.num_experts),
        )?;

        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            expert_proj_names: crate::moe::ExpertProjNames::DEFAULT,
        };

        let experts = MoEExperts::new(
            &moe_cfg,
            vb.clone(),
            layer_device.clone(),
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        // Shared expert
        let shared_expert = crate::layers::Mlp::new(
            vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;

        // Shared expert gate: (1, hidden_size) -> sigmoid
        let shared_expert_gate_vb = vb.pp("shared_expert_gate");
        let mut seg_w = shared_expert_gate_vb.get((1, cfg.hidden_size), "weight")?;
        if loading_isq {
            seg_w = seg_w.to_device(&layer_device)?;
        }
        let shared_expert_gate = Linear::new(seg_w, None);
        let shared_expert_gate_lora = mistralrs_quant::register_dynamic_lora_site(
            &shared_expert_gate_vb.set_device(layer_device),
            mistralrs_quant::LoraLinearSpec::replicated(cfg.hidden_size, 1),
        )?;

        Ok(Self {
            gate,
            gate_lora,
            experts,
            shared_expert,
            shared_expert_gate,
            shared_expert_gate_lora,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn router_logits(&self, xs_flat: &Tensor) -> Result<Tensor> {
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        if self.gate_lora.is_none() {
            if let Some(logits) = crate::ops::moe_router_gemv(xs_flat, self.gate.weight())? {
                return Ok(logits);
            }
        }
        self.gate.forward(xs_flat)
    }

    fn shared_gate_logits(&self, xs_flat: &Tensor) -> Result<Tensor> {
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        if self.shared_expert_gate_lora.is_none() {
            if let Some(logits) =
                crate::ops::moe_router_gemv(xs_flat, self.shared_expert_gate.weight())?
            {
                // The BLAS path produces the weight dtype; keep downstream
                // dtypes identical.
                return logits.to_dtype(xs_flat.dtype());
            }
        }
        self.shared_expert_gate.forward(xs_flat)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.router_logits(&xs_flat)?;
        let router_logits = match &self.gate_lora {
            Some(site) => mistralrs_quant::apply_dynamic_lora_delta(site, &xs_flat, router_logits)?,
            None => router_logits,
        };
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: self.norm_topk_prob,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;

        let mut y = self.experts.forward(xs, topk.values, &topk.indices)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;

        // 3. Shared expert with sigmoid gating
        let shared_out = self.shared_expert.forward(xs)?;

        let shared_gate = self.shared_gate_logits(&xs_flat)?;
        let shared_gate = match &self.shared_expert_gate_lora {
            Some(site) => mistralrs_quant::apply_dynamic_lora_delta(site, &xs_flat, shared_gate)?,
            None => shared_gate,
        };
        let shared_gate = candle_nn::ops::sigmoid(&shared_gate)?;
        let shared_gate = shared_gate.reshape((b_size, seq_len, 1))?;
        let shared_out = shared_out.broadcast_mul(&shared_gate)?;

        // 4. Combine
        y + shared_out
    }
}

// ====================== Decoder Layer ======================

enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

fn gdn_input_projection_kind(vb: &ShardedVarBuilder) -> GdnInputProjectionKind {
    if contains_tensor_or_weight_source(vb, "in_proj_b.weight")
        && contains_tensor_or_weight_source(vb, "in_proj_a.weight")
    {
        GdnInputProjectionKind::Split
    } else if contains_tensor_or_weight_source(vb, "in_proj_qkv.weight") {
        GdnInputProjectionKind::SplitQkvzGroupedBa
    } else {
        GdnInputProjectionKind::Grouped
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedGdnSegment {
    token_range: Range<usize>,
    state_index: usize,
}

fn packed_gdn_segments(
    physical_batch: usize,
    physical_tokens: usize,
    query_lens: &[usize],
) -> Result<Vec<PackedGdnSegment>> {
    if physical_batch != 1 {
        candle_core::bail!(
            "Qwen3-Next packed GDN requires physical batch size 1, got {physical_batch}"
        );
    }
    if query_lens.is_empty() {
        candle_core::bail!("Qwen3-Next packed GDN requires at least one logical sequence");
    }
    let mut offset = 0usize;
    let mut segments = Vec::with_capacity(query_lens.len());
    for (state_index, &query_len) in query_lens.iter().enumerate() {
        if query_len == 0 {
            candle_core::bail!(
                "Qwen3-Next packed GDN logical sequence {state_index} has zero tokens"
            );
        }
        let end = offset
            .checked_add(query_len)
            .ok_or_else(|| candle_core::Error::msg("Qwen3-Next packed GDN length overflow"))?;
        segments.push(PackedGdnSegment {
            token_range: offset..end,
            state_index,
        });
        offset = end;
    }
    if offset != physical_tokens {
        candle_core::bail!(
            "Qwen3-Next packed GDN has {offset} logical tokens but {physical_tokens} physical tokens"
        );
    }
    Ok(segments)
}

fn validate_packed_gdn_state_rows(
    logical_batch: usize,
    conv_state_batch: usize,
    recurrent_state_batch: usize,
) -> Result<()> {
    if conv_state_batch != logical_batch {
        candle_core::bail!(
            "Qwen3-Next packed GDN has {conv_state_batch} convolution state rows but {logical_batch} logical sequences"
        );
    }
    if recurrent_state_batch != logical_batch {
        candle_core::bail!(
            "Qwen3-Next packed GDN has {recurrent_state_batch} recurrent state rows but {logical_batch} logical sequences"
        );
    }
    Ok(())
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    moe: SparseMoeBlock,
}

impl DecoderLayer {
    fn forward_attention(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let attn = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => attn,
            _ => candle_core::bail!("Expected full attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let attn_out = attn.forward(&x, attention_mask, kv_cache, ctx, layer_idx)?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.moe.forward(&normed)?;
        ffn_out + residual
    }

    fn forward_linear(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
        packed_layout: Option<&PackedGdnLayout>,
    ) -> Result<Tensor> {
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let gdn_out = if let Some(layout) = packed_layout {
            let query_lens = layout.query_lens();
            if batch_kind != RecurrentBatchKind::Prefill {
                candle_core::bail!("Qwen3-Next packed GDN cannot run a decode batch");
            }
            let (physical_batch, physical_tokens, _) = x.dims3()?;
            let (conv_state_batch, _, _) = cache.conv_state.dims3()?;
            let (recurrent_state_batch, _, _, _) = cache.recurrent_state.dims4()?;
            let segments = packed_gdn_segments(physical_batch, physical_tokens, query_lens)?;
            validate_packed_gdn_state_rows(
                segments.len(),
                conv_state_batch,
                recurrent_state_batch,
            )?;
            if x.dtype() != cache.conv_state.dtype() {
                candle_core::bail!(
                    "Qwen3-Next packed GDN dtype mismatch: tokens are {:?}, convolution state is {:?}",
                    x.dtype(),
                    cache.conv_state.dtype()
                );
            }
            if !x.device().same_device(cache.conv_state.device())
                || !x.device().same_device(cache.recurrent_state.device())
            {
                candle_core::bail!(
                    "Qwen3-Next packed GDN tokens and recurrent states are on different devices"
                );
            }

            if let Some(output) = try_forward_grouped_packed_gdn(gdn, &x, cache, layout)? {
                output
            } else {
                let mut outputs = Vec::with_capacity(segments.len());
                let mut next_conv_states = Vec::with_capacity(segments.len());
                let mut next_recurrent_states = Vec::with_capacity(segments.len());
                for segment in segments {
                    let segment_x =
                        x.narrow(1, segment.token_range.start, segment.token_range.len())?;
                    let mut segment_cache = GdnLayerCache {
                        conv_state: cache.conv_state.narrow(0, segment.state_index, 1)?,
                        recurrent_state: cache.recurrent_state.narrow(0, segment.state_index, 1)?,
                        state_layout: cache.state_layout,
                        slots: None,
                        pending_transitions: None,
                        deferred_state: None,
                    };
                    outputs.push(mistralrs_quant::with_lora_execution_row_range(
                        segment.token_range.clone(),
                        || gdn.forward(&segment_x, &mut segment_cache, RecurrentBatchKind::Prefill),
                    )?);
                    next_conv_states.push(segment_cache.conv_state);
                    next_recurrent_states.push(segment_cache.recurrent_state);
                }
                cache.conv_state = Tensor::cat(&next_conv_states, 0)?;
                cache.recurrent_state = Tensor::cat(&next_recurrent_states, 0)?;
                Tensor::cat(&outputs, 1)?
            }
        } else {
            gdn.forward(&x, cache, batch_kind)?
        };
        let x = (gdn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.moe.forward(&normed)?;
        ffn_out + residual
    }
}

impl FullAttention {
    fn forward_with_explicit(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        positions: &Tensor,
        kv_cache: (Tensor, Tensor),
        metadata: &PagedAttentionInputMetadata,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let (q_gate, k, v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        (q, k) = self.rotary_emb.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            &positions.reshape((b_sz * seq_len,))?,
        )?;

        let paged_attn = self.paged_attn.as_ref().ok_or_else(|| {
            candle_core::Error::msg("Qwen3Next MTP head requires paged attention")
        })?;
        let mut y = paged_attn.forward(
            &q,
            &k,
            &v,
            attention_mask,
            Some(kv_cache.0),
            Some(kv_cache.1),
            metadata,
            &self.sdpa_params,
            Some(flash_params),
        )?;

        y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        if let Some(res) = crate::ops::try_fused_gated_projection(
            &gate,
            &y,
            crate::layers::Activation::Sigmoid,
            &*self.o_proj,
        )? {
            return Ok(res);
        }
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        y = y.broadcast_mul(&gate)?;
        self.o_proj.forward(&y)
    }
}

impl DecoderLayer {
    fn forward_linear_with_stash(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
        checkpoint_lanes: usize,
        transition_checkpoints: bool,
        stash_out: Option<&mut Option<GdnSpeculativeStash>>,
    ) -> Result<Tensor> {
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let gdn_out = gdn.forward_with_stash(
            &x,
            cache,
            batch_kind,
            checkpoint_lanes,
            transition_checkpoints,
            stash_out,
        )?;
        let x = (gdn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.moe.forward(&normed)?;
        ffn_out + residual
    }
}

const MTP_FC_WEIGHT: &str = "mtp.fc.weight";
const GDN_PENDING_APPLY_MAX_LAYERS: usize = 32;
pub const DEFAULT_MTP_N_PREDICT: usize = 2;
const DEFAULT_MTP_N_PREDICT_LARGE: usize = 3;
const MTP_LARGE_HIDDEN_SIZE: usize = 4096;
const PLACEHOLDER_TOKEN: u32 = 0;

#[derive(Clone)]
struct SpecCapture {
    hidden: Tensor,
    positions: Tensor,
}

#[derive(Clone)]
struct GdnReplayStash {
    slots: Vec<u32>,
    layers: Vec<GdnLayerStash>,
}

#[derive(Clone)]
struct GdnLayerStash {
    layer_idx: usize,
    state_layout: RecurrentStateLayout,
    rollback: GdnLayerRollback,
}

#[derive(Clone)]
enum GdnLayerRollback {
    Replay {
        projected: GdnForwardStash,
        conv_state: Tensor,
        recurrent_state: Tensor,
    },
    Transition(GdnTransitionStash),
}

struct GdnPendingApplyGroup {
    device: Device,
    config: GdnTransitionCommitConfig,
    layers: Vec<usize>,
}

#[derive(Debug, PartialEq, Eq)]
struct GdnReplayBatch {
    keep_rows: usize,
    batch_indices: Vec<u32>,
    slots: Vec<u32>,
}

struct GdnReplayIndices {
    batch_indices: Tensor,
    slots: Tensor,
}

struct GdnCommitIndices {
    keep_rows: Tensor,
    slots: Tensor,
}

fn index_select_replay_rows(source: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if source.is_contiguous() {
        source.index_select(indices, 0)
    } else {
        source.contiguous()?.index_select(indices, 0)
    }
}

fn recurrent_checkpoint_devices_supported(devices: &[Device]) -> bool {
    cfg!(feature = "cuda") && !devices.is_empty() && devices.iter().all(Device::is_cuda)
}

fn should_stash_gdn_replay(
    native_speculative_commit: bool,
    store_spec_hidden: bool,
    query_len: usize,
    batch_kind: Option<RecurrentBatchKind>,
    continuation_without_cache: bool,
) -> bool {
    !native_speculative_commit
        && store_spec_hidden
        && query_len > 1
        && batch_kind == Some(RecurrentBatchKind::SpeculativeDecode)
        && continuation_without_cache
}

fn narrow_spec_graph_tensor(
    tensor: &Tensor,
    batch_dim: usize,
    captured_batch: usize,
    real_batch: usize,
    name: &str,
) -> Result<Tensor> {
    let tensor_batch = tensor.dim(batch_dim)?;
    if tensor_batch != captured_batch {
        candle_core::bail!(
            "speculative graph {name} has batch {tensor_batch}, expected {captured_batch}"
        );
    }
    if real_batch == captured_batch {
        Ok(tensor.clone())
    } else {
        tensor.narrow(batch_dim, 0, real_batch)
    }
}

fn narrow_spec_capture(capture: &mut SpecCapture, real_batch: usize) -> Result<()> {
    let captured_batch = capture.hidden.dim(0)?;
    if real_batch > captured_batch {
        candle_core::bail!(
            "speculative graph batch {real_batch} exceeds captured batch {captured_batch}"
        );
    }
    capture.hidden = narrow_spec_graph_tensor(
        &capture.hidden,
        0,
        captured_batch,
        real_batch,
        "hidden state",
    )?;
    capture.positions = narrow_spec_graph_tensor(
        &capture.positions,
        0,
        captured_batch,
        real_batch,
        "positions",
    )?;
    Ok(())
}

fn narrow_gdn_replay_stash(stash: &mut GdnReplayStash, real_batch: usize) -> Result<()> {
    let captured_batch = stash.slots.len();
    if real_batch > captured_batch {
        candle_core::bail!("GDN replay batch {real_batch} exceeds captured batch {captured_batch}");
    }
    for layer in &mut stash.layers {
        match &mut layer.rollback {
            GdnLayerRollback::Replay {
                projected,
                conv_state,
                recurrent_state,
            } => {
                projected.mixed_qkv = narrow_spec_graph_tensor(
                    &projected.mixed_qkv,
                    0,
                    captured_batch,
                    real_batch,
                    "mixed_qkv",
                )?;
                projected.convolved_qkv = narrow_spec_graph_tensor(
                    &projected.convolved_qkv,
                    0,
                    captured_batch,
                    real_batch,
                    "convolved_qkv",
                )?;
                projected.b =
                    narrow_spec_graph_tensor(&projected.b, 0, captured_batch, real_batch, "b")?;
                projected.a =
                    narrow_spec_graph_tensor(&projected.a, 0, captured_batch, real_batch, "a")?;
                *conv_state = narrow_spec_graph_tensor(
                    conv_state,
                    0,
                    captured_batch,
                    real_batch,
                    "conv_state",
                )?;
                *recurrent_state = narrow_spec_graph_tensor(
                    recurrent_state,
                    0,
                    captured_batch,
                    real_batch,
                    "recurrent_state",
                )?;
            }
            GdnLayerRollback::Transition(_) => {}
        }
    }
    stash.slots.truncate(real_batch);
    Ok(())
}

fn group_gdn_replay_batches(rows: &[(usize, usize)], slots: &[u32]) -> Result<Vec<GdnReplayBatch>> {
    let mut grouped = BTreeMap::<usize, Vec<(u32, u32)>>::new();
    for &(batch_idx, keep_rows) in rows {
        let tensor_idx = u32::try_from(batch_idx).map_err(|_| {
            candle_core::Error::msg(format!("GDN replay batch row {batch_idx} exceeds u32"))
        })?;
        let slot = *slots.get(batch_idx).ok_or_else(|| {
            candle_core::Error::msg(format!("GDN replay stash has no batch row {batch_idx}"))
        })?;
        grouped
            .entry(keep_rows)
            .or_default()
            .push((tensor_idx, slot));
    }
    Ok(grouped
        .into_iter()
        .map(|(keep_rows, rows)| GdnReplayBatch {
            keep_rows,
            batch_indices: rows.iter().map(|(batch_idx, _)| *batch_idx).collect(),
            slots: rows.into_iter().map(|(_, slot)| slot).collect(),
        })
        .collect())
}

fn refresh_gdn_stash_slots(stash: &mut GdnReplayStash, slots: &[u32]) -> Result<()> {
    let batch_size = stash.slots.len();
    if slots.len() < batch_size {
        candle_core::bail!(
            "GDN graph state has {batch_size} rows, but the live slot table has {}",
            slots.len()
        );
    }
    stash.slots.clear();
    stash.slots.extend_from_slice(&slots[..batch_size]);
    Ok(())
}

fn terminal_gdn_transition_slots(
    rows: &[crate::speculative::SpeculativeCommitRow],
    slots: &[u32],
) -> Result<Vec<u32>> {
    rows.iter()
        .filter(|row| row.terminal)
        .map(|row| {
            slots.get(row.batch_idx).copied().ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "GDN transition stash has no terminal batch row {}",
                    row.batch_idx
                ))
            })
        })
        .collect()
}

fn gdn_transition_keep_rows(
    rows: &[crate::speculative::SpeculativeCommitRow],
    batch_size: usize,
    max_rows: usize,
) -> Result<Vec<u32>> {
    let mut keep_rows = vec![0u32; batch_size];
    for row in rows {
        if row.batch_idx >= batch_size {
            candle_core::bail!(
                "GDN transition batch row {} exceeds captured batch {batch_size}",
                row.batch_idx
            );
        }
        let keep = u32::try_from(row.keep_rows).map_err(|_| {
            candle_core::Error::msg(format!(
                "GDN commit row count {} exceeds u32",
                row.keep_rows
            ))
        })?;
        if keep as usize > max_rows {
            candle_core::bail!(
                "GDN commit row count {} exceeds checkpoint lanes {max_rows}",
                row.keep_rows
            );
        }
        keep_rows[row.batch_idx] = keep;
    }
    Ok(keep_rows)
}

struct SpecGraphState {
    spec_capture: Option<SpecCapture>,
    full_capture: Option<SpecCapture>,
    gdn_stash: Option<GdnReplayStash>,
}

impl SpeculativeGraphState for SpecGraphState {
    fn tensors(&self) -> Vec<Tensor> {
        let mut tensors = Vec::new();
        for capture in [self.spec_capture.as_ref(), self.full_capture.as_ref()]
            .into_iter()
            .flatten()
        {
            tensors.push(capture.hidden.clone());
            tensors.push(capture.positions.clone());
        }
        for layer in self
            .gdn_stash
            .as_ref()
            .map(|stash| stash.layers.as_slice())
            .unwrap_or_default()
        {
            if let GdnLayerRollback::Replay {
                projected,
                conv_state,
                recurrent_state,
            } = &layer.rollback
            {
                tensors.push(projected.mixed_qkv.clone());
                tensors.push(projected.convolved_qkv.clone());
                tensors.push(projected.b.clone());
                tensors.push(projected.a.clone());
                tensors.push(conv_state.clone());
                tensors.push(recurrent_state.clone());
            }
        }
        tensors
    }

    fn with_tensors(&self, tensors: Vec<Tensor>) -> Result<Box<dyn SpeculativeGraphState>> {
        let mut iter = tensors.into_iter();
        let mut take_capture = |capture: &Option<SpecCapture>| -> Result<Option<SpecCapture>> {
            let Some(capture) = capture else {
                return Ok(None);
            };
            let hidden = iter.next().ok_or_else(|| {
                candle_core::Error::msg("speculative graph state tensor count mismatch")
            })?;
            let positions = iter.next().ok_or_else(|| {
                candle_core::Error::msg("speculative graph state tensor count mismatch")
            })?;
            let _ = capture;
            Ok(Some(SpecCapture { hidden, positions }))
        };
        let spec_capture = take_capture(&self.spec_capture)?;
        let full_capture = take_capture(&self.full_capture)?;
        let mut gdn_stash = self.gdn_stash.clone();
        if let Some(stash) = gdn_stash.as_mut() {
            for layer in &mut stash.layers {
                if let GdnLayerRollback::Replay {
                    projected,
                    conv_state,
                    recurrent_state,
                } = &mut layer.rollback
                {
                    projected.mixed_qkv = iter.next().ok_or_else(|| {
                        candle_core::Error::msg("speculative graph state tensor count mismatch")
                    })?;
                    projected.convolved_qkv = iter.next().ok_or_else(|| {
                        candle_core::Error::msg("speculative graph state tensor count mismatch")
                    })?;
                    projected.b = iter.next().ok_or_else(|| {
                        candle_core::Error::msg("speculative graph state tensor count mismatch")
                    })?;
                    projected.a = iter.next().ok_or_else(|| {
                        candle_core::Error::msg("speculative graph state tensor count mismatch")
                    })?;
                    *conv_state = iter.next().ok_or_else(|| {
                        candle_core::Error::msg("speculative graph state tensor count mismatch")
                    })?;
                    *recurrent_state = iter.next().ok_or_else(|| {
                        candle_core::Error::msg("speculative graph state tensor count mismatch")
                    })?;
                }
            }
        }
        if iter.next().is_some() {
            candle_core::bail!("speculative graph state tensor count mismatch");
        }
        Ok(Box::new(SpecGraphState {
            spec_capture,
            full_capture,
            gdn_stash,
        }))
    }

    fn for_real_batch(&self, real_batch: usize) -> Result<Box<dyn SpeculativeGraphState>> {
        let mut state = SpecGraphState {
            spec_capture: self.spec_capture.clone(),
            full_capture: self.full_capture.clone(),
            gdn_stash: self.gdn_stash.clone(),
        };
        if let Some(capture) = state.spec_capture.as_mut() {
            narrow_spec_capture(capture, real_batch)?;
        }
        if let Some(capture) = state.full_capture.as_mut() {
            narrow_spec_capture(capture, real_batch)?;
        }
        if let Some(stash) = state.gdn_stash.as_mut() {
            narrow_gdn_replay_stash(stash, real_batch)?;
        }
        Ok(Box::new(state))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct MtpAttentionInputs<'a> {
    pub kv_cache: (Tensor, Tensor),
    pub metadata: &'a PagedAttentionInputMetadata,
    pub attention_mask: &'a AttentionMask,
    pub flash_params: &'a FlashParams,
}

pub struct Qwen3NextMtpHead {
    pre_fc_norm_embedding: GemmaRmsNorm,
    pre_fc_norm_hidden: GemmaRmsNorm,
    fc: Arc<dyn QuantMethod>,
    layer: DecoderLayer,
    norm: GemmaRmsNorm,
    kv_layer_idx: usize,
    device: Device,
    dtype: DType,
}

impl Qwen3NextMtpHead {
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
        real_device: &Device,
        attention_mechanism: &AttentionImplementation,
        is_gptx: bool,
    ) -> Result<Self> {
        if !crate::layers::contains_tensor_or_uqff(&vb, MTP_FC_WEIGHT) {
            candle_core::bail!(
                "`--mtp` requested but the checkpoint has no built-in MTP head (`{MTP_FC_WEIGHT}`)."
            );
        }
        let device = real_device.clone();
        let mtp_idx = cfg.num_hidden_layers;
        let vb_mtp = vb.pp("mtp");
        let vb_quant = mapper.set_nm_device(vb_mtp.clone(), loading_isq);
        let vb_plain = mapper.set_nm_device(vb_mtp, false);
        let comm = mapper.get_comm_for(mtp_idx)?;

        let pre_fc_norm_embedding = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_plain.pp("pre_fc_norm_embedding"),
        )?;
        let pre_fc_norm_hidden = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_plain.pp("pre_fc_norm_hidden"),
        )?;
        let fc = ReplicatedLayer::new(
            2 * cfg.hidden_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            vb_quant.pp("fc"),
        )?;

        let rot_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
        let rotary_emb = Arc::new(RotaryEmbedding::new_partial(
            cfg.rope_theta as f32,
            rot_dim,
            cfg.max_position_embeddings,
            &device,
            is_gptx,
            vb.dtype(),
        )?);
        let paged_attn = match attention_mechanism {
            AttentionImplementation::Eager => None,
            AttentionImplementation::PagedAttention => {
                Some(PagedAttention::new(cfg.head_dim, &device, None)?)
            }
        };
        let vb_layer = vb_plain.pp("layers").pp(0);
        let vb_layer_quant = vb_quant.pp("layers").pp(0);
        let attn = FullAttention::load(
            vb_layer_quant.clone(),
            cfg,
            mapper,
            mtp_idx,
            loading_isq,
            rotary_emb,
            paged_attn,
            &comm,
        )?;
        let input_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(mtp_idx, vb_layer.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(mtp_idx, vb_layer.pp("post_attention_layernorm"), false),
        )?;
        let moe = SparseMoeBlock::new(
            cfg,
            mapper.set_device(mtp_idx, vb_layer.pp("mlp"), loading_isq),
            mapper,
            mtp_idx,
            loading_isq,
            &comm,
            real_device.clone(),
        )?;
        let layer = DecoderLayer {
            layer_impl: LayerImpl::FullAttention(attn),
            input_layernorm,
            post_attention_layernorm,
            moe,
        };
        let norm = GemmaRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_plain.pp("norm"))?;

        Ok(Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            layer,
            norm,
            kv_layer_idx: cfg.mtp_kv_layer_idx(),
            device,
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn kv_layer_idx(&self) -> usize {
        self.kv_layer_idx
    }

    pub fn forward(
        &self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
        attention: MtpAttentionInputs<'_>,
    ) -> Result<Tensor> {
        let embeds = self.pre_fc_norm_embedding.forward(input_embeds)?;
        let hidden = self.pre_fc_norm_hidden.forward(target_hidden)?;
        let xs = self
            .fc
            .forward(&Tensor::cat(&[embeds, hidden], D::Minus1)?)?;
        let attn = match &self.layer.layer_impl {
            LayerImpl::FullAttention(attn) => attn,
            _ => candle_core::bail!("MTP layer is not a full-attention block"),
        };
        let residual = &xs;
        let normed = self.layer.input_layernorm.forward(&xs)?;
        let attn_out = attn.forward_with_explicit(
            &normed,
            attention.attention_mask,
            positions,
            attention.kv_cache,
            attention.metadata,
            attention.flash_params,
        )?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.layer.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.layer.moe.forward(&normed)?;
        let x = (ffn_out + residual)?;
        self.norm.forward(&x)
    }

    pub fn residual_tensors(&self, uvb: &UnVarBuilder) {
        let uvb_mtp = uvb.pp("mtp");
        uvb_mtp
            .pp("pre_fc_norm_embedding")
            .add(&self.pre_fc_norm_embedding);
        uvb_mtp
            .pp("pre_fc_norm_hidden")
            .add(&self.pre_fc_norm_hidden);
        uvb_mtp.pp("norm").add(&self.norm);
        let uvb_l = uvb_mtp.pp("layers").pp(0);
        uvb_l.pp("input_layernorm").add(&self.layer.input_layernorm);
        uvb_l
            .pp("post_attention_layernorm")
            .add(&self.layer.post_attention_layernorm);
        if let LayerImpl::FullAttention(attn) = &self.layer.layer_impl {
            uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
            uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
        }
    }
}

struct PendingPromptTail {
    position: usize,
    hidden: Tensor,
}

struct DraftRow {
    seq_id: usize,
    position: usize,
    token: u32,
}

struct CaptureView {
    hidden: Tensor,
    positions: Vec<Vec<u32>>,
}

fn capture_view(capture: &SpecCapture) -> Result<CaptureView> {
    let hidden = match capture.hidden.rank() {
        3 => capture.hidden.clone(),
        2 => capture.hidden.unsqueeze(1)?,
        rank => candle_core::bail!("unexpected MTP hidden rank {rank}"),
    };
    let positions = capture.positions.to_dtype(candle_core::DType::U32)?;
    let positions = match positions.rank() {
        2 => positions.to_vec2::<u32>()?,
        rank => candle_core::bail!("unexpected MTP position rank {rank}"),
    };
    Ok(CaptureView { hidden, positions })
}

fn position_at(positions: &[Vec<u32>], batch_idx: usize, row: usize) -> Result<u32> {
    positions
        .get(batch_idx)
        .and_then(|r| r.get(row))
        .copied()
        .ok_or_else(|| {
            candle_core::Error::msg(format!(
                "MTP position ids missing for batch {batch_idx} row {row}"
            ))
        })
}

// ====================== Top-level Model ======================

#[allow(dead_code)]
pub struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    norm: GemmaRmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    kv_cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    num_attention_heads: usize,
    max_seq_len: usize,
    mtp: Option<Qwen3NextMtpHead>,
    mtp_n_predict: AtomicUsize,
    store_spec_hidden: AtomicBool,
    last_spec_capture: Mutex<Option<SpecCapture>>,
    last_full_capture: Mutex<Option<SpecCapture>>,
    gdn_replay_stash: Mutex<Option<GdnReplayStash>>,
    pending_prompt_tails: Mutex<HashMap<usize, PendingPromptTail>>,
    draft_lm_head: Mutex<Option<Arc<dyn QuantMethod>>>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");

        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }

        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();

        if !cfg.mlp_only_layers.is_empty() {
            candle_core::bail!("Qwen3Next `mlp_only_layers` is not implemented yet in mistral.rs.");
        }

        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb_lm_head.clone(), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };

        let norm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let layer_types = cfg.layer_types();

        // Build RoPE for attention layers (partial rotary)
        let rot_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
        let mut ropes = HashMap::new();
        for (i, layer_type) in layer_types.iter().enumerate().take(cfg.num_hidden_layers) {
            if matches!(layer_type, LayerType::FullAttention) {
                let device = mapper
                    .device_for(i, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                if let std::collections::hash_map::Entry::Vacant(e) = ropes.entry(device.location())
                {
                    let rope = RotaryEmbedding::new_partial(
                        cfg.rope_theta as f32,
                        rot_dim,
                        cfg.max_position_embeddings,
                        device,
                        is_gptx,
                        vb_m.dtype(),
                    )?;
                    e.insert(Arc::new(rope));
                }
            }
        }

        // Log layer config
        let num_full = layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::FullAttention))
            .count();
        let num_linear = layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::LinearAttention))
            .count();
        tracing::info!(
            "Qwen3Next: {} full attention layers, {} linear attention (GDN) layers",
            num_full,
            num_linear
        );

        // Build layers
        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|i| {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let comm = mapper.get_comm_for(i)?;
            let vb_layer = vb_l.pp(i);

            let layer_impl = match &layer_types[i] {
                LayerType::FullAttention => {
                    let rotary_emb = ropes
                        .get(&device.location())
                        .expect("No RoPE for device location!")
                        .clone();
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(cfg.head_dim, device, None)?)
                        }
                    };
                    LayerImpl::FullAttention(FullAttention::load(
                        vb_layer.clone(),
                        cfg,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        rotary_emb,
                        paged_attn,
                        &comm,
                    )?)
                }
                LayerType::LinearAttention => {
                    let vb_linear_attn = vb_layer.pp("linear_attn");
                    let projection_kind = gdn_input_projection_kind(&vb_linear_attn);
                    LayerImpl::LinearAttention(GatedDeltaNet::load(
                        vb_layer.clone(),
                        cfg as &dyn GdnConfig,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        &comm,
                        projection_kind,
                    )?)
                }
            };

            let input_layernorm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(i, vb_layer.pp("input_layernorm"), false),
            )?;
            let post_attention_layernorm = GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(i, vb_layer.pp("post_attention_layernorm"), false),
            )?;

            let moe = SparseMoeBlock::new(
                cfg,
                mapper.set_device(i, vb_layer.pp("mlp"), normal_loading_metadata.loading_isq),
                &*mapper,
                i,
                normal_loading_metadata.loading_isq,
                &comm,
                normal_loading_metadata.real_device.clone(),
            )?;

            Ok(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                moe,
            })
        })?;

        // Create pipeline hybrid cache config
        let mtp = if cfg.mtp {
            Some(Qwen3NextMtpHead::load(
                vb.clone(),
                cfg,
                &*mapper,
                normal_loading_metadata.loading_isq,
                &normal_loading_metadata.real_device,
                &attention_mechanism,
                is_gptx,
            )?)
        } else {
            None
        };
        let mut pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Recurrent,
            })
            .collect();
        pipeline_layer_types.extend(std::iter::repeat_n(
            HybridLayerType::Attention,
            cfg.mtp_layers(),
        ));

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: cfg.max_position_embeddings,
            recurrent: RecurrentLayerConfig {
                conv_dim: cfg.linear_conv_dim(),
                conv_width: cfg.linear_conv_kernel_dim,
                state: crate::kv_cache::RecurrentStateSpec::Gdn {
                    heads: cfg.linear_num_value_heads,
                    key_dim: cfg.linear_key_head_dim,
                    value_dim: cfg.linear_value_head_dim,
                },
                recurrent_dtype: Some(cfg.mamba_ssm_dtype.dtype()),
            },
        };
        let layer_devices = (0..hybrid_cache_config.layer_types.len())
            .map(|layer_idx| {
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone()
            })
            .collect::<Vec<_>>();

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(hybrid_cache_config, vb_m.dtype(), &layer_devices).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to create hybrid cache: {}", e))
            })?,
        ));

        let num_attention_heads = cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size();

        Ok(Self {
            embed_tokens,
            layers,
            layer_types,
            norm,
            lm_head,
            dtype,
            kv_cache: EitherCache::Hybrid(pipeline_cache),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers + cfg.mtp_layers(),
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: num_attention_heads,
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            num_attention_heads,
            max_seq_len: cfg.max_position_embeddings,
            mtp,
            mtp_n_predict: AtomicUsize::new(0),
            store_spec_hidden: AtomicBool::new(false),
            last_spec_capture: Mutex::new(None),
            last_full_capture: Mutex::new(None),
            gdn_replay_stash: Mutex::new(None),
            pending_prompt_tails: Mutex::new(HashMap::new()),
            draft_lm_head: Mutex::new(None),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut x = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;

        let recurrent_metadata = ctx.recurrent_metadata().cloned();
        let has_linear_attention = self
            .layer_types
            .iter()
            .any(|lt| matches!(lt, LayerType::LinearAttention));
        let packed_layout = if ctx.flash_params().packed {
            let query_lens = ctx
                .paged_input_metadata()
                .and_then(|metadata| metadata.query_lens.clone())
                .ok_or_else(|| {
                    candle_core::Error::msg("Qwen3-Next packed GDN requires logical query lengths")
                })?;
            Some(PackedGdnLayout::new(
                query_lens,
                ctx.flash_params().cumulative_seqlens_q.clone(),
            )?)
        } else {
            None
        };
        if has_linear_attention && recurrent_metadata.is_none() {
            candle_core::bail!(
                "Hybrid recurrent metadata is required for linear-attention layers."
            );
        }
        if has_linear_attention {
            if let Some(layout) = packed_layout.as_ref() {
                let query_lens = layout.query_lens();
                if !ctx.is_first_prompt_chunk() {
                    candle_core::bail!("Qwen3-Next packed GDN requires the first prompt chunk");
                }
                let recurrent_metadata = recurrent_metadata
                    .as_ref()
                    .expect("checked above: linear-attention layers require recurrent metadata");
                if recurrent_metadata.batch_kind() != RecurrentBatchKind::Prefill {
                    candle_core::bail!("Qwen3-Next packed GDN cannot run a decode batch");
                }
                let (physical_batch, physical_tokens, _) = x.dims3()?;
                packed_gdn_segments(physical_batch, physical_tokens, query_lens)?;
                let index_count = recurrent_metadata.state_indices().dims1()?;
                if index_count != query_lens.len() {
                    candle_core::bail!(
                        "Qwen3-Next packed GDN has {index_count} state indices but {} logical sequences",
                        query_lens.len()
                    );
                }
                if let Some(host_indices) = recurrent_metadata.state_indices_host() {
                    if host_indices.len() != query_lens.len() {
                        candle_core::bail!(
                            "Qwen3-Next packed GDN has {} host state indices but {} logical sequences",
                            host_indices.len(),
                            query_lens.len()
                        );
                    }
                }
            }
        }
        let mut hybrid_cache = self.kv_cache.hybrid();
        let checkpoint_lanes = hybrid_cache.checkpoint_lanes();
        let (_input_batch, query_len) = input_ids.dims2().unwrap_or((1, 1));

        let speculative_gdn = checkpoint_lanes > 1
            && (1..=checkpoint_lanes).contains(&query_len)
            && recurrent_metadata.as_ref().is_some_and(|metadata| {
                metadata.batch_kind() == RecurrentBatchKind::SpeculativeDecode
            });
        let transition_gdn = speculative_gdn
            && hybrid_cache.uses_recurrent_transition_log()
            && query_len <= crate::cuda::gdn::GDN_SPEC_FUSED_MAX_TOKENS
            && self.supports_recurrent_speculative_transitions_with_cache(&hybrid_cache);
        if !transition_gdn && hybrid_cache.uses_recurrent_transition_log() {
            let has_slots = hybrid_cache
                .state_indices()
                .is_some_and(|slots| slots.elem_count() != 0);
            if !self.apply_pending_recurrent_transitions_for_current_batch(&hybrid_cache)?
                && has_slots
            {
                candle_core::bail!("Qwen3Next pending recurrent transitions cannot be applied");
            }
        }
        let checkpoint_gdn = speculative_gdn
            && !transition_gdn
            && self.supports_recurrent_speculative_checkpoints_with_cache(&hybrid_cache);
        let gdn_checkpoint_lanes = if checkpoint_gdn { checkpoint_lanes } else { 1 };
        let store_spec_hidden = self.store_spec_hidden.load(Ordering::Relaxed);
        let stash_replay = should_stash_gdn_replay(
            checkpoint_gdn || transition_gdn,
            store_spec_hidden,
            query_len,
            recurrent_metadata
                .as_ref()
                .map(|metadata| metadata.batch_kind()),
            ctx.paged_input_metadata().is_some_and(|meta| {
                !meta.is_first_prompt_chunk && meta.num_cached_tokens.is_none()
            }),
        );
        let stash_gdn = stash_replay || (transition_gdn && store_spec_hidden);
        let mut gdn_stash = stash_gdn.then(|| GdnReplayStash {
            slots: recurrent_metadata
                .as_ref()
                .and_then(|meta| meta.state_indices_host())
                .map(|slots| slots.to_vec())
                .unwrap_or_default(),
            layers: Vec::new(),
        });

        let mask = if ctx.is_paged() {
            let cache = ForwardMaskCache::Paged(ctx.seqlen_offsets());
            CausalMasker.make_causal_mask(
                input_ids,
                &cache,
                x.dtype(),
                &CausalMaskConfig::default(),
            )?
        } else {
            CausalMasker.make_causal_mask(
                input_ids,
                &*hybrid_cache as &dyn PastKvLenCache,
                x.dtype(),
                &CausalMaskConfig::default(),
            )?
        };
        let mask = if ctx.is_first_prompt_chunk() {
            mask
        } else {
            AttentionMask::None
        };
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;

        {
            use std::sync::Once;
            static ONCE: Once = Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[layer-prof-init] LAYER_PROFILE={:?}",
                    std::env::var("LAYER_PROFILE")
                );
            });
        }
        let layer_prof = std::env::var("LAYER_PROFILE").is_ok();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;
            let layer_is_linear = matches!(layer.layer_impl, LayerImpl::LinearAttention(_));
            let t0 = std::time::Instant::now();

            match &layer.layer_impl {
                LayerImpl::FullAttention(_) => {
                    if let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(layer_idx)
                    {
                        let mask_for_layer = &mask.get(x.device());
                        x = layer.forward_attention(
                            &x,
                            mask_for_layer,
                            kv_cache,
                            ctx,
                            layer_idx,
                        )?;
                    }
                }
                LayerImpl::LinearAttention(_) => {
                    let recurrent_metadata = recurrent_metadata.as_ref().expect(
                        "checked above: linear-attention layers require recurrent metadata",
                    );
                    let indices = hybrid_cache
                        .state_indices_for_layer(layer_idx)?
                        .ok_or_else(|| {
                            candle_core::Error::msg(format!(
                                "Hybrid cache layer {layer_idx} is missing recurrent state indices"
                            ))
                        })?;
                    if let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer_idx)
                    {
                        // Packed prefill slices the gathered rows per logical sequence
                        let mut gdn_cache = if packed_layout.is_some() {
                            GdnLayerCache::gathered(
                                pool.gather_conv_state(&indices)?,
                                pool.gather_recurrent_state(&indices)?,
                                pool.state_layout(),
                            )
                        } else {
                            GdnLayerCache::checkout(pool, &indices)?
                        };

                        if packed_layout.is_some() {
                            x = layer.forward_linear(
                                &x,
                                &mut gdn_cache,
                                recurrent_metadata.batch_kind(),
                                packed_layout.as_ref(),
                            )?;
                        } else {
                            let stash_states = stash_replay
                                .then_some(())
                                .as_ref()
                                .map(|_| {
                                    candle_core::Result::Ok((
                                        pool.gather_conv_state(&indices)?,
                                        pool.gather_recurrent_state(&indices)?,
                                    ))
                                })
                                .transpose()?;
                            let mut projected_stash = None;
                            x = layer.forward_linear_with_stash(
                                &x,
                                &mut gdn_cache,
                                recurrent_metadata.batch_kind(),
                                if transition_gdn {
                                    checkpoint_lanes
                                } else {
                                    gdn_checkpoint_lanes
                                },
                                transition_gdn,
                                gdn_stash.as_ref().map(|_| &mut projected_stash),
                            )?;
                            if let Some(stash) = gdn_stash.as_mut() {
                                let captured = projected_stash.ok_or_else(|| {
                                    candle_core::Error::msg("GDN forward returned no stash")
                                })?;
                                let rollback = match (captured, stash_states) {
                                    (
                                        GdnSpeculativeStash::Replay(projected),
                                        Some((conv_state, recurrent_state)),
                                    ) => GdnLayerRollback::Replay {
                                        projected,
                                        conv_state,
                                        recurrent_state,
                                    },
                                    (GdnSpeculativeStash::Transition(transition), None) => {
                                        GdnLayerRollback::Transition(transition)
                                    }
                                    _ => candle_core::bail!(
                                        "GDN speculative capture mode does not match cache storage"
                                    ),
                                };
                                stash.layers.push(GdnLayerStash {
                                    layer_idx,
                                    state_layout: pool.state_layout(),
                                    rollback,
                                });
                            }
                        }

                        gdn_cache.commit(
                            pool,
                            &indices,
                            recurrent_metadata.state_indices_host(),
                        )?;
                    } else {
                        candle_core::bail!(
                            "Hybrid cache layer {layer_idx} is not recurrent for a linear-attention layer."
                        );
                    }
                }
            }

            if layer_prof {
                // to_vec blocks on the stream so elapsed reflects GPU completion
                let _ = x.sum_all()?.to_vec0::<f32>();
                use std::sync::atomic::{AtomicUsize, Ordering};
                static N: AtomicUsize = AtomicUsize::new(0);
                let i = N.fetch_add(1, Ordering::Relaxed);
                if i < 140 {
                    eprintln!(
                        "[layer-profile] layer={layer_idx} {} ms={:.2}",
                        if layer_is_linear { "gdn" } else { "attn" },
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
        }

        if self.store_spec_hidden.load(Ordering::Relaxed) {
            *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = gdn_stash;
        }
        let x = x.to_device(&self.device)?;
        let x = self.norm.forward(&x)?;

        let store_spec = self.store_spec_hidden.load(Ordering::Relaxed);
        if store_spec {
            let (batch_size, seq_len) = input_ids.dims2().unwrap_or((1, query_len));
            let text_positions = ctx
                .text_positions(&self.device, seq_len)?
                .ok_or_else(|| candle_core::Error::msg("Qwen3Next is missing text positions"))?
                .clone();
            let position_ids = text_positions.reshape((batch_size, seq_len))?;
            let full_capture = if recurrent_metadata
                .as_ref()
                .is_some_and(|metadata| metadata.batch_kind() == RecurrentBatchKind::Prefill)
            {
                Some(SpecCapture {
                    hidden: x.clone(),
                    positions: position_ids.clone(),
                })
            } else {
                None
            };
            *self
                .last_full_capture
                .lock()
                .expect("spec capture poisoned") = full_capture;
        }
        let x = ctx.logits(&x)?;

        if store_spec {
            let (batch_size, seq_len) = input_ids.dims2().unwrap_or((1, query_len));
            let text_positions = ctx
                .text_positions(&self.device, seq_len)?
                .ok_or_else(|| candle_core::Error::msg("Qwen3Next is missing text positions"))?
                .clone();
            let position_ids = text_positions.reshape((batch_size, seq_len))?;
            let positions = ctx
                .logits(&position_ids.unsqueeze(D::Minus1)?)?
                .squeeze(2)?;
            *self
                .last_spec_capture
                .lock()
                .expect("spec capture poisoned") = Some(SpecCapture {
                hidden: x.clone(),
                positions,
            });
        }
        let logits = self.lm_head.forward(&x)?;

        Ok(logits)
    }
}

impl Model {
    fn set_store_spec_hidden(&self, store: bool) {
        self.store_spec_hidden.store(store, Ordering::Relaxed);
        if !store {
            *self
                .last_spec_capture
                .lock()
                .expect("spec capture poisoned") = None;
            *self
                .last_full_capture
                .lock()
                .expect("spec capture poisoned") = None;
            *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = None;
        }
    }

    fn reserve_recurrent_transition_storage(&self) -> Result<bool> {
        let mut cache = self.kv_cache.hybrid();
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let max_rows = cache.checkpoint_lanes();
        let mut spec = None;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if !matches!(layer_type, LayerType::LinearAttention) {
                continue;
            }
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                candle_core::bail!("Qwen3Next GDN layer has no recurrent state pool");
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype) {
                return Ok(false);
            }
            let layer_spec = gdn.pending_transition_spec(max_rows);
            if spec
                .replace(layer_spec)
                .is_some_and(|spec| spec != layer_spec)
            {
                candle_core::bail!("Qwen3Next GDN transition dimensions diverge across layers");
            }
        }
        let Some(spec) = spec else {
            return Ok(false);
        };
        cache.reserve_gdn_pending_transitions(spec)
    }

    fn reserve_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        let mut cache = self.kv_cache.hybrid();
        let mut spec = None;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if !matches!(layer_type, LayerType::LinearAttention) {
                continue;
            }
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                candle_core::bail!("Qwen3Next GDN layer has no recurrent state pool");
            };
            if !gdn.deferred_decode_supported(pool, self.dtype) {
                return Ok(false);
            }
            let layer_spec = gdn.deferred_state_spec();
            if spec
                .replace(layer_spec)
                .is_some_and(|spec| spec != layer_spec)
            {
                candle_core::bail!("Qwen3Next GDN deferred-state dimensions diverge across layers");
            }
        }
        let Some(spec) = spec else {
            return Ok(false);
        };
        cache.reserve_gdn_deferred_state(spec)
    }

    fn disable_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        self.kv_cache.hybrid().disable_gdn_deferred_state()
    }

    fn apply_pending_recurrent_transitions_with_cache(
        &self,
        cache: &HybridCache,
        slots: &[u32],
    ) -> Result<bool> {
        let mut slots = slots
            .iter()
            .copied()
            .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return Ok(true);
        }
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let active_slots = Tensor::from_vec(slots.clone(), (slots.len(),), &self.device)?;
        self.apply_pending_recurrent_transitions(cache, &active_slots, false)
    }

    fn apply_pending_recurrent_transitions_for_current_batch(
        &self,
        cache: &HybridCache,
    ) -> Result<bool> {
        let Some(active_slots) = cache.state_indices() else {
            return Ok(false);
        };
        self.apply_pending_recurrent_transitions(cache, active_slots, true)
    }

    fn apply_current_recurrent_transitions(&self) -> Result<bool> {
        let cache = self.kv_cache.hybrid();
        self.apply_pending_recurrent_transitions_for_current_batch(&cache)
    }

    fn apply_pending_recurrent_transitions(
        &self,
        cache: &HybridCache,
        active_slots: &Tensor,
        use_cached_device_slots: bool,
    ) -> Result<bool> {
        if active_slots.elem_count() == 0 {
            return Ok(true);
        }
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let mut groups = Vec::<GdnPendingApplyGroup>::new();
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if !matches!(layer_type, LayerType::LinearAttention) {
                continue;
            }
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                return Ok(false);
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype)
                || pool.pending_transitions().is_none()
            {
                return Ok(false);
            }
            let config = gdn.transition_commit_config(pool);
            let device = pool.device();
            if let Some(group) = groups
                .iter_mut()
                .find(|group| group.config == config && group.device.same_device(device))
            {
                group.layers.push(layer_idx);
            } else {
                groups.push(GdnPendingApplyGroup {
                    device: device.clone(),
                    config,
                    layers: vec![layer_idx],
                });
            }
        }
        if groups.is_empty() {
            return Ok(false);
        }

        for group in groups {
            let active_slots = if use_cached_device_slots {
                cache
                    .state_indices_for_device(&group.device)
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "GDN transition batch has no device-local state slots",
                        )
                    })?
            } else {
                active_slots.to_device(&group.device)?
            };
            for layer_indices in group.layers.chunks(GDN_PENDING_APPLY_MAX_LAYERS) {
                let mut layers = Vec::with_capacity(layer_indices.len());
                for &layer_idx in layer_indices {
                    let Some(HybridLayerCache::Recurrent(pool)) = cache.get(layer_idx) else {
                        unreachable!("GDN transition pool was validated above")
                    };
                    let pending = pool
                        .pending_transitions()
                        .expect("GDN pending transition pool was validated above");
                    layers.push(crate::cuda::gdn::GdnPendingTransitionApplyLayer {
                        pending_conv_input: &pending.conv_input,
                        pending_key_banks: &pending.key_banks,
                        pending_key_bank: &pending.key_bank,
                        pending_delta: &pending.delta,
                        pending_decay: &pending.decay,
                        pending_keep_rows: &pending.keep_rows,
                        pending_epochs: &pending.pending_epochs,
                        conv_applied_epochs: &pending.conv_applied_epochs,
                        recurrent_applied_epochs: &pending.recurrent_applied_epochs,
                        conv_state: &pool.conv_state,
                        recurrent_state: &pool.recurrent_state,
                    });
                }
                crate::cuda::gdn::pending_transition_apply_batched_cuda(
                    crate::cuda::gdn::GdnPendingTransitionApply {
                        layers: &layers,
                        active_slots: &active_slots,
                        num_k_heads: group.config.num_k_heads,
                        num_v_heads: group.config.num_v_heads,
                        head_k_dim: group.config.head_k_dim,
                        head_v_dim: group.config.head_v_dim,
                        conv_dim: group.config.conv_dim,
                        conv_width: group.config.conv_width,
                        tiled_v_heads: group.config.tiled_v_heads,
                        state_layout: group.config.state_layout,
                    },
                )?;
            }
        }
        Ok(true)
    }

    fn flush_deferred_recurrent_state(
        &self,
        cache: &HybridCache,
        slots: Option<&[u32]>,
    ) -> Result<bool> {
        if !cache.uses_gdn_deferred_state() {
            return Ok(false);
        }
        let host_slots = slots.map(|slots| {
            let mut slots = slots
                .iter()
                .copied()
                .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
                .collect::<Vec<_>>();
            slots.sort_unstable();
            slots.dedup();
            slots
        });
        if host_slots.as_ref().is_some_and(Vec::is_empty) {
            return Ok(true);
        }
        let mut flushed = false;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if !matches!(layer_type, LayerType::LinearAttention) {
                continue;
            }
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                return Ok(false);
            };
            let active_slots = match &host_slots {
                Some(slots) => Tensor::from_vec(slots.clone(), (slots.len(),), pool.device())?,
                None => cache
                    .state_indices_for_device(pool.device())
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "GDN deferred-state flush has no device-local state slots",
                        )
                    })?,
            };
            if !gdn.flush_deferred_state(pool, &active_slots, self.dtype)? {
                return Ok(false);
            }
            flushed = true;
        }
        Ok(flushed)
    }

    fn flush_current_recurrent_state(&self) -> Result<()> {
        let cache = self.kv_cache.hybrid();
        let has_slots = cache
            .state_indices()
            .is_some_and(|slots| slots.elem_count() != 0);
        if !has_slots {
            return Ok(());
        }
        if cache.uses_recurrent_transition_log()
            && !self.apply_pending_recurrent_transitions_for_current_batch(&cache)?
        {
            candle_core::bail!("Qwen3Next pending recurrent transitions cannot be applied");
        }
        if cache.uses_gdn_deferred_state() && !self.flush_deferred_recurrent_state(&cache, None)? {
            candle_core::bail!("Qwen3Next deferred recurrent state cannot be materialized");
        }
        Ok(())
    }

    fn flush_recurrent_transitions_for_sequences(&self, sequence_ids: &[usize]) -> Result<()> {
        let cache = self.kv_cache.hybrid();
        let slots = cache.recurrent_slots_for_sequences(sequence_ids);
        if cache.uses_recurrent_transition_log()
            && !self.apply_pending_recurrent_transitions_with_cache(&cache, &slots)?
            && !slots.is_empty()
        {
            candle_core::bail!("Qwen3Next pending recurrent transitions cannot be applied");
        }
        if cache.uses_gdn_deferred_state()
            && !self.flush_deferred_recurrent_state(&cache, Some(&slots))?
            && !slots.is_empty()
        {
            candle_core::bail!("Qwen3Next deferred recurrent state cannot be materialized");
        }
        Ok(())
    }

    fn stage_recurrent_prefixes(
        &self,
        rows: &[crate::speculative::SpeculativeCommitRow],
    ) -> Result<bool> {
        if rows.is_empty() {
            return Ok(true);
        }
        let Some(stash) = self
            .gdn_replay_stash
            .lock()
            .expect("gdn stash poisoned")
            .clone()
        else {
            candle_core::bail!("no GDN transition stash for speculative commit");
        };
        if stash.layers.is_empty()
            || stash
                .layers
                .iter()
                .any(|layer| !matches!(layer.rollback, GdnLayerRollback::Transition(_)))
        {
            return Ok(false);
        }

        let max_rows = self.kv_cache.hybrid().checkpoint_lanes();
        let keep_rows_host = gdn_transition_keep_rows(rows, stash.slots.len(), max_rows)?;
        let mut live_slots = stash
            .slots
            .iter()
            .copied()
            .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
            .collect::<Vec<_>>();
        live_slots.sort_unstable();
        if live_slots.windows(2).any(|slots| slots[0] == slots[1]) {
            candle_core::bail!("GDN transition batch contains duplicate recurrent slots");
        }

        struct PublishGroup {
            device: Device,
            capacity: usize,
            max_rows: usize,
            layers: Vec<usize>,
        }
        let cache = self.kv_cache.hybrid();
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let mut groups = Vec::<PublishGroup>::new();
        for (stash_idx, layer) in stash.layers.iter().enumerate() {
            let GdnLayerRollback::Transition(_) = &layer.rollback else {
                unreachable!("transition stash was validated above")
            };
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) = (
                &self.layers[layer.layer_idx].layer_impl,
                cache.get(layer.layer_idx),
            ) else {
                return Ok(false);
            };
            let Some(pending) = pool.pending_transitions() else {
                return Ok(false);
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype)
                || pool.state_layout() != layer.state_layout
                || pending.capacity() != cache.recurrent_capacity()
                || pending.spec().num_k_heads != gdn.transition_commit_config(pool).num_k_heads
                || pending.spec().max_rows != max_rows
            {
                return Ok(false);
            }
            let device = pool.device();
            if let Some(group) = groups.iter_mut().find(|group| {
                group.capacity == pending.capacity()
                    && group.max_rows == pending.spec().max_rows
                    && group.device.same_device(device)
            }) {
                group.layers.push(stash_idx);
            } else {
                groups.push(PublishGroup {
                    device: device.clone(),
                    capacity: pending.capacity(),
                    max_rows: pending.spec().max_rows,
                    layers: vec![stash_idx],
                });
            }
        }

        for group in groups {
            if live_slots
                .iter()
                .any(|slot| *slot as usize >= group.capacity)
            {
                candle_core::bail!("GDN transition slot exceeds recurrent capacity");
            }
            let keep_rows = Tensor::from_vec(
                keep_rows_host.clone(),
                (keep_rows_host.len(),),
                &group.device,
            )?;
            let slots = Tensor::from_vec(stash.slots.clone(), (stash.slots.len(),), &group.device)?;
            let mut layers = Vec::with_capacity(group.layers.len());
            for stash_idx in group.layers {
                let layer = &stash.layers[stash_idx];
                let GdnLayerRollback::Transition(_) = &layer.rollback else {
                    unreachable!("transition stash was validated above")
                };
                let Some(HybridLayerCache::Recurrent(pool)) = cache.get(layer.layer_idx) else {
                    unreachable!("transition pool was validated above")
                };
                let pending = pool
                    .pending_transitions()
                    .expect("pending transition pool was validated above");
                layers.push(crate::cuda::gdn::GdnPendingTransitionPublishLayer {
                    pending_keep_rows: &pending.keep_rows,
                    pending_epochs: &pending.pending_epochs,
                    pending_key_bank: &pending.key_bank,
                });
            }
            crate::cuda::gdn::pending_transition_publish_batched_cuda(
                crate::cuda::gdn::GdnPendingTransitionPublish {
                    layers: &layers,
                    keep_rows: &keep_rows,
                    destination_slots: &slots,
                    max_rows: group.max_rows,
                    destination_capacity: group.capacity,
                },
            )?;
        }
        let terminal_slots = terminal_gdn_transition_slots(rows, &stash.slots)?;
        if !terminal_slots.is_empty()
            && !self.apply_pending_recurrent_transitions_with_cache(&cache, &terminal_slots)?
        {
            candle_core::bail!("Qwen3Next terminal recurrent transitions cannot be applied");
        }
        Ok(true)
    }

    fn replay_recurrent_prefixes(&self, rows: &[(usize, usize)]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let Some(stash) = self
            .gdn_replay_stash
            .lock()
            .expect("gdn stash poisoned")
            .clone()
        else {
            candle_core::bail!("no GDN replay stash for speculative rollback");
        };
        let transition_layers = stash
            .layers
            .iter()
            .filter(|layer| matches!(layer.rollback, GdnLayerRollback::Transition(_)))
            .count();
        if transition_layers != 0 && transition_layers != stash.layers.len() {
            candle_core::bail!("GDN speculative stash mixes replay and transition layers");
        }
        if transition_layers == stash.layers.len() && !stash.layers.is_empty() {
            candle_core::bail!("GDN direct transitions must be published before replay fallback");
        }

        let devices = stash.layers.iter().fold(Vec::new(), |mut devices, layer| {
            let GdnLayerRollback::Replay { projected, .. } = &layer.rollback else {
                unreachable!("transition layers were handled above")
            };
            let device = projected.mixed_qkv.device();
            if !devices
                .iter()
                .any(|cached: &Device| cached.same_device(device))
            {
                devices.push(device.clone());
            }
            devices
        });
        let fused_commit_supported = !stash.layers.is_empty()
            && stash.layers.iter().all(|layer| match &layer.rollback {
                GdnLayerRollback::Replay { projected, .. } => {
                    projected.mixed_qkv.device().is_cuda()
                }
                GdnLayerRollback::Transition(_) => false,
            })
            && {
                let hybrid_cache = self.kv_cache.hybrid();
                stash.layers.iter().all(|layer| {
                    let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                        (&self.layers[layer.layer_idx].layer_impl, hybrid_cache.get(layer.layer_idx))
                    else {
                        return false;
                    };
                    let GdnLayerRollback::Replay {
                        projected,
                        conv_state,
                        recurrent_state,
                    } = &layer.rollback
                    else {
                        return false;
                    };
                    pool.state_layout() == layer.state_layout
                        && gdn.speculative_state_commit_supported(
                            projected,
                            conv_state,
                            recurrent_state,
                            pool,
                        )
                })
            };
        if fused_commit_supported {
            let mut keep_rows_host = vec![0u32; stash.slots.len()];
            for &(batch_idx, rows) in rows {
                let keep_rows = u32::try_from(rows).map_err(|_| {
                    candle_core::Error::msg(format!("GDN commit row count {rows} exceeds u32"))
                })?;
                *keep_rows_host.get_mut(batch_idx).ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "GDN replay stash has no batch row {batch_idx}"
                    ))
                })? = keep_rows;
            }
            let commit_indices = devices
                .iter()
                .map(|device| {
                    Ok(GdnCommitIndices {
                        keep_rows: Tensor::from_vec(
                            keep_rows_host.clone(),
                            (keep_rows_host.len(),),
                            device,
                        )?,
                        slots: Tensor::from_vec(stash.slots.clone(), (stash.slots.len(),), device)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut hybrid_cache = self.kv_cache.hybrid();
            for layer in &stash.layers {
                let GdnLayerRollback::Replay {
                    projected,
                    conv_state,
                    recurrent_state,
                } = &layer.rollback
                else {
                    unreachable!("transition layers were handled above")
                };
                let gdn = match &self.layers[layer.layer_idx].layer_impl {
                    LayerImpl::LinearAttention(gdn) => gdn,
                    LayerImpl::FullAttention(_) => {
                        candle_core::bail!("GDN replay stash points at a full-attention layer")
                    }
                };
                let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer.layer_idx)
                else {
                    candle_core::bail!(
                        "GDN replay stash layer {} has no recurrent state pool",
                        layer.layer_idx
                    );
                };
                if pool.state_layout() != layer.state_layout {
                    candle_core::bail!(
                        "GDN replay state layout mismatch: stash {:?}, pool {:?}",
                        layer.state_layout,
                        pool.state_layout()
                    );
                }
                let device_idx = devices
                    .iter()
                    .position(|device| device.same_device(projected.mixed_qkv.device()))
                    .expect("stashed GDN layer device was collected above");
                let indices = &commit_indices[device_idx];
                if !gdn.commit_state_batch_from_stash_cuda(
                    projected,
                    conv_state,
                    recurrent_state,
                    &indices.keep_rows,
                    &indices.slots,
                    pool,
                )? {
                    candle_core::bail!("CUDA GDN speculative state commit was unavailable");
                }
            }
            return Ok(());
        }

        let batches = group_gdn_replay_batches(rows, &stash.slots)?;
        let replay_indices = batches
            .iter()
            .map(|batch| {
                devices
                    .iter()
                    .map(|device| {
                        Ok(GdnReplayIndices {
                            batch_indices: Tensor::from_vec(
                                batch.batch_indices.clone(),
                                (batch.batch_indices.len(),),
                                device,
                            )?,
                            slots: Tensor::from_vec(
                                batch.slots.clone(),
                                (batch.slots.len(),),
                                device,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let mut hybrid_cache = self.kv_cache.hybrid();
        for layer in &stash.layers {
            let GdnLayerRollback::Replay {
                projected,
                conv_state,
                recurrent_state,
            } = &layer.rollback
            else {
                unreachable!("transition layers were handled above")
            };
            let gdn = match &self.layers[layer.layer_idx].layer_impl {
                LayerImpl::LinearAttention(gdn) => gdn,
                LayerImpl::FullAttention(_) => {
                    candle_core::bail!("GDN replay stash points at a full-attention layer")
                }
            };
            let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer.layer_idx)
            else {
                candle_core::bail!(
                    "GDN replay stash layer {} has no recurrent state pool",
                    layer.layer_idx
                );
            };
            if pool.state_layout() != layer.state_layout {
                candle_core::bail!(
                    "GDN replay state layout mismatch: stash {:?}, pool {:?}",
                    layer.state_layout,
                    pool.state_layout()
                );
            }
            let device_idx = devices
                .iter()
                .position(|device| device.same_device(projected.mixed_qkv.device()))
                .expect("stashed GDN layer device was collected above");
            for (group_idx, batch) in batches.iter().enumerate() {
                let indices = &replay_indices[group_idx][device_idx];
                let mut cache = GdnLayerCache::gathered(
                    index_select_replay_rows(conv_state, &indices.batch_indices)?,
                    index_select_replay_rows(recurrent_state, &indices.batch_indices)?,
                    layer.state_layout,
                );
                gdn.advance_state_batch_from_stash(
                    projected,
                    &indices.batch_indices,
                    batch.keep_rows,
                    &mut cache,
                )?;
                pool.scatter_conv_state(&indices.slots, &cache.conv_state)?;
                pool.scatter_recurrent_state(&indices.slots, &cache.recurrent_state)?;
            }
        }
        Ok(())
    }

    fn clear_gdn_replay_stash(&self) {
        *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = None;
    }

    fn supports_recurrent_speculative_checkpoints(&self) -> bool {
        self.supports_recurrent_speculative_checkpoints_with_cache(&self.kv_cache.hybrid())
    }

    fn supports_recurrent_speculative_transitions(&self) -> bool {
        self.supports_recurrent_speculative_transitions_with_cache(&self.kv_cache.hybrid())
    }

    fn supports_recurrent_speculative_transitions_with_cache(&self, cache: &HybridCache) -> bool {
        let recurrent_devices = cache.recurrent_devices();
        if !recurrent_checkpoint_devices_supported(&recurrent_devices) {
            return false;
        }
        let mut found_gdn = false;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if !matches!(layer_type, LayerType::LinearAttention) {
                continue;
            }
            found_gdn = true;
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                return false;
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype) {
                return false;
            }
        }
        found_gdn
    }

    fn supports_recurrent_speculative_checkpoints_with_cache(&self, cache: &HybridCache) -> bool {
        let recurrent_devices = cache.recurrent_devices();
        if !recurrent_checkpoint_devices_supported(&recurrent_devices) {
            return false;
        }
        let mut found_gdn = false;
        for (layer_idx, layer_type) in self.layer_types.iter().enumerate() {
            if !matches!(layer_type, LayerType::LinearAttention) {
                continue;
            }
            found_gdn = true;
            let (LayerImpl::LinearAttention(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                (&self.layers[layer_idx].layer_impl, cache.get(layer_idx))
            else {
                return false;
            };
            if !gdn.speculative_checkpoints_supported(pool, self.dtype) {
                return false;
            }
        }
        found_gdn
    }

    fn take_spec_graph_state(&self) -> Option<SpecGraphState> {
        if !self.store_spec_hidden.load(Ordering::Relaxed) {
            return None;
        }
        Some(SpecGraphState {
            spec_capture: self
                .last_spec_capture
                .lock()
                .expect("spec capture poisoned")
                .take(),
            full_capture: self
                .last_full_capture
                .lock()
                .expect("spec capture poisoned")
                .take(),
            gdn_stash: self
                .gdn_replay_stash
                .lock()
                .expect("gdn stash poisoned")
                .take(),
        })
    }

    fn install_spec_graph_state(&self, state: &SpecGraphState) -> Result<()> {
        let spec_capture = state.spec_capture.clone();
        let full_capture = state.full_capture.clone();
        let mut gdn_stash = state.gdn_stash.clone();
        let slots = self
            .kv_cache
            .hybrid()
            .state_indices_host()
            .map(ToOwned::to_owned);
        if let (Some(stash), Some(slots)) = (gdn_stash.as_mut(), slots.as_deref()) {
            refresh_gdn_stash_slots(stash, slots)?;
        }
        *self
            .last_spec_capture
            .lock()
            .expect("spec capture poisoned") = spec_capture;
        *self
            .last_full_capture
            .lock()
            .expect("spec capture poisoned") = full_capture;
        *self.gdn_replay_stash.lock().expect("gdn stash poisoned") = gdn_stash;
        Ok(())
    }

    fn last_spec_capture(&self) -> Option<SpecCapture> {
        self.last_spec_capture
            .lock()
            .expect("spec capture poisoned")
            .clone()
    }

    fn last_full_capture(&self) -> Option<SpecCapture> {
        self.last_full_capture
            .lock()
            .expect("spec capture poisoned")
            .clone()
    }

    fn lm_head_ref(&self) -> &Arc<dyn QuantMethod> {
        &self.lm_head
    }

    pub fn embed_tokens_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)
    }

    fn mtp_n_predict_value(&self) -> usize {
        self.mtp_n_predict.load(Ordering::Relaxed)
    }

    fn mtp_head(&self) -> Result<&Qwen3NextMtpHead> {
        self.mtp
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("Qwen3Next MTP head is not loaded"))
    }

    fn drafter_forward(
        &self,
        head: &Qwen3NextMtpHead,
        rows: &[DraftRow],
        target_hidden: &Tensor,
        kv_cache: &(Tensor, Tensor),
        paged_meta: &crate::pipeline::text_models_inputs_processor::PagedAttentionMeta,
    ) -> Result<Tensor> {
        let device = head.device();
        let n = rows.len();
        let tokens = Tensor::from_vec(
            rows.iter().map(|row| row.token).collect::<Vec<_>>(),
            (1, n),
            device,
        )?;
        let positions_vec = rows
            .iter()
            .map(|row| row.position as u32)
            .collect::<Vec<_>>();
        let positions = Tensor::from_vec(positions_vec, (1, n), device)?;
        let seq_ids = rows.iter().map(|row| row.seq_id).collect::<Vec<_>>();
        let context_lens = rows.iter().map(|row| row.position + 1).collect::<Vec<_>>();
        let metadata = make_paged_rows_metadata(&seq_ids, &context_lens, paged_meta, device)?;
        let embeds = self.embed_tokens_ids(&tokens)?.to_dtype(head.dtype())?;
        let target_hidden = target_hidden.to_device(device)?.to_dtype(head.dtype())?;
        head.forward(
            &embeds,
            &target_hidden,
            &positions,
            MtpAttentionInputs {
                kv_cache: kv_cache.clone(),
                metadata: &metadata,
                attention_mask: &AttentionMask::None,
                flash_params: &FlashParams::empty(false),
            },
        )
    }

    fn drafter_prefill_chunk(
        &self,
        head: &Qwen3NextMtpHead,
        ctx: &SpeculativePrefillCtx<'_>,
        target: TargetAttentionInputs<'_>,
        capture: &SpecCapture,
        kv_cache: &(Tensor, Tensor),
    ) -> Result<()> {
        let device = head.device();
        let (batch, seq_len, _) = capture.hidden.dims3()?;
        if batch != ctx.chunk_ranges.len() {
            candle_core::bail!(
                "MTP prefill capture has {batch} rows for {} sequences",
                ctx.chunk_ranges.len()
            );
        }
        let mut shifted = Vec::with_capacity(batch * seq_len);
        let mut offsets = Vec::with_capacity(batch);
        for ((start, end), toks) in ctx.chunk_ranges.iter().zip(ctx.tokens.iter()) {
            offsets.push(*start);
            for row in 0..seq_len {
                let position = start + row;
                let token = (position + 1 < *end || !ctx.is_final_prompt_chunk)
                    .then(|| toks.get(position + 1).copied())
                    .flatten();
                shifted.push(token.unwrap_or(PLACEHOLDER_TOKEN));
            }
        }
        let tokens = Tensor::from_vec(shifted, (batch, seq_len), device)?;
        let embeds = self.embed_tokens_ids(&tokens)?.to_dtype(head.dtype())?;
        let target_hidden = capture.hidden.to_device(device)?.to_dtype(head.dtype())?;
        let positions = capture.positions.to_device(device)?;
        let attention_mask = if target.metadata.is_first_prompt_chunk {
            CausalMasker.make_causal_mask(
                &tokens,
                &offsets.as_slice(),
                head.dtype(),
                &crate::layers_masker::CausalMaskConfig::default(),
            )?
        } else {
            AttentionMask::None
        };
        head.forward(
            &embeds,
            &target_hidden,
            &positions,
            MtpAttentionInputs {
                kv_cache: kv_cache.clone(),
                metadata: target.metadata,
                attention_mask: &attention_mask,
                flash_params: target.flash_params,
            },
        )?;
        Ok(())
    }

    fn draft_logits(&self, normed_hidden: &Tensor) -> Result<Tensor> {
        let draft_head = self.draft_lm_head.lock().expect("draft lm_head poisoned");
        let head = draft_head.as_ref().unwrap_or_else(|| self.lm_head_ref());
        head.forward(normed_hidden)?.squeeze(0)
    }

    fn mtp_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        let head = self.mtp_head()?;
        let max_n = self.mtp_n_predict_value();
        let n_predict = ctx.proposal_len;
        let batch = ctx.sequences.len();
        if batch == 0 || max_n == 0 {
            return Ok(None);
        }
        if n_predict == 0 || n_predict > max_n {
            candle_core::bail!(
                "MTP proposal length {n_predict} is outside the configured range 1..={max_n}"
            );
        }
        if ctx.target_rows.len() != batch || ctx.base_lens.len() != batch {
            candle_core::bail!(
                "MTP batch shape mismatch: sequences={batch}, target_rows={}, base_lens={}",
                ctx.target_rows.len(),
                ctx.base_lens.len()
            );
        }
        let SpeculativeKvCache::Paged {
            metadata: paged_meta,
            kv_cache,
        } = ctx.cache;
        let kv_cache = kv_cache
            .get(head.kv_layer_idx())
            .ok_or_else(|| candle_core::Error::msg("paged cache has no MTP layer"))?
            .clone();
        let Some(capture) = self.last_spec_capture() else {
            return Ok(None);
        };
        let CaptureView { hidden, positions } = capture_view(&capture)?;

        {
            let mut kv_mgr = get_mut_arcmutex!(paged_meta.kv_cache_manager);
            for (seq_id, base_len) in ctx.seq_ids.iter().zip(ctx.base_lens.iter()) {
                if kv_mgr
                    .allocate_slots(*seq_id, base_len + n_predict, &[])
                    .is_none()
                {
                    return Ok(None);
                }
            }
        }

        let mut rows = Vec::new();
        let mut hidden_rows = Vec::with_capacity(batch);
        let mut last_row_idx = Vec::with_capacity(batch);
        let mut pending_tails = self
            .pending_prompt_tails
            .lock()
            .expect("mtp tails poisoned");
        for (i, seq) in ctx.sequences.iter().enumerate() {
            let (batch_idx, count) = ctx.target_rows[i];
            let base_len = ctx.base_lens[i];
            let toks = seq.get_toks();
            if count == 0 || base_len < count || toks.len() <= base_len {
                candle_core::bail!(
                    "MTP refresh rows out of range: base_len={base_len}, count={count}, toks={}",
                    toks.len()
                );
            }
            if let Some(tail) = pending_tails.remove(seq.id()) {
                if tail.position + 1 < toks.len() {
                    rows.push(DraftRow {
                        seq_id: ctx.seq_ids[i],
                        position: tail.position,
                        token: toks[tail.position + 1],
                    });
                    hidden_rows.push(tail.hidden.to_device(hidden.device())?);
                }
            }
            for r in 0..count {
                let position = base_len - count + r;
                rows.push(DraftRow {
                    seq_id: ctx.seq_ids[i],
                    position,
                    token: toks[position + 1],
                });
                let _ = position_at(&positions, batch_idx, r)?;
            }
            hidden_rows.push(hidden.narrow(0, batch_idx, 1)?.narrow(1, 0, count)?);
            last_row_idx.push(rows.len() - 1);
        }
        pending_tails.retain(|seq_id, _| ctx.seq_ids.contains(seq_id));
        drop(pending_tails);
        let target_hidden = Tensor::cat(&hidden_rows, 1)?;
        let normed = self.drafter_forward(head, &rows, &target_hidden, &kv_cache, paged_meta)?;
        let last_idx = Tensor::from_vec(
            last_row_idx.iter().map(|i| *i as u32).collect::<Vec<_>>(),
            (batch,),
            normed.device(),
        )?;
        let mut hidden = normed.index_select(&last_idx, 1)?;
        let mut cursor = last_row_idx
            .iter()
            .map(|i| rows[*i].position)
            .collect::<Vec<_>>();

        let mut contexts = ctx
            .sequences
            .iter()
            .map(|seq| seq.get_toks().to_vec())
            .collect::<Vec<_>>();
        let mut tokens: Vec<Vec<u32>> = vec![Vec::with_capacity(n_predict); batch];
        let mut probs: Vec<Vec<f32>> = vec![Vec::with_capacity(n_predict); batch];
        let mut last_device = None;
        for step in 0..n_predict {
            let step_logits = self.draft_logits(&hidden)?;
            let drafts = sample_draft_rows(&step_logits, ctx.sequences, &mut contexts, &ctx.rng)?;
            for (i, (draft, q)) in drafts.iter().enumerate() {
                tokens[i].push(*draft);
                probs[i].push(*q);
            }
            last_device = Some(step_logits.device().clone());
            if step + 1 == n_predict {
                break;
            }
            let chained = ctx
                .seq_ids
                .iter()
                .zip(cursor.iter_mut())
                .zip(drafts.iter())
                .map(|((seq_id, position), (draft, _))| {
                    *position += 1;
                    DraftRow {
                        seq_id: *seq_id,
                        position: *position,
                        token: *draft,
                    }
                })
                .collect::<Vec<_>>();
            hidden = self.drafter_forward(head, &chained, &hidden, &kv_cache, paged_meta)?;
        }

        let device = last_device.ok_or_else(|| candle_core::Error::msg("no draft steps"))?;
        let proposals = tokens
            .into_iter()
            .zip(probs)
            .map(|(tokens, probs)| {
                let n = tokens.len();
                let token_ids = Tensor::from_vec(tokens.clone(), (n, 1), &device)?;
                let probs = Tensor::from_vec(probs, (n, 1), &device)?;
                SpeculativeProposal::with_sparse_probs(tokens, token_ids, probs)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(SpeculativeProposalBatch::new(proposals)))
    }

    fn mtp_prefill(&mut self, ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        let head = self.mtp_head()?;
        let SpeculativeKvCache::Paged {
            metadata: paged_meta,
            kv_cache,
        } = ctx.cache;
        let kv_cache = kv_cache
            .get(head.kv_layer_idx())
            .ok_or_else(|| candle_core::Error::msg("paged cache has no MTP layer"))?
            .clone();
        let Some(capture) = self.last_full_capture() else {
            return Ok(());
        };
        let CaptureView { hidden, positions } = capture_view(&capture)?;

        let mut rows = Vec::new();
        let mut hidden_rows = Vec::new();
        let mut pending_tails = self
            .pending_prompt_tails
            .lock()
            .expect("mtp tails poisoned");
        for (i, seq_id) in ctx.seq_ids.iter().enumerate() {
            let batch_idx = ctx.batch_indices[i];
            let toks = ctx.tokens[i];
            let (start, end) = ctx.chunk_ranges[i];
            if end <= start || hidden.dim(1)? < end - start {
                candle_core::bail!(
                    "MTP prefill rows out of range: chunk=({start}, {end}), hidden rows={}",
                    hidden.dim(1)?
                );
            }
            let last = if ctx.is_final_prompt_chunk {
                let tail_row = end - 1 - start;
                pending_tails.insert(
                    *seq_id,
                    PendingPromptTail {
                        position: end - 1,
                        hidden: hidden.narrow(0, batch_idx, 1)?.narrow(1, tail_row, 1)?,
                    },
                );
                end - 1
            } else {
                end
            };
            if ctx.target_attention.is_some() {
                continue;
            }
            let count = last - start;
            if count == 0 {
                continue;
            }
            if toks.len() <= last {
                candle_core::bail!(
                    "MTP prefill tokens out of range: chunk=({start}, {end}), toks={}",
                    toks.len()
                );
            }
            for r in 0..count {
                let position = start + r;
                let _ = position_at(&positions, batch_idx, r)?;
                rows.push(DraftRow {
                    seq_id: *seq_id,
                    position,
                    token: toks[position + 1],
                });
            }
            hidden_rows.push(hidden.narrow(0, batch_idx, 1)?.narrow(1, 0, count)?);
        }
        drop(pending_tails);
        if let Some(target) = ctx.target_attention {
            return self.drafter_prefill_chunk(head, &ctx, target, &capture, &kv_cache);
        }
        if rows.is_empty() {
            return Ok(());
        }
        let target_hidden = Tensor::cat(&hidden_rows, 1)?;
        self.drafter_forward(head, &rows, &target_hidden, &kv_cache, paged_meta)?;
        Ok(())
    }
}

// ====================== Trait Implementations ======================

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);

            match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
                    uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
                }
                LayerImpl::LinearAttention(gdn) => {
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("conv1d.weight", gdn.conv1d_weight.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("dt_bias", gdn.dt_bias.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("A_log", gdn.a_log.clone());
                    uvb_l
                        .pp("linear_attn")
                        .pp("norm")
                        .add_tensor("weight", gdn.norm.weight.clone());
                }
            }

            // MoE gate and shared expert gate
            uvb_l
                .pp("mlp")
                .pp("gate")
                .add_tensor("weight", layer.moe.gate.weight().clone());
            uvb_l
                .pp("mlp")
                .pp("shared_expert_gate")
                .add_tensor("weight", layer.moe.shared_expert_gate.weight().clone());
        }
        if let Some(mtp) = &self.mtp {
            mtp.residual_tensors(&uvb);
        }

        uvb.to_safetensors()
    }
}

impl SpeculativeTargetMixin for Model {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.attach_speculative_with_runtime(config, MtpRuntimeConfig::default())
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: SpeculativeConfig,
        _runtime: MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        let SpeculativeConfig::Mtp(config) = config else {
            self.mtp_n_predict.store(0, Ordering::Relaxed);
            self.set_store_spec_hidden(false);
            return Ok(None);
        };
        if !config.is_builtin() {
            candle_core::bail!("Qwen3Next supports only the built-in MTP drafter");
        }
        if self.mtp.is_none() {
            candle_core::bail!(
                "The built-in MTP head was not loaded; pass `--mtp` when loading the model."
            );
        }
        let default_n_predict = if self.cfg.hidden_size >= MTP_LARGE_HIDDEN_SIZE {
            DEFAULT_MTP_N_PREDICT_LARGE
        } else {
            DEFAULT_MTP_N_PREDICT
        };
        let n_predict = config.n_predict.unwrap_or(default_n_predict);
        if n_predict == 0 {
            candle_core::bail!("MTP n_predict must be at least 1.");
        }
        self.mtp_n_predict.store(n_predict, Ordering::Relaxed);
        self.set_store_spec_hidden(true);
        if let Some(ty) = config.draft_lm_head_isq {
            let head = self.lm_head_ref().clone().apply_isq(
                Some(ty),
                self.device.clone(),
                &std::sync::atomic::AtomicUsize::new(0),
                None,
                mistralrs_quant::QuantizeOntoGuard::new(),
            )?;
            *self.draft_lm_head.lock().expect("draft lm_head poisoned") = Some(head);
        } else {
            *self.draft_lm_head.lock().expect("draft lm_head poisoned") = None;
        }
        Ok(Some(SpeculativeAttachInfo::mtp(
            "built-in".to_string(),
            n_predict,
        )))
    }

    fn has_speculative_proposer(&self) -> bool {
        self.mtp_n_predict_value() > 0
    }

    fn supports_recurrent_speculative_checkpoints(&self) -> bool {
        self.supports_recurrent_speculative_checkpoints()
    }

    fn supports_recurrent_speculative_transitions(&self) -> bool {
        self.supports_recurrent_speculative_transitions()
    }

    fn reserve_recurrent_speculative_transition_storage(&self) -> Result<bool> {
        self.reserve_recurrent_transition_storage()
    }

    fn reserve_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        if self.has_speculative_proposer() {
            Ok(false)
        } else {
            self.reserve_recurrent_decode_deferred_storage()
        }
    }

    fn disable_recurrent_decode_deferred_storage(&self) -> Result<bool> {
        self.disable_recurrent_decode_deferred_storage()
    }

    fn apply_recurrent_speculative_transitions_for_current_batch(&self) -> Result<bool> {
        self.apply_current_recurrent_transitions()
    }

    fn flush_recurrent_state_for_current_batch(&self) -> Result<()> {
        self.flush_current_recurrent_state()
    }

    fn flush_recurrent_speculative_transitions(&self, seq_ids: &[usize]) -> Result<()> {
        self.flush_recurrent_transitions_for_sequences(seq_ids)
    }

    fn speculative_plan(&self, _batch_size: usize) -> Option<SpeculativeBatchPlan> {
        let n = self.mtp_n_predict_value();
        if n == 0 {
            return None;
        }
        Some(SpeculativeBatchPlan::new(n))
    }

    fn speculative_graph_plans(&self) -> Vec<SpeculativeGraphPlan> {
        let n = self.mtp_n_predict_value();
        if n == 0 {
            return Vec::new();
        }
        vec![SpeculativeGraphPlan::new(n, None)]
    }

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        self.mtp_propose(ctx)
    }

    fn speculative_target_hiddens(&self, rows: &[(usize, usize)]) -> Result<Option<Tensor>> {
        let Some(capture) = self.last_spec_capture() else {
            return Ok(None);
        };
        let hidden = capture_view(&capture)?.hidden;
        let gathered = rows
            .iter()
            .map(|(batch_idx, row)| {
                hidden
                    .narrow(0, *batch_idx, 1)?
                    .narrow(1, *row, 1)?
                    .squeeze(0)?
                    .squeeze(0)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Tensor::stack(&gathered, 0)?))
    }

    fn speculative_prefill(&mut self, ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        if self.mtp_n_predict_value() == 0 {
            return Ok(());
        }
        self.mtp_prefill(ctx)
    }

    fn speculative_commit(&mut self, rows: &[SpeculativeCommitRow]) -> Result<()> {
        let checkpoint_rows = rows
            .iter()
            .map(|row| (row.batch_idx, row.keep_rows))
            .collect::<Vec<_>>();
        if self.kv_cache.hybrid().uses_recurrent_transition_log() {
            if !self.stage_recurrent_prefixes(rows)? {
                self.replay_recurrent_prefixes(&checkpoint_rows)?;
            }
            self.clear_gdn_replay_stash();
            return Ok(());
        }
        let checkpointed = {
            let mut cache = self.kv_cache.hybrid();
            self.supports_recurrent_speculative_checkpoints_with_cache(&cache)
                && cache.commit_speculative_rows(&checkpoint_rows)?
        };
        if checkpointed {
            self.clear_gdn_replay_stash();
            return Ok(());
        }
        let rejected = rows
            .iter()
            .filter(|row| !row.accepted_all)
            .map(|row| (row.batch_idx, row.keep_rows))
            .collect::<Vec<_>>();
        self.replay_recurrent_prefixes(&rejected)?;
        self.clear_gdn_replay_stash();
        Ok(())
    }

    fn take_speculative_graph_state(&self) -> Option<Box<dyn SpeculativeGraphState>> {
        self.take_spec_graph_state()
            .map(|state| Box::new(state) as Box<dyn SpeculativeGraphState>)
    }

    fn install_speculative_graph_state(&self, state: &dyn SpeculativeGraphState) -> Result<()> {
        let state = state
            .as_any()
            .downcast_ref::<SpecGraphState>()
            .ok_or_else(|| {
                candle_core::Error::msg("foreign speculative graph state for Qwen3Next")
            })?;
        self.install_spec_graph_state(state)
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("Qwen3Next does not support X-LoRA forward")
    }
    fn cache(&self) -> &EitherCache {
        &self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }

    fn supports_packed_prefill(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

    use candle_core::{DType, Device, Result, Tensor};
    use mistralrs_quant::{
        QuantMethod, QuantizedWeightSource, Shard, ShardedSafeTensors, ShardedVarBuilder,
    };

    use super::{
        gdn_input_projection_kind, packed_gdn_segments, validate_packed_gdn_state_rows,
        GdnInputProjectionKind, PackedGdnSegment,
    };

    struct ProjectionWeightSource(HashSet<String>);

    impl QuantizedWeightSource for ProjectionWeightSource {
        fn contains(&self, name: &str) -> bool {
            self.0.contains(name)
        }

        fn load_linear(
            &self,
            _key: &str,
            _device: &Device,
            _shard: Shard,
        ) -> Result<Option<Arc<dyn QuantMethod>>> {
            unreachable!()
        }

        fn load_optional_tensor(&self, _name: &str, _device: &Device) -> Result<Option<Tensor>> {
            unreachable!()
        }

        fn shard_alignment(&self, _key: &str) -> Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> Result<usize> {
            Ok(1)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> Result<Option<usize>> {
            Ok(Some(1))
        }
    }

    fn projection_vb(residual: &[&str], source: &[&str]) -> Result<ShardedVarBuilder> {
        let tensors = residual
            .iter()
            .map(|name| {
                Ok((
                    format!("model.layers.0.linear_attn.{name}"),
                    Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let source = source
            .iter()
            .map(|name| format!("model.layers.0.linear_attn.{name}"))
            .collect();
        Ok(ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu)
            .with_weight_source(Arc::new(ProjectionWeightSource(source)))
            .pp("model.layers.0.linear_attn"))
    }

    #[test]
    fn gdn_projection_kind_reads_residual_and_weight_source_tensors() -> Result<()> {
        let split = [
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_b.weight",
            "in_proj_a.weight",
        ];
        assert_eq!(
            gdn_input_projection_kind(&projection_vb(&[], &split)?),
            GdnInputProjectionKind::Split
        );
        assert_eq!(
            gdn_input_projection_kind(&projection_vb(&split, &[])?),
            GdnInputProjectionKind::Split
        );
        assert_eq!(
            gdn_input_projection_kind(&projection_vb(&[], &["in_proj_qkv.weight"])?),
            GdnInputProjectionKind::SplitQkvzGroupedBa
        );
        assert_eq!(
            gdn_input_projection_kind(&projection_vb(&[], &[])?),
            GdnInputProjectionKind::Grouped
        );
        Ok(())
    }

    #[test]
    fn packed_gdn_maps_unequal_queries_to_matching_state_rows() {
        assert_eq!(
            packed_gdn_segments(1, 8, &[2, 5, 1]).unwrap(),
            vec![
                PackedGdnSegment {
                    token_range: 0..2,
                    state_index: 0,
                },
                PackedGdnSegment {
                    token_range: 2..7,
                    state_index: 1,
                },
                PackedGdnSegment {
                    token_range: 7..8,
                    state_index: 2,
                },
            ]
        );
    }

    #[test]
    fn packed_gdn_rejects_query_and_state_cardinality_mismatches() {
        assert!(packed_gdn_segments(2, 8, &[2, 5, 1]).is_err());
        assert!(packed_gdn_segments(1, 0, &[]).is_err());
        assert!(packed_gdn_segments(1, 7, &[2, 5, 1]).is_err());
        assert!(packed_gdn_segments(1, 8, &[2, 0, 6]).is_err());
        assert!(packed_gdn_segments(1, usize::MAX, &[usize::MAX, 1]).is_err());
        assert!(validate_packed_gdn_state_rows(3, 2, 3).is_err());
        assert!(validate_packed_gdn_state_rows(3, 3, 2).is_err());
    }

    #[test]
    fn packed_gdn_keeps_token_and_state_order_isolated() {
        let tokens = [10, 11, 20, 21, 22, 30, 31];
        let state_markers = [100, 200, 300];
        let observed = packed_gdn_segments(1, tokens.len(), &[2, 3, 2])
            .unwrap()
            .into_iter()
            .map(|segment| {
                (
                    state_markers[segment.state_index],
                    tokens[segment.token_range].to_vec(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![
                (100, vec![10, 11]),
                (200, vec![20, 21, 22]),
                (300, vec![30, 31]),
            ]
        );
    }
}
