use super::isq_flow::CalibrationStatus;
use super::{
    get_model_paths, AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, IsqPipelineMixin, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, TokenSource,
};
use crate::device_map::{self, DeviceMapper};
use crate::embedding_models::inputs_processor::{EmbeddingProcessor, ModelInputs};
use crate::embedding_models::RerankScoreHead;
use crate::get_embedding_paths;
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::loaders::{auto_device_map, DeviceMappedModelLoader, QuantizationConfigShim};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::{
    ChatTemplate, EmbeddingModelLoader, EmbeddingModelPaths, Processor, Qwen3EmbeddingLoader,
};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::progress::{new_multi_progress, ProgressScopeGuard};
use crate::utils::tokenizer::get_tokenizer;
use crate::Modalities;
use crate::SupportedModality;
use crate::{
    DeviceMapSetting, PagedAttentionConfig, Pipeline, Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::Module;
use hf_hub::Cache;
use hf_hub::{Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, trace, warn};

pub struct RerankerPipeline {
    model: Box<dyn crate::pipeline::EmbeddingModel + Send + Sync>,
    tracked_modules: Vec<mistralrs_quant::TrackedModule>,
    source_weight_files: Vec<std::path::PathBuf>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    processor: Arc<dyn Processor + Send + Sync>,
    score_head: RerankScoreHead,
}

/// A loader for a GGUF reranker (cross-encoder) model.
pub struct RerankerLoader {
    model_id: String,
    config: RerankerSpecificConfig,
    kind: ModelKind,
    tokenizer_json: Option<String>,
    token_source: RwLock<Option<TokenSource>>,
    revision: RwLock<Option<String>>,
}

#[derive(Default)]
/// A builder for a loader for a GGUF reranker (cross-encoder) model.
pub struct RerankerLoaderBuilder {
    model_id: Option<String>,
    config: RerankerSpecificConfig,
    tokenizer_json: Option<String>,
}

#[derive(Clone, Default)]
/// Config specific to loading a reranker model.
pub struct RerankerSpecificConfig {
    pub topology: Option<Topology>,
    pub hf_cache_path: Option<PathBuf>,
    /// GGUF filename(s), resolved within `model_id` at load time. Required:
    /// safetensors CrossEncoder loading is not supported yet.
    pub quantized_filenames: Option<Vec<String>>,
}

impl RerankerLoaderBuilder {
    pub fn new(
        config: RerankerSpecificConfig,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
    ) -> Self {
        Self {
            config,
            tokenizer_json,
            model_id,
        }
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(RerankerLoader {
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: ModelKind::Normal,
            tokenizer_json: self.tokenizer_json,
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
        })
    }
}

/// A GGUF reranker checkpoint opened for the loading arm of [`RerankerLoader`].
struct RerankerGgufContext {
    archive: Arc<mistralrs_quant::GgufArchive>,
    head_spec: crate::gguf::embedding::RerankHeadSpec,
}

impl RerankerLoader {
    fn prepare_gguf(
        &self,
        paths: &dyn ModelPaths,
        in_situ_quant: Option<IsqType>,
    ) -> Result<RerankerGgufContext> {
        anyhow::ensure!(
            self.config.quantized_filenames.is_some(),
            "Reranker models load from GGUF checkpoints; specify `quantized_filename` \
             (safetensors CrossEncoder loading is not supported yet)"
        );
        anyhow::ensure!(
            in_situ_quant.is_none(),
            "In-situ ISQ is not supported for GGUF reranker models; requantize the GGUF file \
             externally instead"
        );
        anyhow::ensure!(
            self.config.topology.is_none(),
            "Topology mapping is not supported for GGUF reranker models"
        );
        for path in paths.get_weight_filenames() {
            anyhow::ensure!(
                path.extension().and_then(|ext| ext.to_str()) == Some("gguf"),
                "GGUF reranker loading requires `.gguf` weight files, got `{}`",
                path.display()
            );
        }
        let archive = mistralrs_quant::GgufArchive::open(paths.get_weight_filenames())?;
        let head_spec = crate::gguf::embedding::rerank_head_spec_from_gguf(&archive)?;
        Ok(RerankerGgufContext {
            archive: Arc::new(archive),
            head_spec,
        })
    }

    fn resolve_gguf_tokenizer(
        &self,
        ctx: &RerankerGgufContext,
        paths: &dyn ModelPaths,
    ) -> Result<Tokenizer> {
        let path = paths.get_tokenizer_filename();
        if path.as_os_str().is_empty() {
            return Ok(super::embedding::convert_embedded_gguf_tokenizer(&ctx.archive)?.tokenizer);
        }
        let external = get_tokenizer(path, None).and_then(|tokenizer| {
            crate::gguf::validate_external_gguf_tokenizer(tokenizer, ctx.archive.metadata())
        });
        match external {
            Ok(conversion) => Ok(conversion.tokenizer),
            Err(error) => {
                if self.tokenizer_json.is_some() {
                    return Err(error).with_context(|| {
                        format!(
                            "Tokenizer `{}` is incompatible with the GGUF model vocabulary",
                            path.display()
                        )
                    });
                }
                warn!(
                    "Tokenizer from model assets `{}` does not match the GGUF vocabulary; using \
                     the tokenizer embedded in the GGUF instead: {error}",
                    path.display()
                );
                super::embedding::convert_embedded_gguf_tokenizer(&ctx.archive)
                    .map(|conversion| conversion.tokenizer)
                    .with_context(|| {
                        format!(
                            "Tokenizer `{}` was rejected ({error}); GGUF tokenizer conversion \
                             also failed",
                            path.display()
                        )
                    })
            }
        }
    }
}

impl Loader for RerankerLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let cache = self
            .config
            .hf_cache_path
            .clone()
            .map(Cache::new)
            .unwrap_or_default();
        GLOBAL_HF_CACHE.get_or_init(|| cache);

        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_embedding_paths!(
            EmbeddingModelPaths,
            &token_source,
            revision.clone(),
            self,
            Some(self.model_id.clone()),
            self.config.quantized_filenames.clone(),
            silent,
            false
        );
        *self
            .token_source
            .write()
            .expect("Failed to write to token source") = Some(token_source);
        *self.revision.write().expect("Failed to write to revision") = revision.clone();
        self.load_model_from_path(
            paths?.as_ref(),
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let gguf = self.prepare_gguf(paths, in_situ_quant)?;
        let config = {
            let external_config = if paths.get_config_filename().as_os_str().is_empty() {
                None
            } else {
                Some(std::fs::read_to_string(paths.get_config_filename())?)
            };
            let tensor_names = gguf.archive.tensors().keys().cloned().collect::<Vec<_>>();
            crate::gguf::embedding::prepare_embedding_config(
                crate::pipeline::EmbeddingLoaderType::Qwen3Embedding,
                external_config.as_deref(),
                gguf.archive.metadata(),
                &tensor_names,
            )?
        };
        let inner = &Qwen3EmbeddingLoader;

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for reranker models, disabling it.");
            paged_attn_config = None;
        }

        let tensor_parallelism = crate::distributed::resolve_tensor_parallelism(
            inner.model_config(&config)?.as_ref(),
            mistralrs_quant::distributed::use_nccl(),
            false,
        )?;
        anyhow::ensure!(
            !tensor_parallelism.is_enabled(),
            "GGUF reranker models do not support tensor parallelism"
        );

        let available_devices = device_map::get_all_similar_devices(device)?;
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        for device in &available_devices {
            if let Ok(dev) = device.as_role_device() {
                unsafe { dev.disable_event_tracking() };
            }
        }

        // If auto, convert to Map
        if let DeviceMapSetting::Auto(params) = mapper.clone() {
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;

            let (layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) = {
                let inventory = crate::pipeline::loaders::checkpoint_device_map_sizes(
                    paths.get_weight_filenames(),
                    inner.num_layers(&config)?,
                    dtype,
                    |name| inner.checkpoint_layer_index(&config, name),
                )?;
                if let Some(inventory) = inventory {
                    info!(
                        model_mib = inventory.total_model_size_in_bytes / (1024 * 1024),
                        "Using checkpoint tensor inventory for automatic device mapping"
                    );
                    (
                        inventory.layer_sizes_in_bytes,
                        inventory.non_mapped_size_in_bytes,
                        inventory.total_model_size_in_bytes,
                    )
                } else {
                    let weight_pack_factor =
                        QuantizationConfigShim::get_quant_config_pack_factor(&config, dtype)?;
                    let layer_sizes_in_bytes =
                        inner.layer_sizes_in_bytes(&config, dtype, weight_pack_factor, None)?;
                    let non_mapped_size_in_bytes = inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
                        None,
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                }
            };

            let new = auto_device_map::get_device_layers(
                inner,
                &config,
                inner.num_layers(&config)?,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &available_devices,
                dtype,
                &params,
                paged_attn_config.as_mut(),
            )?;
            mapper = DeviceMapSetting::Map(new);
        }

        let pipeline_mapper = mapper.into_mapper(
            inner.num_layers(&config)?,
            device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let mapper = mapper.into_mapper(
            inner.num_layers(&config)?,
            device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let dtype = super::isq_flow::resolve_weight_load_dtype(
            dtype,
            mapper.as_ref(),
            &available_devices,
            false,
        )?;

        trace!("Model config: {:?}", inner.get_config_repr(&config)?);

        let multi_progress = Arc::new(new_multi_progress());

        // The lane has no ISQ/UQFF/imatrix paths (prepare_gguf bails), so the
        // loading is always plain and never loading_isq.
        let bindings = crate::gguf::embedding::build_reranker_bindings(&gguf.archive)?;
        let source = Arc::new(mistralrs_quant::GgufWeightSource::new(
            gguf.archive.clone(),
            &bindings,
            dtype,
        )?);
        let vb = source.sharded_var_builder(Device::Cpu);
        let tracker = vb.tracker().clone();
        let model = inner.load(
            &config,
            vb,
            crate::pipeline::NormalLoadingMetadata {
                mapper,
                loading_isq: false,
                real_device: device.clone(),
                multi_progress,
                matformer_slicing_config: None,
                rope_pairing: None,
            },
            AttentionImplementation::Eager,
        )?;

        let head_weight = source
            .materialize_tensor(crate::gguf::embedding::RERANKER_SCORE_HEAD_BINDING, device)?
            .to_dtype(dtype)?;
        let (head_rows, head_cols) = head_weight.dims2()?;
        let hidden_size = crate::gguf::embedding::config_hidden_size(&config)?;
        anyhow::ensure!(
            (head_rows, head_cols) == (gguf.head_spec.labels.len(), hidden_size),
            "GGUF `cls.output.weight` has shape {head_rows}x{head_cols}, expected \
             {}x{hidden_size}",
            gguf.head_spec.labels.len()
        );
        let score_head = RerankScoreHead {
            linear: candle_nn::Linear::new(head_weight, None),
            yes_idx: gguf.head_spec.yes_idx,
        };

        if mistralrs_quant::GgufArchive::drop_host_after_load_enabled() {
            gguf.archive.release_host_shards();
        }

        let tokenizer = self.resolve_gguf_tokenizer(&gguf, paths)?;
        let tracked_modules = tracker.get().clone();

        info!(
            "Loaded reranker `{}` from GGUF (labels: {:?}, scoring label `yes` at index {}).",
            self.model_id, gguf.head_spec.labels, gguf.head_spec.yes_idx
        );

        let max_seq_len = inner.model_config(&config)?.max_seq_len();
        let has_causal_attention = inner.has_causal_attention(&config)?;

        Ok(Arc::new(Mutex::new(RerankerPipeline {
            model,
            tracked_modules,
            source_weight_files: paths.get_weight_filenames().to_vec(),
            tokenizer: tokenizer.into(),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: None,
                is_xlora: false,
                no_prefix_cache: false,
                num_hidden_layers: 1, // FIXME(EricLBuehler): we know this is only for caching, so its OK.
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true, // NOTE(EricLBuehler): no cache for these.
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Rerank],
                },
                loaded_for_uqff_write: false,
                paged_prefill_chunk_size: None,
            }),
            mapper: pipeline_mapper,
            processor: Arc::new(EmbeddingProcessor {
                has_causal_attention,
            }),
            score_head,
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for RerankerPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        self.processor.clone()
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for RerankerPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        if self.tracked_modules.is_empty() {
            anyhow::bail!("Runtime re-ISQ requires the model to have been loaded with ISQ.");
        }
        tracing::info!(
            "Re-quantizing {} layers to {dtype}.",
            self.tracked_modules.len()
        );
        super::isq_flow::requantize_and_swap(
            &self.tracked_modules,
            dtype,
            |module| module.default_type(dtype),
            &|_| None,
        )
    }

    fn begin_calibration(&mut self) -> Result<()> {
        super::isq_flow::begin_calibration(&self.tracked_modules).map(|_| ())
    }

    fn calibration_status(&self) -> Result<CalibrationStatus> {
        Ok(super::isq_flow::calibration_status(&self.tracked_modules))
    }

    fn apply_calibration(
        &mut self,
        save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<CalibrationStatus> {
        super::isq_flow::apply_calibration(
            &self.tracked_modules,
            &self.source_weight_files,
            None,
            save_cimatrix.as_deref(),
        )
    }
}

impl CacheManagerMixin for RerankerPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) -> candle_core::Result<()> {
        Ok(())
    }
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) -> candle_core::Result<()> {
        Ok(())
    }
    fn cache(&self) -> &EitherCache {
        unreachable!()
    }
}

impl MetadataMixin for RerankerPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[async_trait::async_trait]
impl Pipeline for RerankerPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        let ModelInputs {
            input_ids,
            flash_meta,
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");

        let xs = self.model.forward(&input_ids, &flash_meta)?;
        let scores = self.score_head.forward(&xs)?;

        // The score rides the per-sequence float-vector channel the embedding
        // transport already carries end to end.
        Ok(ForwardInputsResult::Embeddings { embeddings: scores })
    }
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Reranker
    }
}

impl AnyMoePipelineMixin for RerankerPipeline {}

#[cfg(test)]
mod gguf_load_tests {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

    use super::*;
    use crate::embedding_models::inputs_processor::ModelInputs;
    use crate::gguf::embedding::{GGUF_POOLING_RANK, RERANKER_SCORE_HEAD_BINDING};
    use crate::pipeline::text_models_inputs_processor::FlashParams;
    use crate::pipeline::{AdapterPaths, EmbeddingModelPaths, ModelCategory};
    use crate::ModelDType;
    use candle_core::quantized::gguf_file::{write as write_gguf, Value};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Tensor};
    use std::io::Write;

    const HIDDEN: usize = 16;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const KV_HEADS: usize = 1;
    const INTERMEDIATE: usize = 32;
    const LAYERS: usize = 2;
    const VOCAB: usize = 4;

    fn linspace(count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|index| (index as f32 / count as f32).sin().abs() * scale)
            .collect()
    }

    fn f32_qtensor(rows: usize, cols: usize) -> QTensor {
        let tensor = Tensor::from_vec(linspace(rows * cols, 1.0), (rows, cols), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        QTensor::quantize(&tensor, GgmlDType::F32).unwrap()
    }

    fn f32_qtensor_1d(len: usize) -> QTensor {
        let tensor = Tensor::from_vec(linspace(len, 1.0), len, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        QTensor::quantize(&tensor, GgmlDType::F32).unwrap()
    }

    fn write_reranker_gguf(path: &std::path::Path) {
        let mut metadata = vec![
            ("general.architecture", Value::String("qwen3".to_string())),
            ("qwen3.context_length", Value::U32(64)),
            ("qwen3.embedding_length", Value::U32(HIDDEN as u32)),
            ("qwen3.block_count", Value::U32(LAYERS as u32)),
            ("qwen3.feed_forward_length", Value::U32(INTERMEDIATE as u32)),
            ("qwen3.attention.head_count", Value::U32(HEADS as u32)),
            ("qwen3.attention.head_count_kv", Value::U32(KV_HEADS as u32)),
            ("qwen3.attention.key_length", Value::U32(HEAD_DIM as u32)),
            ("qwen3.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("qwen3.rope.freq_base", Value::F32(10_000.0)),
            ("qwen3.pooling_type", Value::U32(GGUF_POOLING_RANK)),
            (
                "qwen3.classifier.output_labels",
                Value::Array(
                    ["yes", "no"]
                        .iter()
                        .map(|label| Value::String(label.to_string()))
                        .collect(),
                ),
            ),
            ("tokenizer.ggml.model", Value::String("llama".to_string())),
            (
                "tokenizer.ggml.tokens",
                Value::Array(
                    ["<unk>", "a", "b", "<eos>"]
                        .iter()
                        .map(|token| Value::String(token.to_string()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.scores",
                Value::Array((0..VOCAB).map(|_| Value::F32(0.0)).collect()),
            ),
            ("tokenizer.ggml.eos_token_id", Value::U32(3)),
            ("tokenizer.ggml.bos_token_id", Value::U32(0)),
        ];
        metadata.sort_by(|a, b| a.0.cmp(b.0));

        let mut tensors: Vec<(String, QTensor)> = vec![
            ("token_embd.weight".to_string(), f32_qtensor(VOCAB, HIDDEN)),
            ("output_norm.weight".to_string(), f32_qtensor_1d(HIDDEN)),
            ("cls.output.weight".to_string(), f32_qtensor(2, HIDDEN)),
        ];
        for layer in 0..LAYERS {
            let prefix = format!("blk.{layer}");
            tensors.push((format!("{prefix}.attn_norm.weight"), f32_qtensor_1d(HIDDEN)));
            tensors.push((
                format!("{prefix}.attn_q.weight"),
                f32_qtensor(HEADS * HEAD_DIM, HIDDEN),
            ));
            tensors.push((
                format!("{prefix}.attn_k.weight"),
                f32_qtensor(KV_HEADS * HEAD_DIM, HIDDEN),
            ));
            tensors.push((
                format!("{prefix}.attn_v.weight"),
                f32_qtensor(KV_HEADS * HEAD_DIM, HIDDEN),
            ));
            tensors.push((
                format!("{prefix}.attn_output.weight"),
                f32_qtensor(HIDDEN, HEADS * HEAD_DIM),
            ));
            tensors.push((
                format!("{prefix}.attn_q_norm.weight"),
                f32_qtensor_1d(HEAD_DIM),
            ));
            tensors.push((
                format!("{prefix}.attn_k_norm.weight"),
                f32_qtensor_1d(HEAD_DIM),
            ));
            tensors.push((format!("{prefix}.ffn_norm.weight"), f32_qtensor_1d(HIDDEN)));
            tensors.push((
                format!("{prefix}.ffn_gate.weight"),
                f32_qtensor(INTERMEDIATE, HIDDEN),
            ));
            tensors.push((
                format!("{prefix}.ffn_up.weight"),
                f32_qtensor(INTERMEDIATE, HIDDEN),
            ));
            tensors.push((
                format!("{prefix}.ffn_down.weight"),
                f32_qtensor(HIDDEN, INTERMEDIATE),
            ));
        }

        let mut file = std::fs::File::create(path).unwrap();
        let tensors_ref = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let metadata_ref = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        write_gguf(&mut file, &metadata_ref, &tensors_ref).unwrap();
        file.flush().unwrap();
    }

    fn load_synthetic_gguf_pipeline(
        root: &std::path::Path,
    ) -> Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>> {
        let loader = RerankerLoaderBuilder::new(
            RerankerSpecificConfig {
                quantized_filenames: Some(vec!["model.gguf".to_string()]),
                ..Default::default()
            },
            None,
            Some(root.to_string_lossy().into_owned()),
        )
        .build();
        let paths = EmbeddingModelPaths::new(
            PathBuf::new(),
            PathBuf::new(),
            vec![root.join("model.gguf")],
            AdapterPaths::None,
            Vec::new(),
        );
        loader
            .load_model_from_path(
                &paths,
                &ModelDType::Auto,
                &Device::Cpu,
                true,
                DeviceMapSetting::dummy(),
                None,
                None,
            )
            .unwrap()
    }

    #[test]
    fn loads_qwen3_reranker_gguf_and_scores() {
        let root = std::env::temp_dir().join(format!(
            "mistralrs-rerank-gguf-e2e-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        write_reranker_gguf(&root.join("model.gguf"));

        let pipeline = load_synthetic_gguf_pipeline(&root);
        let mut pipeline = pipeline.blocking_lock();
        assert_eq!(pipeline.category(), ModelCategory::Reranker);

        let inputs = ModelInputs {
            input_ids: Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &Device::Cpu).unwrap(),
            flash_meta: FlashParams::empty(true),
        };
        let result = pipeline.forward_inputs(Box::new(inputs), false).unwrap();
        let ForwardInputsResult::Embeddings { embeddings } = result else {
            panic!("expected embeddings output");
        };

        assert_eq!(embeddings.dims(), &[1, 1]);
        let score = embeddings.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(score.is_finite(), "score is not finite: {score}");
        assert!(
            (0.0..=1.0).contains(&score),
            "score is not a probability: {score}"
        );
    }

    #[test]
    fn reranker_gguf_refuses_embedding_checkpoints() {
        let root = std::env::temp_dir().join(format!(
            "mistralrs-rerank-gguf-reject-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();

        // Same checkpoint shape as the embedding e2e test but with MEAN pooling
        // and no classifier head: the reranker lane must refuse it loudly.
        let mut metadata = vec![
            ("general.architecture", Value::String("qwen3".to_string())),
            ("qwen3.context_length", Value::U32(64)),
            ("qwen3.embedding_length", Value::U32(HIDDEN as u32)),
            ("qwen3.block_count", Value::U32(LAYERS as u32)),
            ("qwen3.feed_forward_length", Value::U32(INTERMEDIATE as u32)),
            ("qwen3.attention.head_count", Value::U32(HEADS as u32)),
            ("qwen3.attention.head_count_kv", Value::U32(KV_HEADS as u32)),
            ("qwen3.attention.key_length", Value::U32(HEAD_DIM as u32)),
            ("qwen3.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("qwen3.rope.freq_base", Value::F32(10_000.0)),
            ("qwen3.pooling_type", Value::U32(1)),
            ("tokenizer.ggml.model", Value::String("llama".to_string())),
            (
                "tokenizer.ggml.tokens",
                Value::Array(
                    ["<unk>", "a", "b", "<eos>"]
                        .iter()
                        .map(|token| Value::String(token.to_string()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.scores",
                Value::Array((0..VOCAB).map(|_| Value::F32(0.0)).collect()),
            ),
        ];
        metadata.sort_by(|a, b| a.0.cmp(b.0));
        let mut file = std::fs::File::create(root.join("model.gguf")).unwrap();
        let metadata_ref = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        write_gguf(&mut file, &metadata_ref, &[]).unwrap();
        file.flush().unwrap();

        let loader = RerankerLoaderBuilder::new(
            RerankerSpecificConfig {
                quantized_filenames: Some(vec!["model.gguf".to_string()]),
                ..Default::default()
            },
            None,
            Some(root.to_string_lossy().into_owned()),
        )
        .build();
        let paths = EmbeddingModelPaths::new(
            PathBuf::new(),
            PathBuf::new(),
            vec![root.join("model.gguf")],
            AdapterPaths::None,
            Vec::new(),
        );
        let err = loader
            .load_model_from_path(
                &paths,
                &ModelDType::Auto,
                &Device::Cpu,
                true,
                DeviceMapSetting::dummy(),
                None,
                None,
            )
            .err()
            .expect("reranker lane must refuse non-RANK checkpoints")
            .to_string();
        assert!(err.contains("RANK"), "{err}");
    }

    #[test]
    fn score_head_binding_is_bound() {
        // The head binding must round-trip through the GGUF weight source by
        // native name so the loader can materialize it.
        let root =
            std::env::temp_dir().join(format!("mistralrs-rerank-bind-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        write_reranker_gguf(&root.join("model.gguf"));
        let archive = mistralrs_quant::GgufArchive::open(&[root.join("model.gguf")]).unwrap();
        let bindings = crate::gguf::embedding::build_reranker_bindings(&archive).unwrap();
        assert!(bindings.get(RERANKER_SCORE_HEAD_BINDING).is_some());
    }
}
