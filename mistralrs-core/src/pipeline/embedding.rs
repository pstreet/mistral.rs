use super::isq::{
    write_uqff_artifacts, UqffFullSer, UqffWriteConfig, UqffWriteRequest, WeightLoadingMode,
    WeightLoadingState,
};
use super::{
    get_model_paths, AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, IsqPipelineMixin, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, TokenSource,
};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{self, WorkerTransferData};
use crate::embedding_models::inputs_processor::{EmbeddingProcessor, ModelInputs};
use crate::embedding_models::{Dense, DenseActivation, Normalize, Pooling};
use crate::embedding_normal_model_loader;
use crate::embedding_normal_model_loader_sharded;
use crate::get_embedding_paths;
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::loaders::auto_device_map;
use crate::pipeline::loaders::{AutoDeviceMapQuantization, QuantizationConfigShim};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::EmbeddingLoaderType;
use crate::pipeline::EmbeddingModel;
use crate::pipeline::EmbeddingModelLoader;
use crate::pipeline::{AutoEmbeddingLoader, EmbeddingModulePaths};
use crate::pipeline::{ChatTemplate, EmbeddingModelPaths, IsqOrganization, Processor};
use crate::pipeline::{EmbeddingGemmaLoader, Qwen3EmbeddingLoader};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::Modalities;
use crate::SupportedModality;
use crate::{
    get_uqff_paths, DeviceMapSetting, PagedAttentionConfig, Pipeline, Topology, TryIntoDType,
    GLOBAL_HF_CACHE,
};
use anyhow::Context;
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use hf_hub::Cache;
use hf_hub::{Repo, RepoType};
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::safetensors::MmapedSafetensors;
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};

pub struct EmbeddingPipeline {
    model: Box<dyn EmbeddingModel + Send + Sync>,
    tracked_modules: Vec<mistralrs_quant::TrackedModule>,
    source_weight_files: Vec<std::path::PathBuf>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    modules: Vec<Box<dyn Module + Send + Sync>>,
    processor: Arc<dyn Processor + Send + Sync>,
}

/// A loader for an embedding (non-quantized) model.
pub struct EmbeddingLoader {
    inner: Box<dyn EmbeddingModelLoader>,
    explicit_arch: Option<EmbeddingLoaderType>,
    model_id: String,
    config: EmbeddingSpecificConfig,
    kind: ModelKind,
    tokenizer_json: Option<String>,
    token_source: RwLock<Option<TokenSource>>,
    revision: RwLock<Option<String>>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    hf_cache_path: Option<PathBuf>,
    load_context: EmbeddingLoadContext,
}

#[derive(Clone, Copy, Default)]
pub(crate) enum EmbeddingLoadContext {
    #[default]
    Primary,
    Search,
}

impl EmbeddingLoadContext {
    fn weight_target(self) -> &'static str {
        match self {
            Self::Primary => "model",
            Self::Search => "search embedding model",
        }
    }
}

#[derive(Default)]
/// A builder for a loader for an embedding (non-quantized) model.
pub struct EmbeddingLoaderBuilder {
    model_id: Option<String>,
    config: EmbeddingSpecificConfig,
    kind: ModelKind,
    tokenizer_json: Option<String>,
    hf_cache_path: Option<PathBuf>,
    load_context: EmbeddingLoadContext,
}

#[derive(Clone, Default)]
/// Config specific to loading an embedding model.
pub struct EmbeddingSpecificConfig {
    pub topology: Option<Topology>,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
    /// GGUF filename(s), resolved within `model_id` at load time. `None` loads
    /// the safetensors checkpoint from the model repository.
    pub quantized_filenames: Option<Vec<String>>,
}

impl EmbeddingLoaderBuilder {
    pub fn new(
        config: EmbeddingSpecificConfig,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
    ) -> Self {
        let hf_cache_path = config.hf_cache_path.clone();
        Self {
            config,
            tokenizer_json,
            model_id,
            kind: ModelKind::Normal,
            hf_cache_path,
            ..Default::default()
        }
    }

    pub fn hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    pub(crate) fn with_load_context(mut self, load_context: EmbeddingLoadContext) -> Self {
        self.load_context = load_context;
        self
    }

    pub fn build(self, loader: Option<EmbeddingLoaderType>) -> Box<dyn Loader> {
        let explicit_arch = loader;
        let loader: Box<dyn EmbeddingModelLoader> = match explicit_arch {
            Some(EmbeddingLoaderType::EmbeddingGemma) => Box::new(EmbeddingGemmaLoader),
            Some(EmbeddingLoaderType::Qwen3Embedding) => Box::new(Qwen3EmbeddingLoader),
            None => Box::new(AutoEmbeddingLoader),
        };
        Box::new(EmbeddingLoader {
            inner: loader,
            explicit_arch,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            tokenizer_json: self.tokenizer_json,
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            load_context: self.load_context,
        })
    }
}

/// A GGUF embedding checkpoint opened for the GGUF loading arm of
/// [`EmbeddingLoader`].
struct EmbeddingGgufContext {
    archive: Arc<mistralrs_quant::GgufArchive>,
    loader: EmbeddingLoaderType,
}

/// Convert the tokenizer embedded in the GGUF. `tokenizer.ggml.add_eos_token`
/// mirrors llama.cpp's embedding semantics: the EOS is appended at embed time
/// (a post-processor here), because last-token pooling reads the EOS position.
fn convert_embedded_gguf_tokenizer(
    ctx: &EmbeddingGgufContext,
) -> Result<crate::gguf::GgufTokenizerConversion> {
    let conversion = crate::gguf::convert_gguf_metadata_to_hf_tokenizer(ctx.archive.metadata())?;
    let add_eos = match ctx.archive.metadata_value("tokenizer.ggml.add_eos_token") {
        Some(candle_core::quantized::gguf_file::Value::Bool(value)) => *value,
        Some(value) => {
            anyhow::bail!("GGUF `tokenizer.ggml.add_eos_token` must be a bool, got {value:?}")
        }
        None => false,
    };
    if !add_eos {
        return Ok(conversion);
    }
    let eos = conversion
        .eos
        .as_ref()
        .context("GGUF `tokenizer.ggml.add_eos_token` is set but the tokenizer has no EOS token")?;
    let eos_id = conversion
        .tokenizer
        .token_to_id(eos)
        .with_context(|| format!("EOS token `{eos}` is missing from the GGUF tokenizer"))?;
    let post_processor = {
        let append_eos = tokenizers::processors::template::TemplateProcessing::builder()
            .try_single(format!("$A:0 {eos}:0"))
            .map_err(anyhow::Error::msg)?
            .try_pair("$A:0 $B:1".to_string())
            .map_err(anyhow::Error::msg)?
            .special_tokens(vec![(eos.clone(), eos_id)])
            .build()
            .map_err(anyhow::Error::msg)?;
        // The BPE conversion attaches a byte-level post-processor; keep it in front.
        match conversion.tokenizer.get_post_processor() {
            Some(existing) => tokenizers::processors::PostProcessorWrapper::Sequence(
                tokenizers::processors::sequence::Sequence::new(vec![
                    existing.clone(),
                    tokenizers::processors::PostProcessorWrapper::Template(append_eos),
                ]),
            ),
            None => tokenizers::processors::PostProcessorWrapper::Template(append_eos),
        }
    };
    let mut tokenizer = conversion.tokenizer;
    tokenizer.with_post_processor(Some(post_processor));
    Ok(crate::gguf::GgufTokenizerConversion {
        tokenizer,
        bos: conversion.bos,
        eos: conversion.eos,
        unk: conversion.unk,
    })
}

impl EmbeddingLoader {
    fn prepare_gguf(
        &self,
        paths: &dyn ModelPaths,
        in_situ_quant: Option<IsqType>,
    ) -> Result<Option<EmbeddingGgufContext>> {
        if self.config.quantized_filenames.is_none() {
            return Ok(None);
        }
        anyhow::ensure!(
            in_situ_quant.is_none(),
            "In-situ ISQ is not supported for GGUF embedding models; requantize the GGUF file \
             externally instead"
        );
        anyhow::ensure!(
            self.config.topology.is_none(),
            "Topology mapping is not supported for GGUF embedding models"
        );
        anyhow::ensure!(
            self.config.from_uqff.is_none()
                && self.config.write_uqff.is_none()
                && self.config.imatrix.is_none()
                && self.config.calibration_file.is_none(),
            "UQFF and imatrix flows are not supported for GGUF embedding models"
        );
        for path in paths.get_weight_filenames() {
            anyhow::ensure!(
                path.extension().and_then(|ext| ext.to_str()) == Some("gguf"),
                "GGUF embedding loading requires `.gguf` weight files, got `{}`",
                path.display()
            );
        }
        let archive = mistralrs_quant::GgufArchive::open(paths.get_weight_filenames())?;
        let architecture = match archive.metadata_value("general.architecture") {
            Some(candle_core::quantized::gguf_file::Value::String(value)) => value.as_str(),
            Some(value) => {
                anyhow::bail!("GGUF `general.architecture` must be a string, got {value:?}")
            }
            None => anyhow::bail!("GGUF metadata is missing `general.architecture`"),
        };
        let loader = crate::gguf::embedding::embedding_loader_for_architecture(
            architecture,
            self.explicit_arch,
        )?;
        Ok(Some(EmbeddingGgufContext {
            archive: Arc::new(archive),
            loader,
        }))
    }

    fn resolve_gguf_tokenizer(
        &self,
        ctx: &EmbeddingGgufContext,
        paths: &dyn ModelPaths,
    ) -> Result<Tokenizer> {
        let path = paths.get_tokenizer_filename();
        if path.as_os_str().is_empty() {
            return Ok(convert_embedded_gguf_tokenizer(ctx)?.tokenizer);
        }
        let external = get_tokenizer(path, None).and_then(|tokenizer| {
            crate::gguf::validate_external_gguf_tokenizer(tokenizer, ctx.archive.metadata())
        });
        match external {
            Ok(conversion) => Ok(conversion.tokenizer),
            Err(error) => {
                if self.tokenizer_json.is_some() {
                    // An explicit `tokenizer_json` override is strict, same as the text GGUF lane.
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
                convert_embedded_gguf_tokenizer(ctx)
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

impl Loader for EmbeddingLoader {
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
            self.config
                .quantized_filenames
                .is_some()
                .then(|| self.model_id.clone()),
            self.config.quantized_filenames.clone(),
            silent,
            self.config.from_uqff.is_some()
        );
        *self
            .token_source
            .write()
            .expect("Failed to write to token source") = Some(token_source);
        *self.revision.write().expect("Failed to write to revision") = revision.clone();
        if let Some(from_uqff) = self.config.from_uqff.clone() {
            *self.from_uqff.write().unwrap() = Some(get_uqff_paths!(&from_uqff, self, silent));
        }
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
        let embedding_gguf = self.prepare_gguf(paths, in_situ_quant)?;
        let config = match &embedding_gguf {
            Some(ctx) => {
                let external_config = if paths.get_config_filename().as_os_str().is_empty() {
                    None
                } else {
                    Some(std::fs::read_to_string(paths.get_config_filename())?)
                };
                let tensor_names = ctx.archive.tensors().keys().cloned().collect::<Vec<_>>();
                crate::gguf::embedding::prepare_embedding_config(
                    ctx.loader,
                    external_config.as_deref(),
                    ctx.archive.metadata(),
                    &tensor_names,
                )?
            }
            None => {
                let config = std::fs::read_to_string(paths.get_config_filename())?;
                if self.config.from_uqff.is_some() {
                    super::isq::sanitize_quantized_weight_source_config(&config)?
                } else {
                    config
                }
            }
        };
        // GGUF checkpoints pin the loader via `general.architecture`; auto detection
        // from the config would race the synthesized architectures field.
        let inner: &dyn EmbeddingModelLoader = match &embedding_gguf {
            Some(ctx) => match ctx.loader {
                EmbeddingLoaderType::Qwen3Embedding => &Qwen3EmbeddingLoader,
                EmbeddingLoaderType::EmbeddingGemma => &EmbeddingGemmaLoader,
            },
            None => &*self.inner,
        };

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for embedding models, disabling it.");
            paged_attn_config = None;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let use_nccl = mistralrs_quant::distributed::use_nccl();
        let write_uqff = self.config.write_uqff.is_some();
        let tensor_parallelism = distributed::resolve_tensor_parallelism(
            inner.model_config(&config)?.as_ref(),
            use_nccl,
            write_uqff,
        )?;
        let use_distributed = tensor_parallelism.is_enabled();
        if use_distributed {
            anyhow::ensure!(
                embedding_gguf.is_none(),
                "GGUF embedding models do not support tensor parallelism"
            );
        }

        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { worker_rank, .. } = payload;
            vec![candle_core::Device::new_cuda(worker_rank + 1)?]
        } else if use_distributed {
            vec![candle_core::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        for device in &available_devices {
            if let Ok(dev) = device.as_role_device() {
                unsafe { dev.disable_event_tracking() };
            }
        }
        let device = if use_distributed {
            available_devices[0].clone()
        } else {
            device.clone()
        };
        let uqff_reader = if let Some(from_uqff) = &*self.from_uqff.read().unwrap() {
            Some(Arc::new(mistralrs_quant::UqffReader::open(from_uqff)?))
        } else {
            None
        };

        // If auto, convert to Map if not using nccl
        if write_uqff {
            mapper = DeviceMapSetting::dummy();
        } else if use_distributed {
            mapper = DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            };
        } else if let DeviceMapSetting::Auto(params) = mapper.clone() {
            // Initial dtype
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
                if let Some(reader) = uqff_reader.as_ref() {
                    let weight_pack_factor = reader.pack_factor(dtype)?;
                    let quantization = AutoDeviceMapQuantization::uqff(reader);
                    let layer_sizes_in_bytes =
                        inner.layer_sizes_in_bytes(&config, dtype, weight_pack_factor, None)?;
                    let non_mapped_size_in_bytes = inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        Some(&quantization),
                        None,
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                } else if let Some(isq) = in_situ_quant {
                    let quantization =
                        AutoDeviceMapQuantization::isq(Some(isq), self.config.topology.as_ref());
                    let weight_pack_factor =
                        quantization.conservative_pack_factor(dtype, isq.pack_factor(dtype));
                    let layer_sizes_in_bytes =
                        inner.layer_sizes_in_bytes(&config, dtype, weight_pack_factor, None)?;
                    let non_mapped_size_in_bytes = inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        Some(&quantization),
                        None,
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                } else {
                    let inventory = if self.config.topology.is_none() {
                        let num_layers = inner.num_layers(&config)?;
                        crate::pipeline::loaders::checkpoint_device_map_sizes(
                            paths.get_weight_filenames(),
                            num_layers,
                            dtype,
                            |name| inner.checkpoint_layer_index(&config, name),
                        )?
                    } else {
                        None
                    };
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
                        // Be sure to get the weight pack factor here; we might be loading a prequantized model.
                        let weight_pack_factor =
                            QuantizationConfigShim::get_quant_config_pack_factor(&config, dtype)?;
                        let quantization =
                            self.config.topology.as_ref().map(|topology| {
                                AutoDeviceMapQuantization::isq(None, Some(topology))
                            });
                        let weight_pack_factor =
                            quantization
                                .as_ref()
                                .map_or(weight_pack_factor, |quantization| {
                                    quantization.conservative_pack_factor(dtype, weight_pack_factor)
                                });
                        let layer_sizes_in_bytes =
                            inner.layer_sizes_in_bytes(&config, dtype, weight_pack_factor, None)?;
                        let non_mapped_size_in_bytes = inner.non_mapped_size_in_bytes(
                            &config,
                            dtype,
                            weight_pack_factor,
                            quantization.as_ref(),
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

        let mapper_device = if write_uqff {
            Device::Cpu
        } else {
            device.clone()
        };
        let mapper_topology = if write_uqff {
            None
        } else {
            self.config.topology.as_ref()
        };

        let pipeline_mapper = mapper.into_mapper(
            inner.num_layers(&config)?,
            &mapper_device,
            mapper_topology,
            &available_devices,
        )?;
        let mapper = mapper.into_mapper(
            inner.num_layers(&config)?,
            &mapper_device,
            mapper_topology,
            &available_devices,
        )?;
        let mut layer_devices = Vec::new();
        for layer in 0..inner.num_layers(&config)? {
            let device = mapper.device_for(layer, false).cloned();
            layer_devices.push(device);
        }
        let dtype = super::isq_flow::resolve_weight_load_dtype(
            dtype,
            mapper.as_ref(),
            &available_devices,
            write_uqff,
        )?;

        trace!("Model config: {:?}", inner.get_config_repr(&config)?);
        if crate::using_flash_attn() {
            once_log_info("FlashAttention is enabled.");
        }

        let topology_overrides = self
            .config
            .topology
            .as_ref()
            .map(|topology| topology.immediate_overrides())
            .unwrap_or_default();

        let plan = super::isq_flow::resolve_and_install_isq_plan(super::isq_flow::IsqPlanInputs {
            in_situ_quant,
            has_imatrix: self.config.imatrix.is_some(),
            has_calibration: self.config.calibration_file.is_some(),
            write_uqff_types: self.config.write_uqff.as_ref().map(|c| c.types.clone()),
            has_write_uqff: self.config.write_uqff.is_some(),
            loading_from_uqff: self.config.from_uqff.is_some(),
            organization: Default::default(),
            topology_overrides,
            loader: inner,
            config: &config,
            device: &device,
        })?;
        let use_immediate = plan.immediate_isq_installed;
        let loading_isq = plan.loading_isq;
        let load_device = plan.load_device.clone();

        let attention_mechanism = if paged_attn_config.is_some() {
            // Capability gate: the configured PA stack may not exist for
            // this device's role in this build (the NVIDIA role in dual
            // builds has no paged-attention companion yet). Degrade loudly
            // to eager attention rather than building layers that fault
            // on their first step.
            if crate::device_spec::backend_caps(&device).paged_attention {
                AttentionImplementation::PagedAttention
            } else {
                warn!(
                    "PagedAttention is configured but this device's role in this build has no \
                     paged-attention kernels; serving with eager attention"
                );
                AttentionImplementation::Eager
            }
        } else {
            AttentionImplementation::Eager
        };

        let multi_progress = Arc::new(new_multi_progress());

        let modules_config: Vec<_> = paths.get_modules().map(|m| m.to_vec()).unwrap_or_default();

        let mut modules: Vec<Box<dyn Module + Send + Sync>> = Vec::new();
        if modules_config.is_empty() {
            let Some(ctx) = &embedding_gguf else {
                anyhow::bail!("Embedding models require the `modules.json` file.");
            };
            // No sentence-transformers assets: pooling comes from the llama.cpp
            // `{arch}.pooling_type` metadata, normalization matches the reference
            // sentence-transformers layout of both supported models.
            let pooling = crate::gguf::embedding::pooling_from_gguf_metadata(
                &ctx.archive,
                ctx.loader,
                crate::gguf::embedding::config_hidden_size(&config)?,
            )?;
            info!(
                "No `modules.json` found; using GGUF pooling metadata (mean: {}, last-token: {}) \
                 followed by normalization",
                pooling.pooling_mode_mean_tokens, pooling.pooling_mode_lasttoken
            );
            modules.push(Box::new(pooling));
            modules.push(Box::new(Normalize));
        } else {
            assert!(matches!(
                modules_config.first(),
                Some(EmbeddingModulePaths::Transformer { .. })
            ));
            for module in &modules_config {
                match module {
                    EmbeddingModulePaths::Transformer { .. } => (),
                    EmbeddingModulePaths::Pooling { config, .. } => {
                        let layer: Pooling =
                            serde_json::from_str(&std::fs::read_to_string(config)?)?;
                        modules.push(Box::new(layer));
                    }
                    EmbeddingModulePaths::Dense { config, model, .. } => {
                        let config: Dense =
                            serde_json::from_str(&std::fs::read_to_string(config)?)?;
                        let safetensors = unsafe { MmapedSafetensors::new(model)? };
                        let weight = safetensors.load("linear.weight", &device, Some(dtype))?;
                        let bias = if config.bias {
                            Some(safetensors.load("linear.bias", &device, Some(dtype))?)
                        } else {
                            None
                        };
                        let (out_f, in_f) = weight.dims2()?;
                        assert_eq!((out_f, in_f), (config.out_features, config.in_features));
                        if !matches!(config.activation_function, DenseActivation::Identity) {
                            anyhow::bail!("Expected Identity activation function.");
                        }

                        modules.push(Box::new(Linear::new(weight, bias)));
                    }
                    EmbeddingModulePaths::Normalize { .. } => {
                        modules.push(Box::new(Normalize));
                    }
                }
            }
        }
        info!(
            "{}",
            WeightLoadingMode::from(WeightLoadingState {
                from_uqff: self.config.from_uqff.is_some(),
                loading_isq,
                immediate_isq: use_immediate,
                write_uqff: self.config.write_uqff.is_some(),
            })
            .message(self.load_context.weight_target())
        );

        let (model, tracker) = if use_distributed {
            let (mapper, sharded_vb) =
                distributed::prepare_distributed_mapper(distributed::DistributedMapperConfig {
                    dtype,
                    device: &device,
                    available_devices: &available_devices,
                    global_world_size_override: tensor_parallelism.world_size(),
                    silent,
                    config: &config,
                    loading_isq,
                    from_uqff: self.config.from_uqff.is_some(),
                    write_uqff: self.config.write_uqff.is_some(),
                    organization: IsqOrganization::Default,
                    model: inner,
                    weights: distributed::DistributedWeightSource::Paths(paths),
                })?;
            let sharded_vb = if let Some(reader) = uqff_reader.clone() {
                sharded_vb.with_uqff_reader(reader)
            } else {
                sharded_vb
            };

            // Special case for where things can be more optimially loaded.
            match self.kind {
                ModelKind::Normal => embedding_normal_model_loader_sharded!(
                    sharded_vb,
                    config,
                    inner,
                    mapper,
                    loading_isq,
                    device.clone(),
                    attention_mechanism,
                    multi_progress.clone(),
                    uqff_reader.clone(),
                ),
                _ => unreachable!(),
            }
        } else {
            match self.kind {
                ModelKind::Normal => {
                    if let Some(ctx) = &embedding_gguf {
                        let bindings = crate::gguf::embedding::build_embedding_bindings(
                            &ctx.archive,
                            ctx.loader,
                        )?;
                        let source = Arc::new(mistralrs_quant::GgufWeightSource::new(
                            ctx.archive.clone(),
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
                                loading_isq,
                                real_device: device.clone(),
                                multi_progress,
                                matformer_slicing_config: None,
                                rope_pairing: None,
                            },
                            attention_mechanism,
                        )?;
                        (model, tracker)
                    } else {
                        embedding_normal_model_loader!(
                            paths,
                            Some(dtype),
                            &load_device,
                            layer_devices.clone(),
                            config,
                            inner,
                            silent,
                            mapper,
                            loading_isq,
                            self.config.from_uqff.is_some(),
                            device.clone(),
                            attention_mechanism,
                            multi_progress,
                            uqff_reader.clone(),
                        )
                    }
                }
                _ => unreachable!(),
            }
        };

        if let Some(ctx) = &embedding_gguf {
            if mistralrs_quant::GgufArchive::drop_host_after_load_enabled() {
                ctx.archive.release_host_shards();
            }
        }

        let tokenizer = match &embedding_gguf {
            Some(ctx) => self.resolve_gguf_tokenizer(ctx, paths)?,
            None => get_tokenizer(paths.get_tokenizer_filename(), None)?,
        };

        plan.validate_tracked_selection(&tracker.get())?;

        let imatrix_map = if plan.wants_imatrix {
            let drive = super::isq_flow::EmbeddingCalibrationDrive(&*model);
            Some(super::isq_flow::resolve_imatrix_map(
                &drive,
                &tracker.get().clone(),
                self.config.imatrix.as_ref(),
                self.config.calibration_file.as_ref(),
                &super::isq_flow::CalibrationCtx {
                    tokenizer: &tokenizer,
                    bos_tok_id: None,
                    load_device: &load_device,
                    mapper: Some(pipeline_mapper.as_ref()),
                },
            )?)
        } else {
            None
        };

        if plan.capture == mistralrs_quant::IsqCaptureMode::CaptureMatches {
            let ty = in_situ_quant.context("imatrix quantization requires an ISQ type")?;
            super::isq_flow::complete_isq_capture(
                &tracker.get().clone(),
                ty,
                imatrix_map
                    .as_ref()
                    .expect("CaptureMatches requires imatrix data"),
            )?;
        }

        if let Some(write_uqff) = &self.config.write_uqff {
            let layers = tracker.get().clone();
            let uqff_types = plan
                .write_types
                .clone()
                .filter(|types| !types.is_empty())
                .context("UQFF serialization requires at least one ISQ type.")?;
            let modules_json = EmbeddingModulePaths::serialize_modules(&modules_config);
            let full_ser = UqffFullSer {
                tokenizer: &tokenizer,
                template_filename: paths.get_template_filename(),
                effective_chat_template: None,
                generation_config: paths.get_gen_conf_filename(),
                config: config.clone(),
                processor_filename: &None,
                preprocessor_filename: &None,
                modules: Some(&modules_json),
                module_paths: Some(&modules_config),
            };
            write_uqff_artifacts(UqffWriteRequest {
                output: write_uqff.output.clone(),
                types: uqff_types,
                base_model: write_uqff.base_model.clone(),
                repo_id: write_uqff.repo_id.clone(),
                layers,
                quantize_predicates: plan.uqff_quantize_predicates.clone(),
                residual: model.residual_tensors(),
                full_ser,
                imatrix: imatrix_map.unwrap_or_default(),
            })?;
        }

        if plan.immediate_isq_installed {
            for module in tracker.get().clone() {
                module.ct.resolve()?;
            }
        }

        let has_causal_attention = inner.has_causal_attention(&config)?;
        let max_seq_len = inner.model_config(&config)?.max_seq_len();
        let tracked_modules = tracker.get().clone();
        // rank-sliced layers re-slice at source read; inexpressible slices fall back per layer
        let source_weight_files = if self.config.from_uqff.is_some() {
            Vec::new()
        } else {
            paths.get_weight_filenames().to_vec()
        };

        Ok(Arc::new(Mutex::new(EmbeddingPipeline {
            model,
            tracked_modules,
            source_weight_files,
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
                    output: vec![SupportedModality::Embedding],
                },
                loaded_for_uqff_write: self.config.write_uqff.is_some(),
                paged_prefill_chunk_size: None,
            }),
            mapper: pipeline_mapper,
            modules,
            processor: Arc::new(EmbeddingProcessor {
                has_causal_attention,
            }),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for EmbeddingPipeline {
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

impl IsqPipelineMixin for EmbeddingPipeline {
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

    fn calibration_status(&self) -> Result<super::isq_flow::CalibrationStatus> {
        Ok(super::isq_flow::calibration_status(&self.tracked_modules))
    }

    fn apply_calibration(
        &mut self,
        save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<super::isq_flow::CalibrationStatus> {
        super::isq_flow::apply_calibration(
            &self.tracked_modules,
            &self.source_weight_files,
            None,
            save_cimatrix.as_deref(),
        )
    }
}

impl CacheManagerMixin for EmbeddingPipeline {
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

impl MetadataMixin for EmbeddingPipeline {
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
impl Pipeline for EmbeddingPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        let ModelInputs {
            input_ids,
            flash_meta,
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");

        let mut xs = self.model.forward(&input_ids, &flash_meta)?;
        for module in &self.modules {
            xs = module.forward(&xs)?;
        }

        Ok(ForwardInputsResult::Embeddings { embeddings: xs })
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
        ModelCategory::Embedding
    }
}

impl AnyMoePipelineMixin for EmbeddingPipeline {}

#[cfg(test)]
mod gguf_load_tests {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

    use super::*;
    use crate::embedding_models::inputs_processor::ModelInputs;
    use crate::pipeline::text_models_inputs_processor::FlashParams;
    use crate::pipeline::{AdapterPaths, EmbeddingModelPaths};
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

    fn write_embedding_gguf(path: &std::path::Path) {
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
            ("qwen3.pooling_type", Value::U32(3)),
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
            ("tokenizer.ggml.add_eos_token", Value::Bool(true)),
        ];
        metadata.sort_by(|a, b| a.0.cmp(b.0));

        let mut tensors: Vec<(String, QTensor)> = vec![
            ("token_embd.weight".to_string(), f32_qtensor(VOCAB, HIDDEN)),
            ("output_norm.weight".to_string(), f32_qtensor_1d(HIDDEN)),
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
        let loader = EmbeddingLoaderBuilder::new(
            EmbeddingSpecificConfig {
                quantized_filenames: Some(vec!["model.gguf".to_string()]),
                ..Default::default()
            },
            None,
            Some(root.to_string_lossy().into_owned()),
        )
        .build(None);
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
    fn loads_qwen3_embedding_gguf_and_embeds() {
        let root =
            std::env::temp_dir().join(format!("mistralrs-emb-gguf-e2e-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        write_embedding_gguf(&root.join("model.gguf"));

        let pipeline = load_synthetic_gguf_pipeline(&root);
        let mut pipeline = pipeline.blocking_lock();

        let inputs = ModelInputs {
            input_ids: Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &Device::Cpu).unwrap(),
            flash_meta: FlashParams::empty(true),
        };
        let result = pipeline.forward_inputs(Box::new(inputs), false).unwrap();
        let ForwardInputsResult::Embeddings { embeddings } = result else {
            panic!("expected embeddings output");
        };

        // `add_eos_token` must append the EOS so last-token pooling reads it.
        let tokenizer = pipeline.tokenizer().expect("embedding pipeline has a tokenizer");
        let ids = tokenizer.encode("a b", true).unwrap().get_ids().to_vec();
        assert_eq!(ids.last(), Some(&3), "EOS must be appended: {ids:?}");

        assert_eq!(embeddings.dims(), &[1, HIDDEN]);
        let flattened = embeddings.flatten_all().unwrap();
        let values = flattened.to_vec1::<f32>().unwrap();
        assert!(
            values.iter().all(|value| value.is_finite()),
            "embedding contains non-finite values: {values:?}"
        );
        let norm: f32 = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "synthesized pooling+normalize should produce a unit vector, got {norm}"
        );

        std::fs::remove_dir_all(root).unwrap();
    }
}
