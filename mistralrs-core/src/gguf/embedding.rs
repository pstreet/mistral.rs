use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value as GgufValue;
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use serde_json::{json, Value as JsonValue};

use super::{
    gemma3_bindings::build_gemma3_text_bindings,
    gemma3_config::prepare_gemma3_text_config,
    normal_bindings::build_normal_bindings,
    normal_config::synthesize_normal_config,
    normal_registry::{resolve_native_adapter, CanonicalGgufArchitecture, GgufDescriptor},
};
use crate::{embedding_models::Pooling, pipeline::EmbeddingLoaderType, NormalLoaderType};
use std::collections::HashMap;

// llama.cpp `gguf.PoolingType` values written to `{arch}.pooling_type`.
const GGUF_POOLING_NONE: u32 = 0;
const GGUF_POOLING_MEAN: u32 = 1;
const GGUF_POOLING_LAST: u32 = 3;
pub(crate) const GGUF_POOLING_RANK: u32 = 4;

const EMBEDDING_GEMMA_ARCHITECTURE: &str = "Gemma3TextModel";

// Native binding for the reranker score head, following the sentence-transformers
// CrossEncoder `score.weight` convention. The GGUF tensor is llama.cpp's
// classifier head, extracted from `lm_head` by the official converter.
pub(crate) const RERANKER_SCORE_HEAD_BINDING: &str = "score.weight";
const GGUF_RERANKER_HEAD_TENSOR: &str = "cls.output.weight";

/// Label order and yes-position resolved from the GGUF classifier metadata.
#[derive(Debug)]
pub(crate) struct RerankHeadSpec {
    pub labels: Vec<String>,
    pub yes_idx: usize,
}

pub(crate) fn rerank_head_spec_from_gguf(archive: &GgufArchive) -> Result<RerankHeadSpec> {
    let architecture = match archive.metadata_value("general.architecture") {
        Some(GgufValue::String(value)) => value.as_str(),
        Some(value) => bail!("GGUF `general.architecture` must be a string, got {value:?}"),
        None => bail!("GGUF metadata is missing `general.architecture`"),
    };
    anyhow::ensure!(
        architecture.trim().eq_ignore_ascii_case("qwen3"),
        "GGUF architecture `{architecture}` has no reranker loader. Supported: `qwen3` \
         (Qwen3-Reranker)"
    );
    let head_spec = rerank_head_spec_from_metadata(
        archive.metadata_value("qwen3.pooling_type"),
        archive.metadata_value("qwen3.classifier.output_labels"),
    )?;
    anyhow::ensure!(
        archive.tensors().contains_key(GGUF_RERANKER_HEAD_TENSOR),
        "GGUF is missing the `{GGUF_RERANKER_HEAD_TENSOR}` tensor required for reranker \
         loading; use an artifact from the official convert_hf_to_gguf.py"
    );
    Ok(head_spec)
}

fn rerank_head_spec_from_metadata(
    pooling: Option<&GgufValue>,
    labels: Option<&GgufValue>,
) -> Result<RerankHeadSpec> {
    let pooling_key = "qwen3.pooling_type";
    match pooling {
        Some(GgufValue::U32(GGUF_POOLING_RANK)) => (),
        Some(other) => bail!(
            "GGUF `{pooling_key}` must be {GGUF_POOLING_RANK} (RANK) for reranker loading, got \
             {other:?}"
        ),
        None => bail!(
            "GGUF is missing `{pooling_key}` = {GGUF_POOLING_RANK} (RANK), so this is not a \
             reranker checkpoint. Community conversions that drop the RANK metadata and the \
             `cls.output.weight` head emit garbage scores; use an artifact from the official \
             convert_hf_to_gguf.py (e.g. ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF)"
        ),
    }
    let labels = match labels {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|item| match item {
                GgufValue::String(value) => Ok(value.as_str().to_string()),
                other => {
                    bail!("GGUF `qwen3.classifier.output_labels` entries must be strings, got {other:?}")
                }
            })
            .collect::<Result<Vec<_>>>()?,
        Some(other) => {
            bail!("GGUF `qwen3.classifier.output_labels` must be an array of strings, got {other:?}")
        }
        None => bail!("GGUF is missing `qwen3.classifier.output_labels` (expected `[\"yes\", \"no\"]`)"),
    };
    anyhow::ensure!(
        labels.len() == 2,
        "GGUF `qwen3.classifier.output_labels` must have exactly two entries, got {labels:?}"
    );
    let yes_idx = labels
        .iter()
        .position(|label| label == "yes")
        .with_context(|| {
            format!("GGUF `qwen3.classifier.output_labels` {labels:?} has no `yes` label")
        })?;
    Ok(RerankHeadSpec { labels, yes_idx })
}

pub(crate) fn build_reranker_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    let mut bindings = build_embedding_bindings(archive, EmbeddingLoaderType::Qwen3Embedding)?;
    bindings.insert(
        RERANKER_SCORE_HEAD_BINDING,
        GgufTensorBinding::tensor(GGUF_RERANKER_HEAD_TENSOR),
    );
    Ok(bindings)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolingMode {
    Mean,
    Last,
}

pub(crate) fn embedding_loader_for_architecture(
    architecture: &str,
    explicit: Option<EmbeddingLoaderType>,
) -> Result<EmbeddingLoaderType> {
    let resolved = match architecture.trim().to_ascii_lowercase().as_str() {
        "qwen3" => EmbeddingLoaderType::Qwen3Embedding,
        "gemma3" => EmbeddingLoaderType::EmbeddingGemma,
        other => bail!(
            "GGUF architecture `{other}` has no embedding model loader. Supported: `qwen3` \
             (Qwen3-Embedding) and `gemma3` (EmbeddingGemma)"
        ),
    };
    if let Some(explicit) = explicit {
        if explicit != resolved {
            bail!(
                "Explicit embedding architecture `{explicit}` does not match the GGUF \
                 architecture `{architecture}`"
            );
        }
    }
    Ok(resolved)
}

pub(crate) fn prepare_embedding_config(
    loader: EmbeddingLoaderType,
    external_config: Option<&str>,
    metadata: &HashMap<String, GgufValue>,
    tensor_names: &[String],
) -> Result<String> {
    match loader {
        EmbeddingLoaderType::Qwen3Embedding => match external_config {
            Some(config) => null_quantization_config(config),
            None => synthesize_normal_config(&NormalLoaderType::Qwen3, metadata, tensor_names)
                .map_err(anyhow::Error::msg),
        },
        EmbeddingLoaderType::EmbeddingGemma => {
            let config = prepare_gemma3_text_config(external_config, metadata, tensor_names)?;
            let mut value: JsonValue =
                serde_json::from_str(&config).context("Gemma 3 embedding config must be JSON")?;
            let object = value
                .as_object_mut()
                .context("Gemma 3 embedding config must be a JSON object")?;
            object.insert(
                "architectures".to_string(),
                json!([EMBEDDING_GEMMA_ARCHITECTURE]),
            );
            object
                .entry("use_bidirectional_attention".to_string())
                .or_insert_with(|| json!(true));
            Ok(serde_json::to_string(&value)?)
        }
    }
}

fn null_quantization_config(config: &str) -> Result<String> {
    let mut value: JsonValue =
        serde_json::from_str(config).context("External embedding model config is not JSON")?;
    let object = value
        .as_object_mut()
        .context("External embedding model config must be a JSON object")?;
    // The checkpoint is GGUF; a safetensors-style quantization_config would build the
    // wrong layer type against GGUF-backed weights.
    object.insert("quantization_config".to_string(), JsonValue::Null);
    Ok(serde_json::to_string(&value)?)
}

pub(crate) fn build_embedding_bindings(
    archive: &GgufArchive,
    loader: EmbeddingLoaderType,
) -> Result<GgufBindingMap> {
    let bindings = match loader {
        EmbeddingLoaderType::Qwen3Embedding => {
            let architecture = CanonicalGgufArchitecture::Qwen3;
            let metadata_keys = archive
                .metadata()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let tensor_names = archive
                .tensors()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let descriptor =
                GgufDescriptor::new(architecture.as_str(), &metadata_keys, &tensor_names)
                    .map_err(anyhow::Error::msg)?;
            resolve_native_adapter(&descriptor, None).map_err(anyhow::Error::msg)?;
            build_normal_bindings(archive, &NormalLoaderType::Qwen3, architecture)?
        }
        EmbeddingLoaderType::EmbeddingGemma => build_gemma3_text_bindings(archive, false)?,
    };
    // The embedding models read the checkpoint without the `model.` prefix used by
    // the causal-LM loaders the binding tables were built for.
    Ok(strip_model_prefix(bindings))
}

fn strip_model_prefix(bindings: GgufBindingMap) -> GgufBindingMap {
    bindings
        .iter()
        .filter_map(|(name, binding)| {
            name.strip_prefix("model.")
                .map(|name| (name.to_string(), binding.clone()))
        })
        .collect()
}

pub(crate) fn pooling_from_gguf_metadata(
    archive: &GgufArchive,
    loader: EmbeddingLoaderType,
    hidden_size: usize,
) -> Result<Pooling> {
    let key = format!("{}.pooling_type", architecture_name(loader));
    pooling_from_metadata_value(archive.metadata_value(&key), loader, hidden_size)
        .with_context(|| format!("resolving pooling from GGUF metadata `{key}`"))
}

fn pooling_from_metadata_value(
    value: Option<&GgufValue>,
    loader: EmbeddingLoaderType,
    hidden_size: usize,
) -> Result<Pooling> {
    let mode = match value {
        None | Some(GgufValue::U32(GGUF_POOLING_NONE)) => None,
        Some(GgufValue::U32(GGUF_POOLING_MEAN)) => Some(PoolingMode::Mean),
        Some(GgufValue::U32(GGUF_POOLING_LAST)) => Some(PoolingMode::Last),
        Some(GgufValue::U32(other)) => bail!(
            "GGUF pooling type {other} is not supported by the embedding pipeline (mean and \
             last-token only)"
        ),
        Some(other) => bail!("GGUF pooling metadata must be a u32, got {other:?}"),
    };
    let mode = match mode {
        Some(mode) => mode,
        // Per-model published usage: Qwen3-Embedding last-token, EmbeddingGemma mean.
        None => match loader {
            EmbeddingLoaderType::Qwen3Embedding => PoolingMode::Last,
            EmbeddingLoaderType::EmbeddingGemma => PoolingMode::Mean,
        },
    };
    Ok(Pooling {
        word_embedding_dimension: hidden_size,
        pooling_mode_cls_token: false,
        pooling_mode_mean_tokens: mode == PoolingMode::Mean,
        pooling_mode_max_tokens: false,
        pooling_mode_mean_sqrt_len_tokens: false,
        pooling_mode_weightedmean_tokens: false,
        pooling_mode_lasttoken: mode == PoolingMode::Last,
        include_prompt: true,
    })
}

pub(crate) fn config_hidden_size(config: &str) -> Result<usize> {
    let value: JsonValue =
        serde_json::from_str(config).context("embedding model config is not JSON")?;
    value
        .get("hidden_size")
        .and_then(JsonValue::as_u64)
        .and_then(|size| usize::try_from(size).ok())
        .context("embedding model config is missing `hidden_size`")
}

fn architecture_name(loader: EmbeddingLoaderType) -> &'static str {
    match loader {
        EmbeddingLoaderType::Qwen3Embedding => "qwen3",
        EmbeddingLoaderType::EmbeddingGemma => "gemma3",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs_quant::GgufTensorBinding;

    const GGUF_POOLING_CLS: u32 = 2;

    fn qwen3_metadata() -> HashMap<String, GgufValue> {
        HashMap::from([
            (
                "general.architecture".to_string(),
                GgufValue::String("qwen3".to_string()),
            ),
            ("qwen3.context_length".to_string(), GgufValue::U32(32768)),
            ("qwen3.embedding_length".to_string(), GgufValue::U32(16)),
            ("qwen3.block_count".to_string(), GgufValue::U32(2)),
            ("qwen3.feed_forward_length".to_string(), GgufValue::U32(32)),
            ("qwen3.attention.head_count".to_string(), GgufValue::U32(2)),
            (
                "qwen3.attention.head_count_kv".to_string(),
                GgufValue::U32(1),
            ),
            ("qwen3.attention.key_length".to_string(), GgufValue::U32(8)),
            (
                "qwen3.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::F32(1e-6),
            ),
            (
                "qwen3.rope.freq_base".to_string(),
                GgufValue::F32(1_000_000.0),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(
                    (0..4)
                        .map(|index| GgufValue::String(format!("token-{index}")))
                        .collect(),
                ),
            ),
        ])
    }

    fn qwen3_tensor_names() -> Vec<String> {
        let mut names = vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
        ];
        for layer in 0..2 {
            for role in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "attn_q_norm",
                "attn_k_norm",
                "attn_norm",
                "ffn_norm",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                names.push(format!("blk.{layer}.{role}.weight"));
            }
        }
        names
    }

    #[test]
    fn maps_gguf_architectures_to_embedding_loaders() {
        assert_eq!(
            embedding_loader_for_architecture("qwen3", None).unwrap(),
            EmbeddingLoaderType::Qwen3Embedding
        );
        assert_eq!(
            embedding_loader_for_architecture("Gemma3", None).unwrap(),
            EmbeddingLoaderType::EmbeddingGemma
        );
        assert!(embedding_loader_for_architecture("bert", None).is_err());
        assert!(embedding_loader_for_architecture(
            "qwen3",
            Some(EmbeddingLoaderType::Qwen3Embedding)
        )
        .is_ok());
        assert!(embedding_loader_for_architecture(
            "qwen3",
            Some(EmbeddingLoaderType::EmbeddingGemma)
        )
        .is_err());
    }

    #[test]
    fn synthesizes_qwen3_embedding_config_from_metadata() {
        let config = prepare_embedding_config(
            EmbeddingLoaderType::Qwen3Embedding,
            None,
            &qwen3_metadata(),
            &qwen3_tensor_names(),
        )
        .unwrap();
        let parsed: crate::embedding_models::qwen3_embedding::Config =
            serde_json::from_str(&config).unwrap();
        assert_eq!(parsed.hidden_size, 16);
        assert_eq!(parsed.num_hidden_layers, 2);
        assert_eq!(parsed.vocab_size, 4);
        assert_eq!(parsed.head_dim(), 8);
        let value: JsonValue = serde_json::from_str(&config).unwrap();
        assert!(value["quantization_config"].is_null());
        assert_eq!(value["architectures"], json!(["Qwen3ForCausalLM"]));
    }

    #[test]
    fn nulls_quantization_config_of_external_qwen3_config() {
        let external = r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 16,
            "quantization_config": {"bits": 4}
        }"#;
        let config = prepare_embedding_config(
            EmbeddingLoaderType::Qwen3Embedding,
            Some(external),
            &qwen3_metadata(),
            &qwen3_tensor_names(),
        )
        .unwrap();
        let value: JsonValue = serde_json::from_str(&config).unwrap();
        assert!(value["quantization_config"].is_null());
        assert_eq!(value["hidden_size"], 16);
    }

    #[test]
    fn pooling_metadata_maps_to_modes() {
        let hidden = 16;
        let mean = pooling_from_metadata_value(
            Some(&GgufValue::U32(GGUF_POOLING_MEAN)),
            EmbeddingLoaderType::Qwen3Embedding,
            hidden,
        )
        .unwrap();
        assert!(mean.pooling_mode_mean_tokens && !mean.pooling_mode_lasttoken);
        assert_eq!(mean.word_embedding_dimension, hidden);

        let last = pooling_from_metadata_value(
            Some(&GgufValue::U32(GGUF_POOLING_LAST)),
            EmbeddingLoaderType::EmbeddingGemma,
            hidden,
        )
        .unwrap();
        assert!(last.pooling_mode_lasttoken && !last.pooling_mode_mean_tokens);

        assert!(pooling_from_metadata_value(
            Some(&GgufValue::U32(GGUF_POOLING_CLS)),
            EmbeddingLoaderType::Qwen3Embedding,
            hidden
        )
        .is_err());
        assert!(pooling_from_metadata_value(
            Some(&GgufValue::U32(GGUF_POOLING_RANK)),
            EmbeddingLoaderType::Qwen3Embedding,
            hidden
        )
        .is_err());
    }

    #[test]
    fn missing_pooling_metadata_uses_architecture_defaults() {
        let qwen3 =
            pooling_from_metadata_value(None, EmbeddingLoaderType::Qwen3Embedding, 16).unwrap();
        assert!(qwen3.pooling_mode_lasttoken);
        let gemma = pooling_from_metadata_value(
            Some(&GgufValue::U32(GGUF_POOLING_NONE)),
            EmbeddingLoaderType::EmbeddingGemma,
            16,
        )
        .unwrap();
        assert!(gemma.pooling_mode_mean_tokens);
    }

    #[test]
    fn strips_model_prefix_from_bindings() {
        let bindings = GgufBindingMap::new()
            .with_binding(
                "model.embed_tokens.weight",
                GgufTensorBinding::tensor("token_embd.weight"),
            )
            .with_binding(
                "model.layers.0.self_attn.q_proj.weight",
                GgufTensorBinding::tensor("blk.0.attn_q.weight"),
            )
            .with_binding("lm_head.weight", GgufTensorBinding::tensor("output.weight"));
        let stripped = strip_model_prefix(bindings);
        assert!(stripped.get("embed_tokens.weight").is_some());
        assert!(stripped.get("layers.0.self_attn.q_proj.weight").is_some());
        assert!(stripped.get("lm_head.weight").is_none());
        assert!(stripped.get("model.embed_tokens.weight").is_none());
    }

    #[test]
    fn reads_hidden_size_from_config() {
        assert_eq!(
            config_hidden_size(r#"{"hidden_size": 1024}"#).unwrap(),
            1024
        );
        assert!(config_hidden_size("{}").is_err());
    }

    fn labels(value: &[&str]) -> GgufValue {
        GgufValue::Array(
            value
                .iter()
                .map(|label| GgufValue::String(label.to_string()))
                .collect(),
        )
    }

    #[test]
    fn rerank_spec_resolves_yes_index_from_labels() {
        let spec = rerank_head_spec_from_metadata(
            Some(&GgufValue::U32(GGUF_POOLING_RANK)),
            Some(&labels(&["yes", "no"])),
        )
        .unwrap();
        assert_eq!(spec.yes_idx, 0);

        let flipped = rerank_head_spec_from_metadata(
            Some(&GgufValue::U32(GGUF_POOLING_RANK)),
            Some(&labels(&["no", "yes"])),
        )
        .unwrap();
        assert_eq!(flipped.yes_idx, 1);
    }

    #[test]
    fn rerank_spec_requires_rank_pooling() {
        assert!(rerank_head_spec_from_metadata(
            Some(&GgufValue::U32(GGUF_POOLING_MEAN)),
            Some(&labels(&["yes", "no"]))
        )
        .is_err());
        let missing = rerank_head_spec_from_metadata(None, Some(&labels(&["yes", "no"])))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("RANK"), "{missing}");
        assert!(missing.contains("garbage scores"), "{missing}");
    }

    #[test]
    fn rerank_spec_requires_two_labels_with_yes() {
        assert!(
            rerank_head_spec_from_metadata(Some(&GgufValue::U32(GGUF_POOLING_RANK)), None).is_err()
        );
        assert!(rerank_head_spec_from_metadata(
            Some(&GgufValue::U32(GGUF_POOLING_RANK)),
            Some(&labels(&["yes"]))
        )
        .is_err());
        assert!(rerank_head_spec_from_metadata(
            Some(&GgufValue::U32(GGUF_POOLING_RANK)),
            Some(&labels(&["true", "false"]))
        )
        .is_err());
    }
}
