use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Value as GgufValue;
use serde_json::{json, Value as JsonValue};

use crate::{MultimodalLoaderType, NormalLoaderType};

use super::normal_config::synthesize_normal_config_value;

const IMAGE_TOKEN: &str = "<|image_pad|>";
const VIDEO_TOKEN: &str = "<|video_pad|>";
const VISION_START_TOKEN: &str = "<|vision_start|>";
const VISION_END_TOKEN: &str = "<|vision_end|>";
const DEEPSTACK_LAYERS_KEY: &str = "clip.vision.is_deepstack_layers";
const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const DEFAULT_SPATIAL_MERGE_SIZE: usize = 2;
const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;
// HF Qwen3-VL processor pixel budget; preprocessing policy, not model identity.
// The GGUF carries no equivalent keys, so standalone files use the family default.
const DEFAULT_SHORTEST_EDGE_PIXELS: usize = 256 * 256;
const DEFAULT_LONGEST_EDGE_PIXELS: usize = 4096 * 4096;
pub(crate) fn synthesize_multimodal_config(
    loader: &MultimodalLoaderType,
    architecture: &str,
    metadata: &HashMap<String, GgufValue>,
    tensor_names: &[String],
) -> Result<String> {
    let value = match loader {
        MultimodalLoaderType::Qwen3_5 => synthesize_qwen35(metadata, tensor_names)?,
        _ => anyhow::bail!(
            "multimodal GGUF architecture `{architecture}` requires its original `config.json`; pass `--tok-model-id <original-model-id>`"
        ),
    };
    serde_json::to_string(&value).context("Failed to serialize synthesized multimodal config")
}

// Whether `synthesize_multimodal_config` can cover this file pair, so asset
// inference can stand down instead of demanding `--tok-model-id`.
pub(crate) fn supports_standalone_config(architecture: &str, projector: Option<&str>) -> bool {
    matches!(
        (architecture, projector),
        ("qwen35", Some("qwen3vl_merger"))
    )
}

pub(crate) fn supports_standalone_assets(
    model_metadata: &HashMap<String, GgufValue>,
    projector_metadata: &[&HashMap<String, GgufValue>],
) -> bool {
    let architecture = model_metadata
        .get("general.architecture")
        .and_then(|value| match value {
            GgufValue::String(architecture) => Some(architecture.as_str()),
            _ => None,
        });
    let projector = projector_metadata.iter().find_map(|metadata| {
        ["clip.projector_type", "clip.vision.projector_type"]
            .iter()
            .find_map(|key| match metadata.get(*key) {
                Some(GgufValue::String(projector)) => Some(projector.as_str()),
                _ => None,
            })
    });
    matches!(
        architecture,
        Some(architecture) if supports_standalone_config(architecture, projector)
    )
}

fn synthesize_qwen35(
    metadata: &HashMap<String, GgufValue>,
    tensor_names: &[String],
) -> Result<JsonValue> {
    let mut text_config =
        synthesize_normal_config_value(&NormalLoaderType::Qwen3_5, metadata, tensor_names)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
    let text_object = text_config
        .as_object_mut()
        .context("Synthesized Qwen3.5 text config must be a JSON object")?;
    // Built-in MTP head depth has no HF-config-independent default; the file states it.
    let mtp_layers = optional_usize(metadata, "qwen35.nextn_predict_layers")?.unwrap_or(0);
    text_object.insert("mtp_num_hidden_layers".to_string(), json!(mtp_layers));
    let vision_config = qwen35_vision_config(metadata)?;
    Ok(json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": text_config,
        "vision_config": vision_config,
        "image_token_id": token_id(metadata, IMAGE_TOKEN)?,
        "video_token_id": token_id(metadata, VIDEO_TOKEN)?,
        "vision_start_token_id": token_id(metadata, VISION_START_TOKEN)?,
        "vision_end_token_id": token_id(metadata, VISION_END_TOKEN)?,
        "tie_word_embeddings": !tensor_names.iter().any(|name| name == "output.weight"),
    }))
}

pub(crate) fn synthesize_qwen35_preprocessor(
    metadata: &HashMap<String, GgufValue>,
) -> Result<String> {
    let f32_array = |key: &str| -> Result<[f64; 3]> {
        let Some(GgufValue::Array(values)) = metadata.get(key) else {
            anyhow::bail!("Standalone multimodal config requires GGUF `{key}`");
        };
        let mut out = [0.5; 3];
        for (slot, value) in out.iter_mut().zip(values.iter()) {
            let GgufValue::F32(raw) = value else {
                anyhow::bail!("GGUF `{key}` must be an f32 array");
            };
            *slot = f64::from(*raw);
        }
        Ok(out)
    };
    let merge = optional_usize(metadata, "clip.vision.spatial_merge_size")?
        .unwrap_or(DEFAULT_SPATIAL_MERGE_SIZE);
    let preprocessor = json!({
        "patch_size": required_usize(metadata, "clip.vision.patch_size")?,
        "temporal_patch_size": DEFAULT_TEMPORAL_PATCH_SIZE,
        "merge_size": merge,
        "image_mean": f32_array("clip.vision.image_mean")?,
        "image_std": f32_array("clip.vision.image_std")?,
        "size": {
            "shortest_edge": DEFAULT_SHORTEST_EDGE_PIXELS,
            "longest_edge": DEFAULT_LONGEST_EDGE_PIXELS,
        },
    });
    serde_json::to_string(&preprocessor)
        .context("Failed to serialize synthesized preprocessor config")
}

fn qwen35_vision_config(metadata: &HashMap<String, GgufValue>) -> Result<JsonValue> {
    let u = |key: &str| required_usize(metadata, key);
    let image_size = u("clip.vision.image_size")?;
    let patch_size = u("clip.vision.patch_size")?;
    if patch_size == 0 || !image_size.is_multiple_of(patch_size) {
        anyhow::bail!("GGUF `clip.vision.image_size` ({image_size}) must be a multiple of `clip.vision.patch_size` ({patch_size})");
    }
    let grid = image_size / patch_size;
    let mut vision = json!({
        "depth": u("clip.vision.block_count")?,
        "hidden_size": u("clip.vision.embedding_length")?,
        "out_hidden_size": u("clip.vision.projection_dim")?,
        "intermediate_size": u("clip.vision.feed_forward_length")?,
        "num_heads": u("clip.vision.attention.head_count")?,
        "patch_size": patch_size,
        "num_position_embeddings": grid * grid,
        "deepstack_visual_indexes": deepstack_indices(metadata)?,
    });
    if let Some(merge) = optional_usize(metadata, "clip.vision.spatial_merge_size")? {
        vision["spatial_merge_size"] = json!(merge);
    } else {
        vision["spatial_merge_size"] = json!(DEFAULT_SPATIAL_MERGE_SIZE);
    }
    Ok(vision)
}

fn deepstack_indices(metadata: &HashMap<String, GgufValue>) -> Result<Vec<usize>> {
    let Some(value) = metadata.get(DEEPSTACK_LAYERS_KEY) else {
        return Ok(Vec::new());
    };
    let GgufValue::Array(flags) = value else {
        anyhow::bail!("GGUF `{DEEPSTACK_LAYERS_KEY}` must be a boolean array");
    };
    flags
        .iter()
        .enumerate()
        .map(|(index, flag)| match flag {
            GgufValue::Bool(true) => Ok(Some(index)),
            GgufValue::Bool(false) => Ok(None),
            _ => Err(anyhow::anyhow!(
                "GGUF `{DEEPSTACK_LAYERS_KEY}` must be a boolean array"
            )),
        })
        .collect::<Result<Vec<_>>>()
        .map(|flags| flags.into_iter().flatten().collect())
}

fn token_id(metadata: &HashMap<String, GgufValue>, token: &str) -> Result<u32> {
    let Some(GgufValue::Array(tokens)) = metadata.get(TOKENIZER_TOKENS_KEY) else {
        anyhow::bail!(
            "Standalone multimodal config requires `{TOKENIZER_TOKENS_KEY}` to resolve `{token}`"
        );
    };
    tokens
        .iter()
        .position(|entry| matches!(entry, GgufValue::String(text) if text == token))
        .and_then(|index| u32::try_from(index).ok())
        .with_context(|| format!("GGUF tokenizer has no `{token}` entry"))
}

fn required_usize(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<usize> {
    let value = metadata
        .get(key)
        .with_context(|| format!("Standalone multimodal config requires GGUF `{key}`"))?;
    value_usize(value).with_context(|| format!("GGUF `{key}` must be an integer"))
}

fn optional_usize(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<Option<usize>> {
    metadata
        .get(key)
        .map(|value| value_usize(value).with_context(|| format!("GGUF `{key}` must be an integer")))
        .transpose()
}

fn value_usize(value: &GgufValue) -> Option<usize> {
    let raw = match value {
        GgufValue::U8(value) => u64::from(*value),
        GgufValue::U16(value) => u64::from(*value),
        GgufValue::U32(value) => u64::from(*value),
        GgufValue::U64(value) => *value,
        GgufValue::I8(value) => u64::try_from(*value).ok()?,
        GgufValue::I16(value) => u64::try_from(*value).ok()?,
        GgufValue::I32(value) => u64::try_from(*value).ok()?,
        GgufValue::I64(value) => u64::try_from(*value).ok()?,
        _ => return None,
    };
    usize::try_from(raw).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: Vec<(&str, GgufValue)>) -> HashMap<String, GgufValue> {
        pairs
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect()
    }

    fn vision_meta() -> HashMap<String, GgufValue> {
        meta(vec![
            ("clip.vision.image_size", GgufValue::U32(768)),
            ("clip.vision.patch_size", GgufValue::U32(16)),
            ("clip.vision.block_count", GgufValue::U32(27)),
            ("clip.vision.embedding_length", GgufValue::U32(1152)),
            ("clip.vision.projection_dim", GgufValue::U32(5120)),
            ("clip.vision.feed_forward_length", GgufValue::U32(4304)),
            ("clip.vision.attention.head_count", GgufValue::U32(16)),
            ("clip.vision.spatial_merge_size", GgufValue::U32(2)),
        ])
    }

    #[test]
    fn qwen35_vision_config_derives_grid_positions() {
        let vision = qwen35_vision_config(&vision_meta()).unwrap();
        assert_eq!(vision["depth"], json!(27));
        assert_eq!(vision["hidden_size"], json!(1152));
        assert_eq!(vision["out_hidden_size"], json!(5120));
        assert_eq!(vision["intermediate_size"], json!(4304));
        assert_eq!(vision["num_heads"], json!(16));
        assert_eq!(vision["num_position_embeddings"], json!(48 * 48));
        assert_eq!(vision["deepstack_visual_indexes"], json!([]));
        assert_eq!(vision["spatial_merge_size"], json!(2));
    }

    #[test]
    fn qwen35_vision_config_rejects_ragged_grid() {
        let mut metadata = vision_meta();
        metadata.insert("clip.vision.image_size".to_string(), GgufValue::U32(770));
        assert!(qwen35_vision_config(&metadata).is_err());
    }

    #[test]
    fn deepstack_indices_collects_true_positions() {
        let metadata = meta(vec![(
            DEEPSTACK_LAYERS_KEY,
            GgufValue::Array(vec![
                GgufValue::Bool(false),
                GgufValue::Bool(true),
                GgufValue::Bool(false),
                GgufValue::Bool(true),
            ]),
        )]);
        assert_eq!(deepstack_indices(&metadata).unwrap(), vec![1, 3]);
        assert!(deepstack_indices(&HashMap::new()).unwrap().is_empty());
    }

    #[test]
    fn token_id_resolves_vocab_position() {
        let metadata = meta(vec![(
            TOKENIZER_TOKENS_KEY,
            GgufValue::Array(vec![
                GgufValue::String("a".to_string()),
                GgufValue::String(VISION_START_TOKEN.to_string()),
                GgufValue::String("b".to_string()),
            ]),
        )]);
        assert_eq!(token_id(&metadata, VISION_START_TOKEN).unwrap(), 1);
        assert!(token_id(&metadata, IMAGE_TOKEN).is_err());
    }
}
