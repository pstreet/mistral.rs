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
const GEMMA4_PARTIAL_ROTARY_FACTOR: f64 = 0.25;
pub(crate) fn synthesize_multimodal_config(
    loader: &MultimodalLoaderType,
    architecture: &str,
    metadata: &HashMap<String, GgufValue>,
    tensor_shapes: &HashMap<String, Vec<usize>>,
) -> Result<String> {
    let tensor_names = tensor_shapes.keys().cloned().collect::<Vec<_>>();
    let value = match loader {
        MultimodalLoaderType::Qwen3_5 => synthesize_qwen35(metadata, &tensor_names)?,
        MultimodalLoaderType::Gemma4 => synthesize_gemma4(metadata, tensor_shapes)?,
        MultimodalLoaderType::MuseGlimmer => synthesize_muse_glimmer(metadata, tensor_shapes)?,
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
            | ("gemma4", Some("gemma4v"))
            | ("muse-glimmer", Some("muse-glimmer"))
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

pub(crate) fn synthesize_gemma4_preprocessor(
    metadata: &HashMap<String, GgufValue>,
) -> Result<String> {
    let _ = metadata;
    let preprocessor = json!({
        "do_rescale": true,
        "rescale_factor": 1.0f64 / 255.0,
        "do_convert_rgb": true,
    });
    serde_json::to_string(&preprocessor)
        .context("Failed to serialize synthesized preprocessor config")
}

pub(crate) fn synthesize_muse_glimmer_preprocessor(
    metadata: &HashMap<String, GgufValue>,
) -> Result<String> {
    let merge = optional_usize(metadata, "clip.vision.spatial_merge_size")?
        .unwrap_or(DEFAULT_SPATIAL_MERGE_SIZE);
    let preprocessor = json!({
        "patch_size": required_usize(metadata, "clip.vision.patch_size")?,
        "temporal_patch_size": DEFAULT_TEMPORAL_PATCH_SIZE,
        "merge_size": merge,
        "image_mean": f32_3(metadata, "clip.vision.image_mean")?,
        "image_std": f32_3(metadata, "clip.vision.image_std")?,
    });
    serde_json::to_string(&preprocessor)
        .context("Failed to serialize synthesized preprocessor config")
}

fn synthesize_gemma4(
    metadata: &HashMap<String, GgufValue>,
    tensor_shapes: &HashMap<String, Vec<usize>>,
) -> Result<JsonValue> {
    let u = |key: &str| required_usize(metadata, key);
    let f = |key: &str| required_f64(metadata, key);
    let layers = u("gemma4.block_count")?;
    let layer_types = gemma4_layer_types(metadata, layers)?;
    let k_eq_v = gemma4_attention_k_eq_v(tensor_shapes, &layer_types)?;
    let (bos_id, eos_ids) = gen_ids(metadata)?;
    let (kv_sliding, kv_full) = gemma4_kv_heads(metadata, &layer_types)?;
    let key_length = u("gemma4.attention.key_length")?;
    let key_length_swa = u("gemma4.attention.key_length_swa")?;
    let freq_base = f("gemma4.rope.freq_base")?;
    let freq_base_swa = f("gemma4.rope.freq_base_swa")?;
    let mut text = json!({
        "hidden_size": u("gemma4.embedding_length")?,
        "intermediate_size": u("gemma4.feed_forward_length")?,
        "num_hidden_layers": layers,
        "num_attention_heads": u("gemma4.attention.head_count")?,
        "sliding_window": u("gemma4.attention.sliding_window")?,
        "final_logit_softcapping": f("gemma4.final_logit_softcapping")?,
        "quantization_config": JsonValue::Null,
        "layer_types": layer_types,
        "num_global_key_value_heads": kv_full.map(JsonValue::from).unwrap_or(JsonValue::Null),
        "num_experts": u("gemma4.expert_count")?,
        "top_k_experts": u("gemma4.expert_used_count")?,
        "enable_moe_block": true,
        "expert_intermediate_size": u("gemma4.expert_feed_forward_length")?,
        "hidden_size_per_layer_input": JsonValue::Null,
        "vocab_size_per_layer_input": JsonValue::Null,
        "rope_parameters": {
            // HF Gemma 4 rotates a quarter of each head; the GGUF head lengths only size the tables.
            "full_attention": {
                "rope_theta": freq_base,
                "rope_type": "proportional",
                "partial_rotary_factor": GEMMA4_PARTIAL_ROTARY_FACTOR,
            },
            "sliding_attention": {
                "rope_theta": freq_base_swa,
                "rope_type": "default",
                "partial_rotary_factor": GEMMA4_PARTIAL_ROTARY_FACTOR,
            },
        },
        "rope_theta": freq_base,
        // Gemma 4 text always runs under the vision attention policy; no GGUF key carries this.
        "use_bidirectional_attention": "vision",
        "max_position_embeddings": u("gemma4.context_length")?,
        "vocab_size": tokenizer_len(metadata)?,
        "rms_norm_eps": f("gemma4.attention.layer_norm_rms_epsilon")?,
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_bias": text_attention_bias(tensor_shapes, layers)?,
        "attention_k_eq_v": k_eq_v,
        // HF Gemma 4 ties the head even though quantizers emit a duplicate `output.weight`.
        "tie_word_embeddings": true,
        "bos_token_id": bos_id,
        "eos_token_id": eos_ids,
        "head_dim": key_length_swa,
        "global_head_dim": key_length,
    });
    if let Some(kv) = kv_sliding {
        text["num_key_value_heads"] = json!(kv);
    }
    let num_vision_heads = u("clip.vision.attention.head_count")?;
    let vision_hidden = u("clip.vision.embedding_length")?;
    let mut vision = json!({
        "hidden_size": vision_hidden,
        "intermediate_size": u("clip.vision.feed_forward_length")?,
        "num_hidden_layers": u("clip.vision.block_count")?,
        "num_attention_heads": num_vision_heads,
        "patch_size": u("clip.vision.patch_size")?,
        "hidden_activation": "gelu_pytorch_tanh",
        "standardize": true,
    });
    if let Some((head_dim, kv_heads)) =
        gemma4_vision_attn_dims(tensor_shapes, num_vision_heads, vision_hidden)?
    {
        vision["head_dim"] = json!(head_dim);
        vision["num_key_value_heads"] = json!(kv_heads);
    }
    if let Some(shape) = tensor_shapes.get("v.position_embd.weight") {
        anyhow::ensure!(
            shape.len() == 3,
            "GGUF `v.position_embd.weight` has unexpected shape {shape:?}"
        );
        vision["position_embedding_size"] = json!(shape[1]);
    }
    Ok(json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": text,
        "vision_config": vision,
        "image_token_id": token_id(metadata, "<|image|>")?,
        "audio_token_id": token_id(metadata, "<|audio|>")?,
        "video_token_id": token_id(metadata, "<|video|>")?,
        "boi_token_id": token_id(metadata, "<|image>")?,
        "eoi_token_id": token_id(metadata, "<image|>")?,
        "boa_token_id": token_id(metadata, "<|audio>")?,
        "eoa_token_id": token_id(metadata, "<audio|>")?,
    }))
}

fn synthesize_muse_glimmer(
    metadata: &HashMap<String, GgufValue>,
    tensor_shapes: &HashMap<String, Vec<usize>>,
) -> Result<JsonValue> {
    let u = |key: &str| required_usize(metadata, key);
    let f = |key: &str| required_f64(metadata, key);
    let layers = u("muse-glimmer.block_count")?;
    let key_length = u("muse-glimmer.attention.key_length")?;
    let value_length = u("muse-glimmer.attention.value_length")?;
    anyhow::ensure!(
        key_length == value_length,
        "muse-glimmer key/value head dims differ ({key_length} vs {value_length})"
    );
    let vision_hidden = u("clip.vision.embedding_length")?;
    let merge = optional_usize(metadata, "clip.vision.spatial_merge_size")?
        .unwrap_or(DEFAULT_SPATIAL_MERGE_SIZE);
    let (bos_id, eos_ids) = gen_ids(metadata)?;
    let mut config = json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "text_config": {
            "vocab_size": tokenizer_len(metadata)?,
            "hidden_size": u("muse-glimmer.embedding_length")?,
            "intermediate_size": u("muse-glimmer.feed_forward_length")?,
            "num_hidden_layers": layers,
            "num_attention_heads": u("muse-glimmer.attention.head_count")?,
            "num_key_value_heads": u("muse-glimmer.attention.head_count_kv")?,
            "head_dim": key_length,
            "max_position_embeddings": u("muse-glimmer.context_length")?,
            "rms_norm_eps": f("muse-glimmer.attention.layer_norm_rms_epsilon")?,
            "sliding_window": u("muse-glimmer.attention.sliding_window")?,
            "rope_parameters": {"rope_theta": f("muse-glimmer.rope.freq_base")?},
            // No GGUF key carries the QK scale; the serde default matches the
            // reference config. `logit_scale` is the output multiplier.
            "output_multiplier": f("muse-glimmer.logit_scale")?,
            "final_logit_softcapping": f("muse-glimmer.final_logit_softcapping")?,
            "attention_bias": text_attention_bias(tensor_shapes, layers)?,
            // The GGUF bindings require `output.weight`, so the head is never tied.
            "tie_word_embeddings": false,
            "bos_token_id": bos_id,
            "eos_token_id": eos_ids,
        },
        "vision_config": {
            "hidden_size": vision_hidden,
            "intermediate_size": u("clip.vision.feed_forward_length")?,
            "num_attention_heads": u("clip.vision.attention.head_count")?,
            "num_hidden_layers": u("clip.vision.block_count")?,
            "patch_size": u("clip.vision.patch_size")?,
            "merge_size": merge,
        },
        "image_token_id": token_id(metadata, "<|patch|>")?,
        "video_token_id": token_id(metadata, "<|video|>")?,
    });
    if let Some(hidden) = glimmer_projector_hidden(tensor_shapes, merge, vision_hidden)? {
        config["projector_hidden_size"] = json!(hidden);
    }
    Ok(config)
}

fn gemma4_layer_types(metadata: &HashMap<String, GgufValue>, layers: usize) -> Result<Vec<String>> {
    let pattern = required_bool_array(metadata, "gemma4.attention.sliding_window_pattern")?;
    anyhow::ensure!(
        pattern.len() == layers,
        "GGUF `gemma4.attention.sliding_window_pattern` has {} entries for {layers} layers",
        pattern.len()
    );
    Ok(pattern
        .iter()
        .map(|sliding| {
            if *sliding {
                "sliding_attention".to_string()
            } else {
                "full_attention".to_string()
            }
        })
        .collect())
}

fn gemma4_kv_heads(
    metadata: &HashMap<String, GgufValue>,
    layer_types: &[String],
) -> Result<(Option<usize>, Option<usize>)> {
    let Some(value) = metadata.get("gemma4.attention.head_count_kv") else {
        return Ok((None, None));
    };
    let counts = int_array_values(value, "gemma4.attention.head_count_kv")?;
    anyhow::ensure!(
        counts.len() == layer_types.len(),
        "GGUF `gemma4.attention.head_count_kv` has {} entries for {} layers",
        counts.len(),
        layer_types.len()
    );
    let mut sliding = None;
    let mut full = None;
    for (index, layer) in layer_types.iter().enumerate() {
        let slot = if layer == "sliding_attention" {
            &mut sliding
        } else {
            &mut full
        };
        let count = usize::try_from(counts[index]).map_err(|_| {
            anyhow::anyhow!("GGUF `gemma4.attention.head_count_kv[{index}]` does not fit usize")
        })?;
        match slot {
            None => *slot = Some(count),
            Some(seen) if *seen == count => {}
            Some(seen) => anyhow::bail!(
                "GGUF `gemma4.attention.head_count_kv` is not uniform within one attention class ({seen} vs {count}); provide the original Hugging Face config"
            ),
        }
    }
    Ok((sliding, full))
}

fn gemma4_vision_attn_dims(
    tensor_shapes: &HashMap<String, Vec<usize>>,
    num_heads: usize,
    hidden: usize,
) -> Result<Option<(usize, usize)>> {
    let mut layers = tensor_shapes
        .keys()
        .filter_map(|name| {
            name.strip_prefix("v.blk.")?
                .strip_suffix(".attn_q.weight")?
                .parse::<usize>()
                .ok()
        })
        .collect::<Vec<_>>();
    layers.sort_unstable();
    let Some(first) = layers.first() else {
        return Ok(None);
    };
    let q = tensor_shapes.get(&format!("v.blk.{first}.attn_q.weight"));
    let k = tensor_shapes.get(&format!("v.blk.{first}.attn_k.weight"));
    let (Some(q), Some(k)) = (q, k) else {
        return Ok(None);
    };
    let q_out = proj_out_dim(q, hidden).with_context(|| {
        format!("GGUF `v.blk.{first}.attn_q.weight` has unexpected shape {q:?}")
    })?;
    let k_out = proj_out_dim(k, hidden).with_context(|| {
        format!("GGUF `v.blk.{first}.attn_k.weight` has unexpected shape {k:?}")
    })?;
    anyhow::ensure!(
        num_heads != 0 && q_out.is_multiple_of(num_heads),
        "vision q dim {q_out} is not divisible into {num_heads} heads"
    );
    let head_dim = q_out / num_heads;
    anyhow::ensure!(
        k_out.is_multiple_of(head_dim),
        "vision k dim {k_out} is not a multiple of head dim {head_dim}"
    );
    Ok(Some((head_dim, k_out / head_dim)))
}

fn proj_out_dim(shape: &[usize], in_features: usize) -> Option<usize> {
    let [first, second] = shape else {
        return None;
    };
    if *first == in_features {
        Some(*second)
    } else if *second == in_features {
        Some(*first)
    } else {
        None
    }
}

fn glimmer_projector_hidden(
    tensor_shapes: &HashMap<String, Vec<usize>>,
    merge: usize,
    vision_hidden: usize,
) -> Result<Option<usize>> {
    let Some(shape) = tensor_shapes.get("mm.0.weight") else {
        return Ok(None);
    };
    anyhow::ensure!(
        shape.len() == 2,
        "GGUF `mm.0.weight` has unexpected shape {shape:?}"
    );
    let expected_in = vision_hidden * merge * merge;
    if shape[0] == expected_in {
        Ok(Some(shape[1]))
    } else if shape[1] == expected_in {
        Ok(Some(shape[0]))
    } else {
        anyhow::bail!(
            "GGUF `mm.0.weight` shape {shape:?} matches neither weight layout for {expected_in} input features"
        )
    }
}

fn gemma4_attention_k_eq_v(
    tensor_shapes: &HashMap<String, Vec<usize>>,
    layer_types: &[String],
) -> Result<bool> {
    let mut k_eq_v = false;
    for (index, layer) in layer_types.iter().enumerate() {
        let has_v = tensor_shapes.contains_key(&format!("blk.{index}.attn_v.weight"));
        if layer == "sliding_attention" {
            anyhow::ensure!(
                has_v,
                "GGUF sliding layer blk.{index} has no `attn_v.weight`; provide the original Hugging Face config"
            );
        } else if !has_v {
            k_eq_v = true;
        }
    }
    Ok(k_eq_v)
}

fn text_attention_bias(tensor_shapes: &HashMap<String, Vec<usize>>, layers: usize) -> Result<bool> {
    let mut present = 0;
    for layer in 0..layers {
        for projection in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            if tensor_shapes.contains_key(&format!("blk.{layer}.{projection}.bias")) {
                present += 1;
            }
        }
    }
    anyhow::ensure!(
        present == 0 || present == layers * 4,
        "GGUF has an incomplete attention bias tensor set ({present} of {})",
        layers * 4
    );
    Ok(present != 0)
}

// Stop ids for the synthesized config: the GGUF-declared EOS plus the `<eos>`
// vocab entry when it is a different token (quantizers sometimes declare a
// template delimiter as EOS while the model emits `<eos>`).
fn gen_ids(metadata: &HashMap<String, GgufValue>) -> Result<(u32, JsonValue)> {
    let as_u32 = |key: &str| {
        let id = value_usize(
            metadata
                .get(key)
                .with_context(|| format!("Standalone multimodal config requires GGUF `{key}`"))?,
        )
        .with_context(|| format!("GGUF `{key}` must be an integer"))?;
        u32::try_from(id).map_err(|_| anyhow::anyhow!("GGUF `{key}` does not fit u32"))
    };
    let bos = as_u32("tokenizer.ggml.bos_token_id")?;
    let eos = as_u32("tokenizer.ggml.eos_token_id")?;
    let mut ids = vec![eos];
    if let Some(GgufValue::Array(tokens)) = metadata.get(TOKENIZER_TOKENS_KEY) {
        let named = tokens
            .iter()
            .position(|entry| matches!(entry, GgufValue::String(text) if text == "<eos>"));
        if let Some(position) = named.and_then(|index| u32::try_from(index).ok()) {
            if position != eos {
                ids.push(position);
            }
        }
    }
    let eos_json = if ids.len() == 1 {
        json!(ids[0])
    } else {
        json!(ids)
    };
    Ok((bos, eos_json))
}

fn tokenizer_len(metadata: &HashMap<String, GgufValue>) -> Result<usize> {
    match metadata.get(TOKENIZER_TOKENS_KEY) {
        Some(GgufValue::Array(tokens)) if !tokens.is_empty() => Ok(tokens.len()),
        Some(_) => anyhow::bail!("GGUF metadata `{TOKENIZER_TOKENS_KEY}` must be a nonempty array"),
        None => anyhow::bail!("GGUF metadata is missing `{TOKENIZER_TOKENS_KEY}`"),
    }
}

fn required_f64(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<f64> {
    let value = metadata
        .get(key)
        .with_context(|| format!("Standalone multimodal config requires GGUF `{key}`"))?;
    match value {
        GgufValue::F32(raw) => Ok(f64::from(*raw)),
        GgufValue::F64(raw) => Ok(*raw),
        _ => anyhow::bail!("GGUF `{key}` must be a float"),
    }
}

fn f32_3(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<[f64; 3]> {
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
}

fn required_bool_array(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<Vec<bool>> {
    let Some(GgufValue::Array(values)) = metadata.get(key) else {
        anyhow::bail!("Standalone multimodal config requires GGUF `{key}`");
    };
    values
        .iter()
        .map(|value| match value {
            GgufValue::Bool(flag) => Ok(*flag),
            _ => anyhow::bail!("GGUF `{key}` must be a boolean array"),
        })
        .collect()
}

fn int_array_values(value: &GgufValue, key: &str) -> Result<Vec<u64>> {
    let GgufValue::Array(values) = value else {
        anyhow::bail!("GGUF `{key}` must be an integer array");
    };
    values
        .iter()
        .map(|value| match value {
            GgufValue::U8(raw) => Ok(u64::from(*raw)),
            GgufValue::I8(raw) => u64::try_from(*raw)
                .map_err(|_| anyhow::anyhow!("GGUF `{key}` must hold nonnegative integers")),
            GgufValue::U16(raw) => Ok(u64::from(*raw)),
            GgufValue::I16(raw) => u64::try_from(*raw)
                .map_err(|_| anyhow::anyhow!("GGUF `{key}` must hold nonnegative integers")),
            GgufValue::U32(raw) => Ok(u64::from(*raw)),
            GgufValue::I32(raw) => u64::try_from(*raw)
                .map_err(|_| anyhow::anyhow!("GGUF `{key}` must hold nonnegative integers")),
            GgufValue::U64(raw) => Ok(*raw),
            GgufValue::I64(raw) => u64::try_from(*raw)
                .map_err(|_| anyhow::anyhow!("GGUF `{key}` must hold nonnegative integers")),
            _ => anyhow::bail!("GGUF `{key}` must hold nonnegative integers"),
        })
        .collect()
}
pub(crate) fn synthesize_qwen35_preprocessor(
    metadata: &HashMap<String, GgufValue>,
) -> Result<String> {
    let merge = optional_usize(metadata, "clip.vision.spatial_merge_size")?
        .unwrap_or(DEFAULT_SPATIAL_MERGE_SIZE);
    let preprocessor = json!({
        "patch_size": required_usize(metadata, "clip.vision.patch_size")?,
        "temporal_patch_size": DEFAULT_TEMPORAL_PATCH_SIZE,
        "merge_size": merge,
        "image_mean": f32_3(metadata, "clip.vision.image_mean")?,
        "image_std": f32_3(metadata, "clip.vision.image_std")?,
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

    #[test]
    fn standalone_cover_matches_new_pairs() {
        assert!(supports_standalone_config("qwen35", Some("qwen3vl_merger")));
        assert!(supports_standalone_config("gemma4", Some("gemma4v")));
        assert!(supports_standalone_config(
            "muse-glimmer",
            Some("muse-glimmer")
        ));
        assert!(!supports_standalone_config("gemma4", Some("gemma4uv")));
        assert!(!supports_standalone_config("gemma4", None));
        assert!(!supports_standalone_config("llama4", Some("llama4")));
    }

    fn gemma4_test_meta() -> HashMap<String, GgufValue> {
        let tokens = vec![
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "<|image|>",
            "<|audio|>",
            "<|video|>",
            "<|image>",
            "<image|>",
            "<|audio>",
            "<audio|>",
        ];
        meta(vec![
            ("gemma4.block_count", GgufValue::U32(4)),
            ("gemma4.embedding_length", GgufValue::U32(32)),
            ("gemma4.feed_forward_length", GgufValue::U32(64)),
            ("gemma4.attention.head_count", GgufValue::U32(4)),
            (
                "gemma4.attention.head_count_kv",
                GgufValue::Array(vec![
                    GgufValue::I32(2),
                    GgufValue::I32(2),
                    GgufValue::I32(1),
                    GgufValue::I32(1),
                ]),
            ),
            ("gemma4.attention.key_length", GgufValue::U32(8)),
            ("gemma4.attention.value_length", GgufValue::U32(8)),
            ("gemma4.attention.key_length_swa", GgufValue::U32(4)),
            ("gemma4.attention.value_length_swa", GgufValue::U32(4)),
            ("gemma4.attention.sliding_window", GgufValue::U32(16)),
            (
                "gemma4.attention.sliding_window_pattern",
                GgufValue::Array(vec![
                    GgufValue::Bool(true),
                    GgufValue::Bool(true),
                    GgufValue::Bool(false),
                    GgufValue::Bool(false),
                ]),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("gemma4.rope.dimension_count", GgufValue::U32(8)),
            ("gemma4.rope.dimension_count_swa", GgufValue::U32(4)),
            ("gemma4.rope.freq_base", GgufValue::F32(1000000.0)),
            ("gemma4.rope.freq_base_swa", GgufValue::F32(10000.0)),
            ("gemma4.context_length", GgufValue::U32(128)),
            ("gemma4.expert_count", GgufValue::U32(8)),
            ("gemma4.expert_feed_forward_length", GgufValue::U32(16)),
            ("gemma4.expert_used_count", GgufValue::U32(2)),
            ("gemma4.final_logit_softcapping", GgufValue::F32(50.0)),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(2)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(106)),
            ("clip.vision.block_count", GgufValue::U32(2)),
            ("clip.vision.embedding_length", GgufValue::U32(16)),
            ("clip.vision.feed_forward_length", GgufValue::U32(32)),
            ("clip.vision.attention.head_count", GgufValue::U32(4)),
            ("clip.vision.patch_size", GgufValue::U32(8)),
            (
                TOKENIZER_TOKENS_KEY,
                GgufValue::Array(
                    tokens
                        .into_iter()
                        .map(|token| GgufValue::String(token.to_string()))
                        .collect(),
                ),
            ),
        ])
    }

    #[test]
    fn gemma4_synthesis_partitions_kv_and_maps_layers() {
        let metadata = gemma4_test_meta();
        let shapes = HashMap::from([
            ("output.weight".to_string(), vec![11, 32]),
            ("v.position_embd.weight".to_string(), vec![2, 64, 16]),
            ("v.blk.0.attn_q.weight".to_string(), vec![16, 16]),
            ("v.blk.0.attn_k.weight".to_string(), vec![8, 16]),
            ("blk.0.attn_v.weight".to_string(), vec![8, 32]),
            ("blk.1.attn_v.weight".to_string(), vec![8, 32]),
        ]);
        let config = synthesize_gemma4(&metadata, &shapes).unwrap();
        assert_eq!(
            config["text_config"]["layer_types"],
            json!([
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "full_attention"
            ])
        );
        assert_eq!(config["text_config"]["num_key_value_heads"], json!(2));
        assert_eq!(
            config["text_config"]["num_global_key_value_heads"],
            json!(1)
        );
        assert_eq!(config["text_config"]["num_experts"], json!(8));
        assert_eq!(config["text_config"]["enable_moe_block"], json!(true));
        assert_eq!(config["text_config"]["attention_k_eq_v"], json!(true));
        assert_eq!(config["text_config"]["tie_word_embeddings"], json!(true));
        assert_eq!(config["vision_config"]["head_dim"], json!(4));
        assert_eq!(config["vision_config"]["num_key_value_heads"], json!(2));
        assert_eq!(
            config["vision_config"]["position_embedding_size"],
            json!(64)
        );
        assert_eq!(config["image_token_id"], json!(4));
        assert_eq!(
            config["architectures"],
            json!(["Gemma4ForConditionalGeneration"])
        );
    }

    #[test]
    fn gemma4_synthesis_rejects_ragged_kv_classes() {
        let mut metadata = gemma4_test_meta();
        metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufValue::Array(vec![
                GgufValue::I32(2),
                GgufValue::I32(3),
                GgufValue::I32(1),
                GgufValue::I32(1),
            ]),
        );
        assert!(synthesize_gemma4(&metadata, &HashMap::new()).is_err());
    }

    fn glimmer_test_meta() -> HashMap<String, GgufValue> {
        meta(vec![
            ("muse-glimmer.block_count", GgufValue::U32(8)),
            ("muse-glimmer.embedding_length", GgufValue::U32(64)),
            ("muse-glimmer.feed_forward_length", GgufValue::U32(128)),
            ("muse-glimmer.attention.head_count", GgufValue::U32(4)),
            ("muse-glimmer.attention.head_count_kv", GgufValue::U32(2)),
            ("muse-glimmer.attention.key_length", GgufValue::U32(16)),
            ("muse-glimmer.attention.value_length", GgufValue::U32(16)),
            ("muse-glimmer.attention.sliding_window", GgufValue::U32(32)),
            (
                "muse-glimmer.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-5),
            ),
            ("muse-glimmer.rope.freq_base", GgufValue::F32(500000.0)),
            ("muse-glimmer.context_length", GgufValue::U32(256)),
            ("muse-glimmer.final_logit_softcapping", GgufValue::F32(20.0)),
            ("muse-glimmer.logit_scale", GgufValue::F32(3.87)),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(200000)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(200001)),
            ("clip.vision.block_count", GgufValue::U32(6)),
            ("clip.vision.embedding_length", GgufValue::U32(32)),
            ("clip.vision.feed_forward_length", GgufValue::U32(64)),
            ("clip.vision.attention.head_count", GgufValue::U32(4)),
            ("clip.vision.patch_size", GgufValue::U32(7)),
            ("clip.vision.spatial_merge_size", GgufValue::U32(2)),
            (
                "clip.vision.image_mean",
                GgufValue::Array(vec![
                    GgufValue::F32(0.5),
                    GgufValue::F32(0.5),
                    GgufValue::F32(0.5),
                ]),
            ),
            (
                "clip.vision.image_std",
                GgufValue::Array(vec![
                    GgufValue::F32(0.5),
                    GgufValue::F32(0.5),
                    GgufValue::F32(0.5),
                ]),
            ),
            (
                TOKENIZER_TOKENS_KEY,
                GgufValue::Array(vec![
                    GgufValue::String("a".to_string()),
                    GgufValue::String("<|patch|>".to_string()),
                    GgufValue::String("<|video|>".to_string()),
                ]),
            ),
        ])
    }

    #[test]
    fn glimmer_synthesis_derives_projector_and_tokens() {
        let metadata = glimmer_test_meta();
        let shapes = HashMap::from([("mm.0.weight".to_string(), vec![128, 128])]);
        let config = synthesize_muse_glimmer(&metadata, &shapes).unwrap();
        assert_eq!(config["text_config"]["head_dim"], json!(16));
        assert_eq!(
            config["text_config"]["output_multiplier"],
            json!(f64::from(3.87f32))
        );
        assert!(config["text_config"].get("qk_scale_factor").is_none());
        assert_eq!(config["text_config"]["tie_word_embeddings"], json!(false));
        assert_eq!(config["vision_config"]["merge_size"], json!(2));
        assert_eq!(config["projector_hidden_size"], json!(128));
        assert_eq!(config["image_token_id"], json!(1));
        assert_eq!(config["video_token_id"], json!(2));
        assert_eq!(
            config["architectures"],
            json!(["MuseGlimmerForConditionalGeneration"])
        );
    }

    #[test]
    fn glimmer_synthesis_rejects_mismatched_projector_layout() {
        let metadata = glimmer_test_meta();
        let shapes = HashMap::from([("mm.0.weight".to_string(), vec![7, 9])]);
        assert!(synthesize_muse_glimmer(&metadata, &shapes).is_err());
    }
    #[test]
    fn native_preprocessors_synthesize() {
        let metadata = glimmer_test_meta();
        let glimmer = synthesize_muse_glimmer_preprocessor(&metadata).unwrap();
        assert!(glimmer.contains("\"patch_size\":7"));
        let gemma = synthesize_gemma4_preprocessor(&metadata).unwrap();
        assert!(gemma.contains("\"do_convert_rgb\":true"));
    }
}
