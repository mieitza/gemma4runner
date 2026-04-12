use std::path::Path;
use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::Device;

use crate::config::Gemma4TextConfig;

/// Parsed GGUF model with metadata and tensor access.
pub struct GgufModel {
    pub content: gguf_file::Content,
    pub config: Gemma4TextConfig,
    file_path: std::path::PathBuf,
}

impl GgufModel {
    pub fn load(path: &Path, _device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .context("Failed to parse GGUF file")?;

        let config = config_from_metadata(&content.metadata)?;

        tracing::info!(
            "GGUF model: {} layers, hidden_size={}, {} tensors",
            config.num_hidden_layers,
            config.hidden_size,
            content.tensor_infos.len(),
        );

        Ok(Self {
            content,
            config,
            file_path: path.to_path_buf(),
        })
    }

    /// Get a quantized tensor by name.
    pub fn tensor(&self, name: &str, device: &Device) -> Result<candle_core::quantized::QTensor> {
        let mut file = std::fs::File::open(&self.file_path)?;
        self.content.tensor(&mut file, name, device)
            .with_context(|| format!("Failed to load tensor: {}", name))
    }
}

/// Extract model config from GGUF metadata.
/// GGUF stores config as flat key-value pairs like "gemma4.embedding_length".
fn config_from_metadata(metadata: &std::collections::HashMap<String, gguf_file::Value>) -> Result<Gemma4TextConfig> {
    // Helper to read metadata values with fallback key prefixes
    let get_u32 = |keys: &[&str]| -> Option<usize> {
        for key in keys {
            if let Some(val) = metadata.get(*key) {
                if let Ok(v) = val.to_u32() {
                    return Some(v as usize);
                }
            }
        }
        None
    };

    let get_f64 = |keys: &[&str]| -> Option<f64> {
        for key in keys {
            if let Some(val) = metadata.get(*key) {
                if let Ok(v) = val.to_f32() {
                    return Some(v as f64);
                }
            }
        }
        None
    };

    let hidden_size = get_u32(&["gemma4.embedding_length", "gemma.embedding_length"])
        .context("Missing embedding_length in GGUF metadata")?;
    let num_hidden_layers = get_u32(&["gemma4.block_count", "gemma.block_count"])
        .context("Missing block_count in GGUF metadata")?;
    let num_attention_heads = get_u32(&["gemma4.attention.head_count", "gemma.attention.head_count"])
        .context("Missing head_count in GGUF metadata")?;
    let num_key_value_heads = get_u32(&["gemma4.attention.head_count_kv", "gemma.attention.head_count_kv"])
        .unwrap_or(num_attention_heads);
    let intermediate_size = get_u32(&["gemma4.feed_forward_length", "gemma.feed_forward_length"])
        .unwrap_or(hidden_size * 4);
    let vocab_size = get_u32(&["gemma4.vocab_size", "tokenizer.ggml.tokens"])
        .unwrap_or(262144);
    let head_dim = get_u32(&["gemma4.attention.key_length"]).unwrap_or(256);
    let rms_norm_eps = get_f64(&["gemma4.attention.layer_norm_rms_epsilon"]).unwrap_or(1e-6);
    let sliding_window = get_u32(&["gemma4.attention.sliding_window"]).unwrap_or(512);
    let max_position_embeddings = get_u32(&["gemma4.context_length"]).unwrap_or(131072);

    // Build layer_types from num_hidden_layers (default 5:1 pattern)
    let layer_types: Vec<String> = (0..num_hidden_layers)
        .map(|i| {
            if (i + 1) % 6 == 0 {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect();

    Ok(Gemma4TextConfig {
        attention_bias: false,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_hidden_layers,
        num_key_value_heads,
        head_dim,
        global_head_dim: get_u32(&["gemma4.attention.global_key_length"]).unwrap_or(512),
        rms_norm_eps,
        vocab_size,
        max_position_embeddings,
        sliding_window,
        final_logit_softcapping: get_f64(&["gemma4.final_logit_softcapping"]),
        tie_word_embeddings: true,
        layer_types,
        rope_parameters: None, // Use defaults
        enable_moe_block: get_u32(&["gemma4.expert_count"]).map(|n| n > 0).unwrap_or(false),
        num_experts: get_u32(&["gemma4.expert_count"]),
        top_k_experts: get_u32(&["gemma4.expert_used_count"]),
        moe_intermediate_size: get_u32(&["gemma4.expert_feed_forward_length"]),
        num_global_key_value_heads: get_u32(&["gemma4.attention.global_head_count_kv"]),
    })
}
