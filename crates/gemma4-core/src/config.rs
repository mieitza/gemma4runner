use serde::Deserialize;

fn default_global_head_dim() -> usize {
    0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
    #[serde(default)]
    pub image_token_id: Option<usize>,
    #[serde(default)]
    pub audio_token_id: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub attention_bias: bool,
    pub hidden_activation: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
}

impl Gemma4TextConfig {
    /// Returns true if the given layer index uses sliding-window attention.
    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "sliding_attention")
            .unwrap_or(false)
    }

    /// Returns the head dimension for a given layer (global vs local).
    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_sliding_layer(layer_idx) {
            self.head_dim
        } else {
            if self.global_head_dim > 0 {
                self.global_head_dim
            } else {
                self.head_dim
            }
        }
    }

    /// Returns the number of key-value heads for a given layer.
    /// Full-attention layers use num_global_key_value_heads when set; sliding
    /// layers use the configured num_key_value_heads.
    pub fn kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        if !self.is_sliding_layer(layer_idx) {
            if let Some(global_kv) = self.num_global_key_value_heads {
                return global_kv;
            }
        }
        self.num_key_value_heads
    }

    /// Number of query-head groups per key-value head (GQA ratio).
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Returns the rope_theta for a given layer, falling back to 10000.0.
    pub fn rope_theta_for_layer(&self, layer_idx: usize) -> f64 {
        let Some(ref rope) = self.rope_parameters else {
            return 10_000.0;
        };
        if self.is_sliding_layer(layer_idx) {
            rope.sliding_attention
                .as_ref()
                .and_then(|p| p.rope_theta)
                .unwrap_or(10_000.0)
        } else {
            rope.full_attention
                .as_ref()
                .and_then(|p| p.rope_theta)
                .unwrap_or(10_000.0)
        }
    }

    /// Returns the partial_rotary_factor for a given layer, falling back to 1.0.
    pub fn partial_rotary_factor_for_layer(&self, layer_idx: usize) -> f64 {
        let Some(ref rope) = self.rope_parameters else {
            return 1.0;
        };
        if self.is_sliding_layer(layer_idx) {
            rope.sliding_attention
                .as_ref()
                .and_then(|p| p.partial_rotary_factor)
                .unwrap_or(1.0)
        } else {
            rope.full_attention
                .as_ref()
                .and_then(|p| p.partial_rotary_factor)
                .unwrap_or(1.0)
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: Option<RopeLayerParams>,
    pub sliding_attention: Option<RopeLayerParams>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeLayerParams {
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
    pub partial_rotary_factor: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_e4b_config() {
        let json = r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "attention_bias": false,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 2560,
                "intermediate_size": 10240,
                "num_attention_heads": 8,
                "num_hidden_layers": 42,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "rms_norm_eps": 1e-06,
                "vocab_size": 262144,
                "max_position_embeddings": 131072,
                "sliding_window": 512,
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "layer_types": [
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention"
                ],
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            },
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "eos_token_id": [1, 106]
        }"#;

        let config: Gemma4Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.text_config.hidden_size, 2560);
        assert_eq!(config.text_config.num_hidden_layers, 42);
        assert_eq!(config.text_config.num_attention_heads, 8);
        assert_eq!(config.text_config.num_key_value_heads, 2);
        assert_eq!(config.text_config.head_dim, 256);
        assert_eq!(config.text_config.global_head_dim, 512);
        assert_eq!(config.text_config.sliding_window, 512);
        assert_eq!(config.text_config.vocab_size, 262144);
        assert!(config.text_config.is_sliding_layer(0));
        assert!(config.text_config.is_sliding_layer(4));
        assert!(!config.text_config.is_sliding_layer(5));
    }

    #[test]
    fn test_parse_moe_config() {
        let json = r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "attention_bias": false,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 2816,
                "intermediate_size": 2112,
                "num_attention_heads": 16,
                "num_hidden_layers": 6,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "rms_norm_eps": 1e-06,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "sliding_window": 1024,
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "enable_moe_block": true,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704,
                "layer_types": [
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention"
                ],
                "rope_parameters": {
                    "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
                }
            },
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "eos_token_id": [1, 106]
        }"#;
        let config: Gemma4Config = serde_json::from_str(json).unwrap();
        let tc = &config.text_config;
        assert!(tc.enable_moe_block);
        assert_eq!(tc.num_experts.unwrap(), 128);
        assert_eq!(tc.top_k_experts.unwrap(), 8);
        assert_eq!(tc.moe_intermediate_size.unwrap(), 704);
        assert_eq!(tc.num_global_key_value_heads.unwrap(), 2);
        assert_eq!(tc.kv_heads_for_layer(5), 2);  // global layer uses global KV heads
        assert_eq!(tc.kv_heads_for_layer(0), 8);  // sliding layer uses regular KV heads
    }

    #[test]
    fn test_rope_params() {
        // Same JSON as above but with 6 layers only, parse and verify rope params
        let json = r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "attention_bias": false,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 2560,
                "intermediate_size": 10240,
                "num_attention_heads": 8,
                "num_hidden_layers": 6,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "rms_norm_eps": 1e-06,
                "vocab_size": 262144,
                "max_position_embeddings": 131072,
                "sliding_window": 512,
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true,
                "layer_types": [
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention"
                ],
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            },
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "eos_token_id": [1, 106]
        }"#;

        let config: Gemma4Config = serde_json::from_str(json).unwrap();
        let rope = config.text_config.rope_parameters.as_ref().unwrap();
        let sliding = rope.sliding_attention.as_ref().unwrap();
        let full = rope.full_attention.as_ref().unwrap();
        assert_eq!(sliding.rope_theta.unwrap(), 10000.0);
        assert_eq!(full.rope_theta.unwrap(), 1000000.0);
        assert_eq!(full.partial_rotary_factor.unwrap(), 0.25);
    }
}
