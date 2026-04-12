# Gemma4Runner Phase 1 — Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working end-to-end binary that loads a Gemma 4 E4B model from a local safetensors path, runs CPU text inference, and serves a non-streaming `/v1/chat/completions` endpoint.

**Architecture:** 3-crate Cargo workspace — `gemma4-core` (model loading, inference engine, sampling), `gemma4-api` (axum HTTP server, OpenAI-compatible types), `gemma4runner` (CLI binary wiring both together). The inference engine runs on a dedicated OS thread communicating with the async HTTP layer via bounded mpsc channels.

**Tech Stack:** Rust, candle-core/candle-nn/candle-transformers 0.10.2, tokenizers 0.22, axum 0.8, tokio, serde, clap 4

**Spec:** `docs/superpowers/specs/2026-04-12-gemma4runner-design.md`

---

## File Structure

```
gemma4runner/
├── Cargo.toml                          # workspace root
├── crates/
│   ├── gemma4-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # pub mod declarations + re-exports
│   │       ├── config.rs              # Gemma4Config + Gemma4TextConfig structs
│   │       ├── loader.rs              # load_safetensors(), load_tokenizer()
│   │       ├── tokenizer.rs           # GemmaTokenizer wrapper
│   │       ├── chat_template.rs       # format_chat_prompt()
│   │       ├── sampling.rs            # SamplingParams + LogitsProcessor
│   │       ├── kv_cache.rs            # KvCache for sliding + global layers
│   │       ├── rope.rs                # RotaryEmbedding + ProportionalRotaryEmbedding
│   │       ├── attention.rs           # GemmaAttention with GQA
│   │       ├── mlp.rs                 # GemmaMlp (gate/up/down projections)
│   │       ├── model.rs              # GemmaTextModel (full transformer)
│   │       └── engine.rs             # InferenceEngine (thread + channels)
│   ├── gemma4-api/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                 # pub mod declarations + re-exports
│   │       ├── server.rs              # build_router(), start_server()
│   │       ├── types/
│   │       │   ├── mod.rs
│   │       │   ├── chat.rs            # ChatCompletionRequest/Response
│   │       │   ├── common.rs          # Message, Role, Usage, FinishReason
│   │       │   └── error.rs           # ApiError, OpenAI error envelope
│   │       └── handlers/
│   │           ├── mod.rs
│   │           ├── chat.rs            # POST /v1/chat/completions handler
│   │           └── health.rs          # GET /health handler
│   └── gemma4runner/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                # entry point
│           └── cli.rs                 # Clap derive structs
```

---

### Task 1: Initialize Workspace and Cargo.toml Files

**Files:**
- Create: `Cargo.toml`
- Create: `crates/gemma4-core/Cargo.toml`
- Create: `crates/gemma4-api/Cargo.toml`
- Create: `crates/gemma4runner/Cargo.toml`

- [ ] **Step 1: Create workspace root Cargo.toml**

```toml
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"

[workspace.dependencies]
candle-core = "0.10"
candle-nn = "0.10"
candle-transformers = "0.10"
tokenizers = "0.22"
safetensors = "0.5"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
axum = "0.8"
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
clap = { version = "4", features = ["derive"] }
anyhow = "1"
uuid = { version = "1", features = ["v4"] }
rand = "0.9"
```

- [ ] **Step 2: Create gemma4-core Cargo.toml**

```toml
[package]
name = "gemma4-core"
version.workspace = true
edition.workspace = true

[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
candle-transformers = { workspace = true }
tokenizers = { workspace = true }
safetensors = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tracing = { workspace = true }
anyhow = { workspace = true }
rand = { workspace = true }

[features]
default = []
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
```

- [ ] **Step 3: Create gemma4-api Cargo.toml**

```toml
[package]
name = "gemma4-api"
version.workspace = true
edition.workspace = true

[dependencies]
gemma4-core = { path = "../gemma4-core" }
axum = { workspace = true }
tokio = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tracing = { workspace = true }
anyhow = { workspace = true }
uuid = { workspace = true }
```

- [ ] **Step 4: Create gemma4runner Cargo.toml**

```toml
[package]
name = "gemma4runner"
version.workspace = true
edition.workspace = true

[[bin]]
name = "gemma4runner"
path = "src/main.rs"

[dependencies]
gemma4-core = { path = "../gemma4-core" }
gemma4-api = { path = "../gemma4-api" }
clap = { workspace = true }
tokio = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
anyhow = { workspace = true }

[features]
default = []
metal = ["gemma4-core/metal"]
cuda = ["gemma4-core/cuda"]
```

- [ ] **Step 5: Create stub lib.rs and main.rs so workspace compiles**

`crates/gemma4-core/src/lib.rs`:
```rust
pub mod config;
```

`crates/gemma4-core/src/config.rs`:
```rust
// Placeholder — implemented in Task 2
```

`crates/gemma4-api/src/lib.rs`:
```rust
pub mod types;
```

`crates/gemma4-api/src/types/mod.rs`:
```rust
// Placeholder — implemented in Task 5
```

`crates/gemma4runner/src/main.rs`:
```rust
fn main() {
    println!("gemma4runner");
}
```

- [ ] **Step 6: Verify workspace compiles**

Run: `cargo check`
Expected: Compiles with no errors (may have warnings about unused code)

- [ ] **Step 7: Initialize git and commit**

```bash
git init
```

Create `.gitignore`:
```
/target
*.swp
*.swo
.DS_Store
```

```bash
git add Cargo.toml .gitignore crates/
git commit -m "feat: initialize 3-crate workspace (gemma4-core, gemma4-api, gemma4runner)"
```

---

### Task 2: Config — Model Configuration Structs

**Files:**
- Modify: `crates/gemma4-core/src/config.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

These structs deserialize the `config.json` shipped with Gemma 4 models on HuggingFace.

- [ ] **Step 1: Write config test**

Add to the bottom of `crates/gemma4-core/src/config.rs`:

```rust
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
    fn test_rope_params() {
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gemma4-core test_parse_e4b_config`
Expected: FAIL — `Gemma4Config` not defined

- [ ] **Step 3: Implement config structs**

Replace the contents of `crates/gemma4-core/src/config.rs` with:

```rust
use serde::Deserialize;

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
}

fn default_global_head_dim() -> usize {
    512
}

impl Gemma4TextConfig {
    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "sliding_attention")
            .unwrap_or(true)
    }

    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_sliding_layer(layer_idx) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }

    pub fn kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        self.num_key_value_heads
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn rope_theta_for_layer(&self, layer_idx: usize) -> f64 {
        let params = match &self.rope_parameters {
            Some(p) => p,
            None => return 10000.0,
        };
        if self.is_sliding_layer(layer_idx) {
            params.sliding_attention.as_ref()
                .and_then(|s| s.rope_theta)
                .unwrap_or(10000.0)
        } else {
            params.full_attention.as_ref()
                .and_then(|f| f.rope_theta)
                .unwrap_or(1000000.0)
        }
    }

    pub fn partial_rotary_factor_for_layer(&self, layer_idx: usize) -> f64 {
        if self.is_sliding_layer(layer_idx) {
            return 1.0;
        }
        self.rope_parameters.as_ref()
            .and_then(|p| p.full_attention.as_ref())
            .and_then(|f| f.partial_rotary_factor)
            .unwrap_or(1.0)
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

// Keep the tests at the bottom...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p gemma4-core`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gemma4-core/src/config.rs
git commit -m "feat(core): add Gemma 4 model config deserialization"
```

---

### Task 3: Chat Template — Message Formatting

**Files:**
- Create: `crates/gemma4-core/src/chat_template.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Write chat template tests**

Create `crates/gemma4-core/src/chat_template.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Formats OpenAI-style messages into Gemma 4's chat format.
///
/// Gemma 4 uses:
/// - `<bos>` at the start
/// - `<|turn>role\ncontent<turn|>\n` for each message
/// - `<|turn>model\n` as the generation prompt
pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_user_message() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = format_chat_prompt(&messages);
        assert_eq!(prompt, "<bos><|turn>user\nHello<turn|>\n<|turn>model\n");
    }

    #[test]
    fn test_system_and_user() {
        let messages = vec![
            ChatMessage { role: "system".into(), content: "You are helpful.".into() },
            ChatMessage { role: "user".into(), content: "Hi".into() },
        ];
        let prompt = format_chat_prompt(&messages);
        assert_eq!(
            prompt,
            "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn test_multi_turn() {
        let messages = vec![
            ChatMessage { role: "user".into(), content: "What is 2+2?".into() },
            ChatMessage { role: "assistant".into(), content: "4".into() },
            ChatMessage { role: "user".into(), content: "And 3+3?".into() },
        ];
        let prompt = format_chat_prompt(&messages);
        assert_eq!(
            prompt,
            "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n4<turn|>\n<|turn>user\nAnd 3+3?<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn test_assistant_mapped_to_model() {
        let messages = vec![
            ChatMessage { role: "assistant".into(), content: "Hi there".into() },
        ];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.contains("<|turn>model\nHi there<turn|>"));
    }
}
```

- [ ] **Step 2: Add module to lib.rs and run tests to verify they fail**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod chat_template;
```

Run: `cargo test -p gemma4-core chat_template`
Expected: FAIL — `todo!()` panics

- [ ] **Step 3: Implement format_chat_prompt**

Replace the `todo!()` in `format_chat_prompt`:

```rust
pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<bos>");

    for msg in messages {
        let role = match msg.role.as_str() {
            "assistant" => "model",
            other => other,
        };
        prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, msg.content));
    }

    // Add generation prompt for the model to respond
    prompt.push_str("<|turn>model\n");
    prompt
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p gemma4-core chat_template`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gemma4-core/src/chat_template.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add Gemma 4 chat template formatting"
```

---

### Task 4: Sampling — Logits Processor

**Files:**
- Create: `crates/gemma4-core/src/sampling.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Write sampling tests**

Create `crates/gemma4-core/src/sampling.rs`:

```rust
use anyhow::Result;
use candle_core::{Device, Tensor};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub max_tokens: usize,
    pub seed: Option<u64>,
    pub repetition_penalty: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            max_tokens: 2048,
            seed: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
}

impl LogitsProcessor {
    pub fn new(seed: Option<u64>) -> Self {
        todo!()
    }

    /// Apply sampling parameters to logits and return a sampled token ID.
    pub fn sample(&mut self, logits: &Tensor, params: &SamplingParams) -> Result<u32> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        // Temperature 0 = greedy (argmax)
        let logits = Tensor::new(&[1.0f32, 5.0, 2.0, 0.5], &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut proc = LogitsProcessor::new(Some(42));
        let token = proc.sample(&logits, &params).unwrap();
        assert_eq!(token, 1); // index of 5.0
    }

    #[test]
    fn test_deterministic_with_seed() {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let mut proc1 = LogitsProcessor::new(Some(42));
        let mut proc2 = LogitsProcessor::new(Some(42));
        let t1 = proc1.sample(&logits, &params).unwrap();
        let t2 = proc2.sample(&logits, &params).unwrap();
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_top_k_filters() {
        // With top_k=1, should always pick the highest logit
        let logits = Tensor::new(&[1.0f32, 10.0, 2.0, 3.0], &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            ..Default::default()
        };
        let mut proc = LogitsProcessor::new(Some(42));
        let token = proc.sample(&logits, &params).unwrap();
        assert_eq!(token, 1); // index of 10.0
    }

    #[test]
    fn test_default_params() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.max_tokens, 2048);
        assert_eq!(params.repetition_penalty, 1.0);
    }
}
```

- [ ] **Step 2: Add module to lib.rs and run tests to verify they fail**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod sampling;
```

Run: `cargo test -p gemma4-core sampling`
Expected: FAIL — `todo!()` panics

- [ ] **Step 3: Implement LogitsProcessor**

Replace the `todo!()`s in `sampling.rs`:

```rust
use rand::SeedableRng;
use rand::distr::{Distribution, WeightedIndex};

impl LogitsProcessor {
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_os_rng(),
        };
        Self { rng }
    }

    pub fn sample(&mut self, logits: &Tensor, params: &SamplingParams) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32)?.flatten_all()?;
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;

        // Greedy: temperature == 0
        if params.temperature == 0.0 {
            let token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            return Ok(token);
        }

        // Apply temperature
        if params.temperature != 1.0 {
            let inv_temp = 1.0 / params.temperature as f32;
            for l in logits_vec.iter_mut() {
                *l *= inv_temp;
            }
        }

        // Top-k filtering
        if let Some(k) = params.top_k {
            if k < logits_vec.len() {
                let mut indexed: Vec<(usize, f32)> =
                    logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let threshold = indexed[k - 1].1;
                for l in logits_vec.iter_mut() {
                    if *l < threshold {
                        *l = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Softmax
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits_vec.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top-p (nucleus) filtering
        if params.top_p < 1.0 {
            let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
            sorted_indices.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());
            let mut cumulative = 0.0f32;
            let mut cutoff_idx = sorted_indices.len();
            for (i, &idx) in sorted_indices.iter().enumerate() {
                cumulative += probs[idx];
                if cumulative > params.top_p as f32 {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            let allowed: std::collections::HashSet<usize> =
                sorted_indices[..cutoff_idx].iter().copied().collect();
            for (i, p) in probs.iter_mut().enumerate() {
                if !allowed.contains(&i) {
                    *p = 0.0;
                }
            }
            // Re-normalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        // Sample from distribution
        let dist = WeightedIndex::new(&probs)?;
        let token = dist.sample(&mut self.rng) as u32;
        Ok(token)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p gemma4-core sampling`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gemma4-core/src/sampling.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add logits processor with temperature, top-k, top-p sampling"
```

---

### Task 5: KV Cache

**Files:**
- Create: `crates/gemma4-core/src/kv_cache.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement KvCache**

Create `crates/gemma4-core/src/kv_cache.rs`:

```rust
use anyhow::Result;
use candle_core::Tensor;

/// KV cache for a single attention layer.
/// Sliding layers evict entries beyond the window; global layers keep everything.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
    sliding_window: Option<usize>,
    current_len: usize,
}

impl LayerKvCache {
    pub fn new(sliding_window: Option<usize>) -> Self {
        Self {
            key: None,
            value: None,
            sliding_window,
            current_len: 0,
        }
    }

    /// Append new key/value tensors. Returns the full (possibly windowed) k/v.
    /// key shape: (batch, num_kv_heads, seq_len, head_dim)
    /// value shape: (batch, num_kv_heads, seq_len, head_dim)
    pub fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (self.key.take(), self.value.take()) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[&prev_k, key], 2)?;
                let v = Tensor::cat(&[&prev_v, value], 2)?;
                (k, v)
            }
            _ => (key.clone(), value.clone()),
        };

        // Apply sliding window if configured
        let (k, v) = if let Some(window) = self.sliding_window {
            let seq_len = k.dim(2)?;
            if seq_len > window {
                let start = seq_len - window;
                let k = k.narrow(2, start, window)?;
                let v = v.narrow(2, start, window)?;
                (k, v)
            } else {
                (k, v)
            }
        } else {
            (k, v)
        };

        self.current_len = k.dim(2)?;
        self.key = Some(k.clone());
        self.value = Some(v.clone());
        Ok((k, v))
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn reset(&mut self) {
        self.key = None;
        self.value = None;
        self.current_len = 0;
    }
}

/// Collection of KV caches for all layers.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
}

impl KvCache {
    pub fn new(layer_types: &[String], sliding_window: usize) -> Self {
        let layers = layer_types
            .iter()
            .map(|t| {
                let window = if t == "sliding_attention" {
                    Some(sliding_window)
                } else {
                    None // global layers keep full context
                };
                LayerKvCache::new(window)
            })
            .collect();
        Self { layers }
    }

    pub fn layer_mut(&mut self, idx: usize) -> &mut LayerKvCache {
        &mut self.layers[idx]
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_global_cache_keeps_all() {
        let mut cache = LayerKvCache::new(None);
        let k1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let (k, _v) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k.dim(2).unwrap(), 3);

        let k2 = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
        let (k, _v) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k.dim(2).unwrap(), 8); // 3 + 5
    }

    #[test]
    fn test_sliding_cache_evicts() {
        let mut cache = LayerKvCache::new(Some(4));
        let k1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let (k, _v) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k.dim(2).unwrap(), 3); // under window, keep all

        let k2 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let (k, _v) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k.dim(2).unwrap(), 4); // 6 total but window=4
    }

    #[test]
    fn test_kv_cache_multi_layer() {
        let layer_types = vec![
            "sliding_attention".into(),
            "sliding_attention".into(),
            "full_attention".into(),
        ];
        let mut cache = KvCache::new(&layer_types, 4);
        assert!(cache.layer_mut(0).sliding_window.is_some());
        assert!(cache.layer_mut(2).sliding_window.is_none());
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod kv_cache;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p gemma4-core kv_cache`
Expected: All 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add crates/gemma4-core/src/kv_cache.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add KV cache with sliding window eviction"
```

---

### Task 6: RoPE — Rotary Position Embeddings

**Files:**
- Create: `crates/gemma4-core/src/rope.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement RoPE**

Create `crates/gemma4-core/src/rope.rs`:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Standard Rotary Position Embedding (for sliding attention layers).
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        theta: f64,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
        let inv_freq = inv_freq.unsqueeze(0)?;

        let freqs = positions.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // (max_seq_len, head_dim)
        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    /// Apply RoPE to query and key tensors.
    /// q, k shapes: (batch, num_heads, seq_len, head_dim)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

        let q_rot = apply_rotary_emb(q, &cos, &sin)?;
        let k_rot = apply_rotary_emb(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}

/// Proportional Rotary Position Embedding (for global/full attention layers).
/// Uses partial_rotary_factor to only rotate a fraction of dimensions.
pub struct ProportionalRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl ProportionalRotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let rotary_dim = (head_dim as f64 * partial_rotary_factor) as usize;
        // Ensure rotary_dim is even
        let rotary_dim = rotary_dim - (rotary_dim % 2);

        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
        let inv_freq = inv_freq.unsqueeze(0)?;

        let freqs = positions.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // (max_seq_len, rotary_dim)
        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin, rotary_dim })
    }

    /// Apply proportional RoPE. Only rotates the first `rotary_dim` dimensions;
    /// the rest pass through unchanged.
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let head_dim = q.dim(3)?;

        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

        if self.rotary_dim == head_dim {
            let q_rot = apply_rotary_emb(q, &cos, &sin)?;
            let k_rot = apply_rotary_emb(k, &cos, &sin)?;
            return Ok((q_rot, k_rot));
        }

        // Partial rotation: split, rotate first part, concat back
        let q_rot_part = q.narrow(3, 0, self.rotary_dim)?;
        let q_pass = q.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?;
        let k_rot_part = k.narrow(3, 0, self.rotary_dim)?;
        let k_pass = k.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?;

        let q_rotated = apply_rotary_emb(&q_rot_part, &cos, &sin)?;
        let k_rotated = apply_rotary_emb(&k_rot_part, &cos, &sin)?;

        let q_out = Tensor::cat(&[&q_rotated, &q_pass], 3)?;
        let k_out = Tensor::cat(&[&k_rotated, &k_pass], 3)?;
        Ok((q_out, k_out))
    }
}

/// Core rotary embedding application: x * cos + rotate_half(x) * sin
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;

    // rotate_half: [-x2, x1]
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

    let result = x.broadcast_mul(cos)?.broadcast_add(&rotated.broadcast_mul(sin)?)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_output_shape() {
        let rope = RotaryEmbedding::new(DType::F32, 8, 10000.0, 128, &Device::Cpu).unwrap();
        let q = Tensor::zeros((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::zeros((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_rot.dims(), &[1, 2, 4, 8]);
        assert_eq!(k_rot.dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_proportional_rope_partial_rotation() {
        // partial_rotary_factor=0.25, head_dim=8 → rotary_dim=2
        let rope = ProportionalRotaryEmbedding::new(
            DType::F32, 8, 1000000.0, 0.25, 128, &Device::Cpu,
        )
        .unwrap();
        let q = Tensor::ones((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_rot.dims(), &[1, 2, 4, 8]);
        assert_eq!(k_rot.dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_rope_offset() {
        let rope = RotaryEmbedding::new(DType::F32, 8, 10000.0, 128, &Device::Cpu).unwrap();
        let q = Tensor::ones((1, 2, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 2, 1, 8), DType::F32, &Device::Cpu).unwrap();
        // Should not panic with offset > 0
        let (q0, _) = rope.apply(&q, &k, 0).unwrap();
        let (q5, _) = rope.apply(&q, &k, 5).unwrap();
        // Different positions should produce different embeddings
        let diff = (q0 - q5).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff > 0.0);
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod rope;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p gemma4-core rope`
Expected: All 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add crates/gemma4-core/src/rope.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add standard and proportional RoPE implementations"
```

---

### Task 7: MLP Layer

**Files:**
- Create: `crates/gemma4-core/src/mlp.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement GemmaMlp**

Create `crates/gemma4-core/src/mlp.rs`:

```rust
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{linear_no_bias, Activation, Linear, Module, VarBuilder};

/// Gemma 4 MLP block: gated feedforward with GeluPytorchTanh activation.
/// output = down_proj(act(gate_proj(x)) * up_proj(x))
pub struct GemmaMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GemmaMlp {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: Activation::GeluPytorchTanh,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = self.act_fn.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        let output = self.down_proj.forward(&fused)?;
        Ok(output)
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod mlp;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p gemma4-core`
Expected: Compiles (no runtime test needed — MLP is simple wiring of candle-nn primitives)

- [ ] **Step 4: Commit**

```bash
git add crates/gemma4-core/src/mlp.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add gated MLP with GeluPytorchTanh activation"
```

---

### Task 8: Attention Layer

**Files:**
- Create: `crates/gemma4-core/src/attention.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement GemmaAttention**

Create `crates/gemma4-core/src/attention.rs`:

```rust
use std::sync::Arc;
use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::config::Gemma4TextConfig;
use crate::kv_cache::LayerKvCache;
use crate::rope::{ProportionalRotaryEmbedding, RotaryEmbedding};

pub struct GemmaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    is_sliding: bool,
    rotary_emb_local: Arc<RotaryEmbedding>,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
}

impl GemmaAttention {
    pub fn new(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        rotary_emb_local: Arc<RotaryEmbedding>,
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding_layer(layer_idx);
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.kv_heads_for_layer(layer_idx);

        let q_proj = linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            is_sliding,
            rotary_emb_local,
            rotary_emb_global,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerKvCache,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply per-head RMS norm to Q and K
        let q = self.apply_head_norm(&q, &self.q_norm)?;
        let k = self.apply_head_norm(&k, &self.k_norm)?;

        // Apply RoPE
        let (q, k) = if self.is_sliding {
            self.rotary_emb_local.apply(&q, &k, seqlen_offset)?
        } else {
            self.rotary_emb_global.apply(&q, &k, seqlen_offset)?
        };

        // Update KV cache
        let (k, v) = cache.append(&k, &v)?;

        // Repeat KV heads for GQA
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let output = attn_weights.matmul(&v)?;

        // Reshape back: (batch, heads, seq, dim) → (batch, seq, heads*dim)
        let output = output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        let output = self.o_proj.forward(&output)?;
        Ok(output)
    }

    /// Apply RmsNorm to each head independently.
    /// Input: (batch, num_heads, seq_len, head_dim)
    fn apply_head_norm(&self, x: &Tensor, norm: &candle_nn::RmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let x = x.reshape((b * h * s, d))?;
        let x = norm.forward(&x)?;
        let x = x.reshape((b, h, s, d))?;
        Ok(x)
    }

    /// Repeat KV heads to match query head count for GQA.
    /// (batch, num_kv_heads, seq, dim) → (batch, num_heads, seq, dim)
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_groups == 1 {
            return Ok(x.clone());
        }
        let (b, h, s, d) = x.dims4()?;
        let x = x.unsqueeze(2)?; // (b, h, 1, s, d)
        let x = x.expand((b, h, self.num_kv_groups, s, d))?;
        let x = x.reshape((b, h * self.num_kv_groups, s, d))?;
        Ok(x)
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod attention;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p gemma4-core`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add crates/gemma4-core/src/attention.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add GQA attention with dual RoPE and sliding/global support"
```

---

### Task 9: Full Text Model

**Files:**
- Create: `crates/gemma4-core/src/model.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement GemmaTextModel**

Create `crates/gemma4-core/src/model.rs`:

```rust
use std::sync::Arc;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

use crate::attention::GemmaAttention;
use crate::config::Gemma4TextConfig;
use crate::kv_cache::KvCache;
use crate::mlp::GemmaMlp;
use crate::rope::{ProportionalRotaryEmbedding, RotaryEmbedding};

struct DecoderLayer {
    self_attn: GemmaAttention,
    mlp: GemmaMlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
    pre_feedforward_layernorm: candle_nn::RmsNorm,
    post_feedforward_layernorm: candle_nn::RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        rotary_emb_local: Arc<RotaryEmbedding>,
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = GemmaAttention::new(
            cfg,
            layer_idx,
            rotary_emb_local,
            rotary_emb_global,
            vb.pp("self_attn"),
        )?;
        let mlp = GemmaMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        let eps = cfg.rms_norm_eps;
        let hs = cfg.hidden_size;
        let input_layernorm = candle_nn::rms_norm(hs, eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(hs, eps, vb.pp("post_attention_layernorm"))?;
        let pre_feedforward_layernorm =
            candle_nn::rms_norm(hs, eps, vb.pp("pre_feedforward_layernorm"))?;
        let post_feedforward_layernorm =
            candle_nn::rms_norm(hs, eps, vb.pp("post_feedforward_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut crate::kv_cache::LayerKvCache,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Pre-norm → attention → post-norm → residual
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, cache, seqlen_offset)?;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = (residual + x)?;

        // Pre-norm → MLP → post-norm → residual
        let residual = &x;
        let ff = self.pre_feedforward_layernorm.forward(&x)?;
        let ff = self.mlp.forward(&ff)?;
        let ff = self.post_feedforward_layernorm.forward(&ff)?;
        let x = (residual + ff)?;

        Ok(x)
    }
}

pub struct GemmaTextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: candle_nn::RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    hidden_size: usize,
    cfg: Gemma4TextConfig,
}

impl GemmaTextModel {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        // Create shared RoPE embeddings
        let sliding_rope_params = cfg.rope_parameters.as_ref()
            .and_then(|p| p.sliding_attention.as_ref());
        let full_rope_params = cfg.rope_parameters.as_ref()
            .and_then(|p| p.full_attention.as_ref());

        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            sliding_rope_params.and_then(|p| p.rope_theta).unwrap_or(10000.0),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(),
            cfg.global_head_dim,
            full_rope_params.and_then(|p| p.rope_theta).unwrap_or(1000000.0),
            full_rope_params.and_then(|p| p.partial_rotary_factor).unwrap_or(0.25),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                i,
                rotary_emb_local.clone(),
                rotary_emb_global.clone(),
                vb_l.pp(i),
            )?);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            hidden_size: cfg.hidden_size,
            cfg: cfg.clone(),
        })
    }

    /// Run a forward pass. Returns logits for the last token only.
    /// input_ids: (batch_size, seq_len) tensor of token IDs
    pub fn forward(
        &self,
        input_ids: &Tensor,
        cache: &mut KvCache,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Embed tokens and scale by sqrt(hidden_size)
        let mut x = self.embed_tokens.forward(input_ids)?;
        let scale = (self.hidden_size as f64).sqrt();
        x = (x * scale)?;

        // Create attention masks
        let (sliding_mask, global_mask) =
            self.create_masks(batch_size, seq_len, seqlen_offset, x.device())?;

        // Run through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if self.cfg.is_sliding_layer(i) {
                sliding_mask.as_ref()
            } else {
                global_mask.as_ref()
            };
            x = layer.forward(&x, mask, cache.layer_mut(i), seqlen_offset)?;
        }

        // Final norm + LM head on last token only
        let x = x.narrow(1, seq_len - 1, 1)?;
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;

        // Apply logit softcapping: tanh(logits / cap) * cap
        let logits = match self.final_logit_softcapping {
            Some(cap) => ((logits / cap)?.tanh()? * cap)?,
            None => logits,
        };

        Ok(logits)
    }

    fn create_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        offset: usize,
        device: &Device,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len == 1 {
            return Ok((None, None));
        }

        // Causal mask: upper triangle of -inf
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j > i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let causal = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?;

        // Sliding window mask
        let sw = self.cfg.sliding_window;
        let sliding_mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    let pos_i = i + offset;
                    let pos_j = j + offset;
                    if j > i || (pos_i >= sw && pos_j < pos_i - sw + 1) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let sliding = Tensor::from_vec(sliding_mask, (1, 1, seq_len, seq_len), device)?;

        Ok((Some(sliding), Some(causal)))
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod mlp;
pub mod model;
```

Wait, `mlp` was already added in Task 7. Just add:
```rust
pub mod model;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p gemma4-core`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add crates/gemma4-core/src/model.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add full Gemma 4 text model with decoder layers"
```

---

### Task 10: Model Loader

**Files:**
- Create: `crates/gemma4-core/src/loader.rs`
- Create: `crates/gemma4-core/src/tokenizer.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement loader**

Create `crates/gemma4-core/src/loader.rs`:

```rust
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::config::Gemma4Config;
use crate::model::GemmaTextModel;

pub struct LoadedModel {
    pub model: GemmaTextModel,
    pub config: Gemma4Config,
}

/// Load a Gemma 4 model from a local directory containing safetensors + config.json.
pub fn load_model(model_dir: &Path, device: &Device) -> Result<LoadedModel> {
    let config_path = model_dir.join("config.json");
    let config: Gemma4Config = serde_json::from_reader(
        std::fs::File::open(&config_path)
            .with_context(|| format!("Failed to open {}", config_path.display()))?,
    )
    .context("Failed to parse config.json")?;

    let safetensor_files = find_safetensor_files(model_dir)?;
    tracing::info!(
        "Loading {} safetensor file(s) from {}",
        safetensor_files.len(),
        model_dir.display()
    );

    let dtype = DType::F32; // CPU inference uses F32 for Phase 1
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)
            .context("Failed to load safetensors")?
    };

    let model = GemmaTextModel::new(&config.text_config, vb)
        .context("Failed to build model from weights")?;

    tracing::info!(
        "Model loaded: {} layers, hidden_size={}",
        config.text_config.num_hidden_layers,
        config.text_config.hidden_size
    );

    Ok(LoadedModel { model, config })
}

fn find_safetensor_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    anyhow::ensure!(!files.is_empty(), "No .safetensors files found in {}", dir.display());

    files.sort();
    Ok(files)
}
```

- [ ] **Step 2: Implement tokenizer wrapper**

Create `crates/gemma4-core/src/tokenizer.rs`:

```rust
use std::path::Path;
use anyhow::{Context, Result};

pub struct GemmaTokenizer {
    inner: tokenizers::Tokenizer,
    eos_token_ids: Vec<u32>,
}

impl GemmaTokenizer {
    pub fn from_file(path: &Path, eos_token_ids: Vec<u32>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self {
            inner,
            eos_token_ids,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let text = self
            .inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        Ok(text)
    }

    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
}
```

- [ ] **Step 3: Add modules to lib.rs**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod loader;
pub mod tokenizer;
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p gemma4-core`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add crates/gemma4-core/src/loader.rs crates/gemma4-core/src/tokenizer.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add model loader and tokenizer wrapper"
```

---

### Task 11: Inference Engine

**Files:**
- Create: `crates/gemma4-core/src/engine.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement InferenceEngine**

Create `crates/gemma4-core/src/engine.rs`:

```rust
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

use crate::chat_template::{format_chat_prompt, ChatMessage};
use crate::config::Gemma4Config;
use crate::kv_cache::KvCache;
use crate::loader;
use crate::sampling::{LogitsProcessor, SamplingParams};
use crate::tokenizer::GemmaTokenizer;

#[derive(Debug)]
pub struct InferenceRequest {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    pub sampling: SamplingParams,
    pub response_tx: mpsc::Sender<InferenceEvent>,
}

#[derive(Debug, Clone)]
pub enum InferenceEvent {
    Token(String),
    Usage(UsageStats),
    Done(FinishReason),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    Stop,
    Length,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
        }
    }
}

/// Handle to send requests to the inference engine thread.
#[derive(Clone)]
pub struct EngineHandle {
    request_tx: mpsc::SyncSender<InferenceRequest>,
}

impl EngineHandle {
    pub fn send(&self, request: InferenceRequest) -> Result<()> {
        self.request_tx
            .try_send(request)
            .map_err(|e| anyhow::anyhow!("Engine queue full or disconnected: {}", e))
    }
}

/// Start the inference engine on a dedicated thread.
/// Returns a handle for sending requests.
pub fn start_engine(
    model_path: &Path,
    queue_depth: usize,
) -> Result<EngineHandle> {
    let device = Device::Cpu;

    // Load model and tokenizer
    let loaded = loader::load_model(model_path, &device)?;
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = GemmaTokenizer::from_file(
        &tokenizer_path,
        loaded.config.eos_token_id.clone(),
    )?;

    let (request_tx, request_rx) = mpsc::sync_channel::<InferenceRequest>(queue_depth);

    let config = loaded.config.clone();
    let model = loaded.model;

    thread::Builder::new()
        .name("inference-engine".to_string())
        .spawn(move || {
            engine_loop(model, tokenizer, config, device, request_rx);
        })?;

    Ok(EngineHandle { request_tx })
}

fn engine_loop(
    model: crate::model::GemmaTextModel,
    tokenizer: GemmaTokenizer,
    config: Gemma4Config,
    device: Device,
    request_rx: mpsc::Receiver<InferenceRequest>,
) {
    // We need model to be mutable for KV cache, but the model itself is not mutable
    // KV cache is managed separately
    let model = model;

    while let Ok(request) = request_rx.recv() {
        if let Err(e) = process_request(&model, &tokenizer, &config, &device, &request) {
            let _ = request.response_tx.send(InferenceEvent::Error(e.to_string()));
        }
    }
    tracing::info!("Inference engine shutting down");
}

fn process_request(
    model: &crate::model::GemmaTextModel,
    tokenizer: &GemmaTokenizer,
    config: &Gemma4Config,
    device: &Device,
    request: &InferenceRequest,
) -> Result<()> {
    // Format chat prompt
    let prompt = format_chat_prompt(&request.messages);
    let prompt_tokens = tokenizer.encode(&prompt)?;
    let prompt_len = prompt_tokens.len();

    tracing::debug!("Prompt: {} tokens", prompt_len);

    // Initialize KV cache
    let mut cache = KvCache::new(
        &config.text_config.layer_types,
        config.text_config.sliding_window,
    );

    // Create logits processor
    let mut logits_processor = LogitsProcessor::new(request.sampling.seed);

    // Prefill: process all prompt tokens at once
    let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, &mut cache, 0)?;
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits, &request.sampling)?;

    let mut generated_tokens: Vec<u32> = vec![next_token];

    // Decode loop: generate tokens one at a time
    for step in 0..request.sampling.max_tokens {
        if tokenizer.is_eos(next_token) {
            break;
        }

        // Send the decoded token
        let token_text = tokenizer.decode(&[next_token])?;
        let _ = request.response_tx.send(InferenceEvent::Token(token_text));

        // Forward pass for next token
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let seqlen_offset = prompt_len + step + 1;
        let logits = model.forward(&input, &mut cache, seqlen_offset)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        next_token = logits_processor.sample(&logits, &request.sampling)?;
        generated_tokens.push(next_token);
    }

    // Determine finish reason
    let finish_reason = if tokenizer.is_eos(next_token) {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };

    // Send usage and done
    let _ = request.response_tx.send(InferenceEvent::Usage(UsageStats {
        prompt_tokens: prompt_len,
        completion_tokens: generated_tokens.len(),
    }));
    let _ = request.response_tx.send(InferenceEvent::Done(finish_reason));

    Ok(())
}
```

- [ ] **Step 2: Add module to lib.rs**

Add to `crates/gemma4-core/src/lib.rs`:
```rust
pub mod engine;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p gemma4-core`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add crates/gemma4-core/src/engine.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add inference engine with dedicated thread and channel-based communication"
```

---

### Task 12: API Types — OpenAI-Compatible Structs

**Files:**
- Modify: `crates/gemma4-api/src/types/mod.rs`
- Create: `crates/gemma4-api/src/types/common.rs`
- Create: `crates/gemma4-api/src/types/chat.rs`
- Create: `crates/gemma4-api/src/types/error.rs`

- [ ] **Step 1: Implement common types**

Replace `crates/gemma4-api/src/types/mod.rs`:
```rust
pub mod chat;
pub mod common;
pub mod error;
```

Create `crates/gemma4-api/src/types/common.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}
```

- [ ] **Step 2: Implement chat completion types**

Create `crates/gemma4-api/src/types/chat.rs`:

```rust
use serde::{Deserialize, Serialize};

use super::common::{FinishReason, Message, Usage};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
}

fn default_temperature() -> Option<f64> {
    Some(1.0)
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChoiceMessage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
pub struct ChoiceMessage {
    pub role: String,
    pub content: String,
}
```

- [ ] **Step 3: Implement error types**

Create `crates/gemma4-api/src/types/error.rs`:

```rust
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ApiErrorResponse {
    pub error: ApiErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

pub struct ApiError {
    pub status: StatusCode,
    pub body: ApiErrorResponse,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ApiErrorResponse {
                error: ApiErrorBody {
                    message: message.into(),
                    error_type: "invalid_request_error".into(),
                    param,
                    code: None,
                },
            },
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ApiErrorResponse {
                error: ApiErrorBody {
                    message: message.into(),
                    error_type: "internal_error".into(),
                    param: None,
                    code: None,
                },
            },
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: ApiErrorResponse {
                error: ApiErrorBody {
                    message: message.into(),
                    error_type: "service_unavailable".into(),
                    param: None,
                    code: None,
                },
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::to_string(&self.body).unwrap_or_default();
        (self.status, [("content-type", "application/json")], body).into_response()
    }
}
```

- [ ] **Step 4: Update lib.rs**

`crates/gemma4-api/src/lib.rs` should be:
```rust
pub mod types;
pub mod handlers;
pub mod server;
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p gemma4-api`
Expected: Compiles (handlers and server don't exist yet, so may need stubs — see next step)

Create `crates/gemma4-api/src/handlers/mod.rs`:
```rust
pub mod chat;
pub mod health;
```

Create `crates/gemma4-api/src/handlers/chat.rs`:
```rust
// Implemented in Task 13
```

Create `crates/gemma4-api/src/handlers/health.rs`:
```rust
// Implemented in Task 13
```

Create `crates/gemma4-api/src/server.rs`:
```rust
// Implemented in Task 13
```

Run: `cargo check -p gemma4-api`
Expected: Compiles

- [ ] **Step 6: Commit**

```bash
git add crates/gemma4-api/
git commit -m "feat(api): add OpenAI-compatible request/response types and error handling"
```

---

### Task 13: HTTP Handlers and Server

**Files:**
- Modify: `crates/gemma4-api/src/handlers/chat.rs`
- Modify: `crates/gemma4-api/src/handlers/health.rs`
- Modify: `crates/gemma4-api/src/server.rs`

- [ ] **Step 1: Implement health handler**

Replace `crates/gemma4-api/src/handlers/health.rs`:

```rust
use axum::http::StatusCode;

pub async fn health() -> StatusCode {
    StatusCode::OK
}
```

- [ ] **Step 2: Implement chat completion handler**

Replace `crates/gemma4-api/src/handlers/chat.rs`:

```rust
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};
use axum::extract::State;
use axum::Json;
use axum::response::IntoResponse;

use gemma4_core::chat_template::ChatMessage;
use gemma4_core::engine::{EngineHandle, FinishReason, InferenceEvent, InferenceRequest};
use gemma4_core::sampling::SamplingParams;

use crate::types::chat::*;
use crate::types::common;
use crate::types::error::ApiError;

pub async fn chat_completions(
    State(engine): State<EngineHandle>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Validate
    if request.messages.is_empty() {
        return Err(ApiError::bad_request(
            "messages must not be empty",
            Some("messages".into()),
        ));
    }

    // Convert messages
    let messages: Vec<ChatMessage> = request
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: match m.role {
                common::Role::System => "system".into(),
                common::Role::User => "user".into(),
                common::Role::Assistant => "assistant".into(),
            },
            content: m.content.clone(),
        })
        .collect();

    // Build sampling params
    let sampling = SamplingParams {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k,
        max_tokens: request.max_tokens.unwrap_or(2048),
        seed: request.seed,
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
    };

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = request.model.clone();

    // Create response channel
    let (response_tx, response_rx) = mpsc::channel();

    let inference_request = InferenceRequest {
        id: request_id.clone(),
        messages,
        sampling,
        response_tx,
    };

    // Send to engine
    engine
        .send(inference_request)
        .map_err(|e| ApiError::service_unavailable(e.to_string()))?;

    // Collect response (blocking — we use spawn_blocking to avoid blocking tokio)
    let result = tokio::task::spawn_blocking(move || {
        collect_response(response_rx)
    })
    .await
    .map_err(|e| ApiError::internal(format!("Task join error: {}", e)))?
    .map_err(|e| ApiError::internal(e.to_string()))?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".into(),
        created,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".into(),
                content: result.content,
            },
            finish_reason: result.finish_reason,
        }],
        usage: common::Usage {
            prompt_tokens: result.usage.prompt_tokens,
            completion_tokens: result.usage.completion_tokens,
            total_tokens: result.usage.prompt_tokens + result.usage.completion_tokens,
        },
    };

    Ok(Json(response))
}

struct CollectedResponse {
    content: String,
    finish_reason: common::FinishReason,
    usage: gemma4_core::engine::UsageStats,
}

fn collect_response(rx: mpsc::Receiver<InferenceEvent>) -> anyhow::Result<CollectedResponse> {
    let mut content = String::new();
    let mut finish_reason = common::FinishReason::Stop;
    let mut usage = gemma4_core::engine::UsageStats {
        prompt_tokens: 0,
        completion_tokens: 0,
    };

    while let Ok(event) = rx.recv() {
        match event {
            InferenceEvent::Token(t) => content.push_str(&t),
            InferenceEvent::Usage(u) => usage = u,
            InferenceEvent::Done(reason) => {
                finish_reason = match reason {
                    FinishReason::Stop => common::FinishReason::Stop,
                    FinishReason::Length => common::FinishReason::Length,
                };
                break;
            }
            InferenceEvent::Error(e) => {
                return Err(anyhow::anyhow!("Inference error: {}", e));
            }
            _ => {}
        }
    }

    Ok(CollectedResponse {
        content,
        finish_reason,
        usage,
    })
}
```

- [ ] **Step 3: Implement server**

Replace `crates/gemma4-api/src/server.rs`:

```rust
use axum::routing::{get, post};
use axum::Router;
use gemma4_core::engine::EngineHandle;

use crate::handlers;

pub fn build_router(engine: EngineHandle) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handlers::chat::chat_completions))
        .route("/health", get(handlers::health::health))
        .with_state(engine)
}

pub async fn start_server(engine: EngineHandle, host: &str, port: u16) -> anyhow::Result<()> {
    let app = build_router(engine);
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p gemma4-api`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add crates/gemma4-api/
git commit -m "feat(api): add chat completions handler and axum server"
```

---

### Task 14: CLI Binary — Wire Everything Together

**Files:**
- Modify: `crates/gemma4runner/src/main.rs`
- Create: `crates/gemma4runner/src/cli.rs`

- [ ] **Step 1: Implement CLI**

Create `crates/gemma4runner/src/cli.rs`:

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gemma4runner")]
#[command(about = "Run Gemma 4 models with an OpenAI-compatible API")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the API server
    Serve {
        /// Path to model directory or HuggingFace model ID
        #[arg(long)]
        model: String,

        /// Host to listen on
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(long, default_value_t = 8080)]
        port: u16,

        /// Log level
        #[arg(long, default_value = "info")]
        log_level: String,

        /// Max queued inference requests
        #[arg(long, default_value_t = 64)]
        queue_depth: usize,
    },
}
```

- [ ] **Step 2: Implement main.rs**

Replace `crates/gemma4runner/src/main.rs`:

```rust
mod cli;

use std::path::PathBuf;
use anyhow::{Context, Result};
use clap::Parser;

use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            model,
            host,
            port,
            log_level,
            queue_depth,
        } => {
            // Initialize logging
            let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&log_level));
            tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .init();

            let model_path = PathBuf::from(&model);
            anyhow::ensure!(
                model_path.exists(),
                "Model path does not exist: {}",
                model_path.display()
            );

            tracing::info!("Loading model from {}", model_path.display());
            let engine = gemma4_core::engine::start_engine(&model_path, queue_depth)
                .context("Failed to start inference engine")?;

            tracing::info!("Starting server on {}:{}", host, port);
            gemma4_api::server::start_server(engine, &host, port).await?;

            Ok(())
        }
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p gemma4runner`
Expected: Compiles and produces a binary at `target/debug/gemma4runner`

- [ ] **Step 4: Verify CLI help works**

Run: `cargo run -p gemma4runner -- --help`
Expected: Shows help text with `serve` subcommand

Run: `cargo run -p gemma4runner -- serve --help`
Expected: Shows serve options (--model, --host, --port, etc.)

- [ ] **Step 5: Commit**

```bash
git add crates/gemma4runner/
git commit -m "feat(runner): add CLI binary with serve command"
```

---

### Task 15: Integration Test — API Types Serialization

**Files:**
- Create: `crates/gemma4-api/tests/types_test.rs`

- [ ] **Step 1: Write serialization round-trip tests**

Create `crates/gemma4-api/tests/types_test.rs`:

```rust
use gemma4_api::types::chat::*;
use gemma4_api::types::common::*;

#[test]
fn test_deserialize_minimal_request() {
    let json = r#"{
        "model": "gemma-4-e4b",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "gemma-4-e4b");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.temperature, Some(1.0)); // default
    assert_eq!(req.max_tokens, None);
}

#[test]
fn test_deserialize_full_request() {
    let json = r#"{
        "model": "gemma-4-e4b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 512,
        "seed": 42,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.top_p, Some(0.9));
    assert_eq!(req.top_k, Some(40));
    assert_eq!(req.max_tokens, Some(512));
    assert_eq!(req.seed, Some(42));
}

#[test]
fn test_serialize_response() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-123".into(),
        object: "chat.completion".into(),
        created: 1234567890,
        model: "gemma-4-e4b".into(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".into(),
                content: "Hello!".into(),
            },
            finish_reason: FinishReason::Stop,
        }],
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 1,
            total_tokens: 6,
        },
    };
    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["id"], "chatcmpl-123");
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["usage"]["total_tokens"], 6);
}

#[test]
fn test_serialize_error() {
    use gemma4_api::types::error::*;
    let err = ApiErrorResponse {
        error: ApiErrorBody {
            message: "Invalid temperature".into(),
            error_type: "invalid_request_error".into(),
            param: Some("temperature".into()),
            code: None,
        },
    };
    let json = serde_json::to_value(&err).unwrap();
    assert_eq!(json["error"]["message"], "Invalid temperature");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["param"], "temperature");
    assert!(json["error"]["code"].is_null());
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p gemma4-api`
Expected: All 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add crates/gemma4-api/tests/
git commit -m "test(api): add serialization round-trip tests for OpenAI types"
```

---

### Task 16: Full Build Verification and Cleanup

**Files:**
- Modify: `crates/gemma4-core/src/lib.rs` (final module list)

- [ ] **Step 1: Ensure lib.rs has all modules**

`crates/gemma4-core/src/lib.rs` should be:
```rust
pub mod attention;
pub mod chat_template;
pub mod config;
pub mod engine;
pub mod kv_cache;
pub mod loader;
pub mod mlp;
pub mod model;
pub mod rope;
pub mod sampling;
pub mod tokenizer;
```

- [ ] **Step 2: Run full workspace build**

Run: `cargo build`
Expected: Compiles with no errors

- [ ] **Step 3: Run all tests**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings (fix any that appear)

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: cleanup and fix clippy warnings"
```

- [ ] **Step 6: Verify binary runs**

Run: `cargo run -p gemma4runner -- --help`
Expected: Help output showing `serve` subcommand

Run: `cargo run -p gemma4runner -- serve --model /nonexistent --port 8080`
Expected: Error message about model path not existing (graceful error, not a panic)
