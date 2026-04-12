# Gemma4Runner Phase 5 — Production Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GGUF model loading, quantized inference, performance metrics, and production improvements.

**Architecture:** GGUF loading uses candle's `quantized::gguf_file::Content` and `QMatMul` for quantized matmul. A new `quantized_model.rs` mirrors the existing `model.rs` but uses `QMatMul` instead of `Linear` for weight matrices. The loader auto-detects GGUF vs safetensors. A `/metrics` endpoint exposes tok/s, latency, and queue depth.

**Tech Stack:** Same as Phase 4. Uses `candle_core::quantized::{gguf_file, QTensor, QMatMul, GgmlDType}` and `candle_transformers::quantized_nn`.

**Key Reference:** `candle-transformers/src/models/quantized_gemma3.rs` — closest existing quantized model.

---

### Task 1: GGUF Loader

**Files:**
- Create: `crates/gemma4-core/src/gguf_loader.rs`
- Modify: `crates/gemma4-core/src/loader.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Create GGUF loader**

Create `crates/gemma4-core/src/gguf_loader.rs`:

```rust
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
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
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
```

- [ ] **Step 2: Update loader.rs to detect GGUF**

In `crates/gemma4-core/src/loader.rs`, add a detection function:

```rust
/// Detect whether a model path is a GGUF file or a safetensors directory.
pub fn is_gguf_file(path: &Path) -> bool {
    path.is_file() && path.extension().map(|e| e == "gguf").unwrap_or(false)
}
```

- [ ] **Step 3: Add module to lib.rs**

Add `pub mod gguf_loader;` to lib.rs.

- [ ] **Step 4: Verify and commit**

Run: `cargo check -p gemma4-core`

```bash
git add crates/gemma4-core/
git commit -m "feat(core): add GGUF file loader with metadata-to-config extraction"
```

---

### Task 2: Quantized Model

**Files:**
- Create: `crates/gemma4-core/src/quantized_model.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

This mirrors `model.rs` but uses `QMatMul` instead of `Linear` for weight matrices, and `candle_transformers::quantized_nn::RmsNorm` for norms.

- [ ] **Step 1: Create quantized model**

Create `crates/gemma4-core/src/quantized_model.rs`:

```rust
use std::sync::Arc;
use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};
use candle_core::quantized::{QMatMul, QTensor};
use candle_nn::Module;

use crate::config::Gemma4TextConfig;
use crate::kv_cache::KvCache;
use crate::rope::{ProportionalRotaryEmbedding, RotaryEmbedding};

struct QRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    fn new(qtensor: QTensor, eps: f64) -> Result<Self> {
        let weight = qtensor.dequantize(&Device::Cpu)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = (&x * &x)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_norm = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let result = x_norm.broadcast_mul(&self.weight)?;
        result.to_dtype(dtype)
    }
}

struct QLinear {
    weight: QMatMul,
}

impl QLinear {
    fn new(qtensor: QTensor) -> Result<Self> {
        Ok(Self { weight: QMatMul::from_qtensor(qtensor)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.weight.forward(x)
    }
}

struct QGemmaMlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl QGemmaMlp {
    fn new(gguf: &crate::gguf_loader::GgufModel, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            gate_proj: QLinear::new(gguf.tensor(&format!("{}.ffn_gate.weight", prefix), device)?)?,
            up_proj: QLinear::new(gguf.tensor(&format!("{}.ffn_up.weight", prefix), device)?)?,
            down_proj: QLinear::new(gguf.tensor(&format!("{}.ffn_down.weight", prefix), device)?)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        self.down_proj.forward(&fused)
    }
}

struct QGemmaAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    is_sliding: bool,
    rotary_emb_local: Arc<RotaryEmbedding>,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
}

impl QGemmaAttention {
    fn new(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        rotary_emb_local: Arc<RotaryEmbedding>,
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        gguf: &crate::gguf_loader::GgufModel,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding_layer(layer_idx);
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.kv_heads_for_layer(layer_idx);

        Ok(Self {
            q_proj: QLinear::new(gguf.tensor(&format!("{}.attn_q.weight", prefix), device)?)?,
            k_proj: QLinear::new(gguf.tensor(&format!("{}.attn_k.weight", prefix), device)?)?,
            v_proj: QLinear::new(gguf.tensor(&format!("{}.attn_v.weight", prefix), device)?)?,
            o_proj: QLinear::new(gguf.tensor(&format!("{}.attn_output.weight", prefix), device)?)?,
            q_norm: QRmsNorm::new(gguf.tensor(&format!("{}.attn_q_norm.weight", prefix), device)?, cfg.rms_norm_eps)?,
            k_norm: QRmsNorm::new(gguf.tensor(&format!("{}.attn_k_norm.weight", prefix), device)?, cfg.rms_norm_eps)?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            is_sliding,
            rotary_emb_local,
            rotary_emb_global,
        })
    }

    // forward() follows same pattern as attention.rs but uses QLinear
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut crate::kv_cache::LayerKvCache,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let q = self.apply_head_norm(&q, &self.q_norm)?;
        let k = self.apply_head_norm(&k, &self.k_norm)?;

        let (q, k) = if self.is_sliding {
            self.rotary_emb_local.apply(&q, &k, seqlen_offset)?
        } else {
            self.rotary_emb_global.apply(&q, &k, seqlen_offset)?
        };

        let (k, v) = cache.append(&k, &v)?;

        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let output = attn_weights.matmul(&v)?;
        let output = output.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&output)
    }

    fn apply_head_norm(&self, x: &Tensor, norm: &QRmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let x = x.reshape((b * h * s, d))?;
        let x = norm.forward(&x)?;
        x.reshape((b, h, s, d))
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_groups == 1 { return Ok(x.clone()); }
        let (b, h, s, d) = x.dims4()?;
        x.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?.reshape((b, h * self.num_kv_groups, s, d))
    }
}

struct QDecoderLayer {
    self_attn: QGemmaAttention,
    mlp: QGemmaMlp,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
    pre_feedforward_layernorm: QRmsNorm,
    post_feedforward_layernorm: QRmsNorm,
}

impl QDecoderLayer {
    fn new(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        rotary_emb_local: Arc<RotaryEmbedding>,
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        gguf: &crate::gguf_loader::GgufModel,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);
        Ok(Self {
            self_attn: QGemmaAttention::new(cfg, layer_idx, rotary_emb_local, rotary_emb_global, gguf, &prefix, device)?,
            mlp: QGemmaMlp::new(gguf, &prefix, device)?,
            input_layernorm: QRmsNorm::new(gguf.tensor(&format!("{}.attn_norm.weight", &prefix), device)?, cfg.rms_norm_eps)?,
            post_attention_layernorm: QRmsNorm::new(gguf.tensor(&format!("{}.post_attention_norm.weight", &prefix), device)?, cfg.rms_norm_eps)?,
            pre_feedforward_layernorm: QRmsNorm::new(gguf.tensor(&format!("{}.ffn_norm.weight", &prefix), device)?, cfg.rms_norm_eps)?,
            post_feedforward_layernorm: QRmsNorm::new(gguf.tensor(&format!("{}.post_ffw_norm.weight", &prefix), device)?, cfg.rms_norm_eps)?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, cache: &mut crate::kv_cache::LayerKvCache, seqlen_offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, cache, seqlen_offset)?;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = (residual + x)?;

        let residual = &x;
        let ff = self.pre_feedforward_layernorm.forward(&x)?;
        let ff = self.mlp.forward(&ff)?;
        let ff = self.post_feedforward_layernorm.forward(&ff)?;
        let x = (residual + ff)?;
        Ok(x)
    }
}

pub struct QuantizedGemmaModel {
    embed_tokens: Tensor,  // dequantized embedding
    layers: Vec<QDecoderLayer>,
    norm: QRmsNorm,
    lm_head: QMatMul,
    final_logit_softcapping: Option<f64>,
    hidden_size: usize,
    cfg: Gemma4TextConfig,
}

impl QuantizedGemmaModel {
    pub fn from_gguf(gguf: &crate::gguf_loader::GgufModel, device: &Device) -> Result<Self> {
        let cfg = &gguf.config;

        // Embedding is dequantized to f32
        let embed_tokens = gguf.tensor("token_embd.weight", device)?
            .dequantize(device)?;

        let sliding_rope = cfg.rope_parameters.as_ref().and_then(|p| p.sliding_attention.as_ref());
        let full_rope = cfg.rope_parameters.as_ref().and_then(|p| p.full_attention.as_ref());

        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            DType::F32, cfg.head_dim,
            sliding_rope.and_then(|p| p.rope_theta).unwrap_or(10000.0),
            cfg.max_position_embeddings, device,
        )?);
        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            DType::F32, cfg.global_head_dim,
            full_rope.and_then(|p| p.rope_theta).unwrap_or(1000000.0),
            full_rope.and_then(|p| p.partial_rotary_factor).unwrap_or(0.25),
            cfg.max_position_embeddings, device,
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QDecoderLayer::new(
                cfg, i, rotary_emb_local.clone(), rotary_emb_global.clone(), gguf, device,
            )?);
        }

        let norm = QRmsNorm::new(gguf.tensor("output_norm.weight", device)?, cfg.rms_norm_eps)?;

        let lm_head = if cfg.tie_word_embeddings {
            QMatMul::Tensor(embed_tokens.clone())
        } else {
            QMatMul::from_qtensor(gguf.tensor("output.weight", device)?)?
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

    pub fn forward(&self, input_ids: &Tensor, cache: &mut KvCache, seqlen_offset: usize) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let embed = candle_nn::Embedding::new(self.embed_tokens.clone(), self.hidden_size);
        let mut x = embed.forward(input_ids)?;
        let scale = (self.hidden_size as f64).sqrt();
        x = (x * scale)?;

        let (sliding_mask, global_mask) = self.create_masks(seq_len, seqlen_offset, x.device())?;

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if self.cfg.is_sliding_layer(i) { sliding_mask.as_ref() } else { global_mask.as_ref() };
            x = layer.forward(&x, mask, cache.layer_mut(i), seqlen_offset)?;
        }

        let x = x.narrow(1, seq_len - 1, 1)?;
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        match self.final_logit_softcapping {
            Some(cap) => Ok(((logits / cap)?.tanh()? * cap)?),
            None => Ok(logits),
        }
    }

    fn create_masks(&self, seq_len: usize, offset: usize, device: &Device) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len == 1 { return Ok((None, None)); }
        let mask: Vec<f32> = (0..seq_len).flat_map(|i| {
            (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
        }).collect();
        let causal = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?;

        let sw = self.cfg.sliding_window;
        let sliding_mask: Vec<f32> = (0..seq_len).flat_map(|i| {
            (0..seq_len).map(move |j| {
                let pi = i + offset; let pj = j + offset;
                if j > i || (pi >= sw && pj < pi - sw + 1) { f32::NEG_INFINITY } else { 0.0 }
            })
        }).collect();
        let sliding = Tensor::from_vec(sliding_mask, (1, 1, seq_len, seq_len), device)?;
        Ok((Some(sliding), Some(causal)))
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

Add `pub mod quantized_model;` to lib.rs.

- [ ] **Step 3: Verify and commit**

```bash
git add crates/gemma4-core/
git commit -m "feat(core): add quantized model for GGUF inference"
```

---

### Task 3: Integrate GGUF into Engine

**Files:**
- Modify: `crates/gemma4-core/src/engine.rs`
- Modify: `crates/gemma4-core/src/loader.rs`

- [ ] **Step 1: Add ModelBackend enum to engine**

The engine needs to support both safetensors and GGUF models:

```rust
enum ModelBackend {
    Safetensors(crate::model::GemmaTextModel),
    Quantized(crate::quantized_model::QuantizedGemmaModel),
}

impl ModelBackend {
    fn forward(&self, input_ids: &Tensor, cache: &mut KvCache, seqlen_offset: usize) -> Result<Tensor> {
        match self {
            ModelBackend::Safetensors(m) => m.forward(input_ids, cache, seqlen_offset),
            ModelBackend::Quantized(m) => m.forward(input_ids, cache, seqlen_offset),
        }
    }
}
```

- [ ] **Step 2: Update start_engine to detect GGUF**

```rust
pub fn start_engine(model_path: &Path, device: Device, queue_depth: usize) -> Result<EngineHandle> {
    let (model, config) = if loader::is_gguf_file(model_path) {
        let gguf = crate::gguf_loader::GgufModel::load(model_path, &device)?;
        let config = /* build Gemma4Config from gguf.config */;
        let model = ModelBackend::Quantized(
            crate::quantized_model::QuantizedGemmaModel::from_gguf(&gguf, &device)?
        );
        (model, config)
    } else {
        // existing safetensors path
        let dtype = match &device { Device::Cpu => DType::F32, _ => DType::BF16 };
        let loaded = loader::load_model(model_path, &device, dtype)?;
        (ModelBackend::Safetensors(loaded.model), loaded.config)
    };
    // ... rest of start_engine
}
```

- [ ] **Step 3: Update process_request to use ModelBackend**

Replace `model.forward(...)` calls with `model.forward(...)` via the enum dispatch.

- [ ] **Step 4: Verify and commit**

```bash
git add crates/gemma4-core/
git commit -m "feat(core): integrate GGUF loading into inference engine"
```

---

### Task 4: Metrics Endpoint

**Files:**
- Create: `crates/gemma4-api/src/metrics.rs`
- Modify: `crates/gemma4-api/src/handlers/mod.rs`
- Modify: `crates/gemma4-api/src/server.rs`
- Modify: `crates/gemma4-api/src/lib.rs`

- [ ] **Step 1: Create metrics state**

Create `crates/gemma4-api/src/metrics.rs`:

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use serde::Serialize;

#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    total_requests: AtomicU64,
    total_prompt_tokens: AtomicU64,
    total_completion_tokens: AtomicU64,
    total_inference_ms: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                total_requests: AtomicU64::new(0),
                total_prompt_tokens: AtomicU64::new(0),
                total_completion_tokens: AtomicU64::new(0),
                total_inference_ms: AtomicU64::new(0),
            }),
        }
    }

    pub fn record_request(&self, prompt_tokens: u64, completion_tokens: u64, inference_ms: u64) {
        self.inner.total_requests.fetch_add(1, Ordering::Relaxed);
        self.inner.total_prompt_tokens.fetch_add(prompt_tokens, Ordering::Relaxed);
        self.inner.total_completion_tokens.fetch_add(completion_tokens, Ordering::Relaxed);
        self.inner.total_inference_ms.fetch_add(inference_ms, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let total_requests = self.inner.total_requests.load(Ordering::Relaxed);
        let total_completion_tokens = self.inner.total_completion_tokens.load(Ordering::Relaxed);
        let total_inference_ms = self.inner.total_inference_ms.load(Ordering::Relaxed);

        let avg_tokens_per_sec = if total_inference_ms > 0 {
            (total_completion_tokens as f64 / total_inference_ms as f64) * 1000.0
        } else {
            0.0
        };

        MetricsSnapshot {
            total_requests,
            total_prompt_tokens: self.inner.total_prompt_tokens.load(Ordering::Relaxed),
            total_completion_tokens,
            total_inference_ms,
            avg_tokens_per_sec,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_inference_ms: u64,
    pub avg_tokens_per_sec: f64,
}
```

- [ ] **Step 2: Create metrics handler**

Add handler that returns the metrics snapshot:

```rust
// In a new handler file or inline
pub async fn get_metrics(State(metrics): State<Metrics>) -> Json<MetricsSnapshot> {
    Json(metrics.snapshot())
}
```

- [ ] **Step 3: Wire into server**

Add `.route("/metrics", get(metrics_handler))` to the router. Pass `Metrics` as shared state.

- [ ] **Step 4: Verify and commit**

```bash
git add crates/gemma4-api/
git commit -m "feat(api): add /metrics endpoint with token throughput stats"
```

---

### Task 5: Full Verification

- [ ] **Step 1: cargo build**
- [ ] **Step 2: cargo test --workspace**  
- [ ] **Step 3: Verify serve --help shows all options**
- [ ] **Step 4: Verify info --help**
- [ ] **Step 5: Commit cleanup**

```bash
git add -A
git commit -m "chore: Phase 5 cleanup and verification"
```
