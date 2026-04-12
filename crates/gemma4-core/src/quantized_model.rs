use std::sync::Arc;
use anyhow::Result;
use candle_core::{Device, DType, Tensor, D};
use candle_core::quantized::{QMatMul, QTensor};
use candle_nn::Module;

use crate::config::Gemma4TextConfig;
use crate::kv_cache::KvCache;
use crate::rope::{ProportionalRotaryEmbedding, RotaryEmbedding};

// ---------------------------------------------------------------------------
// TensorLoader trait
// ---------------------------------------------------------------------------

/// Trait for loading tensors by name. Implemented by GgufModel (Task 1) and
/// any test stub that needs to supply tensors.
pub trait TensorLoader {
    fn load(&self, name: &str, device: &Device) -> Result<QTensor>;

    fn load_dequantized(&self, name: &str, device: &Device) -> Result<Tensor> {
        self.load(name, device)?
            .dequantize(device)
            .map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// QRmsNorm — dequantizes weight on load, applies RMS norm in f32
// ---------------------------------------------------------------------------

struct QRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    fn new(qtensor: QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = (&x_f32 * &x_f32)?.mean_keepdim(D::Minus1)?;
        let x_norm = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight = self.weight.to_dtype(DType::F32)?;
        let result = x_norm.broadcast_mul(&weight)?;
        Ok(result.to_dtype(dtype)?)
    }
}

// ---------------------------------------------------------------------------
// QLinear — wraps QMatMul
// ---------------------------------------------------------------------------

struct QLinear {
    weight: QMatMul,
}

impl QLinear {
    fn new(qtensor: QTensor) -> Result<Self> {
        Ok(Self {
            weight: QMatMul::from_qtensor(qtensor)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.weight.forward(x)?)
    }
}

// ---------------------------------------------------------------------------
// PleGlobal — per-layer embedding global precomputation
// ---------------------------------------------------------------------------

struct PleGlobal {
    token_embd: Tensor,    // dequantized [vocab_size, num_layers * ple_dim]
    model_proj: QLinear,   // [num_layers * ple_dim, hidden_size]
    proj_norm: QRmsNorm,   // [ple_dim]
    ple_dim: usize,
    num_layers: usize,
    hidden_size: usize,
}

impl PleGlobal {
    fn new(cfg: &Gemma4TextConfig, loader: &dyn TensorLoader, device: &Device) -> Result<Option<Self>> {
        if cfg.hidden_size_per_layer_input == 0 {
            return Ok(None);
        }
        let ple_dim = cfg.hidden_size_per_layer_input;

        let token_embd = loader.load("per_layer_token_embd.weight", device)?
            .dequantize(device)?;
        let model_proj = QLinear::new(loader.load("per_layer_model_proj.weight", device)?)?;
        let proj_norm = QRmsNorm::new(
            loader.load("per_layer_proj_norm.weight", device)?,
            cfg.rms_norm_eps,
            device,
        )?;

        Ok(Some(Self {
            token_embd,
            model_proj,
            proj_norm,
            ple_dim,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
        }))
    }

    /// Precompute per-layer inputs from input_ids and the main embedding.
    /// Returns tensor of shape [batch, seq_len, num_layers, ple_dim]
    fn precompute(&self, input_ids: &Tensor, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Token identity signal: embed through per-layer table
        let embed = candle_nn::Embedding::new(self.token_embd.clone(), self.ple_dim * self.num_layers);
        let per_layer_tok = embed.forward(input_ids)?; // [batch, seq, num_layers * ple_dim]
        let scale = (self.ple_dim as f64).sqrt();
        let per_layer_tok = (per_layer_tok * scale)?;
        let per_layer_tok = per_layer_tok.reshape((batch_size, seq_len, self.num_layers, self.ple_dim))?;

        // Contextual signal: project main embedding
        let per_layer_proj = self.model_proj.forward(inputs_embeds)?; // [batch, seq, num_layers * ple_dim]
        let proj_scale = (self.hidden_size as f64).powf(-0.5);
        let per_layer_proj = (per_layer_proj * proj_scale)?;
        let per_layer_proj = per_layer_proj.reshape((batch_size * seq_len * self.num_layers, self.ple_dim))?;
        let per_layer_proj = self.proj_norm.forward(&per_layer_proj)?;
        let per_layer_proj = per_layer_proj.reshape((batch_size, seq_len, self.num_layers, self.ple_dim))?;

        // Combine: (tok + proj) * 1/sqrt(2)
        let combined = (per_layer_tok + per_layer_proj)?;
        let input_scale = (2.0f64).powf(-0.5);
        let combined = (combined * input_scale)?;

        Ok(combined) // [batch, seq_len, num_layers, ple_dim]
    }
}

// ---------------------------------------------------------------------------
// QGemmaMlp — gate/up/down projections using QLinear
// ---------------------------------------------------------------------------

struct QGemmaMlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl QGemmaMlp {
    fn new(loader: &dyn TensorLoader, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            gate_proj: QLinear::new(loader.load(&format!("{prefix}.ffn_gate.weight"), device)?)?,
            up_proj:   QLinear::new(loader.load(&format!("{prefix}.ffn_up.weight"),   device)?)?,
            down_proj: QLinear::new(loader.load(&format!("{prefix}.ffn_down.weight"), device)?)?,
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

// ---------------------------------------------------------------------------
// QGemmaAttention — same as attention.rs but with QLinear / QRmsNorm
// ---------------------------------------------------------------------------

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
        loader: &dyn TensorLoader,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding_layer(layer_idx);
        let head_dim = cfg.head_dim_for_layer(layer_idx);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.kv_heads_for_layer(layer_idx);

        Ok(Self {
            q_proj: QLinear::new(loader.load(&format!("{prefix}.attn_q.weight"),      device)?)?,
            k_proj: QLinear::new(loader.load(&format!("{prefix}.attn_k.weight"),      device)?)?,
            v_proj: QLinear::new(loader.load(&format!("{prefix}.attn_v.weight"),      device)?)?,
            o_proj: QLinear::new(loader.load(&format!("{prefix}.attn_output.weight"), device)?)?,
            q_norm: QRmsNorm::new(loader.load(&format!("{prefix}.attn_q_norm.weight"), device)?, cfg.rms_norm_eps, device)?,
            k_norm: QRmsNorm::new(loader.load(&format!("{prefix}.attn_k_norm.weight"), device)?, cfg.rms_norm_eps, device)?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            is_sliding,
            rotary_emb_local,
            rotary_emb_global,
        })
    }

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

        let q = q.reshape((batch_size, seq_len, self.num_heads,    self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Per-head RMS norm
        let q = self.apply_head_norm(&q, &self.q_norm)?;
        let k = self.apply_head_norm(&k, &self.k_norm)?;

        // RoPE
        let (q, k) = if self.is_sliding {
            self.rotary_emb_local.apply(&q, &k, seqlen_offset)?
        } else {
            self.rotary_emb_global.apply(&q, &k, seqlen_offset)?
        };

        // KV cache
        let (k, v) = cache.append(&k, &v)?;

        // Repeat KV for GQA
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
        let output = output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&output)
    }

    fn apply_head_norm(&self, x: &Tensor, norm: &QRmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let x = x.reshape((b * h * s, d))?;
        let x = norm.forward(&x)?;
        Ok(x.reshape((b, h, s, d))?)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_groups == 1 {
            return Ok(x.clone());
        }
        let (b, h, s, d) = x.dims4()?;
        Ok(x.unsqueeze(2)?
            .expand((b, h, self.num_kv_groups, s, d))?
            .reshape((b, h * self.num_kv_groups, s, d))?)
    }
}

// ---------------------------------------------------------------------------
// QDecoderLayer
// ---------------------------------------------------------------------------

struct QDecoderLayer {
    self_attn: QGemmaAttention,
    mlp: QGemmaMlp,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
    pre_feedforward_layernorm: QRmsNorm,
    post_feedforward_layernorm: QRmsNorm,
    // PLE fields (optional, only for E4B/E2B)
    ple_gate: Option<QLinear>,          // inp_gate: [ple_dim, hidden_size]
    ple_proj: Option<QLinear>,          // proj: [hidden_size, ple_dim]
    ple_post_norm: Option<QRmsNorm>,    // post_norm
    layer_output_scale: Option<Tensor>, // [1] scalar
}

impl QDecoderLayer {
    fn new(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        rotary_emb_local: Arc<RotaryEmbedding>,
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        loader: &dyn TensorLoader,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let eps = cfg.rms_norm_eps;

        let (ple_gate, ple_proj, ple_post_norm, layer_output_scale) = if cfg.hidden_size_per_layer_input > 0 {
            let gate = QLinear::new(loader.load(&format!("{prefix}.inp_gate.weight"), device)?)?;
            let proj = QLinear::new(loader.load(&format!("{prefix}.proj.weight"), device)?)?;
            let post_norm = QRmsNorm::new(
                loader.load(&format!("{prefix}.post_norm.weight"), device)?,
                eps, device,
            )?;
            let scale = loader.load(&format!("{prefix}.layer_output_scale.weight"), device)?
                .dequantize(device)?;
            (Some(gate), Some(proj), Some(post_norm), Some(scale))
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            self_attn: QGemmaAttention::new(
                cfg, layer_idx,
                rotary_emb_local, rotary_emb_global,
                loader, &prefix, device,
            )?,
            mlp: QGemmaMlp::new(loader, &prefix, device)?,
            input_layernorm: QRmsNorm::new(
                loader.load(&format!("{prefix}.attn_norm.weight"), device)?,
                eps, device,
            )?,
            post_attention_layernorm: QRmsNorm::new(
                loader.load(&format!("{prefix}.post_attention_norm.weight"), device)?,
                eps, device,
            )?,
            pre_feedforward_layernorm: QRmsNorm::new(
                loader.load(&format!("{prefix}.ffn_norm.weight"), device)?,
                eps, device,
            )?,
            post_feedforward_layernorm: QRmsNorm::new(
                loader.load(&format!("{prefix}.post_ffw_norm.weight"), device)?,
                eps, device,
            )?,
            ple_gate,
            ple_proj,
            ple_post_norm,
            layer_output_scale,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut crate::kv_cache::LayerKvCache,
        seqlen_offset: usize,
        per_layer_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Attention sub-layer with pre/post norm and residual
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, cache, seqlen_offset)?;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = (residual + x)?;

        // Feed-forward sub-layer with pre/post norm and residual
        let residual = &x;
        let ff = self.pre_feedforward_layernorm.forward(&x)?;
        let ff = self.mlp.forward(&ff)?;
        let ff = self.post_feedforward_layernorm.forward(&ff)?;
        let x = (residual + ff)?;

        // PLE sub-block (only active for E4B/E2B when per_layer_input is provided)
        let x = if let (Some(gate), Some(proj), Some(post_norm), Some(scale), Some(ple_input)) =
            (&self.ple_gate, &self.ple_proj, &self.ple_post_norm, &self.layer_output_scale, per_layer_input)
        {
            let residual = &x;
            let gated = gate.forward(&x)?;                                          // [batch, seq, ple_dim]
            let gated = candle_nn::Activation::GeluPytorchTanh.forward(&gated)?;
            let gated = (gated * ple_input)?;                                       // element-wise multiply
            let projected = proj.forward(&gated)?;                                  // [batch, seq, hidden_size]
            let normed = post_norm.forward(&projected)?;
            let x = (residual + normed)?;
            x.broadcast_mul(scale)?
        } else {
            x
        };

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// QuantizedGemmaModel — public entry point
// ---------------------------------------------------------------------------

/// Full Gemma4 model loaded from a GGUF file via the `TensorLoader` trait.
pub struct QuantizedGemmaModel {
    /// Dequantized embedding table (f32 on CPU, reused as tied lm_head).
    embed_tokens: Tensor,
    layers: Vec<QDecoderLayer>,
    norm: QRmsNorm,
    /// LM head — either `QMatMul::Tensor(embed_tokens)` (tied) or a separate
    /// quantized projection.
    lm_head: QMatMul,
    final_logit_softcapping: Option<f64>,
    hidden_size: usize,
    cfg: Gemma4TextConfig,
    /// Per-Layer Embeddings global state (E4B/E2B only).
    ple: Option<PleGlobal>,
}

impl QuantizedGemmaModel {
    /// Build from pre-loaded quantized tensors.
    ///
    /// In practice this is called by the engine after loading GGUF via
    /// `GgufModel` (which implements `TensorLoader`). Any stub that implements
    /// `TensorLoader` works too, making the type unit-testable without disk I/O.
    pub fn new(
        cfg: &Gemma4TextConfig,
        loader: &dyn TensorLoader,
        device: &Device,
    ) -> Result<Self> {
        // Embedding — dequantize to f32 so matmul with integer token ids works.
        let embed_tokens = loader.load_dequantized("token_embd.weight", device)?;

        let sliding_rope = cfg.rope_parameters.as_ref().and_then(|p| p.sliding_attention.as_ref());
        let full_rope    = cfg.rope_parameters.as_ref().and_then(|p| p.full_attention.as_ref());

        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            DType::F32,
            cfg.head_dim,
            sliding_rope.and_then(|p| p.rope_theta).unwrap_or(10_000.0),
            cfg.max_position_embeddings,
            device,
        )?);
        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            DType::F32,
            cfg.global_head_dim.max(cfg.head_dim), // fall back to head_dim if 0
            full_rope.and_then(|p| p.rope_theta).unwrap_or(1_000_000.0),
            full_rope.and_then(|p| p.partial_rotary_factor).unwrap_or(0.25),
            cfg.max_position_embeddings,
            device,
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QDecoderLayer::new(
                cfg, i,
                rotary_emb_local.clone(),
                rotary_emb_global.clone(),
                loader, device,
            )?);
        }

        let norm = QRmsNorm::new(
            loader.load("output_norm.weight", device)?,
            cfg.rms_norm_eps,
            device,
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            // Reuse the dequantized embedding as the output projection.
            QMatMul::Tensor(embed_tokens.clone())
        } else {
            QMatMul::from_qtensor(loader.load("output.weight", device)?)?
        };

        let ple = PleGlobal::new(cfg, loader, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            hidden_size: cfg.hidden_size,
            cfg: cfg.clone(),
            ple,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        cache: &mut KvCache,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Embedding lookup + scale
        let embed = candle_nn::Embedding::new(self.embed_tokens.clone(), self.hidden_size);
        let mut x = embed.forward(input_ids)?;
        let scale = (self.hidden_size as f64).sqrt();
        x = (x * scale)?;

        // PLE global precomputation (E4B/E2B only)
        let per_layer_inputs = match &self.ple {
            Some(ple) => Some(ple.precompute(input_ids, &x)?),
            None => None,
        };

        let (sliding_mask, global_mask) =
            self.create_masks(seq_len, seqlen_offset, x.device())?;

        for (i, layer) in self.layers.iter().enumerate() {
            let ple_input = per_layer_inputs.as_ref().map(|pli| {
                pli.narrow(2, i, 1).unwrap().squeeze(2).unwrap() // [batch, seq, ple_dim]
            });
            let mask = if self.cfg.is_sliding_layer(i) {
                sliding_mask.as_ref()
            } else {
                global_mask.as_ref()
            };
            x = layer.forward(&x, mask, cache.layer_mut(i), seqlen_offset, ple_input.as_ref())?;
        }

        // Take only the last token's hidden state for decoding
        let x = x.narrow(1, seq_len - 1, 1)?;
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;

        match self.final_logit_softcapping {
            Some(cap) => Ok(((logits / cap)?.tanh()? * cap)?),
            None => Ok(logits),
        }
    }

    fn create_masks(
        &self,
        seq_len: usize,
        offset: usize,
        device: &Device,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len == 1 {
            return Ok((None, None));
        }

        // Standard causal mask
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect();
        let causal = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?;

        // Sliding-window causal mask
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
