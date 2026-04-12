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
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut crate::kv_cache::LayerKvCache,
        seqlen_offset: usize,
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
        Ok((residual + ff)?)
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

        let (sliding_mask, global_mask) =
            self.create_masks(seq_len, seqlen_offset, x.device())?;

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if self.cfg.is_sliding_layer(i) {
                sliding_mask.as_ref()
            } else {
                global_mask.as_ref()
            };
            x = layer.forward(&x, mask, cache.layer_mut(i), seqlen_offset)?;
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
