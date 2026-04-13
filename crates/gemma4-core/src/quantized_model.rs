//! Quantized Gemma 4 model implementation.
//!
//! Closely follows the patterns from candle-transformers' `quantized_gemma3.rs`
//! (which works correctly with GGUF files) with minimal adaptations for Gemma 4
//! differences: GeluPytorchTanh activation, per-layer head dimensions, per-layer
//! RoPE, logit softcapping, and PLE support.

use std::sync::Arc;
use anyhow::Result;
use candle_core::{Device, DType, IndexOp, Tensor, D};
use candle_core::quantized::{QMatMul, QTensor};
use candle_nn::Module;

use crate::config::Gemma4TextConfig;
use crate::kv_cache::KvCache;

// ---------------------------------------------------------------------------
// TensorLoader trait
// ---------------------------------------------------------------------------

/// Trait for loading tensors by name. Implemented by GgufModel (Task 1) and
/// any test stub that needs to supply tensors.
pub trait TensorLoader {
    fn load(&self, name: &str, device: &Device) -> Result<QTensor>;

    /// Dequantize a tensor and ensure the result lives on `device`.
    ///
    /// `QTensor::dequantize` internally calls `.to_device(device)`, but we add
    /// an explicit `.to_device()` as a defensive measure for GPU backends
    /// (CUDA, Metal) where a mismatch between QTensor storage location and
    /// the target device would cause silent errors or panics.
    fn load_dequantized(&self, name: &str, device: &Device) -> Result<Tensor> {
        let tensor = self.load(name, device)?
            .dequantize(device)?;
        // Defensive: ensure the result is on the target device even if
        // dequantize returned a CPU tensor (e.g. for non-quantized dtypes).
        if tensor.device().location() != device.location() {
            tensor.to_device(device).map_err(Into::into)
        } else {
            Ok(tensor)
        }
    }
}

// ---------------------------------------------------------------------------
// QRmsNorm — dequantizes weight on load, applies RMS norm in f32
// ---------------------------------------------------------------------------

struct QRmsNorm {
    weight: Tensor,
    eps: f32,
}

impl QRmsNorm {
    fn new(qtensor: QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        // Defensive: ensure the norm weight lives on the target device.
        let weight = if weight.device().location() != device.location() {
            weight.to_device(device)?
        } else {
            weight
        };
        Ok(Self { weight, eps: eps as f32 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// QLinear — wraps QMatMul (matches QMatMul in quantized_gemma3.rs)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct QLinear {
    inner: QMatMul,
}

impl QLinear {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.inner.forward(x)?)
    }
}

// ---------------------------------------------------------------------------
// PleGlobal — per-layer embedding global precomputation (E4B/E2B only)
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
        // Defensive: ensure PLE embedding is on the target device.
        let token_embd = if token_embd.device().location() != device.location() {
            token_embd.to_device(device)?
        } else {
            token_embd
        };
        let model_proj = QLinear::from_qtensor(loader.load("per_layer_model_proj.weight", device)?)?;
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
// RotaryEmbedding — follows quantized_gemma3.rs exactly
// ---------------------------------------------------------------------------

const MAX_SEQ_LEN: usize = 131072;

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    /// Partial RoPE: only rotate the first `rotary_dim` dimensions, pass through the rest.
    fn new_partial(head_dim: usize, partial_rotary_factor: f64, rope_frequency: f32, device: &Device) -> Result<(Self, usize)> {
        let rotary_dim = ((head_dim as f64 * partial_rotary_factor) as usize) & !1; // ensure even
        let emb = Self::new(rotary_dim, rope_frequency, device)?;
        Ok((emb, rotary_dim))
    }

    #[allow(dead_code)]
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// Mlp — GeluPytorchTanh for Gemma 4 (vs SiLU for Gemma 3)
// ---------------------------------------------------------------------------

struct Mlp {
    feed_forward_gate: QLinear,
    feed_forward_up: QLinear,
    feed_forward_down: QLinear,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let activated = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
        let gated = (activated * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// repeat_kv — GQA helper (same as quantized_gemma3.rs utils::repeat_kv)
// ---------------------------------------------------------------------------

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
    Ok(x
        .unsqueeze(2)?
        .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
        .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?)
}

// ---------------------------------------------------------------------------
// LayerWeights — follows quantized_gemma3.rs LayerWeights exactly
// ---------------------------------------------------------------------------

struct LayerWeights {
    // Attention
    attention_wq: QLinear,
    attention_wk: QLinear,
    attention_wv: QLinear,
    attention_wo: QLinear,

    // Q/K norms
    attention_q_norm: QRmsNorm,
    attention_k_norm: QRmsNorm,

    // Layer norms (pre/post attention, pre/post FFN)
    attention_norm: QRmsNorm,
    post_attention_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    post_ffn_norm: QRmsNorm,

    // MLP
    mlp: Mlp,

    // Attention geometry
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,

    // Sliding window
    sliding_window_size: Option<usize>,

    // RoPE
    rotary_embedding: Arc<RotaryEmbedding>,
    rotary_dim: usize,  // for partial RoPE (global layers)

    neg_inf: Tensor,

    // Pre-computed epsilon tensor on the target device (avoids CPU↔GPU transfers
    // when computing V-norm during attention).
    rms_norm_eps_tensor: Tensor,

    // RMS norm epsilon scalar value (retained for debugging; the pre-computed
    // tensor rms_norm_eps_tensor is used in forward_attn).
    #[allow(dead_code)]
    rms_norm_eps: f32,

    // Whether this layer computes its own KV (false for shared KV layers)
    has_kv: bool,

    // KV cache
    kv_cache: Option<(Tensor, Tensor)>,

    // PLE (optional, E4B/E2B)
    ple_gate: Option<QLinear>,
    ple_proj: Option<QLinear>,
    ple_post_norm: Option<QRmsNorm>,
    layer_output_scale: Option<Tensor>,

    // The device this layer's tensors live on (CPU, Metal, or CUDA).
    device: Device,
}

impl LayerWeights {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // Build mask as u32 (1 = attend, 0 = mask out) — matches gemma3 pattern
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        // For cached tokens, prepend ones (allow attending to all cached positions)
        let mask = if index_pos > 0 {
            let mask0 = Tensor::ones((seq_len, index_pos), DType::U32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(DType::U32)
            .map_err(Into::into)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        shared_kv: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head Q norm
        let q = self.attention_q_norm.forward(&q.contiguous()?)?;

        // RoPE on Q only (K gets RoPE only if this layer computes its own K)
        let q = if self.rotary_dim < self.head_dim {
            let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?;
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?;
            let cos = self.rotary_embedding.cos.narrow(0, index_pos, seq_len)?;
            let sin = self.rotary_embedding.sin.narrow(0, index_pos, seq_len)?;
            let q_rot = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
            Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?
        } else {
            let cos = self.rotary_embedding.cos.narrow(0, index_pos, seq_len)?;
            let sin = self.rotary_embedding.sin.narrow(0, index_pos, seq_len)?;
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?
        };

        // K/V: either compute from this layer's weights, or reuse from shared layer
        let (k, v) = if self.has_kv {
            // This layer computes its own K/V
            let k = self.attention_wk.forward(x)?;
            let v = self.attention_wv.forward(x)?;

            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;

            // V norm — RMS norm without learned weight (with_scale=False in HuggingFace,
            // raw ggml_rms_norm in llama.cpp). Normalizes each value head vector.
            //
            // The epsilon addition uses a pre-computed tensor on the target device
            // (self.rms_norm_eps_tensor) to avoid creating CPU scalars that could
            // trigger cross-device errors on CUDA/Metal.
            let v = {
                let v_f32 = v.to_dtype(DType::F32)?;
                let sq = v_f32.sqr()?;
                let mean_sq = (sq.sum_keepdim(D::Minus1)? / self.head_dim as f64)?;
                let rms = mean_sq.broadcast_add(&self.rms_norm_eps_tensor)?.sqrt()?;
                v_f32.broadcast_div(&rms)?.to_dtype(v.dtype())?
            };

            // K norm
            let k = self.attention_k_norm.forward(&k.contiguous()?)?;

            // RoPE on K
            let k = if self.rotary_dim < self.head_dim {
                let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?;
                let k_pass = k.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?;
                let cos = self.rotary_embedding.cos.narrow(0, index_pos, seq_len)?;
                let sin = self.rotary_embedding.sin.narrow(0, index_pos, seq_len)?;
                let k_rot = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;
                Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?
            } else {
                let cos = self.rotary_embedding.cos.narrow(0, index_pos, seq_len)?;
                let sin = self.rotary_embedding.sin.narrow(0, index_pos, seq_len)?;
                candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?
            };

            // KV cache update
            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((k_cache, v_cache)) => {
                    if index_pos == 0 {
                        (k, v)
                    } else {
                        let k = Tensor::cat(&[k_cache, &k], 2)?;
                        let v = Tensor::cat(&[v_cache, &v], 2)?;
                        (k, v)
                    }
                }
            };

            // Sliding window eviction
            let (k, v) = if let Some(window) = self.sliding_window_size {
                let kv_seq_len = k.dim(2)?;
                if kv_seq_len > window {
                    let start = kv_seq_len - window;
                    (k.narrow(2, start, window)?, v.narrow(2, start, window)?)
                } else {
                    (k, v)
                }
            } else {
                (k, v)
            };

            self.kv_cache = Some((k.clone(), v.clone()));
            (k, v)
        } else {
            // Shared KV layer — reuse from the reference layer's cache
            match shared_kv {
                Some((k, v)) => (k.clone(), v.clone()),
                None => anyhow::bail!("Shared KV layer {} has no reference KV cache", 0),
            }
        };

        // Repeat KV for GQA
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled dot-product attention
        // Gemma4 uses f_attention_scale = 1.0 (no scaling on QK).
        // Both llama.cpp and HuggingFace set self.scaling = 1.0 for Gemma 4.
        // The Q/K norms already normalize the vectors, so no 1/sqrt(d) is needed.
        // .contiguous() required for Metal/CUDA GPU kernels after transpose/cat/narrow.
        let q = q.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?;

        // Apply mask: eq(0u32)?.where_cond(&neg_inf, &attn_weights)?  — exact gemma3 pattern
        let attn_weights = if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let v = v.contiguous()?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, self.q_dim))?;

        self.attention_wo.forward(&attn_output)
    }
}

// ---------------------------------------------------------------------------
// QuantizedGemmaModel — public entry point
// ---------------------------------------------------------------------------

/// Full Gemma4 model loaded from a GGUF file via the `TensorLoader` trait.
pub struct QuantizedGemmaModel {
    embed_tokens: Tensor,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    lm_head: QMatMul,
    final_logit_softcapping: Option<f64>,
    hidden_size: usize,
    #[allow(dead_code)]
    cfg: Gemma4TextConfig,
    ple: Option<PleGlobal>,
    /// Number of layers from the start that have their own KV.
    /// Layers >= this index reuse KV from a reference layer.
    n_layer_kv_from_start: usize,
    /// The device all tensors live on (CPU, Metal, or CUDA).
    device: Device,
}

impl QuantizedGemmaModel {
    /// Build from pre-loaded quantized tensors.
    pub fn new(
        cfg: &Gemma4TextConfig,
        loader: &dyn TensorLoader,
        device: &Device,
    ) -> Result<Self> {
        let embed_tokens = loader.load_dequantized("token_embd.weight", device)?;
        tracing::debug!(
            "embed_tokens device: {:?}, dtype: {:?}",
            embed_tokens.device(),
            embed_tokens.dtype()
        );

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        // Pre-compute the RMS norm epsilon as a tensor on the target device.
        // This avoids creating CPU scalars during V-norm computation in
        // forward_attn, which would cause cross-device errors on CUDA/Metal.
        let rms_norm_eps_tensor = Tensor::new(cfg.rms_norm_eps as f32, device)?;

        // Build per-layer RoPE embeddings (different head_dim and freq per layer type)
        // Sliding layers: full rotation, local freq base
        // Global layers: partial rotation, global freq base
        let sliding_rope_freq = cfg.rope_parameters.as_ref()
            .and_then(|p| p.sliding_attention.as_ref())
            .and_then(|p| p.rope_theta)
            .unwrap_or(10_000.0) as f32;
        let global_rope_freq = cfg.rope_parameters.as_ref()
            .and_then(|p| p.full_attention.as_ref())
            .and_then(|p| p.rope_theta)
            .unwrap_or(1_000_000.0) as f32;
        let global_partial_rotary_factor = cfg.rope_parameters.as_ref()
            .and_then(|p| p.full_attention.as_ref())
            .and_then(|p| p.partial_rotary_factor)
            .unwrap_or(1.0);

        // Pre-build shared RoPE embeddings (one per distinct config)
        let sliding_head_dim = cfg.head_dim;
        let global_head_dim = if cfg.global_head_dim > 0 { cfg.global_head_dim } else { cfg.head_dim };

        let sliding_rope = Arc::new(RotaryEmbedding::new(sliding_head_dim, sliding_rope_freq, device)?);
        let (global_rope_inner, global_rotary_dim) = RotaryEmbedding::new_partial(
            global_head_dim, global_partial_rotary_factor, global_rope_freq, device,
        )?;
        let global_rope = Arc::new(global_rope_inner);

        let norm = QRmsNorm::new(
            loader.load("output_norm.weight", device)?,
            cfg.rms_norm_eps,
            device,
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            QMatMul::Tensor(embed_tokens.clone())
        } else {
            QMatMul::from_qtensor(loader.load("output.weight", device)?)?
        };

        let ple = PleGlobal::new(cfg, loader, device)?;

        // KV sharing: last N layers reuse KV from earlier layers
        let n_layer_kv_from_start = cfg.num_hidden_layers - cfg.num_kv_shared_layers;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{layer_idx}");
            let is_sliding = cfg.is_sliding_layer(layer_idx);
            let head_dim = cfg.head_dim_for_layer(layer_idx);
            let n_kv_head = cfg.kv_heads_for_layer(layer_idx);
            let n_head = cfg.num_attention_heads;
            let q_dim = n_head * head_dim;

            let attention_wq = QLinear::from_qtensor(loader.load(&format!("{prefix}.attn_q.weight"), device)?)?;
            let attention_wk = QLinear::from_qtensor(loader.load(&format!("{prefix}.attn_k.weight"), device)?)?;
            let attention_wv = QLinear::from_qtensor(loader.load(&format!("{prefix}.attn_v.weight"), device)?)?;
            let attention_wo = QLinear::from_qtensor(loader.load(&format!("{prefix}.attn_output.weight"), device)?)?;

            let attention_q_norm = QRmsNorm::new(
                loader.load(&format!("{prefix}.attn_q_norm.weight"), device)?,
                cfg.rms_norm_eps, device,
            )?;
            let attention_k_norm = QRmsNorm::new(
                loader.load(&format!("{prefix}.attn_k_norm.weight"), device)?,
                cfg.rms_norm_eps, device,
            )?;

            let attention_norm = QRmsNorm::new(
                loader.load(&format!("{prefix}.attn_norm.weight"), device)?,
                cfg.rms_norm_eps, device,
            )?;
            let post_attention_norm = QRmsNorm::new(
                loader.load(&format!("{prefix}.post_attention_norm.weight"), device)?,
                cfg.rms_norm_eps, device,
            )?;
            let ffn_norm = QRmsNorm::new(
                loader.load(&format!("{prefix}.ffn_norm.weight"), device)?,
                cfg.rms_norm_eps, device,
            )?;
            let post_ffn_norm = QRmsNorm::new(
                loader.load(&format!("{prefix}.post_ffw_norm.weight"), device)?,
                cfg.rms_norm_eps, device,
            )?;

            let mlp = Mlp {
                feed_forward_gate: QLinear::from_qtensor(loader.load(&format!("{prefix}.ffn_gate.weight"), device)?)?,
                feed_forward_up: QLinear::from_qtensor(loader.load(&format!("{prefix}.ffn_up.weight"), device)?)?,
                feed_forward_down: QLinear::from_qtensor(loader.load(&format!("{prefix}.ffn_down.weight"), device)?)?,
            };

            let sliding_window_size = if is_sliding { Some(cfg.sliding_window) } else { None };
            let (rotary_embedding, rotary_dim) = if is_sliding {
                (sliding_rope.clone(), sliding_head_dim)
            } else {
                (global_rope.clone(), global_rotary_dim)
            };

            // PLE components (optional)
            let (ple_gate, ple_proj, ple_post_norm, layer_output_scale) = if cfg.hidden_size_per_layer_input > 0 {
                let gate = QLinear::from_qtensor(loader.load(&format!("{prefix}.inp_gate.weight"), device)?)?;
                let proj = QLinear::from_qtensor(loader.load(&format!("{prefix}.proj.weight"), device)?)?;
                let post_norm = QRmsNorm::new(
                    loader.load(&format!("{prefix}.post_norm.weight"), device)?,
                    cfg.rms_norm_eps, device,
                )?;
                let scale = loader.load(&format!("{prefix}.layer_output_scale.weight"), device)?
                    .dequantize(device)?;
                // Defensive: ensure the scale tensor is on the target device.
                let scale = if scale.device().location() != device.location() {
                    scale.to_device(device)?
                } else {
                    scale
                };
                (Some(gate), Some(proj), Some(post_norm), Some(scale))
            } else {
                (None, None, None, None)
            };

            layers.push(LayerWeights {
                attention_wq,
                attention_wk,
                attention_wv,
                attention_wo,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head,
                n_kv_head,
                head_dim,
                q_dim,
                sliding_window_size,
                rotary_embedding,
                rotary_dim,
                neg_inf: neg_inf.clone(),
                rms_norm_eps_tensor: rms_norm_eps_tensor.clone(),
                rms_norm_eps: cfg.rms_norm_eps as f32,
                has_kv: layer_idx < n_layer_kv_from_start,
                kv_cache: None,
                ple_gate,
                ple_proj,
                ple_post_norm,
                layer_output_scale,
                device: device.clone(),
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            hidden_size: cfg.hidden_size,
            cfg: cfg.clone(),
            ple,
            n_layer_kv_from_start,
            device: device.clone(),
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        _cache: &mut KvCache,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;

        // Ensure input_ids are on the model's device (defensive for GPU backends).
        let input_ids = if input_ids.device().location() != self.device.location() {
            std::borrow::Cow::Owned(input_ids.to_device(&self.device)?)
        } else {
            std::borrow::Cow::Borrowed(input_ids)
        };

        // Embedding lookup + scale (matches gemma3: tok_embeddings * sqrt(hidden_size))
        let embed = candle_nn::Embedding::new(self.embed_tokens.clone(), self.hidden_size);
        let mut layer_in = embed.forward(&input_ids)?;
        layer_in = (layer_in * (self.hidden_size as f64).sqrt())?;

        // PLE global precomputation (E4B/E2B only)
        let per_layer_inputs = match &self.ple {
            Some(ple) => Some(ple.precompute(&input_ids, &layer_in)?),
            None => None,
        };

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];

            // Attention mask — always created on the layer's device to avoid
            // cross-device errors on CUDA/Metal.
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, seqlen_offset, &layer.device)?)
            };

            // For shared KV layers, clone KV cache from the reference layer
            let shared_kv = if !self.layers[layer_idx].has_kv {
                let is_sliding = self.layers[layer_idx].sliding_window_size.is_some();
                let ref_layer = if is_sliding {
                    self.n_layer_kv_from_start.saturating_sub(2)
                } else {
                    self.n_layer_kv_from_start.saturating_sub(1)
                };
                self.layers[ref_layer].kv_cache.clone()
            } else {
                None
            };

            let layer = &mut self.layers[layer_idx];

            // Attention block: norm → attn → post_norm → residual
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), seqlen_offset, shared_kv.as_ref())?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block: norm → mlp → post_norm → residual
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;

            // PLE sub-block (only for E4B/E2B)
            let x = if let (Some(gate), Some(proj), Some(post_norm)) =
                (&layer.ple_gate, &layer.ple_proj, &layer.ple_post_norm)
            {
                if let Some(ref pli) = per_layer_inputs {
                    let ple_input = pli.narrow(2, layer_idx, 1)?.squeeze(2)?;
                    let residual = &x;
                    let gated = gate.forward(&x)?;
                    let gated = candle_nn::Activation::GeluPytorchTanh.forward(&gated)?;
                    let gated = (gated * ple_input)?;
                    let projected = proj.forward(&gated)?;
                    let normed = post_norm.forward(&projected)?;
                    (residual + normed)?
                } else {
                    x
                }
            } else {
                x
            };

            // Layer output scale — applied unconditionally after PLE (matches HuggingFace)
            let x = if let Some(scale) = &layer.layer_output_scale {
                x.broadcast_mul(scale)?
            } else {
                x
            };

            layer_in = x;
        }

        // Final norm + output projection (take last token only)
        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;

        // Logit softcapping: tanh(logits / cap) * cap
        match self.final_logit_softcapping {
            Some(cap) => Ok(((logits / cap)?.tanh()? * cap)?),
            None => Ok(logits),
        }
    }
}
