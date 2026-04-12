use std::sync::Arc;
use anyhow::Result;
use candle_core::{Tensor, D};
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
        cfg: &Gemma4TextConfig, layer_idx: usize,
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
            q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
            num_heads, num_kv_heads, num_kv_groups: num_heads / num_kv_heads,
            head_dim, is_sliding, rotary_emb_local, rotary_emb_global,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, cache: &mut LayerKvCache, seqlen_offset: usize) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
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
        let output = output.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        let output = self.o_proj.forward(&output)?;
        Ok(output)
    }

    fn apply_head_norm(&self, x: &Tensor, norm: &candle_nn::RmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let x = x.reshape((b * h * s, d))?;
        let x = norm.forward(&x)?;
        Ok(x.reshape((b, h, s, d))?)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_groups == 1 { return Ok(x.clone()); }
        let (b, h, s, d) = x.dims4()?;
        let x = x.unsqueeze(2)?.expand((b, h, self.num_kv_groups, s, d))?;
        Ok(x.reshape((b, h * self.num_kv_groups, s, d))?)
    }
}
