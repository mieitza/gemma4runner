use std::sync::Arc;
use anyhow::Result;
use candle_core::{Device, Tensor};
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
    moe_block: Option<crate::moe::MoeBlock>,
    post_feedforward_layernorm_1: Option<candle_nn::RmsNorm>,
}

impl DecoderLayer {
    fn new(
        cfg: &Gemma4TextConfig, layer_idx: usize,
        rotary_emb_local: Arc<RotaryEmbedding>,
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = GemmaAttention::new(cfg, layer_idx, rotary_emb_local, rotary_emb_global, vb.pp("self_attn"))?;
        let mlp = GemmaMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        let eps = cfg.rms_norm_eps;
        let hs = cfg.hidden_size;
        let (moe_block, post_feedforward_layernorm_1) = if cfg.enable_moe_block {
            let moe = crate::moe::MoeBlock::new(
                cfg.hidden_size,
                cfg.moe_intermediate_size.unwrap_or(704),
                cfg.num_experts.unwrap_or(128),
                cfg.top_k_experts.unwrap_or(8),
                cfg.rms_norm_eps,
                vb.clone(),
            )?;
            let post_ff_1 = candle_nn::rms_norm(
                cfg.hidden_size, cfg.rms_norm_eps,
                vb.pp("post_feedforward_layernorm_1"),
            )?;
            (Some(moe), Some(post_ff_1))
        } else {
            (None, None)
        };
        Ok(Self {
            self_attn, mlp,
            input_layernorm: candle_nn::rms_norm(hs, eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: candle_nn::rms_norm(hs, eps, vb.pp("post_attention_layernorm"))?,
            pre_feedforward_layernorm: candle_nn::rms_norm(hs, eps, vb.pp("pre_feedforward_layernorm"))?,
            post_feedforward_layernorm: candle_nn::rms_norm(hs, eps, vb.pp("post_feedforward_layernorm"))?,
            moe_block,
            post_feedforward_layernorm_1,
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

        let ff = if let (Some(moe), Some(post_ff_1)) = (&self.moe_block, &self.post_feedforward_layernorm_1) {
            let dense_normed = post_ff_1.forward(&ff)?;
            let moe_output = moe.forward(residual)?;
            (dense_normed + moe_output)?
        } else {
            ff
        };

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

        let sliding_rope = cfg.rope_parameters.as_ref().and_then(|p| p.sliding_attention.as_ref());
        let full_rope = cfg.rope_parameters.as_ref().and_then(|p| p.full_attention.as_ref());

        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb.dtype(), cfg.head_dim,
            sliding_rope.and_then(|p| p.rope_theta).unwrap_or(10000.0),
            cfg.max_position_embeddings, vb_m.device(),
        )?);
        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(), cfg.global_head_dim,
            full_rope.and_then(|p| p.rope_theta).unwrap_or(1000000.0),
            full_rope.and_then(|p| p.partial_rotary_factor).unwrap_or(0.25),
            cfg.max_position_embeddings, vb_m.device(),
        )?);

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, i, rotary_emb_local.clone(), rotary_emb_global.clone(), vb_l.pp(i))?);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self { embed_tokens, layers, norm, lm_head, final_logit_softcapping: cfg.final_logit_softcapping, hidden_size: cfg.hidden_size, cfg: cfg.clone() })
    }

    pub fn forward(&self, input_ids: &Tensor, cache: &mut KvCache, seqlen_offset: usize) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        let scale = (self.hidden_size as f64).sqrt();
        x = (x * scale)?;

        let (sliding_mask, global_mask) = self.create_masks(batch_size, seq_len, seqlen_offset, x.device())?;

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

    fn create_masks(&self, _batch_size: usize, seq_len: usize, offset: usize, device: &Device) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len == 1 { return Ok((None, None)); }
        let mask: Vec<f32> = (0..seq_len).flat_map(|i| {
            (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
        }).collect();
        let causal = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?;

        let sw = self.cfg.sliding_window;
        let sliding_mask: Vec<f32> = (0..seq_len).flat_map(|i| {
            (0..seq_len).map(move |j| {
                let pos_i = i + offset;
                let pos_j = j + offset;
                if j > i || (pos_i >= sw && pos_j < pos_i - sw + 1) { f32::NEG_INFINITY } else { 0.0 }
            })
        }).collect();
        let sliding = Tensor::from_vec(sliding_mask, (1, 1, seq_len, seq_len), device)?;
        Ok((Some(sliding), Some(causal)))
    }
}
