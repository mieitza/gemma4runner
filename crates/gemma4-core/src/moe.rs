use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{linear_no_bias, Activation, Linear, Module, RmsNorm, VarBuilder};

// ── Router ────────────────────────────────────────────────────────────────────

/// Selects the top-k experts per token.
pub struct Router {
    gate: Linear,
    scale: Tensor,
    per_expert_scale: Tensor,
    num_experts: usize,
    top_k: usize,
    hidden_size: usize,
}

impl Router {
    pub fn new(hidden_size: usize, num_experts: usize, top_k: usize, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(hidden_size, num_experts, vb.pp("gate"))?;
        let scale = vb.get(1, "scale")?;
        let per_expert_scale = vb.get(num_experts, "per_expert_scale")?;
        Ok(Self { gate, scale, per_expert_scale, num_experts, top_k, hidden_size })
    }

    /// Returns `(top_k_weights, top_k_indices)` both shaped `(num_tokens, top_k)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // RMS norm without a learned weight: x / sqrt(mean(x^2) + eps)
        let eps = 1e-6f64;
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
        let rms = (mean_sq + eps)?.sqrt()?;
        let x_normed = x.broadcast_div(&rms)?;

        // Scale by router scale and hidden_size^(-0.5)
        let hs_scale = (self.hidden_size as f64).powf(-0.5);
        let scale_val = self.scale.to_dtype(x.dtype())?;
        let x_scaled = x_normed.broadcast_mul(&scale_val)?;
        let x_scaled = (x_scaled * hs_scale)?;

        // Gate linear → logits of shape (num_tokens, num_experts)
        let logits = self.gate.forward(&x_scaled)?;

        // Softmax over experts dimension
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;

        // Top-k selection: work per-token via Vec round-trip
        let (num_tokens, _) = probs.dims2()?;
        let probs_data: Vec<f32> = probs.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let top_k = self.top_k;
        let ne = self.num_experts;

        let mut all_weights: Vec<f32> = Vec::with_capacity(num_tokens * top_k);
        let mut all_indices: Vec<u32> = Vec::with_capacity(num_tokens * top_k);

        for tok in 0..num_tokens {
            let row = &probs_data[tok * ne..(tok + 1) * ne];

            // Sort indices by probability descending
            let mut idx_prob: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            idx_prob.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let topk_slice = &idx_prob[..top_k];

            // Sum of selected weights for renormalization
            let sum: f32 = topk_slice.iter().map(|(_, w)| w).sum();
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };

            for &(expert_idx, w) in topk_slice {
                all_weights.push(w * inv_sum);
                all_indices.push(expert_idx as u32);
            }
        }

        let device = x.device();
        let top_k_weights = Tensor::from_vec(all_weights, (num_tokens, top_k), device)?
            .to_dtype(x.dtype())?;
        let top_k_indices = Tensor::from_vec(all_indices, (num_tokens, top_k), device)?;

        // Multiply weights by per_expert_scale for each selected expert
        // per_expert_scale shape: (num_experts,)
        let pes_data: Vec<f32> = self.per_expert_scale
            .to_dtype(DType::F32)?
            .to_vec1()?;

        let idx_data: Vec<u32> = top_k_indices.flatten_all()?.to_vec1()?;
        let weight_data: Vec<f32> = top_k_weights.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let scaled_weights: Vec<f32> = weight_data.iter().zip(idx_data.iter())
            .map(|(w, &i)| w * pes_data[i as usize])
            .collect();

        let top_k_weights = Tensor::from_vec(scaled_weights, (num_tokens, top_k), device)?
            .to_dtype(x.dtype())?;

        Ok((top_k_weights, top_k_indices))
    }
}

// ── Experts ───────────────────────────────────────────────────────────────────

/// Stores all expert weights as 3-D tensors for efficient access.
pub struct Experts {
    /// Shape: (num_experts, 2 * moe_intermediate_size, hidden_size)
    gate_up_proj: Tensor,
    /// Shape: (num_experts, hidden_size, moe_intermediate_size)
    down_proj: Tensor,
    #[allow(dead_code)]
    num_experts: usize,
    moe_intermediate_size: usize,
}

impl Experts {
    pub fn new(hidden_size: usize, num_experts: usize, moe_intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_up_proj = vb.get(
            (num_experts, 2 * moe_intermediate_size, hidden_size),
            "gate_up_proj",
        )?;
        let down_proj = vb.get(
            (num_experts, hidden_size, moe_intermediate_size),
            "down_proj",
        )?;
        Ok(Self { gate_up_proj, down_proj, num_experts, moe_intermediate_size })
    }

    /// x:             (num_tokens, hidden_size)
    /// top_k_indices: (num_tokens, top_k)   — u32
    /// top_k_weights: (num_tokens, top_k)   — same dtype as x
    /// Returns:       (num_tokens, hidden_size)
    pub fn forward(
        &self,
        x: &Tensor,
        top_k_indices: &Tensor,
        top_k_weights: &Tensor,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_size) = x.dims2()?;
        let (_, top_k) = top_k_indices.dims2()?;

        let act = Activation::GeluPytorchTanh;

        // Collect per-token outputs
        let dtype = x.dtype();
        let device = x.device();

        // indices and weights as host vecs for the naive loop
        let idx_vec: Vec<u32> = top_k_indices.flatten_all()?.to_vec1()?;
        let wgt_vec: Vec<f32> = top_k_weights.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let mut output_rows: Vec<Tensor> = Vec::with_capacity(num_tokens);

        for tok in 0..num_tokens {
            // x_tok: (1, hidden_size)
            let x_tok = x.narrow(0, tok, 1)?;

            // Accumulator for this token: (1, hidden_size)
            let mut accum: Option<Tensor> = None;

            for k in 0..top_k {
                let expert_idx = idx_vec[tok * top_k + k] as usize;
                let weight = wgt_vec[tok * top_k + k];

                // gate_up for this expert: (2*moe_intermediate_size, hidden_size)
                let gate_up_w = self.gate_up_proj.narrow(0, expert_idx, 1)?
                    .squeeze(0)?; // (2*mis, hs)

                // x_tok @ gate_up_w.T → (1, 2*mis)
                let gate_up_out = x_tok.matmul(&gate_up_w.t()?)?;

                // Split into gate and up: each (1, moe_intermediate_size)
                let gate = gate_up_out.narrow(1, 0, self.moe_intermediate_size)?;
                let up   = gate_up_out.narrow(1, self.moe_intermediate_size, self.moe_intermediate_size)?;

                // Apply activation to gate, fuse
                let gate = act.forward(&gate)?;
                let fused = (gate * up)?; // (1, moe_intermediate_size)

                // down_proj for this expert: (hidden_size, moe_intermediate_size)
                let down_w = self.down_proj.narrow(0, expert_idx, 1)?
                    .squeeze(0)?; // (hs, mis)

                // fused @ down_w.T → (1, hidden_size)
                let out = fused.matmul(&down_w.t()?)?;

                // Weight and accumulate
                let out_weighted = (out * weight as f64)?;
                accum = Some(match accum {
                    None => out_weighted,
                    Some(prev) => (prev + out_weighted)?,
                });
            }

            let row = accum.unwrap_or_else(|| {
                Tensor::zeros((1, hidden_size), dtype, device).unwrap()
            });
            output_rows.push(row);
        }

        // Stack rows → (num_tokens, hidden_size)
        let output = Tensor::cat(&output_rows, 0)?;
        Ok(output)
    }
}

// ── MoeBlock ──────────────────────────────────────────────────────────────────

/// Combines Router + Experts with pre/post feed-forward norms.
pub struct MoeBlock {
    router: Router,
    experts: Experts,
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
}

impl MoeBlock {
    pub fn new(
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let router = Router::new(hidden_size, num_experts, top_k, vb.pp("router"))?;
        let experts = Experts::new(hidden_size, num_experts, moe_intermediate_size, vb.pp("experts"))?;
        let pre_norm = candle_nn::rms_norm(hidden_size, rms_norm_eps, vb.pp("pre_feedforward_layernorm_2"))?;
        let post_norm = candle_nn::rms_norm(hidden_size, rms_norm_eps, vb.pp("post_feedforward_layernorm_2"))?;
        Ok(Self { router, experts, pre_norm, post_norm })
    }

    /// residual: (batch, seq_len, hidden_size)
    /// Returns:  (batch, seq_len, hidden_size)
    pub fn forward(&self, residual: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = residual.dims3()?;
        let num_tokens = batch * seq_len;

        // Flatten spatial dimensions → (num_tokens, hidden_size)
        let x = residual.reshape((num_tokens, hidden_size))?;

        // Route
        let (top_k_weights, top_k_indices) = self.router.forward(&x)?;

        // Pre-norm
        let x_normed = self.pre_norm.forward(&x)?;

        // Expert computation
        let expert_out = self.experts.forward(&x_normed, &top_k_indices, &top_k_weights)?;

        // Post-norm
        let expert_out = self.post_norm.forward(&expert_out)?;

        // Reshape back to (batch, seq_len, hidden_size)
        let output = expert_out.reshape((batch, seq_len, hidden_size))?;

        Ok(output)
    }
}
