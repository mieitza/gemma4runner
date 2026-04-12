# Gemma4Runner Phase 3 — All Model Variants

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support all 4 Gemma 4 variants (E2B, E4B, 26B-A4B MoE, 31B dense) and add `gemma4runner info` subcommand.

**Architecture:** The existing dense model (model.rs) already handles E2B, E4B, and 31B — they differ only in config values. The real work is: (1) MoE support for 26B-A4B (router + experts + 3 extra norms per layer), (2) `num_global_key_value_heads` config field for different GQA ratios on global vs sliding layers, (3) MoE config fields, (4) `info` subcommand.

**Tech Stack:** Same as Phase 2. No new dependencies.

**Key Architecture Insight:** In 26B-A4B, each decoder layer runs BOTH a dense MLP (intermediate_size=2112) AND sparse MoE (128 experts, top-8, each with moe_intermediate_size=704) in parallel. The dense and sparse outputs are summed. The router input and expert input are derived from the residual (before dense MLP), not from MLP output.

---

## File Structure Changes

```
crates/gemma4-core/src/
  config.rs         — MODIFY: add MoE fields, num_global_key_value_heads
  moe.rs            — CREATE: Router + Experts + MoE block
  mlp.rs            — MODIFY: no changes needed (reused as dense MLP in MoE layers)
  attention.rs      — MODIFY: support num_global_key_value_heads
  model.rs          — MODIFY: conditional MoE in DecoderLayer, extra norms

crates/gemma4runner/src/
  cli.rs            — MODIFY: add Info subcommand
  main.rs           — MODIFY: handle Info command
```

---

### Task 1: Extend Config for MoE and Global KV Heads

**Files:**
- Modify: `crates/gemma4-core/src/config.rs`

- [ ] **Step 1: Write tests for MoE config parsing**

Add to the `#[cfg(test)] mod tests` block in config.rs:

```rust
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
    let tc = &config.text_config;
    assert!(tc.enable_moe_block);
    assert_eq!(tc.num_experts.unwrap(), 128);
    assert_eq!(tc.top_k_experts.unwrap(), 8);
    assert_eq!(tc.moe_intermediate_size.unwrap(), 704);
    assert_eq!(tc.num_global_key_value_heads.unwrap(), 2);
    // Global layers use num_global_key_value_heads
    assert_eq!(tc.kv_heads_for_layer(5), 2);
    // Sliding layers use num_key_value_heads
    assert_eq!(tc.kv_heads_for_layer(0), 8);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gemma4-core test_parse_moe_config`
Expected: FAIL — fields don't exist yet

- [ ] **Step 3: Add MoE fields to Gemma4TextConfig**

Add these fields to `Gemma4TextConfig`:

```rust
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
```

- [ ] **Step 4: Update kv_heads_for_layer to use num_global_key_value_heads**

Replace the `kv_heads_for_layer` method:

```rust
pub fn kv_heads_for_layer(&self, layer_idx: usize) -> usize {
    if !self.is_sliding_layer(layer_idx) {
        if let Some(global_kv) = self.num_global_key_value_heads {
            return global_kv;
        }
    }
    self.num_key_value_heads
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p gemma4-core`
Expected: All tests pass including new one

- [ ] **Step 6: Commit**

```bash
git add crates/gemma4-core/src/config.rs
git commit -m "feat(core): add MoE and global KV heads config fields"
```

---

### Task 2: MoE Module — Router and Experts

**Files:**
- Create: `crates/gemma4-core/src/moe.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement Router**

Create `crates/gemma4-core/src/moe.rs`:

```rust
use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

/// MoE Router: selects top-k experts per token.
///
/// Flow: rms_norm(x) * scale * hidden_size^(-0.5) → linear → softmax → top_k → renormalize
pub struct Router {
    gate: Linear,
    scale: Tensor,
    per_expert_scale: Tensor,
    num_experts: usize,
    top_k: usize,
    hidden_size: usize,
}

impl Router {
    pub fn new(
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate = candle_nn::linear_no_bias(hidden_size, num_experts, vb.pp("gate"))?;
        let scale = vb.get(1, "scale")?;
        let per_expert_scale = vb.get(num_experts, "per_expert_scale")?;
        Ok(Self { gate, scale, per_expert_scale, num_experts, top_k, hidden_size })
    }

    /// Returns (router_logits, top_k_weights, top_k_indices)
    /// Input: (num_tokens, hidden_size)
    /// top_k_weights: (num_tokens, top_k)
    /// top_k_indices: (num_tokens, top_k) as u32
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // RMS norm without learned weight
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;

        // Scale
        let hidden_scale = (self.hidden_size as f64).powf(-0.5);
        let x_scaled = (x_norm.broadcast_mul(&self.scale)? * hidden_scale)?;

        // Route
        let logits = self.gate.forward(&x_scaled)?;
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;

        // Top-k selection
        let (top_k_weights, top_k_indices) = self.top_k_select(&probs)?;

        // Renormalize weights to sum to 1
        let weight_sum = top_k_weights.sum_keepdim(D::Minus1)?;
        let top_k_weights = top_k_weights.broadcast_div(&weight_sum)?;

        // Apply per-expert scale
        let expert_scales = self.per_expert_scale.index_select(&top_k_indices.flatten_all()?, 0)?;
        let expert_scales = expert_scales.reshape(top_k_indices.shape())?;
        let top_k_weights = (top_k_weights * expert_scales)?;

        Ok((top_k_weights, top_k_indices))
    }

    fn top_k_select(&self, probs: &Tensor) -> Result<(Tensor, Tensor)> {
        // Get top-k by sorting and taking first k
        let num_tokens = probs.dim(0)?;
        let probs_vec: Vec<Vec<f32>> = (0..num_tokens)
            .map(|i| probs.get(i).unwrap().to_vec1::<f32>().unwrap())
            .collect();

        let mut top_k_w = Vec::with_capacity(num_tokens * self.top_k);
        let mut top_k_i = Vec::with_capacity(num_tokens * self.top_k);

        for token_probs in &probs_vec {
            let mut indexed: Vec<(usize, f32)> = token_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for &(idx, weight) in indexed.iter().take(self.top_k) {
                top_k_w.push(weight);
                top_k_i.push(idx as u32);
            }
        }

        let device = probs.device();
        let top_k_weights = Tensor::from_vec(top_k_w, (num_tokens, self.top_k), device)?
            .to_dtype(probs.dtype())?;
        let top_k_indices = Tensor::from_vec(top_k_i, (num_tokens, self.top_k), device)?;

        Ok((top_k_weights, top_k_indices))
    }
}

/// MoE Experts: stores all expert weights as 3D tensors.
///
/// Each expert is a gated MLP: output = down(act(gate(x)) * up(x))
/// gate and up are fused into gate_up_proj for efficiency.
pub struct Experts {
    gate_up_proj: Tensor,  // (num_experts, 2*moe_intermediate_size, hidden_size)
    down_proj: Tensor,     // (num_experts, hidden_size, moe_intermediate_size)
    num_experts: usize,
    moe_intermediate_size: usize,
}

impl Experts {
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
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

    /// Forward pass through selected experts.
    /// x: (num_tokens, hidden_size)
    /// top_k_indices: (num_tokens, top_k) — which experts to use
    /// top_k_weights: (num_tokens, top_k) — expert weights
    /// Returns: (num_tokens, hidden_size)
    pub fn forward(
        &self,
        x: &Tensor,
        top_k_indices: &Tensor,
        top_k_weights: &Tensor,
    ) -> Result<Tensor> {
        let (num_tokens, top_k) = top_k_indices.dims2()?;
        let hidden_size = x.dim(1)?;
        let device = x.device();
        let dtype = x.dtype();

        let mut output = Tensor::zeros((num_tokens, hidden_size), dtype, device)?;

        // Process each expert assignment
        // For each token, for each selected expert, compute expert output and accumulate
        let indices_vec: Vec<u32> = top_k_indices.flatten_all()?.to_vec1()?;
        let weights_vec: Vec<f32> = top_k_weights.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        for token_idx in 0..num_tokens {
            let x_token = x.get(token_idx)?; // (hidden_size,)

            for k in 0..top_k {
                let flat_idx = token_idx * top_k + k;
                let expert_idx = indices_vec[flat_idx] as usize;
                let weight = weights_vec[flat_idx];

                // Get expert weights
                let gate_up = self.gate_up_proj.get(expert_idx)?; // (2*moe_intermediate, hidden)
                let down = self.down_proj.get(expert_idx)?; // (hidden, moe_intermediate)

                // gate_up(x): (2*moe_intermediate,)
                let gu = gate_up.matmul(&x_token.unsqueeze(1)?)?.squeeze(1)?;

                // Split into gate and up
                let gate = gu.narrow(0, 0, self.moe_intermediate_size)?;
                let up = gu.narrow(0, self.moe_intermediate_size, self.moe_intermediate_size)?;

                // GeluPytorchTanh activation on gate
                let gate = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;

                // gate * up
                let fused = (gate * up)?;

                // down(fused): (hidden_size,)
                let expert_out = down.matmul(&fused.unsqueeze(1)?)?.squeeze(1)?;

                // Weighted accumulation
                let weighted = (expert_out * weight as f64)?;
                output = output.slice_set(&weighted.unsqueeze(0)?, 0, token_idx)?;
            }
        }

        Ok(output)
    }
}

/// Full MoE block: router + experts, with pre/post norms.
pub struct MoeBlock {
    router: Router,
    experts: Experts,
    pre_norm: candle_nn::RmsNorm,
    post_norm: candle_nn::RmsNorm,
}

impl MoeBlock {
    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let router = Router::new(hidden_size, num_experts, top_k, vb.pp("router"))?;
        let experts = Experts::new(hidden_size, moe_intermediate_size, num_experts, vb.pp("experts"))?;
        let pre_norm = candle_nn::rms_norm(hidden_size, rms_norm_eps, vb.pp("pre_feedforward_layernorm_2"))?;
        let post_norm = candle_nn::rms_norm(hidden_size, rms_norm_eps, vb.pp("post_feedforward_layernorm_2"))?;
        Ok(Self { router, experts, pre_norm, post_norm })
    }

    /// Forward pass. Takes the residual (pre-MLP hidden states).
    /// Returns the MoE contribution to add to the dense MLP output.
    pub fn forward(&self, residual: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = residual.dims3()?;
        let flat = residual.reshape((batch_size * seq_len, hidden_size))?;

        // Route
        let (top_k_weights, top_k_indices) = self.router.forward(&flat)?;

        // Normalize input to experts
        let expert_input = self.pre_norm.forward(&flat)?;

        // Run experts
        let expert_output = self.experts.forward(&expert_input, &top_k_indices, &top_k_weights)?;

        // Post-norm
        let expert_output = self.post_norm.forward(&expert_output)?;

        // Reshape back
        let output = expert_output.reshape((batch_size, seq_len, hidden_size))?;
        Ok(output)
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

Add `pub mod moe;` to lib.rs.

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p gemma4-core`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add crates/gemma4-core/src/moe.rs crates/gemma4-core/src/lib.rs
git commit -m "feat(core): add MoE router, experts, and MoE block"
```

---

### Task 3: Update Attention for Global KV Heads

**Files:**
- Modify: `crates/gemma4-core/src/attention.rs`

The 26B-A4B and 31B models use `num_global_key_value_heads` which is different from `num_key_value_heads`. Global attention layers use fewer KV heads than sliding layers. Our attention layer already calls `cfg.kv_heads_for_layer(layer_idx)` which we updated in Task 1 — so this should already work. But we need to verify the `num_kv_groups` calculation is per-layer, not global.

- [ ] **Step 1: Fix repeat_kv to be dynamic**

In `attention.rs`, the `num_kv_groups` is computed once in the constructor. For models where sliding and global layers have different KV head counts, we need to compute it per-layer. Since each `GemmaAttention` is created per-layer, the `num_kv_groups` is already correct for that layer. Verify this by checking the constructor:

```rust
num_kv_groups: num_heads / num_kv_heads,
```

This uses the per-layer `num_kv_heads` from `cfg.kv_heads_for_layer(layer_idx)`, so it's correct. No code change needed, just verify with `cargo check`.

- [ ] **Step 2: Commit (skip if no changes needed)**

Run: `cargo check -p gemma4-core`
Expected: Compiles — no changes needed

---

### Task 4: Update DecoderLayer for MoE

**Files:**
- Modify: `crates/gemma4-core/src/model.rs`

- [ ] **Step 1: Add MoE support to DecoderLayer**

Read model.rs first. Then modify `DecoderLayer` to optionally include a MoE block:

Add to `DecoderLayer` struct:
```rust
    moe_block: Option<crate::moe::MoeBlock>,
    post_feedforward_layernorm_1: Option<candle_nn::RmsNorm>,
```

Update `DecoderLayer::new` to conditionally create MoE:
```rust
    let (moe_block, post_feedforward_layernorm_1) = if cfg.enable_moe_block {
        let moe = crate::moe::MoeBlock::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size.unwrap_or(704),
            cfg.num_experts.unwrap_or(128),
            cfg.top_k_experts.unwrap_or(8),
            cfg.rms_norm_eps,
            vb.clone(),  // MoE norms are at the layer level
        )?;
        let post_ff_1 = candle_nn::rms_norm(
            cfg.hidden_size, cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm_1"),
        )?;
        (Some(moe), Some(post_ff_1))
    } else {
        (None, None)
    };
```

Update `DecoderLayer::forward` to handle MoE:
```rust
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, cache: &mut crate::kv_cache::LayerKvCache, seqlen_offset: usize) -> Result<Tensor> {
        // Attention (unchanged)
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, cache, seqlen_offset)?;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = (residual + x)?;

        // Feedforward
        let residual = &x;
        let ff = self.pre_feedforward_layernorm.forward(&x)?;
        let ff = self.mlp.forward(&ff)?;

        let ff = if let (Some(moe), Some(post_ff_1)) = (&self.moe_block, &self.post_feedforward_layernorm_1) {
            // MoE path: dense MLP output normed + MoE output from residual, summed
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
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p gemma4-core`
Expected: Compiles

- [ ] **Step 3: Run tests**

Run: `cargo test -p gemma4-core`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/gemma4-core/src/model.rs
git commit -m "feat(core): add MoE support to decoder layers (26B-A4B)"
```

---

### Task 5: Info Subcommand

**Files:**
- Modify: `crates/gemma4runner/src/cli.rs`
- Modify: `crates/gemma4runner/src/main.rs`

- [ ] **Step 1: Add Info subcommand to CLI**

In cli.rs, add to the `Commands` enum:

```rust
    /// Print model details without loading weights
    Info {
        /// Path to model directory or HuggingFace model ID
        #[arg(long)]
        model: String,

        /// HuggingFace token for gated models
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,
    },
```

- [ ] **Step 2: Handle Info in main.rs**

Add the match arm in main.rs:

```rust
        Commands::Info { model, hf_token } => {
            let model_path = gemma4_core::loader::resolve_model_source(&model, hf_token.as_deref())?;
            let config_path = model_path.join("config.json");
            let config: gemma4_core::config::Gemma4Config = serde_json::from_reader(
                std::fs::File::open(&config_path)?
            )?;

            let tc = &config.text_config;
            println!("Model: {}", model);
            println!("Architecture: Gemma 4");
            println!("Hidden size: {}", tc.hidden_size);
            println!("Layers: {}", tc.num_hidden_layers);
            println!("Attention heads: {}", tc.num_attention_heads);
            println!("KV heads: {}", tc.num_key_value_heads);
            if let Some(global_kv) = tc.num_global_key_value_heads {
                println!("Global KV heads: {}", global_kv);
            }
            println!("Head dim: {} (global: {})", tc.head_dim, tc.global_head_dim);
            println!("Vocab size: {}", tc.vocab_size);
            println!("Max context: {}", tc.max_position_embeddings);
            println!("Sliding window: {}", tc.sliding_window);

            if tc.enable_moe_block {
                println!("MoE: {} experts, top-{}", tc.num_experts.unwrap_or(0), tc.top_k_experts.unwrap_or(0));
                println!("MoE intermediate size: {}", tc.moe_intermediate_size.unwrap_or(0));
                println!("Dense intermediate size: {}", tc.intermediate_size);
            } else {
                println!("MLP intermediate size: {}", tc.intermediate_size);
            }

            // Count layer types
            let sliding = tc.layer_types.iter().filter(|t| *t == "sliding_attention").count();
            let full = tc.layer_types.iter().filter(|t| *t == "full_attention").count();
            println!("Layer pattern: {} sliding + {} full attention", sliding, full);

            Ok(())
        }
```

- [ ] **Step 3: Add serde_json to gemma4runner Cargo.toml**

Add: `serde_json = { workspace = true }`

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p gemma4runner`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add crates/gemma4runner/
git commit -m "feat(runner): add info subcommand to display model architecture"
```

---

### Task 6: Full Verification

- [ ] **Step 1: Run full build**

Run: `cargo build`
Expected: Compiles

- [ ] **Step 2: Run all tests**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 3: Verify CLI**

Run: `cargo run -p gemma4runner -- --help`
Expected: Shows both `serve` and `info` subcommands

Run: `cargo run -p gemma4runner -- info --help`
Expected: Shows --model and --hf-token flags

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: Phase 3 cleanup and verification"
```
