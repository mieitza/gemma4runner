//! Debug tool: loads a GGUF model and prints intermediate values
//! to narrow down where the forward pass goes wrong.
//!
//! Usage: cargo run --release -p gemma4-core --bin debug_forward -- /path/to/model.gguf

use anyhow::Result;
use candle_core::{Device, Module, Tensor, DType, D};
use std::path::Path;

fn print_stats(name: &str, t: &Tensor) {
    let t = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let vals: Vec<f32> = t.to_vec1().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
    println!("{:40} shape={:?}  mean={:.4}  std={:.4}  min={:.4}  max={:.4}",
             name, t.dims(), mean, std, min, max);
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: debug_forward <model.gguf>");
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);
    let device = Device::Cpu;

    println!("Loading GGUF...");
    let gguf = gemma4_core::gguf_loader::GgufModel::load(model_path, &device)?;
    let cfg = &gguf.config;
    println!("Config: hidden_size={}, layers={}, heads={}, kv_heads={}, head_dim={}, global_head_dim={}",
             cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
             cfg.num_key_value_heads, cfg.head_dim, cfg.global_head_dim);

    // Load just the embedding
    let embed_weight = gguf.tensor("token_embd.weight", &device)?.dequantize(&device)?;
    println!("\nEmbedding weight:");
    print_stats("token_embd.weight", &embed_weight);

    // Create a simple input: token IDs [2, 105, 2364, 107] = "<bos><|turn>user\n"
    let input_ids = Tensor::new(&[2u32, 105, 2364, 107], &device)?.unsqueeze(0)?;
    println!("\nInput IDs: {:?}", input_ids.to_vec2::<u32>()?);

    // Embed
    let embed = candle_nn::Embedding::new(embed_weight.clone(), cfg.hidden_size);
    let embedded = embed.forward(&input_ids)?;
    print_stats("After embedding (raw)", &embedded);

    let scale = (cfg.hidden_size as f64).sqrt();
    let embedded = (embedded * scale)?;
    print_stats("After embedding * sqrt(hidden)", &embedded);

    // Load first layer norms and check
    let input_norm_w = gguf.tensor("blk.0.attn_norm.weight", &device)?.dequantize(&device)?;
    print_stats("blk.0.attn_norm.weight", &input_norm_w);

    // Apply input layernorm manually
    let x = &embedded;
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = (&x_f32 * &x_f32)?.mean_keepdim(D::Minus1)?;
    let x_norm = x_f32.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;
    let normed = x_norm.broadcast_mul(&input_norm_w)?;
    print_stats("After input_layernorm", &normed);

    // Load Q projection and apply
    let q_weight = gguf.tensor("blk.0.attn_q.weight", &device)?;
    println!("\nQ weight dtype={:?}, shape={:?}", q_weight.dtype(), q_weight.shape());
    let q_matmul = candle_core::quantized::QMatMul::from_qtensor(q_weight)?;
    let q_out = q_matmul.forward(&normed)?;
    print_stats("After Q projection (blk.0)", &q_out);

    // Load K projection
    let k_weight = gguf.tensor("blk.0.attn_k.weight", &device)?;
    let k_matmul = candle_core::quantized::QMatMul::from_qtensor(k_weight)?;
    let k_out = k_matmul.forward(&normed)?;
    print_stats("After K projection (blk.0)", &k_out);

    // Check Q norm
    let q_norm_w = gguf.tensor("blk.0.attn_q_norm.weight", &device)?.dequantize(&device)?;
    print_stats("blk.0.attn_q_norm.weight", &q_norm_w);

    // Check first FFN
    let gate_w = gguf.tensor("blk.0.ffn_gate.weight", &device)?;
    let gate_matmul = candle_core::quantized::QMatMul::from_qtensor(gate_w)?;
    let gate_out = gate_matmul.forward(&normed)?;
    print_stats("After FFN gate (blk.0)", &gate_out);

    // Now run the full model
    println!("\n=== Full model forward pass ===");
    let mut q_model = gemma4_core::quantized_model::QuantizedGemmaModel::new(cfg, &gguf, &device)?;
    let mut cache = gemma4_core::kv_cache::KvCache::new(&cfg.layer_types, cfg.sliding_window);
    let logits = q_model.forward(&input_ids, &mut cache, 0)?;
    print_stats("Final logits", &logits);

    // Get top-5 tokens
    let logits_flat = logits.squeeze(0)?.squeeze(0)?;
    let logits_vec: Vec<f32> = logits_flat.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop-5 predicted tokens:");
    for (i, (idx, logit)) in indexed.iter().take(5).enumerate() {
        println!("  #{}: token_id={} logit={:.4}", i+1, idx, logit);
    }

    Ok(())
}
