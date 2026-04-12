mod cli;
mod config;

use std::path::PathBuf;
use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Commands};
use config::AppConfig;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve { model, config, host, port, device, hf_token, api_key, log_level, queue_depth } => {
            let file_config = match &config {
                Some(path) => AppConfig::load(&PathBuf::from(path))?,
                None => AppConfig::default(),
            };

            let log_level = log_level.or(file_config.server.log_level).unwrap_or_else(|| "info".into());
            let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&log_level));
            tracing_subscriber::fmt().with_env_filter(env_filter).init();

            let model_source = model.or(file_config.model.source)
                .context("--model is required (or set model.source in config file)")?;
            let device_str = device.or(file_config.model.device).unwrap_or_else(|| "auto".into());
            let hf_token = hf_token.or(file_config.model.hf_token);
            let host = host.or(file_config.server.host).unwrap_or_else(|| "0.0.0.0".into());
            let port = port.or(file_config.server.port).unwrap_or(8080);
            let queue_depth = queue_depth.or(file_config.server.queue_depth).unwrap_or(64);
            let api_key = api_key.or(file_config.auth.api_key);

            let model_path = gemma4_core::loader::resolve_model_source(&model_source, hf_token.as_deref())?;
            let dev = gemma4_core::engine::device_from_string(&device_str)?;
            tracing::info!("Using device: {}", device_str);

            tracing::info!("Loading model from {}", model_path.display());
            let engine = gemma4_core::engine::start_engine(&model_path, dev, queue_depth)?;

            tracing::info!("Starting server on {}:{}", host, port);
            gemma4_api::server::start_server(engine, &host, port, api_key).await?;
            Ok(())
        }
        Commands::Info { model, hf_token } => {
            let model_path = gemma4_core::loader::resolve_model_source(&model, hf_token.as_deref())?;

            // Load config — handle both GGUF files and safetensors directories
            let tc = if gemma4_core::loader::is_gguf_file(&model_path) {
                let gguf = gemma4_core::gguf_loader::GgufModel::load(&model_path, &candle_core::Device::Cpu)?;
                gguf.config
            } else {
                let config_path = model_path.join("config.json");
                let config: gemma4_core::config::Gemma4Config = serde_json::from_reader(
                    fs::File::open(&config_path)
                        .with_context(|| format!("Failed to open {}", config_path.display()))?,
                ).context("Failed to parse config.json")?;
                config.text_config
            };
            let tc = &tc;

            println!("Model source : {}", model_path.display());
            println!("Hidden size  : {}", tc.hidden_size);
            println!("Layers       : {}", tc.num_hidden_layers);
            println!("Attn heads   : {}", tc.num_attention_heads);
            println!("KV heads     : {}", tc.num_key_value_heads);
            if let Some(global_kv) = tc.num_global_key_value_heads {
                println!("Global KV heads : {}", global_kv);
            }
            println!("Head dim     : {}", tc.head_dim);
            if tc.global_head_dim > 0 {
                println!("Global head dim : {}", tc.global_head_dim);
            }
            println!("Vocab size   : {}", tc.vocab_size);
            println!("Max context  : {}", tc.max_position_embeddings);
            println!("Sliding win  : {}", tc.sliding_window);

            if tc.enable_moe_block {
                println!("MoE          : enabled");
                if let Some(n) = tc.num_experts {
                    println!("  Num experts       : {}", n);
                }
                if let Some(k) = tc.top_k_experts {
                    println!("  Top-k experts     : {}", k);
                }
                if let Some(moe_sz) = tc.moe_intermediate_size {
                    println!("  MoE interm size   : {}", moe_sz);
                }
                println!("  Dense interm size : {}", tc.intermediate_size);
            } else {
                println!("MLP interm   : {}", tc.intermediate_size);
            }

            let sliding_count = tc.layer_types.iter().filter(|t| t.as_str() == "sliding_attention").count();
            let full_count = tc.layer_types.len() - sliding_count;
            println!("Layer pattern: {} sliding, {} full attention", sliding_count, full_count);

            Ok(())
        }
    }
}
