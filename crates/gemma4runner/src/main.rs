mod cli;
mod config;

use std::path::PathBuf;
use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Commands};
use config::AppConfig;

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
    }
}
