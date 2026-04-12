mod cli;

use std::path::PathBuf;
use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve { model, host, port, log_level, queue_depth } => {
            let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&log_level));
            tracing_subscriber::fmt().with_env_filter(env_filter).init();

            let model_path = PathBuf::from(&model);
            anyhow::ensure!(model_path.exists(), "Model path does not exist: {}", model_path.display());

            tracing::info!("Loading model from {}", model_path.display());
            let engine = gemma4_core::engine::start_engine(&model_path, queue_depth)?;

            tracing::info!("Starting server on {}:{}", host, port);
            gemma4_api::server::start_server(engine, &host, port).await?;
            Ok(())
        }
    }
}
