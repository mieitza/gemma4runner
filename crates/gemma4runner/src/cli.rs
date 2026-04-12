use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gemma4runner")]
#[command(about = "Run Gemma 4 models with an OpenAI-compatible API")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the API server
    Serve {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value = "info")]
        log_level: String,
        #[arg(long, default_value_t = 64)]
        queue_depth: usize,
    },
}
