use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gemma4runner", about = "Run Gemma 4 models with an OpenAI-compatible API")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the API server
    Serve {
        #[arg(long)] model: Option<String>,
        #[arg(long)] config: Option<String>,
        #[arg(long)] host: Option<String>,
        #[arg(long)] port: Option<u16>,
        #[arg(long)] device: Option<String>,
        #[arg(long, env = "HF_TOKEN")] hf_token: Option<String>,
        #[arg(long, env = "GEMMA4_AUTH_API_KEY")] api_key: Option<String>,
        #[arg(long)] log_level: Option<String>,
        #[arg(long)] queue_depth: Option<usize>,
    },
}
