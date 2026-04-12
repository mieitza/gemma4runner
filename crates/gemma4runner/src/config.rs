use std::path::Path;
use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub auth: AuthConfig,
}

#[derive(Debug, Deserialize, Default)]
pub struct ModelConfig {
    pub source: Option<String>,
    pub device: Option<String>,
    pub hf_token: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ServerConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub queue_depth: Option<usize>,
    pub log_level: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct InferenceConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
pub struct AuthConfig {
    pub api_key: Option<String>,
}

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        toml::from_str(&content)
            .with_context(|| format!("Failed to parse config: {}", path.display()))
    }
}
