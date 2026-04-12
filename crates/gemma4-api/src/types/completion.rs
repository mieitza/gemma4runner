use serde::{Deserialize, Serialize};
use super::common::{FinishReason, Usage};

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
}
fn default_temperature() -> Option<f64> { Some(1.0) }

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String, pub object: String, pub created: u64, pub model: String,
    pub choices: Vec<CompletionChoice>, pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize, pub text: String, pub finish_reason: FinishReason,
}
