use serde::{Deserialize, Serialize};
use super::common::{FinishReason, Message, Usage};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
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
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
}

fn default_temperature() -> Option<f64> { Some(1.0) }

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChoiceMessage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Serialize)]
pub struct ChoiceMessage { pub role: String, pub content: String }
