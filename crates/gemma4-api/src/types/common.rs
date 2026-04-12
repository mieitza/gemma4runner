use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role { System, User, Assistant }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message { pub role: Role, pub content: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage { pub prompt_tokens: usize, pub completion_tokens: usize, pub total_tokens: usize }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason { Stop, Length }
