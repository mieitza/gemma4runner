use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role { System, User, Assistant, Tool, Developer }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    /// Content can be a string or an array of content parts (OpenAI format).
    /// We accept both and normalize to string.
    #[serde(default, deserialize_with = "deserialize_content")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Deserialize content as either a string or an array of content parts.
fn deserialize_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where D: serde::Deserializer<'de> {
    let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(serde_json::Value::Array(arr)) => {
            let mut text = String::new();
            for item in arr {
                if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                    text.push_str(t);
                }
            }
            Ok(if text.is_empty() { None } else { Some(text) })
        }
        Some(serde_json::Value::Null) => Ok(None),
        Some(other) => Ok(Some(other.to_string())),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage { pub prompt_tokens: usize, pub completion_tokens: usize, pub total_tokens: usize }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason { Stop, Length, ToolCalls }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}
