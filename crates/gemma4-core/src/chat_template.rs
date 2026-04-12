use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct ChatFormatOptions {
    pub tools: Vec<ToolDef>,
    pub enable_thinking: bool,
}

impl Default for ChatFormatOptions {
    fn default() -> Self {
        ChatFormatOptions { tools: vec![], enable_thinking: false }
    }
}

/// Format a Gemma 4 tool declaration string using the DSL syntax.
pub fn format_tool_definition(
    name: &str,
    description: Option<&str>,
    parameters: Option<&serde_json::Value>,
) -> String {
    let mut obj = String::from("{");
    let mut first = true;

    if let Some(desc) = description {
        obj.push_str(&format!("<|\"|>description<|\"|>:<|\"|>{}<|\"|>", desc));
        first = false;
    }

    if let Some(params) = parameters {
        if !first { obj.push(','); }
        obj.push_str(&format!("<|\"|>parameters<|\"|>:{}", format_value_gemma(params)));
    }

    obj.push('}');
    format!("<|tool>declaration:{}{}<tool|>", name, obj)
}

/// Recursively replace JSON double-quote characters with `<|"|>`.
pub fn format_value_gemma(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => format!("<|\"|>{}<|\"|>", s),
        serde_json::Value::Object(map) => {
            let pairs: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("<|\"|>{}<|\"|>:{}", k, format_value_gemma(v)))
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_value_gemma).collect();
            format!("[{}]", items.join(","))
        }
        other => other.to_string(),
    }
}

pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    format_chat_prompt_with_options(messages, &ChatFormatOptions::default())
}

pub fn format_chat_prompt_with_options(messages: &[ChatMessage], options: &ChatFormatOptions) -> String {
    let mut prompt = String::from("<bos>");

    // Emit system preamble turn if tools or thinking are enabled
    if !options.tools.is_empty() || options.enable_thinking {
        let mut system_content = String::new();

        if options.enable_thinking {
            system_content.push_str("<|think|>\n");
        }

        for tool in &options.tools {
            let def = format_tool_definition(
                &tool.name,
                tool.description.as_deref(),
                tool.parameters.as_ref(),
            );
            system_content.push_str(&def);
            system_content.push('\n');
        }

        prompt.push_str(&format!("<|turn>system\n{}<turn|>\n", system_content.trim_end()));
    }

    for msg in messages {
        match msg.role.as_str() {
            "tool" => {
                // Tool response message
                let tool_id = msg.tool_call_id.as_deref().unwrap_or("");
                prompt.push_str(&format!(
                    "<|turn>tool\n<|tool_response>id:{}\n{}<tool_response|><turn|>\n",
                    tool_id, msg.content
                ));
            }
            "assistant" => {
                // Check if this assistant message contains tool calls
                if let Some(calls) = &msg.tool_calls {
                    let mut turn_content = String::new();
                    for call in calls {
                        turn_content.push_str(&format!(
                            "<|tool_call>{}({})<tool_call|>",
                            call.name, call.arguments
                        ));
                    }
                    prompt.push_str(&format!("<|turn>model\n{}<turn|>\n", turn_content));
                } else {
                    prompt.push_str(&format!("<|turn>model\n{}<turn|>\n", msg.content));
                }
            }
            other => {
                prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", other, msg.content));
            }
        }
    }

    // Begin generation turn
    prompt.push_str("<|turn>model\n");

    // If thinking is disabled, suppress the thought channel
    if !options.enable_thinking {
        prompt.push_str("<|channel>thought\n<channel|>");
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.into(), content: content.into(), tool_calls: None, tool_call_id: None }
    }

    #[test]
    fn test_single_user_message() {
        let messages = vec![msg("user", "Hello")];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.starts_with("<bos>"));
        assert!(prompt.contains("<|turn>user\nHello<turn|>"));
        assert!(prompt.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    #[test]
    fn test_system_and_user() {
        let messages = vec![
            msg("system", "You are helpful."),
            msg("user", "Hi"),
        ];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.contains("<|turn>system\nYou are helpful.<turn|>"));
        assert!(prompt.contains("<|turn>user\nHi<turn|>"));
    }

    #[test]
    fn test_multi_turn() {
        let messages = vec![
            msg("user", "What is 2+2?"),
            msg("assistant", "4"),
            msg("user", "And 3+3?"),
        ];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.contains("<|turn>user\nWhat is 2+2?<turn|>"));
        assert!(prompt.contains("<|turn>model\n4<turn|>"));
        assert!(prompt.contains("<|turn>user\nAnd 3+3?<turn|>"));
    }

    #[test]
    fn test_assistant_mapped_to_model() {
        let messages = vec![msg("assistant", "Hi there")];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.contains("<|turn>model\nHi there<turn|>"));
    }

    #[test]
    fn test_tool_definition_format() {
        let params = serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let def = format_tool_definition("get_weather", Some("Gets weather"), Some(&params));
        assert!(def.starts_with("<|tool>declaration:get_weather{"));
        assert!(def.contains("<|\"|>Gets weather<|\"|>"));
        assert!(def.ends_with("<tool|>"));
    }

    #[test]
    fn test_thinking_enabled() {
        let messages = vec![ChatMessage { role: "user".into(), content: "Hi".into(), tool_calls: None, tool_call_id: None }];
        let options = ChatFormatOptions { enable_thinking: true, ..Default::default() };
        let prompt = format_chat_prompt_with_options(&messages, &options);
        assert!(prompt.contains("<|think|>"));
        assert!(!prompt.contains("<|channel>thought\n<channel|>"));
    }

    #[test]
    fn test_thinking_disabled_suppresses() {
        let messages = vec![ChatMessage { role: "user".into(), content: "Hi".into(), tool_calls: None, tool_call_id: None }];
        let prompt = format_chat_prompt_with_options(&messages, &ChatFormatOptions::default());
        assert!(prompt.ends_with("<|channel>thought\n<channel|>"));
    }
}
