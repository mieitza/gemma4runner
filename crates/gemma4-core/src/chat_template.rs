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
    /// JSON-encoded arguments string (e.g. `{"city":"Bangkok"}`).
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

/// Convert a JSON arguments string to Gemma DSL call format.
///
/// Input:  `{"city":"Bangkok","units":"metric"}`
/// Output: `city:<|"|>Bangkok<|"|>,units:<|"|>metric<|"|>`
///
/// The caller wraps this in `call:NAME{...}`.
fn json_args_to_gemma_dsl(arguments: &str) -> String {
    let value: serde_json::Value = serde_json::from_str(arguments)
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    match &value {
        serde_json::Value::Object(map) => {
            let pairs: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}:{}", k, format_value_gemma(v)))
                .collect();
            pairs.join(",")
        }
        _ => String::new(),
    }
}

pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    format_chat_prompt_with_options(messages, &ChatFormatOptions::default())
}

pub fn format_chat_prompt_with_options(messages: &[ChatMessage], options: &ChatFormatOptions) -> String {
    let mut prompt = String::from("<bos>");

    // Determine if the first message is a system/developer message.
    let first_is_system = messages
        .first()
        .map(|m| m.role == "system" || m.role == "developer")
        .unwrap_or(false);

    // Emit system preamble turn if: thinking enabled, tools present, OR first message is system/developer.
    if !options.tools.is_empty() || options.enable_thinking || first_is_system {
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

        // If the first message is a system/developer message, include its content.
        if first_is_system {
            if let Some(first_msg) = messages.first() {
                if !system_content.is_empty() {
                    system_content.push('\n');
                }
                system_content.push_str(&first_msg.content);
            }
        }

        prompt.push_str(&format!("<|turn>system\n{}<turn|>\n", system_content.trim_end()));
    }

    // Determine the starting index: skip the first message if it was a system/developer message
    // already included in the system preamble.
    let msg_iter_start = if first_is_system { 1 } else { 0 };

    for msg in &messages[msg_iter_start..] {
        match msg.role.as_str() {
            "tool" => {
                // Tool response message.
                // tool_call_id holds the function name in Gemma's response DSL.
                let func_name = msg.tool_call_id.as_deref().unwrap_or("unknown");
                prompt.push_str(&format!(
                    "<|turn>tool\n<|tool_response>response:{}{{value:<|\"|>{}<|\"|>}}<tool_response|><turn|>\n",
                    func_name, msg.content
                ));
            }
            "assistant" => {
                // Check if this assistant message contains tool calls.
                if let Some(calls) = &msg.tool_calls {
                    let mut turn_content = String::new();
                    for call in calls {
                        let dsl_args = json_args_to_gemma_dsl(&call.arguments);
                        turn_content.push_str(&format!(
                            "<|tool_call>call:{}{{{}}}<tool_call|>",
                            call.name, dsl_args
                        ));
                    }
                    prompt.push_str(&format!("<|turn>model\n{}<turn|>\n", turn_content));
                } else {
                    prompt.push_str(&format!("<|turn>model\n{}<turn|>\n", msg.content));
                }
            }
            "system" | "developer" => {
                // Should not appear here (consumed above), but emit as-is for safety.
                prompt.push_str(&format!("<|turn>system\n{}<turn|>\n", msg.content));
            }
            other => {
                prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", other, msg.content));
            }
        }
    }

    // Begin generation turn.
    prompt.push_str("<|turn>model\n");

    // When thinking is disabled, suppress thinking channel.
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
        // Thinking suppression appended when enable_thinking is false.
        assert!(prompt.ends_with("<|channel>thought\n<channel|>"));
    }

    #[test]
    fn test_system_and_user() {
        let messages = vec![
            msg("system", "You are helpful."),
            msg("user", "Hi"),
        ];
        let prompt = format_chat_prompt(&messages);
        // System message should be in the preamble turn.
        assert!(prompt.contains("<|turn>system\nYou are helpful.<turn|>"));
        assert!(prompt.contains("<|turn>user\nHi<turn|>"));
        // System message must NOT appear as a second system turn.
        let system_turn_count = prompt.matches("<|turn>system\n").count();
        assert_eq!(system_turn_count, 1, "system turn should appear exactly once");
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
        // No suppression when thinking is enabled.
        assert!(!prompt.contains("<|channel>thought\n<channel|>"));
    }

    #[test]
    fn test_thinking_disabled_suppression_tokens() {
        let messages = vec![ChatMessage { role: "user".into(), content: "Hi".into(), tool_calls: None, tool_call_id: None }];
        let prompt = format_chat_prompt_with_options(&messages, &ChatFormatOptions::default());
        // Official template: suppress thinking when enable_thinking is false.
        assert!(prompt.ends_with("<|channel>thought\n<channel|>"));
        assert!(prompt.contains("<|channel>"));
    }

    #[test]
    fn test_tool_call_format() {
        let messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(vec![ToolCallInfo {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Bangkok"}"#.into(),
                }]),
                tool_call_id: None,
            },
        ];
        let prompt = format_chat_prompt(&messages);
        assert!(
            prompt.contains("<|tool_call>call:get_weather{city:<|\"|>Bangkok<|\"|>}<tool_call|>"),
            "got: {}", prompt
        );
    }

    #[test]
    fn test_tool_response_format() {
        let messages = vec![
            ChatMessage {
                role: "tool".into(),
                content: "25°C sunny".into(),
                tool_calls: None,
                tool_call_id: Some("get_weather".into()),
            },
        ];
        let prompt = format_chat_prompt(&messages);
        assert!(
            prompt.contains("<|tool_response>response:get_weather{value:<|\"|>25°C sunny<|\"|>}<tool_response|>"),
            "got: {}", prompt
        );
    }

    #[test]
    fn test_system_message_in_preamble_with_tools() {
        let params = serde_json::json!({"type": "object", "properties": {}});
        let options = ChatFormatOptions {
            tools: vec![ToolDef {
                name: "search".into(),
                description: Some("Search the web".into()),
                parameters: Some(params),
            }],
            enable_thinking: false,
        };
        let messages = vec![
            msg("system", "You are a helpful assistant."),
            msg("user", "Search for Rust"),
        ];
        let prompt = format_chat_prompt_with_options(&messages, &options);
        // System message content must appear once in the preamble.
        assert!(prompt.contains("You are a helpful assistant."));
        let system_turn_count = prompt.matches("<|turn>system\n").count();
        assert_eq!(system_turn_count, 1);
        // Tool declaration must be in the same preamble.
        assert!(prompt.contains("<|tool>declaration:search"));
    }

    #[test]
    fn test_format_chat_prompt_backward_compat() {
        // format_chat_prompt (no options) must still work.
        let messages = vec![msg("user", "Hello")];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.starts_with("<bos>"));
        assert!(prompt.contains("<|turn>user\nHello<turn|>"));
    }
}
