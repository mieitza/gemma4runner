# Gemma4Runner Phase 4 — Advanced Features

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tool calling, thinking mode, vision (image input), and audio input support.

**Architecture:** Tool calling and thinking extend the chat template and API types. Vision and audio reuse candle-transformers' existing Gemma 4 vision/audio implementations — we wrap them into a multimodal model that combines text + vision + audio embeddings before feeding to the text decoder.

**Tech Stack:** Same as Phase 3. Reuses `candle_transformers::models::gemma4::{vision, audio, multimodal_embedding}`.

---

## Key Implementation Details

### Tool Calling Format (Gemma 4 specific)
- Tool definitions in system prompt: `<|tool>declaration:FUNC_NAME{description:<|"|>...<|"|>,parameters:{...}}<tool|>`
- Model output: `<|tool_call>call:FUNC_NAME{arg:<|"|>value<|"|>}<tool_call|>`
- Tool response: `<|tool_response>response:FUNC_NAME{value:<|"|>result<|"|>}<tool_response|>`
- String delimiter: `<|"|>` replaces quotes in the tool DSL

### Thinking Mode
- Enable: add `<|think|>\n` at top of system prompt
- Model output: `<|channel>thought\n...reasoning...\n<channel|>` before content
- Disable: append `<|channel>thought\n<channel|>` to generation prompt (empty thinking block)

### Vision/Audio
- candle-transformers has `VisionTower` and `AudioModel` for Gemma 4
- `MultimodalEmbedder` projects vision/audio embeddings to text hidden dimension
- Image tokens (`<|image|>`, id=258880) mark where vision embeddings are inserted
- Audio tokens (`<|audio|>`, id=258881) mark where audio embeddings are inserted

---

### Task 1: Tool Calling — Chat Template + API Types

**Files:**
- Modify: `crates/gemma4-core/src/chat_template.rs`
- Modify: `crates/gemma4-api/src/types/chat.rs`
- Modify: `crates/gemma4-api/src/types/common.rs`

- [ ] **Step 1: Add tool types to API**

In `crates/gemma4-api/src/types/common.rs`, add:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
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
```

Update `Message` to support tool calls and tool role:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}
```

Add `Tool` role variant:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role { System, User, Assistant, Tool }
```

- [ ] **Step 2: Add tools and tool_choice to ChatCompletionRequest**

In `crates/gemma4-api/src/types/chat.rs`, add fields:

```rust
    #[serde(default)]
    pub tools: Option<Vec<common::Tool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
```

Update `ChoiceMessage` to include optional tool_calls:

```rust
#[derive(Debug, Serialize)]
pub struct ChoiceMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<common::ToolCall>>,
}
```

Add `ToolCalls` to `FinishReason`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason { Stop, Length, ToolCalls }
```

- [ ] **Step 3: Update chat_template.rs for tool formatting**

In `crates/gemma4-core/src/chat_template.rs`, add tool-related functions:

```rust
/// Format a tool definition for the Gemma 4 system prompt.
/// Uses the Gemma 4 tool DSL with <|"|> string delimiters.
pub fn format_tool_definition(name: &str, description: Option<&str>, parameters: Option<&serde_json::Value>) -> String {
    let mut def = format!("<|tool>declaration:{}", name);
    def.push('{');

    if let Some(desc) = description {
        def.push_str(&format!("description:<|\"|>{}<|\"|>", desc));
    }

    if let Some(params) = parameters {
        if !description.is_none() {
            def.push(',');
        }
        def.push_str("parameters:");
        def.push_str(&format_value_gemma(params));
    }

    def.push('}');
    def.push_str("<tool|>");
    def
}

/// Format a JSON value using Gemma 4's DSL (strings wrapped in <|"|>).
fn format_value_gemma(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => format!("<|\"|>{}<|\"|>", s),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_value_gemma).collect();
            format!("[{}]", items.join(","))
        }
        serde_json::Value::Object(obj) => {
            let pairs: Vec<String> = obj.iter()
                .map(|(k, v)| format!("{}:{}", k, format_value_gemma(v)))
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}
```

Add `serde_json` to gemma4-core dependencies if not already there.

- [ ] **Step 4: Update format_chat_prompt for tools**

Extend `ChatMessage` and `format_chat_prompt`:

```rust
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

#[derive(Debug, Clone, Default)]
pub struct ChatFormatOptions {
    pub tools: Vec<ToolDef>,
    pub enable_thinking: bool,
}

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

pub fn format_chat_prompt_with_options(messages: &[ChatMessage], options: &ChatFormatOptions) -> String {
    let mut prompt = String::from("<bos>");

    // If tools or thinking, inject/modify system prompt
    let has_tools = !options.tools.is_empty();
    let needs_system = has_tools || options.enable_thinking;

    // Check if first message is system
    let (system_content, msg_start) = if !messages.is_empty() && messages[0].role == "system" {
        (Some(messages[0].content.clone()), 1)
    } else {
        (None, 0)
    };

    if needs_system || system_content.is_some() {
        prompt.push_str("<|turn>system\n");
        if options.enable_thinking {
            prompt.push_str("<|think|>\n");
        }
        if let Some(content) = &system_content {
            prompt.push_str(content);
            prompt.push('\n');
        }
        for tool in &options.tools {
            prompt.push_str(&format_tool_definition(&tool.name, tool.description.as_deref(), tool.parameters.as_ref()));
            prompt.push('\n');
        }
        prompt.push_str("<turn|>\n");
    }

    for msg in &messages[msg_start..] {
        let role = match msg.role.as_str() {
            "assistant" => "model",
            other => other,
        };

        prompt.push_str(&format!("<|turn>{}\n", role));

        if role == "model" {
            // Handle tool calls in assistant messages
            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    prompt.push_str(&format!("<|tool_call>call:{}{{", tc.name));
                    // Parse arguments JSON and format in Gemma DSL
                    if let Ok(args) = serde_json::from_str::<serde_json::Value>(&tc.arguments) {
                        if let serde_json::Value::Object(obj) = args {
                            let pairs: Vec<String> = obj.iter()
                                .map(|(k, v)| format!("{}:{}", k, format_value_gemma(v)))
                                .collect();
                            prompt.push_str(&pairs.join(","));
                        }
                    }
                    prompt.push_str("}<tool_call|>");
                }
            }
        }

        if msg.role == "tool" {
            // Tool response
            if let Some(tc_id) = &msg.tool_call_id {
                prompt.push_str(&format!("<|tool_response>response:{}{{value:<|\"|>{}<|\"|>}}<tool_response|>", tc_id, msg.content));
            }
        } else {
            prompt.push_str(&msg.content);
        }

        prompt.push_str("<turn|>\n");
    }

    // Generation prompt
    prompt.push_str("<|turn>model\n");
    if !options.enable_thinking {
        prompt.push_str("<|channel>thought\n<channel|>");
    }
    prompt
}
```

Keep the existing `format_chat_prompt` as-is for backward compatibility — it's used by the engine. Update it to delegate:

```rust
pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    format_chat_prompt_with_options(messages, &ChatFormatOptions::default())
}
```

- [ ] **Step 5: Fix existing tests and add new ones**

Update existing tests to work with the new ChatMessage (which now has optional fields). Add tests for tool formatting:

```rust
#[test]
fn test_tool_definition_format() {
    let params = serde_json::json!({
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    });
    let def = format_tool_definition("get_weather", Some("Gets weather"), Some(&params));
    assert!(def.starts_with("<|tool>declaration:get_weather{"));
    assert!(def.contains("description:<|\"|>Gets weather<|\"|>"));
    assert!(def.ends_with("<tool|>"));
}

#[test]
fn test_thinking_enabled_prompt() {
    let messages = vec![ChatMessage {
        role: "user".into(), content: "Hello".into(),
        tool_calls: None, tool_call_id: None,
    }];
    let options = ChatFormatOptions { enable_thinking: true, ..Default::default() };
    let prompt = format_chat_prompt_with_options(&messages, &options);
    assert!(prompt.contains("<|turn>system\n<|think|>"));
    assert!(!prompt.contains("<|channel>thought\n<channel|>"));
}

#[test]
fn test_thinking_disabled_prompt() {
    let messages = vec![ChatMessage {
        role: "user".into(), content: "Hello".into(),
        tool_calls: None, tool_call_id: None,
    }];
    let prompt = format_chat_prompt_with_options(&messages, &ChatFormatOptions::default());
    assert!(prompt.contains("<|channel>thought\n<channel|>"));
}
```

- [ ] **Step 6: Verify and commit**

Run: `cargo test --workspace`

```bash
git add crates/
git commit -m "feat: add tool calling types and chat template formatting"
```

---

### Task 2: Tool Call Parser

**Files:**
- Create: `crates/gemma4-core/src/tool_parser.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement tool call parser with tests**

Create `crates/gemma4-core/src/tool_parser.rs`:

```rust
use anyhow::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Parse tool calls from model output.
/// Gemma 4 emits: <|tool_call>call:FUNC_NAME{key:<|"|>value<|"|>,...}<tool_call|>
pub fn parse_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();
    let mut remaining = text;

    while let Some(start) = remaining.find("<|tool_call>call:") {
        let after_prefix = &remaining[start + "<|tool_call>call:".len()..];
        if let Some(end) = after_prefix.find("<tool_call|>") {
            let call_body = &after_prefix[..end];
            if let Some(parsed) = parse_single_call(call_body) {
                calls.push(parsed);
            }
            remaining = &after_prefix[end + "<tool_call|>".len()..];
        } else {
            break;
        }
    }
    calls
}

fn parse_single_call(body: &str) -> Option<ParsedToolCall> {
    // body format: FUNC_NAME{key:<|"|>value<|"|>,key2:42}
    let brace_pos = body.find('{')?;
    let name = body[..brace_pos].to_string();
    let args_str = &body[brace_pos..];

    // Convert Gemma DSL to JSON
    let json_str = gemma_dsl_to_json(args_str);
    let arguments: serde_json::Value = serde_json::from_str(&json_str).ok()?;

    Some(ParsedToolCall { name, arguments })
}

/// Convert Gemma 4's tool DSL to JSON.
/// - Replace <|"|> with "
/// - The DSL uses key:value pairs without quotes on keys, so we need to quote them
fn gemma_dsl_to_json(dsl: &str) -> String {
    // First, replace <|"|> with actual quotes
    let s = dsl.replace("<|\"|>", "\"");

    // Quote unquoted keys: find patterns like {key: or ,key: and add quotes
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '{' || chars[i] == ',' {
            result.push(chars[i]);
            i += 1;
            // Skip whitespace
            while i < chars.len() && chars[i].is_whitespace() {
                result.push(chars[i]);
                i += 1;
            }
            // Collect key (alphanumeric + underscore)
            if i < chars.len() && (chars[i].is_alphabetic() || chars[i] == '_') {
                let key_start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let key = &s[key_start..i];
                // Check if followed by colon (it's a key)
                if i < chars.len() && chars[i] == ':' {
                    result.push('"');
                    result.push_str(key);
                    result.push('"');
                } else {
                    result.push_str(key);
                }
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

/// Check if model output contains tool calls.
pub fn has_tool_calls(text: &str) -> bool {
    text.contains("<|tool_call>")
}

/// Extract content before any tool calls.
pub fn content_before_tool_calls(text: &str) -> Option<String> {
    if let Some(pos) = text.find("<|tool_call>") {
        let before = text[..pos].trim();
        if before.is_empty() {
            None
        } else {
            Some(before.to_string())
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_tool_call() {
        let text = r#"<|tool_call>call:get_weather{city:<|"|>Bangkok<|"|>}<tool_call|>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Bangkok");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let text = r#"<|tool_call>call:get_weather{city:<|"|>Bangkok<|"|>}<tool_call|><|tool_call>call:get_time{timezone:<|"|>Asia/Bangkok<|"|>}<tool_call|>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_time");
    }

    #[test]
    fn test_has_tool_calls() {
        assert!(has_tool_calls("Some text <|tool_call>call:foo{}<tool_call|>"));
        assert!(!has_tool_calls("Just normal text"));
    }

    #[test]
    fn test_no_tool_calls() {
        let calls = parse_tool_calls("Just a regular response with no tools.");
        assert!(calls.is_empty());
    }
}
```

- [ ] **Step 2: Add module and test**

Add `pub mod tool_parser;` to lib.rs.

Run: `cargo test -p gemma4-core tool_parser`

- [ ] **Step 3: Commit**

```bash
git add crates/gemma4-core/
git commit -m "feat(core): add tool call parser for Gemma 4 DSL format"
```

---

### Task 3: Think Parser (Streaming State Machine)

**Files:**
- Create: `crates/gemma4-core/src/think_parser.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Implement think parser with tests**

Create `crates/gemma4-core/src/think_parser.rs`:

```rust
/// Streaming parser for Gemma 4's thinking tags.
/// Thinking content: <|channel>thought\n...content...\n<channel|>
///
/// Buffers tokens until it can determine whether they're thinking or content.

#[derive(Debug, Clone, PartialEq)]
pub enum ThinkEvent {
    Thinking(String),
    Content(String),
}

pub struct ThinkParser {
    buffer: String,
    in_thinking: bool,
    thinking_started: bool,
    include_thinking: bool,
}

const THINK_OPEN: &str = "<|channel>thought\n";
const THINK_CLOSE: &str = "\n<channel|>";

impl ThinkParser {
    pub fn new(include_thinking: bool) -> Self {
        Self {
            buffer: String::new(),
            in_thinking: false,
            thinking_started: false,
            include_thinking,
        }
    }

    /// Feed a token and get back zero or more events.
    pub fn feed(&mut self, token: &str) -> Vec<ThinkEvent> {
        self.buffer.push_str(token);
        let mut events = Vec::new();

        loop {
            if !self.in_thinking {
                // Look for thinking open tag
                if let Some(pos) = self.buffer.find(THINK_OPEN) {
                    // Emit any content before the tag
                    if pos > 0 {
                        events.push(ThinkEvent::Content(self.buffer[..pos].to_string()));
                    }
                    self.buffer = self.buffer[pos + THINK_OPEN.len()..].to_string();
                    self.in_thinking = true;
                    self.thinking_started = true;
                    continue;
                }
                // Check if buffer might be start of tag (partial match)
                if THINK_OPEN.starts_with(&self.buffer) || self.buffer.ends_with('<') {
                    break; // Wait for more tokens
                }
                // No tag possible, emit as content
                if !self.buffer.is_empty() {
                    events.push(ThinkEvent::Content(self.buffer.clone()));
                    self.buffer.clear();
                }
                break;
            } else {
                // In thinking mode, look for close tag
                if let Some(pos) = self.buffer.find(THINK_CLOSE) {
                    // Emit thinking content before close tag
                    if pos > 0 && self.include_thinking {
                        events.push(ThinkEvent::Thinking(self.buffer[..pos].to_string()));
                    }
                    self.buffer = self.buffer[pos + THINK_CLOSE.len()..].to_string();
                    self.in_thinking = false;
                    continue;
                }
                // Check for partial close tag
                if self.buffer.ends_with('\n') || self.buffer.ends_with("\n<") {
                    break; // Wait for more tokens
                }
                // Emit thinking content
                if !self.buffer.is_empty() && self.include_thinking {
                    events.push(ThinkEvent::Thinking(self.buffer.clone()));
                    self.buffer.clear();
                } else if !self.include_thinking {
                    self.buffer.clear();
                }
                break;
            }
        }

        events
    }

    /// Flush remaining buffer at end of generation.
    pub fn flush(&mut self) -> Vec<ThinkEvent> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            if self.in_thinking && self.include_thinking {
                events.push(ThinkEvent::Thinking(self.buffer.clone()));
            } else if !self.in_thinking {
                events.push(ThinkEvent::Content(self.buffer.clone()));
            }
            self.buffer.clear();
        }
        events
    }

    pub fn had_thinking(&self) -> bool {
        self.thinking_started
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_thinking() {
        let mut parser = ThinkParser::new(true);
        let events = parser.feed("Hello world");
        assert_eq!(events, vec![ThinkEvent::Content("Hello world".into())]);
    }

    #[test]
    fn test_thinking_then_content() {
        let mut parser = ThinkParser::new(true);
        let mut all = Vec::new();
        all.extend(parser.feed("<|channel>thought\n"));
        all.extend(parser.feed("Let me think..."));
        all.extend(parser.feed("\n<channel|>"));
        all.extend(parser.feed("The answer is 4."));
        all.extend(parser.flush());

        let thinking: Vec<_> = all.iter().filter(|e| matches!(e, ThinkEvent::Thinking(_))).collect();
        let content: Vec<_> = all.iter().filter(|e| matches!(e, ThinkEvent::Content(_))).collect();
        assert!(!thinking.is_empty());
        assert!(!content.is_empty());
    }

    #[test]
    fn test_thinking_stripped_when_disabled() {
        let mut parser = ThinkParser::new(false);
        let mut all = Vec::new();
        all.extend(parser.feed("<|channel>thought\nSecret reasoning\n<channel|>Visible answer"));
        all.extend(parser.flush());

        // Should have no Thinking events
        assert!(all.iter().all(|e| matches!(e, ThinkEvent::Content(_))));
        // Should have content
        let content: String = all.iter().filter_map(|e| match e {
            ThinkEvent::Content(s) => Some(s.as_str()),
            _ => None,
        }).collect();
        assert!(content.contains("Visible answer"));
    }
}
```

- [ ] **Step 2: Add module and test**

Add `pub mod think_parser;` to lib.rs.

Run: `cargo test -p gemma4-core think_parser`

- [ ] **Step 3: Commit**

```bash
git add crates/gemma4-core/
git commit -m "feat(core): add streaming think parser for <|channel>thought tags"
```

---

### Task 4: Wire Tool Calling + Thinking into Engine and API

**Files:**
- Modify: `crates/gemma4-core/src/engine.rs`
- Modify: `crates/gemma4-api/src/handlers/chat.rs`
- Modify: `crates/gemma4-api/src/types/chat.rs`

- [ ] **Step 1: Add thinking and tools to InferenceRequest/Event**

In `crates/gemma4-core/src/engine.rs`, update:

```rust
pub struct InferenceRequest {
    pub id: String,
    pub input: InferenceInput,
    pub sampling: SamplingParams,
    pub response_tx: mpsc::Sender<InferenceEvent>,
    pub tools: Vec<crate::chat_template::ToolDef>,
    pub include_thinking: bool,
}

pub enum InferenceEvent {
    Token(String),
    ThinkingToken(String),
    ToolCalls(Vec<crate::tool_parser::ParsedToolCall>),
    Usage(UsageStats),
    Done(FinishReason),
    Error(String),
}

pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}
```

- [ ] **Step 2: Update process_request for tools and thinking**

In `process_request`, update prompt construction to use `format_chat_prompt_with_options`:

```rust
let prompt = match &request.input {
    InferenceInput::Chat(messages) => {
        let options = crate::chat_template::ChatFormatOptions {
            tools: request.tools.clone(),
            enable_thinking: request.include_thinking,
        };
        crate::chat_template::format_chat_prompt_with_options(messages, &options)
    }
    InferenceInput::Raw(text) => text.clone(),
};
```

After generation completes, check for tool calls:

```rust
// After collecting all tokens into a string, check for tool calls
let full_output: String = /* collected tokens */;
if crate::tool_parser::has_tool_calls(&full_output) {
    let parsed_calls = crate::tool_parser::parse_tool_calls(&full_output);
    let _ = request.response_tx.send(InferenceEvent::ToolCalls(parsed_calls));
    finish_reason = FinishReason::ToolCalls;
}
```

- [ ] **Step 3: Add include_thinking to ChatCompletionRequest**

In `crates/gemma4-api/src/types/chat.rs`:

```rust
    #[serde(default)]
    pub include_thinking: Option<bool>,
```

Add thinking field to response:

```rust
#[derive(Debug, Serialize)]
pub struct ChoiceMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<common::ToolCall>>,
}
```

- [ ] **Step 4: Update chat handler**

Update the handler to pass tools and thinking to InferenceRequest, and handle ToolCalls events in the response collection.

- [ ] **Step 5: Update FinishReason mappings**

Make sure `FinishReason::ToolCalls` maps correctly between core and API types.

- [ ] **Step 6: Verify and commit**

Run: `cargo test --workspace`

```bash
git add crates/
git commit -m "feat: wire tool calling and thinking into engine and API"
```

---

### Task 5: Vision Support

**Files:**
- Create: `crates/gemma4-core/src/multimodal.rs`
- Modify: `crates/gemma4-core/src/engine.rs`
- Modify: `crates/gemma4-core/src/loader.rs`
- Modify: `crates/gemma4-core/src/lib.rs`

- [ ] **Step 1: Create multimodal wrapper**

Create `crates/gemma4-core/src/multimodal.rs` that wraps candle-transformers' vision/audio implementations:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use candle_transformers::models::gemma4::vision::VisionTower;
use candle_transformers::models::gemma4::multimodal_embedding::MultimodalEmbedder;
use candle_transformers::models::gemma4::Gemma4Config as CandleGemma4Config;

pub struct VisionEncoder {
    tower: VisionTower,
    embedder: MultimodalEmbedder,
}

impl VisionEncoder {
    pub fn new(config: &CandleGemma4Config, vb: VarBuilder) -> Result<Self> {
        let tower = VisionTower::new(&config.vision_config, vb.pp("vision_tower"))?;
        let embedder = MultimodalEmbedder::new(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("embed_vision"),
        )?;
        Ok(Self { tower, embedder })
    }

    /// Process a list of images (as pixel tensors) into text-space embeddings.
    pub fn encode(&self, pixel_values: &[Tensor]) -> Result<Vec<Tensor>> {
        let vision_embeds = self.tower.forward(pixel_values)?;
        let projected: Vec<Tensor> = vision_embeds.iter()
            .map(|e| self.embedder.forward(e))
            .collect::<Result<Vec<_>>>()?;
        Ok(projected)
    }
}
```

Note: This depends on the exact candle-transformers API. The implementer should read the actual candle source and adapt. If the API doesn't match, wrap accordingly.

- [ ] **Step 2: Update loader for multimodal**

Add a `LoadedMultimodalModel` struct or extend `LoadedModel` to optionally include the vision encoder.

- [ ] **Step 3: Update engine for vision input**

Add image data to `InferenceRequest` (as raw pixel tensor or file path). The engine processes images through the vision encoder, inserts vision embeddings at `image_token_id` positions, then runs the text decoder on the combined embeddings.

- [ ] **Step 4: Verify and commit**

This is the most complex task. The implementer should focus on getting the struct hierarchy right and compiling. Actual image preprocessing (loading, resizing, normalization) can be simple initially.

```bash
git add crates/
git commit -m "feat(core): add vision encoder using candle-transformers VisionTower"
```

---

### Task 6: Audio Support (E2B/E4B only)

**Files:**
- Modify: `crates/gemma4-core/src/multimodal.rs`

- [ ] **Step 1: Add audio encoder**

Similar to vision, wrap candle-transformers' `AudioModel`:

```rust
use candle_transformers::models::gemma4::audio::AudioModel;

pub struct AudioEncoder {
    model: AudioModel,
    embedder: MultimodalEmbedder,
}

impl AudioEncoder {
    pub fn new(config: &CandleGemma4Config, vb: VarBuilder) -> Result<Option<Self>> {
        let audio_config = match &config.audio_config {
            Some(cfg) => cfg,
            None => return Ok(None),
        };
        let model = AudioModel::new(audio_config, vb.pp("audio_tower"))?;
        let output_dim = audio_config.output_proj_dims.unwrap_or(audio_config.hidden_size);
        let embedder = MultimodalEmbedder::new(
            output_dim,
            config.text_config.hidden_size,
            vb.pp("embed_audio"),
        )?;
        Ok(Some(Self { model, embedder }))
    }

    pub fn encode(&self, audio_mel: &Tensor, audio_mask: &Tensor) -> Result<Tensor> {
        let (audio_embeds, _mask) = self.model.forward(audio_mel, audio_mask)?;
        self.embedder.forward(&audio_embeds)
    }
}
```

- [ ] **Step 2: Verify and commit**

```bash
git add crates/
git commit -m "feat(core): add audio encoder for E2B/E4B models"
```

---

### Task 7: Full Verification

- [ ] **Step 1: cargo build**
- [ ] **Step 2: cargo test --workspace**
- [ ] **Step 3: Verify tool call parsing tests pass**
- [ ] **Step 4: Verify think parser tests pass**
- [ ] **Step 5: Commit any fixes**
