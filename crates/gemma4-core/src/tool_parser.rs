use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
}

const TOOL_CALL_OPEN: &str = "<|tool_call>";
const TOOL_CALL_CLOSE: &str = "<tool_call|>";
const CALL_PREFIX: &str = "call:";

/// Convert Gemma DSL argument block to valid JSON.
///
/// Steps:
/// 1. Replace `<|"|>` with `"` (Gemma's string delimiter token).
/// 2. Quote unquoted object keys: `{key:` → `{"key":` and `,key:` → `,"key":`.
fn gemma_dsl_to_json(dsl: &str) -> String {
    // Step 1 – swap the Gemma string-delimiter token for a real double-quote.
    let step1 = dsl.replace("<|\"|>", "\"");

    // Step 2 – quote bare object keys.
    // A bare key starts after `{` or `,` (with optional whitespace) and ends
    // before `:`.  We build the result character-by-character so we can
    // handle nested braces / already-quoted strings without pulling in a
    // regex dependency.
    let mut out = String::with_capacity(step1.len() + 32);
    let chars: Vec<char> = step1.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        // Inside a JSON string – copy verbatim until the closing `"`.
        if ch == '"' {
            out.push(ch);
            i += 1;
            while i < len {
                let sc = chars[i];
                out.push(sc);
                i += 1;
                if sc == '\\' {
                    // escaped character – consume next char as-is
                    if i < len {
                        out.push(chars[i]);
                        i += 1;
                    }
                } else if sc == '"' {
                    break;
                }
            }
            continue;
        }

        // After `{` or `,` we may have a bare (unquoted) key.
        if ch == '{' || ch == ',' {
            out.push(ch);
            i += 1;

            // Skip whitespace.
            let ws_start = i;
            while i < len && chars[i].is_whitespace() {
                i += 1;
            }
            let ws: String = chars[ws_start..i].iter().collect();

            // Peek ahead: is the next token a bare identifier followed by `:`?
            if i < len && chars[i] != '"' && chars[i] != '}' && chars[i] != ']' {
                // Collect the potential key.
                let key_start = i;
                while i < len && chars[i] != ':' && chars[i] != ',' && chars[i] != '}' {
                    i += 1;
                }
                let key: String = chars[key_start..i].iter().collect();
                let key_trimmed = key.trim();

                if i < len && chars[i] == ':' && !key_trimmed.is_empty() {
                    // It is a bare key – emit it quoted.
                    out.push_str(&ws);
                    out.push('"');
                    out.push_str(key_trimmed);
                    out.push('"');
                    // leave `i` pointing at `:` so the main loop emits it
                } else {
                    // Not a bare key – put everything back.
                    out.push_str(&ws);
                    out.push_str(&key);
                }
            } else {
                out.push_str(&ws);
            }
            continue;
        }

        out.push(ch);
        i += 1;
    }

    out
}

/// Parse a single tool call body (the text after "call:").
///
/// Handles two formats:
///   1. Standard: `NAME{key:val,...}` (Gemma DSL brace format)
///   2. Fallback: `NAME\nCODE` or `NAME:suffix\nCODE` (bare code block)
fn parse_single_call(body: &str) -> Option<ParsedToolCall> {
    // Standard format: NAME{key:val,...}
    if let Some(brace_pos) = body.find('{') {
        let name = body[..brace_pos].trim().to_string();
        let args_str = &body[brace_pos..];
        let json_str = gemma_dsl_to_json(args_str);
        if let Ok(arguments) = serde_json::from_str(&json_str) {
            if !name.is_empty() {
                return Some(ParsedToolCall { name, arguments });
            }
        }
    }

    // Fallback: NAME\nCODE or NAME:CODE_BLOCK_TYPE\nCODE
    // Handle "python\ncode..." or "python:code_block\ncode..."
    let (name, code) = if let Some(newline_pos) = body.find('\n') {
        let raw_name = body[..newline_pos].trim_end_matches(|c: char| c == ':' || c == ' ');
        // Strip any suffix like ":code_block" from the name
        let clean_name = raw_name.split(':').next().unwrap_or(raw_name).trim();
        let code = &body[newline_pos + 1..];
        (clean_name.to_string(), code.to_string())
    } else {
        return None;
    };

    if code.is_empty() {
        return None;
    }

    // Map common tool names to our sandbox names
    let mapped_name = match name.as_str() {
        "python" | "python_interpreter" | "code_interpreter" => "python_interpreter",
        other => other,
    };

    Some(ParsedToolCall {
        name: mapped_name.to_string(),
        arguments: serde_json::json!({"code": code, "language": "python"}),
    })
}

/// Parse all tool calls from model output text.
///
/// Gemma 4 emits tool calls as:
/// `<|tool_call>call:FUNC_NAME{...args...}<tool_call|>`
pub fn parse_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(open_pos) = text[search_from..].find(TOOL_CALL_OPEN) {
        let block_start = search_from + open_pos + TOOL_CALL_OPEN.len();

        let close_pos = match text[block_start..].find(TOOL_CALL_CLOSE) {
            Some(p) => p,
            None => break, // malformed – no closing tag
        };

        let block = &text[block_start..block_start + close_pos];
        search_from = block_start + close_pos + TOOL_CALL_CLOSE.len();

        // The block must start with "call:".
        if !block.starts_with(CALL_PREFIX) {
            continue;
        }
        let rest = &block[CALL_PREFIX.len()..];

        if let Some(call) = parse_single_call(rest) {
            calls.push(call);
        }
    }

    calls
}

/// Return `true` if the text contains at least one tool-call block.
pub fn has_tool_calls(text: &str) -> bool {
    text.contains(TOOL_CALL_OPEN)
}

/// Return the text that appears before the first tool-call block.
///
/// Returns `None` if that prefix is empty or absent.
pub fn content_before_tool_calls(text: &str) -> Option<String> {
    let content = match text.find(TOOL_CALL_OPEN) {
        Some(pos) => &text[..pos],
        None => text,
    };
    let trimmed = content.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let text = r#"<|tool_call>call:get_weather{city:<|"|>Bangkok<|"|>}<tool_call|><|tool_call>call:get_time{timezone:<|"|>UTC<|"|>}<tool_call|>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_time");
    }

    #[test]
    fn test_has_tool_calls() {
        assert!(has_tool_calls("text <|tool_call>call:f{}<tool_call|>"));
        assert!(!has_tool_calls("normal text"));
    }

    #[test]
    fn test_no_tool_calls() {
        assert!(parse_tool_calls("Just text").is_empty());
    }

    #[test]
    fn test_numeric_argument() {
        let text = r#"<|tool_call>call:add{a:1,b:2}<tool_call|>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "add");
        assert_eq!(calls[0].arguments["a"], 1);
        assert_eq!(calls[0].arguments["b"], 2);
    }

    #[test]
    fn test_content_before_tool_calls_present() {
        let text = r#"Here is the answer. <|tool_call>call:f{}<tool_call|>"#;
        assert_eq!(
            content_before_tool_calls(text),
            Some("Here is the answer.".to_string())
        );
    }

    #[test]
    fn test_content_before_tool_calls_empty() {
        let text = r#"<|tool_call>call:f{}<tool_call|>"#;
        assert_eq!(content_before_tool_calls(text), None);
    }

    #[test]
    fn test_content_before_tool_calls_no_calls() {
        let text = "Just plain text";
        assert_eq!(
            content_before_tool_calls(text),
            Some("Just plain text".to_string())
        );
    }

    #[test]
    fn test_gemma_dsl_to_json_mixed() {
        // key:value where value is a string token
        let dsl = r#"{name:<|"|>Alice<|"|>,age:30}"#;
        let json = gemma_dsl_to_json(dsl);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["name"], "Alice");
        assert_eq!(v["age"], 30);
    }
}
