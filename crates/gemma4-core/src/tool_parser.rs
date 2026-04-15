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

/// Unescape literal `\n`, `\"`, `\\`, `\t` in a string.
/// The model often outputs these as two-character sequences instead of real control chars.
fn unescape_literals(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some('n') => { chars.next(); out.push('\n'); }
                Some('t') => { chars.next(); out.push('\t'); }
                Some('"') => { chars.next(); out.push('"'); }
                Some('\\') => { chars.next(); out.push('\\'); }
                _ => out.push(c),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Truncate code at the first non-code marker.
/// The model often generates code then continues with fabricated output/explanation.
fn truncate_code(code: &str) -> String {
    // Strip trailing DSL artifacts: "} or }\n or "\n} etc.
    let code = code.trim_end();
    let code = if code.ends_with("\"}") {
        &code[..code.len()-2]
    } else if code.ends_with("}") {
        &code[..code.len()-1]
    } else {
        code
    };
    let code = code.trim_end_matches('"').trim_end();
    // Cut at first triple-backtick (model closes its own code block)
    let code = if let Some(pos) = code.find("```") {
        &code[..pos]
    } else {
        code
    };
    // Cut at "**Output:**" or similar markers
    let code = if let Some(pos) = code.find("**Output") {
        &code[..pos]
    } else {
        code
    };
    // Cut at "(Note:" explanation
    let code = if let Some(pos) = code.find("(Note:") {
        &code[..pos]
    } else {
        code
    };
    code.trim_end().to_string()
}

/// Parse a single tool call body (the text after "call:").
///
/// Handles two formats:
///   1. Standard: `NAME{key:val,...}` (Gemma DSL brace format)
///   2. Fallback: `NAME\nCODE` or `NAME:suffix\nCODE` (bare code block)
///   3. Fallback with literal escapes: `NAME\\nCODE` (model outputs \n as two chars)
fn parse_single_call(body: &str) -> Option<ParsedToolCall> {
    // Standard format: NAME{key:val,...} or NAME:suffix {key:val,...}
    if let Some(brace_pos) = body.find('{') {
        let raw_name = body[..brace_pos].trim();
        // Strip suffixes like ":code_block", ":code" from the name
        let name = raw_name.split(':').next().unwrap_or(raw_name).trim().to_string();
        let args_str = &body[brace_pos..];
        let json_str = gemma_dsl_to_json(args_str);
        if let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&json_str) {
            if !name.is_empty() {
                // If arguments contain a "code" field with escaped newlines, unescape it
                let arguments = if let Some(code) = arguments.get("code").and_then(|v| v.as_str()) {
                    let mut unescaped = unescape_literals(code);
                    // Strip trailing DSL artifacts that leak from the Gemma tool call format.
                    // The model's code ends with something like: print(result)"}
                    // where "} is the closing of the DSL {code: "..."}
                    unescaped = unescaped.trim_end_matches('}')
                        .trim_end_matches('"')
                        .trim_end()
                        .to_string();
                    let unescaped = truncate_code(&unescaped);
                    let mut map = arguments.as_object().cloned().unwrap_or_default();
                    map.insert("code".to_string(), serde_json::Value::String(unescaped));
                    serde_json::Value::Object(map)
                } else {
                    arguments
                };
                return Some(ParsedToolCall { name, arguments });
            }
        }
    }

    // Unescape the entire body — model outputs literal \n instead of newlines
    let unescaped = unescape_literals(body);
    let body_ref = if unescaped.contains('\n') { &unescaped } else { body };

    // Fallback: NAME\nCODE or NAME:CODE_BLOCK_TYPE\nCODE
    let (name, code) = if let Some(newline_pos) = body_ref.find('\n') {
        let raw_name = body_ref[..newline_pos].trim_end_matches(|c: char| c == ':' || c == ' ');
        let clean_name = raw_name.split(':').next().unwrap_or(raw_name).trim();
        let raw_code = &body_ref[newline_pos + 1..];
        // Truncate at first non-code marker — model often appends fabricated
        // output, markdown, or explanation after the code.
        let code = truncate_code(raw_code);
        (clean_name.to_string(), code)
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

        let (block, next_search) = match text[block_start..].find(TOOL_CALL_CLOSE) {
            Some(p) => (
                &text[block_start..block_start + p],
                block_start + p + TOOL_CALL_CLOSE.len(),
            ),
            None => {
                // No closing tag — treat rest of text as the tool call body.
                // This handles models that hit EOS without emitting <tool_call|>.
                (&text[block_start..], text.len())
            }
        };
        search_from = next_search;

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

/// Strip all tool call blocks from text, keeping only non-tool-call content.
/// Handles both closed (`<|tool_call>...<tool_call|>`) and unclosed blocks.
pub fn strip_tool_calls(text: &str) -> String {
    let mut result = String::new();
    let mut remaining = text;
    while let Some(open_pos) = remaining.find(TOOL_CALL_OPEN) {
        result.push_str(&remaining[..open_pos]);
        let after_open = &remaining[open_pos + TOOL_CALL_OPEN.len()..];
        if let Some(close_pos) = after_open.find(TOOL_CALL_CLOSE) {
            remaining = &after_open[close_pos + TOOL_CALL_CLOSE.len()..];
        } else {
            // Unclosed — strip everything to end
            remaining = "";
            break;
        }
    }
    result.push_str(remaining);
    result.trim().to_string()
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
        let dsl = r#"{name:<|"|>Alice<|"|>,age:30}"#;
        let json = gemma_dsl_to_json(dsl);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["name"], "Alice");
        assert_eq!(v["age"], 30);
    }

    #[test]
    fn test_parse_literal_backslash_n() {
        // Model outputs literal \n instead of real newlines
        let text = r#"<|tool_call>call:python\nimport requests\nprint("hello")<tool_call|>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1, "should parse tool call with literal \\n");
        assert_eq!(calls[0].name, "python_interpreter");
        let code = calls[0].arguments["code"].as_str().unwrap();
        assert!(code.contains('\n'), "code should have real newlines after unescaping");
        assert!(code.contains("import requests"));
        assert!(code.contains("print(\"hello\")"));
    }

    #[test]
    fn test_unescape_literals() {
        assert_eq!(unescape_literals(r#"hello\nworld"#), "hello\nworld");
        assert_eq!(unescape_literals(r#"a\tb"#), "a\tb");
        assert_eq!(unescape_literals(r#"say \"hi\""#), "say \"hi\"");
        assert_eq!(unescape_literals(r#"no escapes"#), "no escapes");
    }
}
