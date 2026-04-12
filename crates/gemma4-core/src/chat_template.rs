use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<bos>");
    for msg in messages {
        let role = match msg.role.as_str() {
            "assistant" => "model",
            other => other,
        };
        prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, msg.content));
    }
    prompt.push_str("<|turn>model\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_user_message() {
        let messages = vec![ChatMessage { role: "user".into(), content: "Hello".into() }];
        let prompt = format_chat_prompt(&messages);
        assert_eq!(prompt, "<bos><|turn>user\nHello<turn|>\n<|turn>model\n");
    }

    #[test]
    fn test_system_and_user() {
        let messages = vec![
            ChatMessage { role: "system".into(), content: "You are helpful.".into() },
            ChatMessage { role: "user".into(), content: "Hi".into() },
        ];
        let prompt = format_chat_prompt(&messages);
        assert_eq!(prompt, "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n");
    }

    #[test]
    fn test_multi_turn() {
        let messages = vec![
            ChatMessage { role: "user".into(), content: "What is 2+2?".into() },
            ChatMessage { role: "assistant".into(), content: "4".into() },
            ChatMessage { role: "user".into(), content: "And 3+3?".into() },
        ];
        let prompt = format_chat_prompt(&messages);
        assert_eq!(prompt, "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n4<turn|>\n<|turn>user\nAnd 3+3?<turn|>\n<|turn>model\n");
    }

    #[test]
    fn test_assistant_mapped_to_model() {
        let messages = vec![ChatMessage { role: "assistant".into(), content: "Hi there".into() }];
        let prompt = format_chat_prompt(&messages);
        assert!(prompt.contains("<|turn>model\nHi there<turn|>"));
    }
}
