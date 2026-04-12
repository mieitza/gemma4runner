use gemma4_api::types::chat::*;
use gemma4_api::types::common::*;

#[test]
fn test_deserialize_minimal_request() {
    let json = r#"{"model": "gemma-4-e4b", "messages": [{"role": "user", "content": "Hello"}]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "gemma-4-e4b");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.temperature, Some(1.0));
    assert_eq!(req.max_tokens, None);
}

#[test]
fn test_deserialize_full_request() {
    let json = r#"{
        "model": "gemma-4-e4b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"}
        ],
        "temperature": 0.7, "top_p": 0.9, "top_k": 40,
        "max_tokens": 512, "seed": 42,
        "repetition_penalty": 1.1, "frequency_penalty": 0.5, "presence_penalty": 0.3
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.top_p, Some(0.9));
    assert_eq!(req.top_k, Some(40));
    assert_eq!(req.max_tokens, Some(512));
    assert_eq!(req.seed, Some(42));
}

#[test]
fn test_serialize_response() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-123".into(), object: "chat.completion".into(),
        created: 1234567890, model: "gemma-4-e4b".into(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChoiceMessage { role: "assistant".into(), content: "Hello!".into() },
            finish_reason: FinishReason::Stop,
        }],
        usage: Usage { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
    };
    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["id"], "chatcmpl-123");
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["usage"]["total_tokens"], 6);
}

#[test]
fn test_serialize_error() {
    use gemma4_api::types::error::*;
    let err = ApiErrorResponse {
        error: ApiErrorBody {
            message: "Invalid temperature".into(),
            error_type: "invalid_request_error".into(),
            param: Some("temperature".into()),
            code: None,
        },
    };
    let json = serde_json::to_value(&err).unwrap();
    assert_eq!(json["error"]["message"], "Invalid temperature");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["param"], "temperature");
    assert!(json["error"]["code"].is_null());
}
