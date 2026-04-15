use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};
use axum::extract::State;
use axum::Extension;
use axum::Json;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};

use gemma4_core::chat_template::ChatMessage;
use gemma4_core::engine::{EngineHandle, FinishReason, InferenceEvent, InferenceInput, InferenceRequest};
use gemma4_core::sampling::SamplingParams;

use crate::metrics::Metrics;
use crate::types::chat::*;
use crate::types::common;
use crate::types::error::ApiError;
use crate::streaming::inference_event_stream;

pub async fn chat_completions(
    State(engine): State<EngineHandle>,
    Extension(metrics): Extension<Metrics>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty", Some("messages".into())));
    }

    let messages: Vec<ChatMessage> = request.messages.iter().map(|m| ChatMessage {
        role: match m.role {
            common::Role::System => "system".into(),
            common::Role::User => "user".into(),
            common::Role::Assistant => "assistant".into(),
            common::Role::Tool => "tool".into(),
        },
        content: m.content.clone().unwrap_or_default(),
        tool_calls: m.tool_calls.as_ref().map(|tcs| tcs.iter().map(|tc| {
            gemma4_core::chat_template::ToolCallInfo {
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
            }
        }).collect()),
        tool_call_id: m.tool_call_id.clone(),
    }).collect();

    // Validate sampling parameters
    if let Some(temp) = request.temperature {
        if temp < 0.0 || temp > 2.0 {
            return Err(ApiError::bad_request(
                "temperature must be between 0 and 2",
                Some("temperature".into()),
            ));
        }
    }
    if let Some(top_p) = request.top_p {
        if top_p < 0.0 || top_p > 1.0 {
            return Err(ApiError::bad_request(
                "top_p must be between 0 and 1",
                Some("top_p".into()),
            ));
        }
    }
    if let Some(max_tokens) = request.max_tokens {
        if max_tokens == 0 {
            return Err(ApiError::bad_request(
                "max_tokens must be greater than 0",
                Some("max_tokens".into()),
            ));
        }
    }

    let sampling = SamplingParams {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k,
        max_tokens: request.max_tokens.unwrap_or(2048),
        seed: request.seed,
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
    };

    let tools: Vec<gemma4_core::chat_template::ToolDef> = request.tools
        .as_ref()
        .map(|ts| ts.iter().map(|t| gemma4_core::chat_template::ToolDef {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters: t.function.parameters.clone(),
        }).collect())
        .unwrap_or_default();

    let include_thinking = request.include_thinking.unwrap_or(false);

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = request.model.clone();
    let (response_tx, response_rx) = mpsc::channel();

    let inference_request = InferenceRequest {
        id: request_id.clone(),
        input: InferenceInput::Chat(messages),
        sampling,
        response_tx,
        tools,
        include_thinking,
    };
    engine.send(inference_request).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("engine_dead") {
            ApiError::service_unavailable("Inference engine has crashed. Restart the server.")
        } else {
            ApiError::too_many_requests("Server is busy. Please retry later.")
        }
    })?;

    if request.stream.unwrap_or(false) {
        let stream = inference_event_stream(response_rx, request_id, model_name);
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        let start = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || collect_response(response_rx))
            .await
            .map_err(|e| ApiError::internal(format!("Task join error: {}", e)))?
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let inference_ms = start.elapsed().as_millis() as u64;

        metrics.record_request(
            result.usage.prompt_tokens as u64,
            result.usage.completion_tokens as u64,
            inference_ms,
        );

        let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        let api_tool_calls: Option<Vec<common::ToolCall>> = if result.tool_calls.is_empty() {
            None
        } else {
            Some(result.tool_calls.iter().enumerate().map(|(i, tc)| common::ToolCall {
                id: format!("call_{}", i),
                tool_type: "function".into(),
                function: common::FunctionCall {
                    name: tc.name.clone(),
                    arguments: tc.arguments.to_string(),
                },
            }).collect())
        };

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created,
            model: model_name,
            choices: vec![ChatChoice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant".into(),
                    // Strip any raw tool call tokens that leaked into the content
                    // (happens when sandbox auto-executes tool calls)
                    content: {
                        let mut c = gemma4_core::tool_parser::strip_tool_calls(&result.content);
                        // Strip all thinking/channel control tokens that leak into content
                        c = c.replace("<|channel>thought\n<channel|>", "");
                        c = c.replace("<|channel>thought", "");
                        c = c.replace("<channel|>", "");
                        c = c.replace("<|think|>", "");
                        c = c.trim().to_string();
                        Some(c)
                    },
                    thinking: result.thinking,
                    tool_calls: api_tool_calls,
                },
                finish_reason: result.finish_reason,
            }],
            usage: common::Usage {
                prompt_tokens: result.usage.prompt_tokens,
                completion_tokens: result.usage.completion_tokens,
                total_tokens: result.usage.prompt_tokens + result.usage.completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

struct CollectedResponse {
    content: String,
    thinking: Option<String>,
    tool_calls: Vec<gemma4_core::tool_parser::ParsedToolCall>,
    finish_reason: common::FinishReason,
    usage: gemma4_core::engine::UsageStats,
}

fn collect_response(rx: mpsc::Receiver<InferenceEvent>) -> anyhow::Result<CollectedResponse> {
    let mut content = String::new();
    let mut thinking: Option<String> = None;
    let mut tool_calls: Vec<gemma4_core::tool_parser::ParsedToolCall> = vec![];
    let mut finish_reason = common::FinishReason::Stop;
    let mut usage = gemma4_core::engine::UsageStats { prompt_tokens: 0, completion_tokens: 0 };

    while let Ok(event) = rx.recv() {
        match event {
            InferenceEvent::Token(t) => content.push_str(&t),
            InferenceEvent::ThinkingToken(t) => {
                thinking.get_or_insert_with(String::new).push_str(&t);
            }
            InferenceEvent::ToolCalls(tc) => { tool_calls = tc; }
            InferenceEvent::Usage(u) => usage = u,
            InferenceEvent::Done(reason) => {
                finish_reason = match reason {
                    FinishReason::Stop => common::FinishReason::Stop,
                    FinishReason::Length => common::FinishReason::Length,
                    FinishReason::ToolCalls => common::FinishReason::ToolCalls,
                };
                break;
            }
            InferenceEvent::Error(e) => return Err(anyhow::anyhow!("Inference error: {}", e)),
        }
    }

    Ok(CollectedResponse { content, thinking, tool_calls, finish_reason, usage })
}
