use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};
use axum::extract::State;
use axum::Json;
use axum::response::IntoResponse;

use gemma4_core::engine::{EngineHandle, FinishReason, InferenceEvent, InferenceInput, InferenceRequest};
use gemma4_core::sampling::SamplingParams;

use crate::types::completion::*;
use crate::types::common;
use crate::types::error::ApiError;

pub async fn completions(
    State(engine): State<EngineHandle>,
    Json(request): Json<CompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if request.prompt.is_empty() {
        return Err(ApiError::bad_request("prompt must not be empty", Some("prompt".into())));
    }

    let sampling = SamplingParams {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k,
        max_tokens: request.max_tokens.unwrap_or(2048),
        seed: request.seed,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_name = request.model.clone();
    let (response_tx, response_rx) = mpsc::channel();

    let inference_request = InferenceRequest {
        id: request_id.clone(),
        input: InferenceInput::Raw(request.prompt),
        sampling,
        response_tx,
    };
    engine.send(inference_request).map_err(|_| ApiError::too_many_requests("Server is busy. Please retry later."))?;

    let result = tokio::task::spawn_blocking(move || collect_response(response_rx))
        .await
        .map_err(|e| ApiError::internal(format!("Task join error: {}", e)))?
        .map_err(|e| ApiError::internal(e.to_string()))?;

    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    let response = CompletionResponse {
        id: request_id,
        object: "text_completion".into(),
        created,
        model: model_name,
        choices: vec![CompletionChoice {
            index: 0,
            text: result.content,
            finish_reason: result.finish_reason,
        }],
        usage: common::Usage {
            prompt_tokens: result.usage.prompt_tokens,
            completion_tokens: result.usage.completion_tokens,
            total_tokens: result.usage.prompt_tokens + result.usage.completion_tokens,
        },
    };

    Ok(Json(response))
}

struct CollectedResponse {
    content: String,
    finish_reason: common::FinishReason,
    usage: gemma4_core::engine::UsageStats,
}

fn collect_response(rx: mpsc::Receiver<InferenceEvent>) -> anyhow::Result<CollectedResponse> {
    let mut content = String::new();
    let mut finish_reason = common::FinishReason::Stop;
    let mut usage = gemma4_core::engine::UsageStats { prompt_tokens: 0, completion_tokens: 0 };

    while let Ok(event) = rx.recv() {
        match event {
            InferenceEvent::Token(t) => content.push_str(&t),
            InferenceEvent::Usage(u) => usage = u,
            InferenceEvent::Done(reason) => {
                finish_reason = match reason {
                    FinishReason::Stop => common::FinishReason::Stop,
                    FinishReason::Length => common::FinishReason::Length,
                };
                break;
            }
            InferenceEvent::Error(e) => return Err(anyhow::anyhow!("Inference error: {}", e)),
        }
    }

    Ok(CollectedResponse { content, finish_reason, usage })
}
