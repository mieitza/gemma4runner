use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::Event;
use futures::stream::Stream;
use gemma4_core::engine::{FinishReason, InferenceEvent};

use crate::types::chat::{ChatCompletionChunk, ChunkChoice, ChunkDelta};
use crate::types::common;

pub fn inference_event_stream(
    rx: mpsc::Receiver<InferenceEvent>,
    request_id: String,
    model: String,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    futures::stream::unfold(
        (rx, request_id, model, true),
        move |(rx, request_id, model, mut first)| async move {
            let event = tokio::task::spawn_blocking(move || {
                let result = rx.recv();
                (result, rx)
            })
            .await;

            let (result, rx) = match event {
                Ok((result, rx)) => (result, rx),
                Err(_) => return None,
            };

            match result {
                Err(_) => None,
                Ok(InferenceEvent::Usage(_)) => {
                    // Skip usage events — emit an empty SSE comment to keep the connection alive
                    let ev = Event::default().comment("");
                    Some((Ok(ev), (rx, request_id, model, first)))
                }
                Ok(InferenceEvent::ThinkingToken(_)) => {
                    // Skip thinking tokens in streaming for now
                    let ev = Event::default().comment("");
                    Some((Ok(ev), (rx, request_id, model, first)))
                }
                Ok(InferenceEvent::ToolCalls(_)) => {
                    // Skip tool calls in streaming for now
                    let ev = Event::default().comment("");
                    Some((Ok(ev), (rx, request_id, model, first)))
                }
                Ok(InferenceEvent::Error(e)) => {
                    tracing::error!("Inference error during streaming: {}", e);
                    None
                }
                Ok(InferenceEvent::Token(token)) => {
                    let created = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();

                    let role = if first {
                        first = false;
                        Some("assistant".to_string())
                    } else {
                        None
                    };

                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role,
                                content: Some(token),
                            },
                            finish_reason: None,
                        }],
                    };

                    let data = serde_json::to_string(&chunk).unwrap_or_default();
                    let ev = Event::default().data(data);
                    Some((Ok(ev), (rx, request_id, model, first)))
                }
                Ok(InferenceEvent::Done(reason)) => {
                    let created = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();

                    let finish_reason = match reason {
                        FinishReason::Stop => common::FinishReason::Stop,
                        FinishReason::Length => common::FinishReason::Length,
                        FinishReason::ToolCalls => common::FinishReason::ToolCalls,
                    };

                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(finish_reason),
                        }],
                    };

                    let data = serde_json::to_string(&chunk).unwrap_or_default();
                    let ev = Event::default().data(data);
                    // After Done, the next recv will fail (channel closed) and return None
                    Some((Ok(ev), (rx, request_id, model, first)))
                }
            }
        },
    )
}
