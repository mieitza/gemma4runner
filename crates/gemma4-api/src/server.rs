use axum::routing::{get, post};
use axum::Router;
use gemma4_core::engine::EngineHandle;
use crate::handlers;

pub fn build_router(engine: EngineHandle) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handlers::chat::chat_completions))
        .route("/v1/completions", post(handlers::completion::completions))
        .route("/v1/models", get(handlers::models::list_models))
        .route("/health", get(handlers::health::health))
        .with_state(engine)
}

pub async fn start_server(engine: EngineHandle, host: &str, port: u16) -> anyhow::Result<()> {
    let app = build_router(engine);
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
