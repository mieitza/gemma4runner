use axum::routing::{get, post};
use axum::Router;
use gemma4_core::engine::EngineHandle;
use crate::handlers;
use crate::metrics::Metrics;
use crate::middleware::{ApiKey, auth_middleware};

pub fn build_router(engine: EngineHandle, api_key: Option<String>, metrics: Metrics) -> Router {
    let mut app = Router::new()
        .route("/v1/chat/completions", post(handlers::chat::chat_completions))
        .route("/v1/completions", post(handlers::completion::completions))
        .route("/v1/models", get(handlers::models::list_models))
        .route("/health", get(handlers::health::health))
        .with_state(engine);

    let metrics_router = Router::new()
        .route("/metrics", get(handlers::metrics::get_metrics))
        .with_state(metrics);

    app = app.merge(metrics_router);

    if let Some(key) = api_key {
        app = app
            .layer(axum::middleware::from_fn(auth_middleware))
            .layer(axum::Extension(ApiKey(key)));
    }
    app
}

pub async fn start_server(engine: EngineHandle, host: &str, port: u16, api_key: Option<String>) -> anyhow::Result<()> {
    let metrics = Metrics::new();
    let app = build_router(engine, api_key, metrics);
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on http://{}", addr);
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.expect("Ctrl+C handler failed"); };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler failed").recv().await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => { tracing::info!("Received Ctrl+C, shutting down..."); }
        _ = terminate => { tracing::info!("Received SIGTERM, shutting down..."); }
    }
}
