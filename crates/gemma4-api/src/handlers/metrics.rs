use axum::extract::State;
use axum::Json;
use crate::metrics::{Metrics, MetricsSnapshot};

pub async fn get_metrics(State(metrics): State<Metrics>) -> Json<MetricsSnapshot> {
    Json(metrics.snapshot())
}
