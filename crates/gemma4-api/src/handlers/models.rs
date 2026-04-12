use axum::Json;
use crate::types::models::*;

pub async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelObject { id: "gemma-4".into(), object: "model".into(), created: 0, owned_by: "google".into() }],
    })
}
