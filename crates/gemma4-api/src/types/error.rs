use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ApiErrorResponse { pub error: ApiErrorBody }

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

pub struct ApiError { pub status: StatusCode, pub body: ApiErrorResponse }

impl ApiError {
    pub fn bad_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, body: ApiErrorResponse { error: ApiErrorBody {
            message: message.into(), error_type: "invalid_request_error".into(), param, code: None,
        }}}
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, body: ApiErrorResponse { error: ApiErrorBody {
            message: message.into(), error_type: "internal_error".into(), param: None, code: None,
        }}}
    }
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self { status: StatusCode::SERVICE_UNAVAILABLE, body: ApiErrorResponse { error: ApiErrorBody {
            message: message.into(), error_type: "service_unavailable".into(), param: None, code: None,
        }}}
    }
    pub fn too_many_requests(message: impl Into<String>) -> Self {
        Self { status: StatusCode::TOO_MANY_REQUESTS, body: ApiErrorResponse { error: ApiErrorBody {
            message: message.into(), error_type: "rate_limit_error".into(), param: None, code: None,
        }}}
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::to_string(&self.body).unwrap_or_default();
        (self.status, [("content-type", "application/json")], body).into_response()
    }
}
