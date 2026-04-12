use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};

#[derive(Clone)]
pub struct ApiKey(pub String);

pub async fn auth_middleware(request: Request, next: Next) -> Result<Response, StatusCode> {
    if let Some(ApiKey(key)) = request.extensions().get::<ApiKey>() {
        if !key.is_empty() {
            let auth = request.headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            match auth {
                Some(token) if token == key => {}
                _ => return Err(StatusCode::UNAUTHORIZED),
            }
        }
    }
    Ok(next.run(request).await)
}
