use axum::extract::Extension;
use axum::Json;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use gemma4_core::sandbox::Sandbox;

use crate::types::error::ApiError;

#[derive(Debug, Deserialize)]
pub struct ExecuteRequest {
    pub language: Option<String>,
    pub code: Option<String>,
    pub command: Option<String>,
    pub filename: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ExecuteResponse {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub elapsed_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct FileWriteRequest {
    pub filename: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct FileReadRequest {
    pub filename: String,
}

#[derive(Debug, Serialize)]
pub struct FileListResponse {
    pub files: Vec<String>,
}

pub async fn execute(
    Extension(sandbox): Extension<Arc<Mutex<Sandbox>>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let result = tokio::task::spawn_blocking(move || {
        let sandbox = sandbox.lock().map_err(|e| anyhow::anyhow!("Lock: {}", e))?;
        if let Some(command) = &req.command {
            sandbox.run_command(command)
        } else if let Some(code) = &req.code {
            let lang = req.language.as_deref().unwrap_or("python");
            let filename = req.filename.as_deref();
            sandbox.execute_code(lang, code, filename)
        } else {
            anyhow::bail!("Provide either 'code' (with optional 'language') or 'command'")
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(|e| ApiError::bad_request(e.to_string(), None))?;

    Ok(Json(ExecuteResponse {
        exit_code: result.exit_code,
        stdout: result.stdout,
        stderr: result.stderr,
        elapsed_ms: result.elapsed_ms,
    }))
}

pub async fn write_file(
    Extension(sandbox): Extension<Arc<Mutex<Sandbox>>>,
    Json(req): Json<FileWriteRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let sandbox = sandbox.lock().map_err(|e| ApiError::internal(e.to_string()))?;
    sandbox
        .write_file(&req.filename, &req.content)
        .map_err(|e| ApiError::bad_request(e.to_string(), None))?;
    Ok(Json(serde_json::json!({"ok": true, "filename": req.filename})))
}

pub async fn read_file(
    Extension(sandbox): Extension<Arc<Mutex<Sandbox>>>,
    Json(req): Json<FileReadRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let sandbox = sandbox.lock().map_err(|e| ApiError::internal(e.to_string()))?;
    let content = sandbox
        .read_file(&req.filename)
        .map_err(|e| ApiError::bad_request(e.to_string(), None))?;
    Ok(Json(serde_json::json!({"filename": req.filename, "content": content})))
}

pub async fn list_files(
    Extension(sandbox): Extension<Arc<Mutex<Sandbox>>>,
) -> Result<impl IntoResponse, ApiError> {
    let sandbox = sandbox.lock().map_err(|e| ApiError::internal(e.to_string()))?;
    let files = sandbox
        .list_files()
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(FileListResponse { files }))
}
