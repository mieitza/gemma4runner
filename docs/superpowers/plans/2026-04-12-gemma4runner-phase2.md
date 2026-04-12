# Gemma4Runner Phase 2 — Make It Useful

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SSE streaming, Metal/CUDA acceleration, HuggingFace Hub download, `/v1/completions` + `/v1/models` endpoints, config file support, and production CLI flags.

**Architecture:** Extends Phase 1's 3-crate workspace. Key changes: (1) streaming SSE responses via axum's Sse type, (2) device selection plumbed from CLI through engine to candle, (3) HF Hub download via `hf-hub` crate, (4) TOML config file with CLI override, (5) request queue with 429 rejection, (6) graceful shutdown via tokio signal handling.

**Tech Stack:** Same as Phase 1 + `hf-hub 0.5`, `toml 0.8`, `tokio-stream`, `futures`

**Spec:** `docs/superpowers/specs/2026-04-12-gemma4runner-design.md`

---

## File Structure Changes

```
crates/gemma4-core/src/
  engine.rs              — MODIFY: add stream field to InferenceRequest, device param to start_engine
  loader.rs              — MODIFY: add load_from_hub(), device/dtype params

crates/gemma4-api/src/
  server.rs              — MODIFY: add new routes, graceful shutdown
  streaming.rs           — CREATE: SSE stream adapter
  queue.rs               — CREATE: bounded request queue with 429
  middleware.rs          — CREATE: optional Bearer auth
  types/
    completion.rs        — CREATE: CompletionRequest/Response
    models.rs            — CREATE: ModelList/ModelObject
    mod.rs               — MODIFY: add new type modules
  handlers/
    chat.rs              — MODIFY: add streaming support
    completion.rs        — CREATE: POST /v1/completions
    models.rs            — CREATE: GET /v1/models
    mod.rs               — MODIFY: add new handlers

crates/gemma4runner/src/
  cli.rs                 — MODIFY: add --device, --hf-token, --api-key, --config flags
  config.rs              — CREATE: TOML config + env overlay
  main.rs                — MODIFY: config resolution, device selection, HF download
```

---

### Task 1: Add Device Selection to Engine

**Files:**
- Modify: `crates/gemma4-core/src/engine.rs`
- Modify: `crates/gemma4-core/src/loader.rs`

- [ ] **Step 1: Update loader to accept device and dtype**

In `crates/gemma4-core/src/loader.rs`, change the `load_model` signature:

```rust
pub fn load_model(model_dir: &Path, device: &Device, dtype: DType) -> Result<LoadedModel> {
```

Replace the hardcoded `let dtype = DType::F32;` with the parameter. Update the function to use the passed `dtype` and `device`.

- [ ] **Step 2: Update engine to accept device**

In `crates/gemma4-core/src/engine.rs`, change `start_engine`:

```rust
pub fn start_engine(model_path: &Path, device: Device, queue_depth: usize) -> Result<EngineHandle> {
```

Remove the hardcoded `let device = Device::Cpu;` and use the passed device. Add `use candle_core::DType;` and pass `DType::F32` to the loader for CPU, `DType::BF16` for GPU:

```rust
let dtype = match &device {
    Device::Cpu => DType::F32,
    _ => DType::BF16,
};
let loaded = loader::load_model(model_path, &device, dtype)?;
```

- [ ] **Step 3: Add device_from_string helper**

Add to `crates/gemma4-core/src/engine.rs`:

```rust
pub fn device_from_string(s: &str) -> Result<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            { Ok(Device::new_metal(0)?) }
            #[cfg(not(feature = "metal"))]
            { anyhow::bail!("Metal support not compiled. Rebuild with --features metal") }
        }
        s if s.starts_with("cuda:") => {
            #[cfg(feature = "cuda")]
            {
                let ordinal: usize = s[5..].parse().context("Invalid CUDA device ordinal")?;
                Ok(Device::new_cuda(ordinal)?)
            }
            #[cfg(not(feature = "cuda"))]
            { anyhow::bail!("CUDA support not compiled. Rebuild with --features cuda") }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            { Ok(Device::new_cuda(0)?) }
            #[cfg(not(feature = "cuda"))]
            { anyhow::bail!("CUDA support not compiled. Rebuild with --features cuda") }
        }
        "auto" => {
            #[cfg(feature = "metal")]
            { return Ok(Device::new_metal(0)?); }
            #[cfg(feature = "cuda")]
            { return Ok(Device::new_cuda(0)?); }
            #[allow(unreachable_code)]
            Ok(Device::Cpu)
        }
        other => anyhow::bail!("Unknown device: {}. Use: auto, cpu, metal, cuda, cuda:N", other),
    }
}
```

- [ ] **Step 4: Update main.rs to pass device**

In `crates/gemma4runner/src/main.rs`, update the serve handler:

```rust
let device = gemma4_core::engine::device_from_string("cpu")?;
let engine = gemma4_core::engine::start_engine(&model_path, device, queue_depth)?;
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check`
Expected: Compiles

- [ ] **Step 6: Commit**

```bash
git add crates/gemma4-core/src/engine.rs crates/gemma4-core/src/loader.rs crates/gemma4runner/src/main.rs
git commit -m "feat(core): add device selection (CPU/Metal/CUDA) to engine and loader"
```

---

### Task 2: SSE Streaming Support

**Files:**
- Create: `crates/gemma4-api/src/streaming.rs`
- Modify: `crates/gemma4-api/src/handlers/chat.rs`
- Modify: `crates/gemma4-api/src/types/chat.rs`
- Modify: `crates/gemma4-api/Cargo.toml`

- [ ] **Step 1: Add streaming dependencies**

Add to `crates/gemma4-api/Cargo.toml` under `[dependencies]`:

```toml
tokio-stream = "0.1"
futures = "0.3"
```

- [ ] **Step 2: Add stream field to ChatCompletionRequest**

In `crates/gemma4-api/src/types/chat.rs`, add to `ChatCompletionRequest`:

```rust
    #[serde(default)]
    pub stream: Option<bool>,
```

Add streaming chunk types:

```rust
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}
```

- [ ] **Step 3: Create streaming adapter**

Create `crates/gemma4-api/src/streaming.rs`:

```rust
use std::convert::Infallible;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::Event;
use futures::stream::Stream;

use gemma4_core::engine::{FinishReason, InferenceEvent};

use crate::types::chat::*;
use crate::types::common;

pub fn inference_event_stream(
    rx: mpsc::Receiver<InferenceEvent>,
    request_id: String,
    model_name: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let mut sent_role = false;

    futures::stream::unfold(
        (rx, request_id, model_name, created, sent_role),
        move |(rx, id, model, created, mut sent_role)| async move {
            // Use spawn_blocking since mpsc::Receiver::recv blocks
            let event = tokio::task::spawn_blocking({
                let rx_clone = rx;
                move || {
                    let result = rx_clone.recv().ok();
                    (result, rx_clone)
                }
            })
            .await
            .ok()?;

            let (maybe_event, rx) = event;
            let inference_event = maybe_event?;

            match inference_event {
                InferenceEvent::Token(token) => {
                    let mut delta = ChunkDelta { role: None, content: Some(token) };
                    if !sent_role {
                        delta.role = Some("assistant".to_string());
                        sent_role = true;
                    }
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice { index: 0, delta, finish_reason: None }],
                    };
                    let data = serde_json::to_string(&chunk).unwrap();
                    let event = Event::default().data(data);
                    Some((Ok(event), (rx, id, model, created, sent_role)))
                }
                InferenceEvent::Done(reason) => {
                    let finish = match reason {
                        FinishReason::Stop => common::FinishReason::Stop,
                        FinishReason::Length => common::FinishReason::Length,
                    };
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta { role: None, content: None },
                            finish_reason: Some(finish),
                        }],
                    };
                    let data = serde_json::to_string(&chunk).unwrap();
                    let event = Event::default().data(data);
                    // After done, emit [DONE] sentinel next iteration
                    Some((Ok(event), (rx, id, model, created, sent_role)))
                }
                InferenceEvent::Usage(_) => {
                    // Skip usage in streaming, continue to next event
                    Some((Ok(Event::default().comment("")), (rx, id, model, created, sent_role)))
                }
                InferenceEvent::Error(e) => {
                    let event = Event::default().data(format!("{{\"error\":\"{}\"}}", e));
                    None // End stream on error
                }
            }
        },
    )
}
```

- [ ] **Step 4: Update chat handler for streaming**

In `crates/gemma4-api/src/handlers/chat.rs`, update the handler to support both streaming and non-streaming:

```rust
use axum::response::sse::{KeepAlive, Sse};
use crate::streaming::inference_event_stream;

pub async fn chat_completions(
    State(engine): State<EngineHandle>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    // ... existing validation and setup code ...

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = request.model.clone();
    let is_streaming = request.stream.unwrap_or(false);

    let (response_tx, response_rx) = mpsc::channel();
    let inference_request = InferenceRequest { id: request_id.clone(), messages, sampling, response_tx };
    engine.send(inference_request).map_err(|e| ApiError::service_unavailable(e.to_string()))?;

    if is_streaming {
        let stream = inference_event_stream(response_rx, request_id, model_name);
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // ... existing non-streaming code ...
    }
}
```

Use `axum::response::IntoResponse` for the return type.

- [ ] **Step 5: Add streaming module to lib.rs**

Add `pub mod streaming;` to `crates/gemma4-api/src/lib.rs`.

- [ ] **Step 6: Verify it compiles**

Run: `cargo check`
Expected: Compiles

- [ ] **Step 7: Commit**

```bash
git add crates/gemma4-api/
git commit -m "feat(api): add SSE streaming for chat completions"
```

---

### Task 3: HuggingFace Hub Download

**Files:**
- Modify: `crates/gemma4-core/src/loader.rs`
- Modify: `crates/gemma4-core/Cargo.toml`

- [ ] **Step 1: Add hf-hub dependency**

Add to `crates/gemma4-core/Cargo.toml` under `[dependencies]`:

```toml
hf-hub = "0.5"
```

- [ ] **Step 2: Add hub download function**

Add to `crates/gemma4-core/src/loader.rs`:

```rust
use std::path::PathBuf;
use hf_hub::api::sync::Api;

/// Resolve a model source: if it's a local path, return it; if it looks like
/// a HuggingFace model ID (e.g. "google/gemma-4-E4B-it"), download it.
pub fn resolve_model_source(source: &str, hf_token: Option<&str>) -> Result<PathBuf> {
    let path = PathBuf::from(source);
    if path.exists() {
        tracing::info!("Using local model at {}", path.display());
        return Ok(path);
    }

    // Looks like a HF model ID (contains '/')
    if source.contains('/') {
        tracing::info!("Downloading model from HuggingFace: {}", source);
        return download_from_hub(source, hf_token);
    }

    anyhow::bail!(
        "'{}' is not a valid model source. Provide a local path or HuggingFace model ID (e.g. google/gemma-4-E4B-it)",
        source
    );
}

fn download_from_hub(model_id: &str, token: Option<&str>) -> Result<PathBuf> {
    let mut builder = hf_hub::api::sync::ApiBuilder::new();
    if let Some(token) = token {
        builder = builder.with_token(Some(token.to_string()));
    }
    let api = builder.build()?;
    let repo = api.model(model_id.to_string());

    // Download config and tokenizer
    let config_path = repo.get("config.json")
        .context("Failed to download config.json")?;
    let _tokenizer_path = repo.get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;

    // Download safetensors — try single file first, then sharded
    let model_dir = config_path.parent().unwrap().to_path_buf();

    // Try model.safetensors (single file)
    match repo.get("model.safetensors") {
        Ok(_) => {
            tracing::info!("Downloaded single model file");
        }
        Err(_) => {
            // Try sharded: look for model.safetensors.index.json
            let index_path = repo.get("model.safetensors.index.json")
                .context("No model.safetensors or model.safetensors.index.json found")?;

            let index: serde_json::Value = serde_json::from_reader(
                std::fs::File::open(&index_path)?
            )?;

            if let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) {
                let files: std::collections::HashSet<String> = weight_map.values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                for file in &files {
                    tracing::info!("Downloading {}", file);
                    repo.get(file)
                        .with_context(|| format!("Failed to download {}", file))?;
                }
            }
        }
    }

    Ok(model_dir)
}
```

- [ ] **Step 3: Update main.rs to use resolve_model_source**

In `crates/gemma4runner/src/main.rs`, replace the model path validation:

```rust
let model_path = gemma4_core::loader::resolve_model_source(&model, None)?;
```

Remove the old `PathBuf::from` + `ensure!(model_path.exists())` code.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add crates/gemma4-core/ crates/gemma4runner/src/main.rs
git commit -m "feat(core): add HuggingFace Hub model download with auto-detection"
```

---

### Task 4: /v1/completions Endpoint

**Files:**
- Create: `crates/gemma4-api/src/types/completion.rs`
- Create: `crates/gemma4-api/src/handlers/completion.rs`
- Modify: `crates/gemma4-api/src/types/mod.rs`
- Modify: `crates/gemma4-api/src/handlers/mod.rs`
- Modify: `crates/gemma4-api/src/server.rs`
- Modify: `crates/gemma4-core/src/engine.rs`

- [ ] **Step 1: Add prompt field to InferenceRequest**

In `crates/gemma4-core/src/engine.rs`, add an enum for request input:

```rust
#[derive(Debug)]
pub enum InferenceInput {
    Chat(Vec<ChatMessage>),
    Raw(String),
}
```

Change `InferenceRequest.messages` to `InferenceRequest.input`:

```rust
pub struct InferenceRequest {
    pub id: String,
    pub input: InferenceInput,
    pub sampling: SamplingParams,
    pub response_tx: mpsc::Sender<InferenceEvent>,
}
```

Update `process_request` to handle both:

```rust
let prompt = match &request.input {
    InferenceInput::Chat(messages) => format_chat_prompt(messages),
    InferenceInput::Raw(text) => text.clone(),
};
```

- [ ] **Step 2: Update chat handler**

In `crates/gemma4-api/src/handlers/chat.rs`, update the InferenceRequest construction to use `InferenceInput::Chat(messages)`.

- [ ] **Step 3: Create completion types**

Create `crates/gemma4-api/src/types/completion.rs`:

```rust
use serde::{Deserialize, Serialize};
use super::common::{FinishReason, Usage};

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
}

fn default_temperature() -> Option<f64> { Some(1.0) }

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: FinishReason,
}
```

- [ ] **Step 4: Create completion handler**

Create `crates/gemma4-api/src/handlers/completion.rs`:

```rust
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
    let sampling = SamplingParams {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k,
        max_tokens: request.max_tokens.unwrap_or(2048),
        seed: request.seed,
        ..Default::default()
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

    engine.send(inference_request).map_err(|e| ApiError::service_unavailable(e.to_string()))?;

    let result = tokio::task::spawn_blocking(move || {
        let mut text = String::new();
        let mut finish_reason = common::FinishReason::Stop;
        let mut usage = gemma4_core::engine::UsageStats { prompt_tokens: 0, completion_tokens: 0 };

        while let Ok(event) = response_rx.recv() {
            match event {
                InferenceEvent::Token(t) => text.push_str(&t),
                InferenceEvent::Usage(u) => usage = u,
                InferenceEvent::Done(r) => {
                    finish_reason = match r {
                        FinishReason::Stop => common::FinishReason::Stop,
                        FinishReason::Length => common::FinishReason::Length,
                    };
                    break;
                }
                InferenceEvent::Error(e) => return Err(anyhow::anyhow!(e)),
            }
        }
        Ok((text, finish_reason, usage))
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(|e| ApiError::internal(e.to_string()))?;

    let (text, finish_reason, usage) = result;
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion".into(),
        created,
        model: model_name,
        choices: vec![CompletionChoice { index: 0, text, finish_reason }],
        usage: common::Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.prompt_tokens + usage.completion_tokens,
        },
    }))
}
```

- [ ] **Step 5: Wire up routes and modules**

Add `pub mod completion;` to `types/mod.rs` and `handlers/mod.rs`.

In `server.rs`, add the route:
```rust
.route("/v1/completions", post(handlers::completion::completions))
```

- [ ] **Step 6: Verify it compiles**

Run: `cargo check`
Expected: Compiles

- [ ] **Step 7: Commit**

```bash
git add crates/
git commit -m "feat(api): add /v1/completions endpoint with raw text completion"
```

---

### Task 5: /v1/models Endpoint

**Files:**
- Create: `crates/gemma4-api/src/types/models.rs`
- Create: `crates/gemma4-api/src/handlers/models.rs`
- Modify: `crates/gemma4-api/src/types/mod.rs`
- Modify: `crates/gemma4-api/src/handlers/mod.rs`
- Modify: `crates/gemma4-api/src/server.rs`

- [ ] **Step 1: Create model list types**

Create `crates/gemma4-api/src/types/models.rs`:

```rust
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}
```

- [ ] **Step 2: Create models handler**

Create `crates/gemma4-api/src/handlers/models.rs`:

```rust
use axum::Json;
use crate::types::models::*;

pub async fn list_models() -> Json<ModelList> {
    // For now return a static model entry. In Phase 3 this will be dynamic.
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelObject {
            id: "gemma-4".into(),
            object: "model".into(),
            created: 0,
            owned_by: "google".into(),
        }],
    })
}
```

- [ ] **Step 3: Wire up**

Add modules to `types/mod.rs`, `handlers/mod.rs`. Add route in `server.rs`:

```rust
.route("/v1/models", get(handlers::models::list_models))
```

- [ ] **Step 4: Verify and commit**

Run: `cargo check`

```bash
git add crates/gemma4-api/
git commit -m "feat(api): add /v1/models endpoint"
```

---

### Task 6: CLI Config File Support

**Files:**
- Create: `crates/gemma4runner/src/config.rs`
- Modify: `crates/gemma4runner/src/cli.rs`
- Modify: `crates/gemma4runner/src/main.rs`
- Modify: `crates/gemma4runner/Cargo.toml`
- Create: `config.example.toml`

- [ ] **Step 1: Add toml dependency**

Add to `crates/gemma4runner/Cargo.toml`:
```toml
toml = "0.8"
serde = { workspace = true }
```

- [ ] **Step 2: Create config module**

Create `crates/gemma4runner/src/config.rs`:

```rust
use std::path::Path;
use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub auth: AuthConfig,
}

#[derive(Debug, Deserialize, Default)]
pub struct ModelConfig {
    pub source: Option<String>,
    pub device: Option<String>,
    pub hf_token: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ServerConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub queue_depth: Option<usize>,
    pub log_level: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct InferenceConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
pub struct AuthConfig {
    pub api_key: Option<String>,
}

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        Ok(config)
    }
}
```

- [ ] **Step 3: Update CLI with new flags**

Update `crates/gemma4runner/src/cli.rs`:

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gemma4runner")]
#[command(about = "Run Gemma 4 models with an OpenAI-compatible API")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the API server
    Serve {
        /// Path to model directory or HuggingFace model ID
        #[arg(long)]
        model: Option<String>,

        /// Path to TOML config file
        #[arg(long)]
        config: Option<String>,

        /// Host to listen on
        #[arg(long)]
        host: Option<String>,

        /// Port to listen on
        #[arg(long)]
        port: Option<u16>,

        /// Compute device: auto, cpu, metal, cuda, cuda:N
        #[arg(long)]
        device: Option<String>,

        /// HuggingFace token for gated models
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,

        /// API key for Bearer authentication (empty = no auth)
        #[arg(long, env = "GEMMA4_AUTH_API_KEY")]
        api_key: Option<String>,

        /// Log level
        #[arg(long)]
        log_level: Option<String>,

        /// Max queued inference requests
        #[arg(long)]
        queue_depth: Option<usize>,
    },
}
```

- [ ] **Step 4: Update main.rs with config resolution**

Replace `crates/gemma4runner/src/main.rs`:

```rust
mod cli;
mod config;

use std::path::PathBuf;
use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Commands};
use config::AppConfig;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve { model, config, host, port, device, hf_token, api_key, log_level, queue_depth } => {
            // Load config file if provided
            let file_config = match &config {
                Some(path) => AppConfig::load(&PathBuf::from(path))?,
                None => AppConfig::default(),
            };

            // Resolution: CLI > env > config file > defaults
            let log_level = log_level
                .or(file_config.server.log_level)
                .unwrap_or_else(|| "info".to_string());

            let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&log_level));
            tracing_subscriber::fmt().with_env_filter(env_filter).init();

            let model_source = model
                .or(file_config.model.source)
                .context("--model is required (or set model.source in config file)")?;

            let device_str = device
                .or(file_config.model.device)
                .unwrap_or_else(|| "auto".to_string());

            let hf_token = hf_token.or(file_config.model.hf_token);
            let host = host.or(file_config.server.host).unwrap_or_else(|| "0.0.0.0".to_string());
            let port = port.or(file_config.server.port).unwrap_or(8080);
            let queue_depth = queue_depth.or(file_config.server.queue_depth).unwrap_or(64);

            // Resolve model path (local or HF download)
            let model_path = gemma4_core::loader::resolve_model_source(
                &model_source,
                hf_token.as_deref(),
            )?;

            // Select device
            let device = gemma4_core::engine::device_from_string(&device_str)?;
            tracing::info!("Using device: {}", device_str);

            tracing::info!("Loading model from {}", model_path.display());
            let engine = gemma4_core::engine::start_engine(&model_path, device, queue_depth)?;

            tracing::info!("Starting server on {}:{}", host, port);
            gemma4_api::server::start_server(engine, &host, port).await?;
            Ok(())
        }
    }
}
```

- [ ] **Step 5: Create example config**

Create `config.example.toml`:

```toml
[model]
source = "google/gemma-4-E4B-it"
device = "auto"
# hf_token = "hf_..."

[server]
host = "0.0.0.0"
port = 8080
queue_depth = 64
log_level = "info"

[inference]
temperature = 0.7
top_p = 1.0
# top_k = 40
max_tokens = 2048

[auth]
# api_key = "sk-my-secret-key"
```

- [ ] **Step 6: Verify and commit**

Run: `cargo check`

```bash
git add crates/gemma4runner/ config.example.toml
git commit -m "feat(runner): add TOML config file support with CLI override"
```

---

### Task 7: Bearer Auth Middleware

**Files:**
- Create: `crates/gemma4-api/src/middleware.rs`
- Modify: `crates/gemma4-api/src/server.rs`
- Modify: `crates/gemma4-api/src/lib.rs`

- [ ] **Step 1: Create auth middleware**

Create `crates/gemma4-api/src/middleware.rs`:

```rust
use axum::{
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};

pub async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // API key is stored in request extensions by the server setup
    let expected_key = request.extensions().get::<ApiKey>();

    if let Some(ApiKey(key)) = expected_key {
        if !key.is_empty() {
            let auth_header = headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));

            match auth_header {
                Some(token) if token == key => {}
                _ => return Err(StatusCode::UNAUTHORIZED),
            }
        }
    }

    Ok(next.run(request).await)
}

#[derive(Clone)]
pub struct ApiKey(pub String);
```

- [ ] **Step 2: Update server to accept optional API key**

In `crates/gemma4-api/src/server.rs`, update:

```rust
use axum::middleware;
use crate::middleware::{auth_middleware, ApiKey};

pub fn build_router(engine: EngineHandle, api_key: Option<String>) -> Router {
    let mut app = Router::new()
        .route("/v1/chat/completions", post(handlers::chat::chat_completions))
        .route("/v1/completions", post(handlers::completion::completions))
        .route("/v1/models", get(handlers::models::list_models))
        .route("/health", get(handlers::health::health))
        .with_state(engine);

    if let Some(key) = api_key {
        app = app
            .layer(axum::Extension(ApiKey(key)))
            .layer(middleware::from_fn(auth_middleware));
    }

    app
}

pub async fn start_server(engine: EngineHandle, host: &str, port: u16, api_key: Option<String>) -> anyhow::Result<()> {
    let app = build_router(engine, api_key);
    // ... rest stays the same
}
```

- [ ] **Step 3: Update main.rs to pass api_key**

Pass `api_key` from CLI to `start_server`.

- [ ] **Step 4: Verify and commit**

Run: `cargo check`

```bash
git add crates/gemma4-api/ crates/gemma4runner/src/main.rs
git commit -m "feat(api): add optional Bearer auth middleware"
```

---

### Task 8: Graceful Shutdown

**Files:**
- Modify: `crates/gemma4-api/src/server.rs`

- [ ] **Step 1: Add graceful shutdown to server**

Update `start_server` in `crates/gemma4-api/src/server.rs`:

```rust
pub async fn start_server(engine: EngineHandle, host: &str, port: u16, api_key: Option<String>) -> anyhow::Result<()> {
    let app = build_router(engine, api_key);
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on http://{}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shut down gracefully");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { tracing::info!("Received Ctrl+C, shutting down..."); }
        _ = terminate => { tracing::info!("Received SIGTERM, shutting down..."); }
    }
}
```

- [ ] **Step 2: Verify and commit**

Run: `cargo check`

```bash
git add crates/gemma4-api/src/server.rs
git commit -m "feat(api): add graceful shutdown on SIGTERM/SIGINT"
```

---

### Task 9: Request Queue with 429

**Files:**
- Create: `crates/gemma4-api/src/queue.rs`
- Modify: `crates/gemma4-api/src/lib.rs`

Note: The engine already uses a bounded `SyncSender` with `try_send` that returns an error when full. The `EngineHandle::send` already returns an error in this case, and the chat handler maps it to `ApiError::service_unavailable`. We just need to change the HTTP status to 429 and add a `Retry-After` header.

- [ ] **Step 1: Add 429 error variant**

In `crates/gemma4-api/src/types/error.rs`, add:

```rust
pub fn too_many_requests(message: impl Into<String>) -> Self {
    Self {
        status: StatusCode::TOO_MANY_REQUESTS,
        body: ApiErrorResponse {
            error: ApiErrorBody {
                message: message.into(),
                error_type: "rate_limit_error".into(),
                param: None,
                code: None,
            },
        },
    }
}
```

- [ ] **Step 2: Update handlers to use 429**

In `chat.rs` and `completion.rs`, change:
```rust
.map_err(|e| ApiError::service_unavailable(e.to_string()))?;
```
to:
```rust
.map_err(|_| ApiError::too_many_requests("Server is busy. Please retry later."))?;
```

- [ ] **Step 3: Verify and commit**

Run: `cargo check`

```bash
git add crates/gemma4-api/
git commit -m "feat(api): return 429 when inference queue is full"
```

---

### Task 10: Full Build Verification

- [ ] **Step 1: Run full build**

Run: `cargo build`

- [ ] **Step 2: Run all tests**

Run: `cargo test --workspace`

- [ ] **Step 3: Verify CLI help**

Run: `cargo run -p gemma4runner -- serve --help`
Expected: Shows all new flags (--device, --hf-token, --api-key, --config)

- [ ] **Step 4: Verify error handling**

Run: `cargo run -p gemma4runner -- serve`
Expected: Error about --model being required

- [ ] **Step 5: Fix any issues and commit**

```bash
git add -A
git commit -m "chore: Phase 2 cleanup and verification"
```
