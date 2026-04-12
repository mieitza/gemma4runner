# Gemma4Runner — Design Spec

## Context

Build a single Rust binary (`gemma4runner`) that runs Google Gemma 4 models locally and exposes an OpenAI v1-compatible HTTP API. The binary supports all 4 Gemma 4 variants (E2B, E4B, 26B-A4B MoE, 31B dense), full multimodal (text + image + audio), tool calling, thinking mode, and Metal + CUDA acceleration. Built on Candle as the tensor engine with a pure Rust model architecture.

Delivered in 5 phases to manage scope. Phase 1 is a walking skeleton; each phase adds incremental value.

---

## Architecture: 3-Crate Workspace

```
gemma4runner/
├── Cargo.toml                  # workspace root
├── config.example.toml
├── crates/
│   ├── gemma4-core/            # Model architecture, inference engine
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs           # ModelConfig from config.json
│   │       ├── loader.rs           # Safetensors + GGUF loading, HF Hub download
│   │       ├── quantization.rs     # Candle quantized tensors, GGML quant types
│   │       ├── sampling.rs         # Temperature, top_k, top_p, penalties, seed
│   │       ├── tokenizer.rs        # Wraps tokenizers crate (262K vocab SentencePiece)
│   │       ├── chat_template.rs    # OpenAI messages → Gemma 4 format conversion
│   │       ├── kv_cache.rs         # Sliding window + global attention cache
│   │       ├── rope.rs             # Standard RoPE + proportional RoPE (factor 0.25)
│   │       ├── attention.rs        # GQA with 5-sliding + 1-global pattern
│   │       ├── think_parser.rs     # <think> tag state machine for streaming
│   │       ├── tool_parser.rs      # Tool call detection + extraction from output
│   │       ├── engine.rs           # InferenceEngine: model + tokenizer + sampling loop
│   │       └── models/
│   │           ├── mod.rs          # GemmaModel trait + ModelVariant enum
│   │           ├── dense.rs        # Dense layers (shared by E2B, E4B, 31B)
│   │           ├── ple.rs          # Per-Layer Embeddings (E2B, E4B)
│   │           ├── moe.rs          # MoE routing + expert dispatch (26B-A4B)
│   │           ├── vision.rs       # SigLIP image encoder
│   │           └── audio.rs        # Audio encoder (E2B/E4B only)
│   ├── gemma4-api/             # Axum HTTP server, OpenAI types
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── server.rs           # Router, graceful shutdown, concurrency limits
│   │       ├── types/
│   │       │   ├── mod.rs
│   │       │   ├── chat.rs         # ChatCompletionRequest/Response/Chunk + tools
│   │       │   ├── completion.rs   # CompletionRequest/Response/Chunk
│   │       │   ├── models.rs       # ModelList, ModelObject
│   │       │   ├── common.rs       # Usage, FinishReason, Message, Role, ToolCall
│   │       │   └── error.rs        # OpenAI-style error envelope
│   │       ├── handlers/
│   │       │   ├── mod.rs
│   │       │   ├── chat.rs         # POST /v1/chat/completions
│   │       │   ├── completion.rs   # POST /v1/completions
│   │       │   ├── models.rs       # GET /v1/models
│   │       │   └── health.rs       # GET /health
│   │       ├── streaming.rs        # Token channel → SSE Event stream adapter
│   │       ├── queue.rs            # Bounded request queue, 429 rejection
│   │       └── middleware.rs       # Logging, optional Bearer auth, CORS
│   └── gemma4runner/           # CLI binary
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs             # Entry point
│           ├── cli.rs              # Clap derive definitions
│           ├── config.rs           # TOML + env + CLI merge (CLI > env > file > defaults)
│           └── run.rs              # Wire core + api, start server
└── tests/                      # Integration tests
    ├── api_tests.rs                # HTTP request/response against mock engine
    ├── chat_template_tests.rs
    ├── sampling_tests.rs
    └── think_parser_tests.rs
```

---

## Core Design Details

### Model Trait

```rust
pub trait GemmaModel: Send + Sync {
    fn forward(&self, input: &ModelInput, cache: &mut KvCache) -> Result<Tensor>;
    fn model_type(&self) -> ModelVariant;
    fn config(&self) -> &ModelConfig;
}

pub enum ModelVariant { E2B, E4B, MoE26B, Dense31B }
```

All 4 variants implement `GemmaModel`. The engine and API only see `Box<dyn GemmaModel>`.

### Gemma 4 Architecture Key Points

- **Attention pattern**: 5 sliding-window layers + 1 global attention layer, repeated. Sliding window sizes: 512 (E4B), 1024 (others).
- **Dual RoPE**: Standard RoPE for sliding layers, proportional RoPE (factor 0.25) for global layers. `rope.rs` provides both.
- **PLE**: E2B/E4B share KV projection weights across the last 18 layers. Implemented as Rc/Arc-based cache sharing, not copies.
- **MoE (26B-A4B)**: 128 experts, top-8 routing via learned linear router. Expert dispatch gathers only the selected experts to avoid materializing all 128.
- **GQA ratios**: Local layers: 2 query heads per KV head. Global layers: 8 per KV head.
- **Activation**: `gelu_pytorch_tanh` throughout.
- **Vocab**: 262,144 tokens (SentencePiece).

### Chat Template (`chat_template.rs`)

Converts OpenAI message arrays to Gemma 4's native format:
```
<start_of_turn>user
{content}<end_of_turn>
<start_of_turn>model
```
Handles: system prompts, multi-turn, tool definitions in system prompt, tool call results as `role: "tool"` messages.

### Think Parser (`think_parser.rs`)

State machine for streaming `<think>` tag detection:
- States: `Normal`, `MaybeOpenTag(buffer)`, `InThinking`, `MaybeCloseTag(buffer)`
- Buffers partial tag characters until it can confirm or reject
- Emits `ThinkToken(String)` or `ContentToken(String)` events
- Handles edge cases: partial tags at chunk boundaries, angle brackets in content

### Tool Call Parser (`tool_parser.rs`)

Detects Gemma 4's tool call output format, converts to OpenAI `ToolCall` structs. Includes:
- Pattern matching for Gemma 4's function call format
- JSON extraction and validation
- Error recovery for malformed tool calls (returns content instead of crashing)

### Inference Engine (`engine.rs`)

```rust
pub struct InferenceEngine {
    model: Box<dyn GemmaModel>,
    tokenizer: Tokenizer,
    device: Device,
    request_rx: mpsc::Receiver<InferenceRequest>,  // bounded channel
}
```

Runs on a **dedicated OS thread** (not tokio), owns the model and GPU. Receives requests via bounded mpsc channel. Sends tokens back via per-request response channel.

```rust
pub struct InferenceRequest {
    pub id: String,
    pub prompt_tokens: Vec<u32>,
    pub sampling: SamplingParams,
    pub stream: bool,
    pub tools: Option<Vec<Tool>>,
    pub include_thinking: bool,
    pub response_tx: mpsc::Sender<InferenceEvent>,
}

pub enum InferenceEvent {
    Token(String),
    ThinkingToken(String),
    ToolCall(ToolCall),
    Usage(UsageStats),
    Done(FinishReason),
    Error(String),
}
```

### Sampling (`sampling.rs`)

```rust
pub struct SamplingParams {
    pub temperature: f64,          // default 1.0, range [0, 2]
    pub top_p: f64,                // default 1.0, range [0, 1]
    pub top_k: Option<usize>,     // default None
    pub max_tokens: usize,         // default 2048
    pub seed: Option<u64>,         // None = random
    pub stop: Vec<String>,
    pub repetition_penalty: f64,   // default 1.0
    pub frequency_penalty: f64,    // default 0.0
    pub presence_penalty: f64,     // default 0.0
}
```

Applied in order: repetition/frequency/presence penalties → temperature scaling → top_k filtering → top_p (nucleus) filtering → sample.

### KV Cache (`kv_cache.rs`)

- Per-layer key/value tensors with sliding window eviction
- Tracks layer type (sliding vs global) to apply correct window size
- Global layers keep full context; sliding layers evict oldest entries beyond window
- Phase 5 adds PagedAttention for memory-efficient long context

---

## API Design Details

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| POST | `/v1/completions` | Raw text completions |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

### Request Queue (`queue.rs`)

- Bounded channel with configurable depth (default: 64)
- When full, returns HTTP 429 with `Retry-After` header
- Future: continuous batching (Phase 5)

### Tool Calling (in `types/chat.rs`)

Request fields: `tools: Vec<Tool>`, `tool_choice: ToolChoice`
Response: `message.tool_calls: Vec<ToolCall>`, `finish_reason: "tool_calls"`
Message role `"tool"` with `tool_call_id` for returning results.

### Thinking Mode (in `types/chat.rs`)

Request field: `include_thinking: bool` (default false)
Response: `message.thinking: Option<String>`
Streaming: separate `delta.thinking` field before `delta.content` begins.
When `include_thinking` is false, thinking tokens are stripped entirely.

### Error Format

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": null
  }
}
```

### Graceful Shutdown

On SIGTERM/SIGINT:
1. Stop accepting new connections
2. Return 503 for new requests
3. Drain in-flight requests (30s timeout)
4. Shut down inference engine
5. Exit

---

## CLI Design

### Commands

```
gemma4runner serve [OPTIONS]          # Start API server
gemma4runner info --model <MODEL>     # Print model details without loading
```

### Key Flags

```
--model <PATH_OR_HF_ID>    Model source (required)
--config <PATH>             TOML config file
--port <PORT>               Listen port (default: 8080)
--host <HOST>               Listen address (default: 0.0.0.0)
--device <DEVICE>           auto | cpu | metal | cuda:0 | cuda:1
--hf-token <TOKEN>          HuggingFace token for gated models
--temperature <FLOAT>       Override default temperature
--max-tokens <INT>          Override default max tokens
--api-key <KEY>             Require Bearer auth
--queue-depth <INT>         Max queued requests (default: 64)
--log-level <LEVEL>         trace | debug | info | warn | error
```

### Config Resolution Order

CLI flags > `GEMMA4_*` env vars > config.toml > built-in defaults

### Model Source Detection

1. If path exists on filesystem → local path
2. If matches pattern `owner/repo` or `owner/repo-name` → HF Hub ID
3. Otherwise → error with helpful message

### HF Auth Resolution

`--hf-token` flag > `HF_TOKEN` env var > `~/.cache/huggingface/token` file

---

## Phased Delivery

### Phase 1 — Walking Skeleton
- **Variant**: E4B only (smallest non-trivial, has PLE)
- **Modality**: Text-only
- **Compute**: CPU-only
- **API**: `/v1/chat/completions` non-streaming only
- **Loading**: Local safetensors path only
- **Sampling**: temperature, top_p, max_tokens
- **Goal**: End-to-end proof — load model, tokenize, forward pass, generate text, return via HTTP

Files to create:
- Workspace `Cargo.toml`
- `gemma4-core`: `lib.rs`, `config.rs`, `loader.rs`, `tokenizer.rs`, `chat_template.rs`, `sampling.rs`, `kv_cache.rs`, `rope.rs`, `attention.rs`, `engine.rs`, `models/mod.rs`, `models/dense.rs`, `models/ple.rs`
- `gemma4-api`: `lib.rs`, `server.rs`, `types/` (chat, common, error), `handlers/chat.rs`
- `gemma4runner`: `main.rs`, `cli.rs`, `config.rs`, `run.rs`
- Tests: `sampling_tests.rs`, `chat_template_tests.rs`, `api_tests.rs`

### Phase 2 — Make It Useful
- Add SSE streaming to `/v1/chat/completions`
- Add Metal acceleration (`candle-core/metal` feature)
- Add CUDA acceleration (`candle-core/cuda` feature)
- Add HF Hub download with auth + progress bar
- Add `/v1/completions` and `/v1/models` endpoints
- Add full config file support + env var overlay
- Add `--device` flag, `--hf-token`, `--api-key`
- Add graceful shutdown
- Add request queue with 429 rejection

### Phase 3 — All Model Variants
- Add E2B (simpler than E4B, similar architecture)
- Add 31B dense (scale up, remove PLE)
- Add 26B-A4B MoE (router, expert dispatch, sparse forward)
- Add `gemma4runner info` subcommand

### Phase 4 — Advanced Features
- Tool calling (types, parser, chat template integration)
- Thinking mode (`<think>` state machine, `include_thinking` param)
- Vision — SigLIP encoder (`models/vision.rs`)
- Audio encoder (`models/audio.rs`, E2B/E4B only)

### Phase 5 — Production Hardening
- GGUF file loading support
- Quantization support (Q4_K_M, Q8_0, etc.)
- PagedAttention for KV cache
- Continuous batching
- GPU+CPU memory offloading for large models
- `/metrics` endpoint (tok/s, latency, memory, queue depth)
- Performance benchmarks

---

## Key Dependencies

| Crate | Purpose | Crate Used In |
|-------|---------|---------------|
| `candle-core` | Tensor operations, Metal/CUDA backends | gemma4-core |
| `candle-nn` | Neural network layers (Linear, Embedding, LayerNorm) | gemma4-core |
| `candle-transformers` | Reference model implementations | gemma4-core |
| `tokenizers` | HuggingFace tokenizer (SentencePiece, 262K vocab) | gemma4-core |
| `hf-hub` | Model download from HuggingFace Hub | gemma4-core |
| `safetensors` | Safetensors file loading | gemma4-core |
| `axum` | HTTP server framework | gemma4-api |
| `tokio` | Async runtime | gemma4-api, gemma4runner |
| `serde` + `serde_json` | JSON serialization for API types | gemma4-api |
| `clap` | CLI argument parsing (derive) | gemma4runner |
| `toml` | Config file parsing | gemma4runner |
| `tracing` + `tracing-subscriber` | Structured logging | all crates |
| `indicatif` | Progress bar for model download | gemma4runner |
| `uuid` | Request/completion ID generation | gemma4-api |

---

## Testing Strategy

**Unit tests** (per crate, run on CI without GPU):
- `sampling.rs`: Verify penalty application, top_k/top_p filtering, temperature scaling, deterministic with seed
- `chat_template.rs`: Message conversion for all role types, multi-turn, tool calls, system prompts
- `think_parser.rs`: State machine transitions, partial tags, edge cases
- `tool_parser.rs`: Valid/invalid tool call extraction
- `config.rs`: TOML parsing, env overlay, CLI merge precedence
- `types/`: Serialization/deserialization round-trips match OpenAI spec

**Integration tests** (mock engine, no GPU):
- `api_tests.rs`: Stand up axum server with a mock `InferenceEngine` that returns canned tokens. Test full HTTP request → response for all endpoints, streaming, errors, auth, 429 rejection.

**Model tests** (CPU, small tensors):
- Verify attention computation with known inputs
- Verify RoPE produces expected rotations
- Verify MoE routing selects correct experts for known logits

**Reference comparison** (manual, requires GPU + model):
- Run same prompt through llama.cpp and gemma4runner
- Compare output token probabilities (not exact tokens — sampling is stochastic)

---

## Verification Plan

After each phase, verify:

1. **Phase 1**: `cargo build` succeeds. `gemma4runner serve --model /path/to/e4b` starts server. `curl -X POST http://localhost:8080/v1/chat/completions -d '{"model":"e4b","messages":[{"role":"user","content":"Hello"}]}'` returns a valid response. All unit tests pass.

2. **Phase 2**: Streaming works with `curl --no-buffer`. Metal/CUDA produce same output as CPU. HF download works with token. Config file + env vars work. 429 returned when queue full. Graceful shutdown drains requests.

3. **Phase 3**: All 4 variants load and generate text. `gemma4runner info` prints correct architecture details per variant.

4. **Phase 4**: Tool calls returned as structured JSON. Thinking mode returns `thinking` field when enabled, strips when disabled. Image input produces relevant text output.

5. **Phase 5**: GGUF files load correctly. Quantized models produce reasonable output. `/metrics` reports accurate tok/s. Long context (>4K) works without OOM.
