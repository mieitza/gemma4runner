# gemma4runner

A single Rust binary that runs Google Gemma 4 models locally and exposes an OpenAI-compatible v1 API.

Built on [candle](https://github.com/huggingface/candle) for tensor operations. Supports GGUF quantized models for efficient CPU inference.

## Features

- **All Gemma 4 variants**: E2B, E4B, 26B-A4B (MoE), 31B dense
- **OpenAI-compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- **SSE streaming**: Real-time token streaming
- **GGUF quantized inference**: Load Q4_K_M, Q8_0, and other GGUF formats
- **HuggingFace Hub**: Auto-download models by ID
- **Tool calling**: OpenAI-standard function calling
- **Thinking mode**: `include_thinking` parameter for chain-of-thought
- **Config file**: TOML configuration with CLI override
- **Auth**: Optional Bearer token authentication
- **Metrics**: `/metrics` endpoint with tok/s, request counts
- **Graceful shutdown**: SIGTERM/SIGINT handling

## Quick Start

```bash
# Build
cargo build --release

# Run with a local GGUF model
gemma4runner serve --model /path/to/gemma-4-E4B-it-Q4_K_M.gguf

# Run with HuggingFace model ID (auto-downloads)
gemma4runner serve --model google/gemma-4-E4B-it --hf-token hf_xxx

# Inspect model architecture without loading weights
gemma4runner info --model /path/to/model.gguf
```

## Usage

### Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0,
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

### Text Completion

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "prompt": "The capital of France is",
    "max_tokens": 10
  }'
```

## CLI Options

```
gemma4runner serve [OPTIONS]

Options:
    --model <PATH_OR_HF_ID>    Model source (required)
    --config <PATH>             TOML config file
    --port <PORT>               Listen port [default: 8080]
    --host <HOST>               Listen address [default: 0.0.0.0]
    --device <DEVICE>           auto | cpu | metal | cuda:N
    --hf-token <TOKEN>          HuggingFace token for gated models [env: HF_TOKEN]
    --api-key <KEY>             Require Bearer auth [env: GEMMA4_AUTH_API_KEY]
    --queue-depth <INT>         Max queued requests [default: 64]
    --log-level <LEVEL>         trace | debug | info | warn | error
```

## Configuration

Create a `config.toml`:

```toml
[model]
source = "google/gemma-4-E4B-it"
device = "cpu"

[server]
host = "0.0.0.0"
port = 8080
queue_depth = 64

[auth]
# api_key = "sk-my-secret"
```

```bash
gemma4runner serve --config config.toml
```

CLI flags override config file values. Environment variables (`HF_TOKEN`, `GEMMA4_AUTH_API_KEY`) override both.

## Architecture

Three-crate Cargo workspace:

```
crates/
  gemma4-core/    # Model architecture, inference engine, sampling
  gemma4-api/     # Axum HTTP server, OpenAI types, SSE streaming
  gemma4runner/   # CLI binary, config resolution
```

### Gemma 4 Implementation Details

This project implements several Gemma 4-specific architectural features:

- **KV Sharing**: The last 18 layers (E4B) reuse K/V from earlier layers instead of computing their own
- **No attention scaling**: Q and K are RMS-normalized per head, so no `1/sqrt(d)` scaling is needed (`scale=1.0`)
- **V normalization**: Bare RMS norm (no learned weight) applied to value vectors
- **Per-Layer Embeddings (PLE)**: Each decoder layer receives an additional vocabulary-derived signal
- **Dual RoPE**: Different frequencies and partial rotation for sliding vs global attention layers
- **4-norm decoder layers**: Pre/post norms for both attention and FFN sub-blocks
- **MoE support**: 128 experts with top-8 routing for the 26B-A4B variant

## Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Sampling temperature (0 = greedy) |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `top_k` | disabled | Top-k filtering |
| `max_tokens` | 2048 | Maximum tokens to generate |
| `seed` | random | Deterministic sampling seed |
| `repetition_penalty` | 1.0 | Penalize repeated tokens |
| `frequency_penalty` | 0.0 | Penalize frequent tokens |
| `presence_penalty` | 0.0 | Penalize present tokens |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| POST | `/v1/completions` | Text completion |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |
| GET | `/metrics` | Throughput metrics (tok/s, request counts) |

## Building

```bash
# CPU only (default)
cargo build --release

# With Metal GPU support (macOS)
cargo build --release --features metal

# With CUDA GPU support (requires CUDA toolkit)
cargo build --release --features cuda
```

## NVIDIA GPU / DGX Deployment

gemma4runner supports NVIDIA CUDA GPUs (A100, H100, etc.) for accelerated GGUF inference. All quantized tensor operations (QMatMul) run natively on GPU via candle's CUDA kernels, and all intermediate tensors (embeddings, norms, RoPE, attention masks) are kept on-device to avoid CPU-GPU transfers.

### Prerequisites

- **CUDA Toolkit** 11.8+ (12.x recommended). Install from [NVIDIA](https://developer.nvidia.com/cuda-downloads) or via your package manager.
- **cuDNN** (optional, not required for candle's quantized path).
- The `nvcc` compiler must be on `$PATH` so candle can build its CUDA kernels at compile time.
- Rust 1.82+ (edition 2024).

### Building on DGX / Linux with CUDA

```bash
# Verify CUDA is visible
nvcc --version
nvidia-smi

# Build with CUDA support
cargo build --release --features cuda

# Run with GPU acceleration (uses GPU 0 by default)
./target/release/gemma4runner serve \
  --model /path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  --device cuda

# Use a specific GPU (e.g., GPU 2 on a multi-GPU DGX)
./target/release/gemma4runner serve \
  --model /path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  --device cuda:2
```

### Multi-GPU Notes

- Each `gemma4runner` process uses a single GPU. For multi-GPU setups, run separate instances on different ports with `--device cuda:0`, `--device cuda:1`, etc., and load-balance across them.
- The `--device auto` flag selects CUDA automatically when the `cuda` feature is compiled in.

### Supported Quantizations

GGUF quantized formats (Q4_K_M, Q8_0, Q5_K_M, etc.) work on both CPU and CUDA. The quantized matmul kernels run directly on GPU without dequantizing full weight matrices to memory, keeping VRAM usage efficient.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `nvcc` not found at build time | Add CUDA toolkit `bin/` to `$PATH` |
| CUDA out of memory | Use a smaller quantization (Q4_K_M) or a smaller model variant |
| Slow first token (prefill) | Expected for large context; subsequent tokens are fast |
| `Invalid matmul arguments` shape errors | Ensure you are using a GGUF file converted for Gemma 4 (not Gemma 3) |

## Docker

### CUDA (DGX / GPU servers)

```bash
docker build -t gemma4runner:cuda .

# Run with GPU access — mount your GGUF model
docker run --gpus all -p 8080:8080 \
  -v /path/to/model.gguf:/model \
  gemma4runner:cuda
```

### CPU only

```bash
docker build -f Dockerfile.cpu -t gemma4runner:cpu .

docker run -p 8080:8080 \
  -v /path/to/model.gguf:/model \
  gemma4runner:cpu
```

## License

MIT
