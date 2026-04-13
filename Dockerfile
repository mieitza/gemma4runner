# Multi-stage build for gemma4runner with CUDA support
# Target: NVIDIA DGX / any Linux host with CUDA

# Stage 1: Build
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

RUN cargo build --release --features cuda

# Stage 2: Runtime
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gemma4runner /usr/local/bin/gemma4runner

# Default port
EXPOSE 8080

# Model must be mounted at /model
VOLUME /model

ENTRYPOINT ["gemma4runner", "serve"]
CMD ["--model", "/model", "--host", "0.0.0.0", "--port", "8080", "--device", "cuda"]
