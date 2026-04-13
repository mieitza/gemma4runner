# Building a Gemma 4 Inference Engine in Rust: Three Bugs That Took Hours to Find

*How I built a single binary that runs Google's Gemma 4 models locally with an OpenAI-compatible API -- and the three architectural misunderstandings that produced garbled multilingual garbage instead of English.*

---

I wanted a simple thing: a single Rust binary that loads a Gemma 4 GGUF file and serves an OpenAI-compatible API. No Python. No Docker. No CUDA toolkit installation. Just `./gemma4runner serve --model gemma-4-E4B-it-Q4_K_M.gguf` and start sending requests.

By the end of a single session I had 45 commits, 44 passing tests, full streaming SSE, tool calling, thinking mode, a `/metrics` endpoint, graceful shutdown, and HuggingFace Hub integration. The infrastructure worked flawlessly. The model loaded, tokens flowed, latency was tracked.

There was just one problem: the model was speaking in tongues.

## The Architecture

Before I get to the debugging, let me sketch the system. It's a three-crate Cargo workspace:

- **`gemma4-core`**: Model architecture -- attention, RoPE, MLP, MoE, PLE, KV cache, sampling, and the inference engine
- **`gemma4-api`**: Axum HTTP server with OpenAI-compatible types, SSE streaming, auth middleware
- **`gemma4runner`**: CLI binary with clap, TOML config, HuggingFace Hub download

The inference engine runs on a dedicated thread, consuming requests from a bounded `mpsc::sync_channel`. The API layer uses Axum's SSE support to stream tokens as they're generated. From the outside, it looks exactly like an OpenAI API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

## Building It: Five Phases

**Phase 1** was the walking skeleton. Get the E4B model loading from safetensors, implement basic attention with KV cache, wire up a non-streaming chat endpoint. The goal was end-to-end signal flow: HTTP request goes in, tokens come out.

**Phase 2** added production features: SSE streaming via `futures::stream::unfold`, Metal/CUDA feature flags, HuggingFace Hub download with `hf-hub`, TOML config files, bearer token auth middleware, and graceful shutdown with `tokio::signal`.

**Phase 3** expanded to all model variants. This is where things got interesting architecturally. Gemma 4 isn't one model -- it's a family:

- **E2B/E4B**: Dense models with Per-Layer Embeddings (PLE)
- **26B-A4B**: A Mixture-of-Experts model with 128 experts, top-8 routing, and a 704-dim expert intermediate size alongside a 2112-dim dense MLP
- **31B**: A conventional dense model

Each variant has different head dimensions, different numbers of KV heads for global vs. sliding layers, and different feature combinations. The config system had to handle all of this:

```rust
pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
    if self.is_sliding_layer(layer_idx) {
        self.head_dim           // 256 for E4B
    } else {
        if self.global_head_dim > 0 {
            self.global_head_dim // 512 for E4B
        } else {
            self.head_dim
        }
    }
}
```

**Phase 4** added tool calling and thinking mode. Gemma 4 doesn't use JSON for tool calls -- it has its own DSL with custom tokens:

```
<|tool_call>call:get_weather{city:<|"|>Bangkok<|"|>}<tool_call|>
```

That `<|"|>` is a special token that represents a double quote. I wrote a streaming state-machine parser that handles nested braces, already-quoted strings, and the conversion from this DSL to valid JSON.

Thinking mode uses `<|channel>thought\n` and `\n<channel|>` tags. Another streaming state-machine parser, this time handling the ambiguity of partial tag matches -- when you receive `<|channel>tho` you can't tell yet whether it's the start of a thinking block or literal text.

**Phase 5** was quantized inference: a `QuantizedGemmaModel` using candle's `QMatMul` for GGUF files, a GGUF metadata parser that extracts model config from flat key-value pairs, and a `/metrics` endpoint for tracking latency, throughput, and token counts.

At this point, all 44 tests passed. Every endpoint worked. The streaming was smooth. The metrics tracked correctly.

And the model output looked like this:

```
User: What is 2+2?
Model: ที่กรุงเทพ を הירושלים 北京市 འབྲུག
```

## The Debugging Journey

Thai. Hebrew. Chinese. Tibetan. All mixed together in a single response. The model was producing high-entropy noise across the entire vocabulary, as if every token had roughly equal probability.

The infrastructure was perfect. The bug was in the tensor computation -- somewhere in the 42 layers of the transformer.

### Hypothesis 1: RoPE

My first implementation of rotary position embeddings was hand-rolled:

```rust
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
    Ok(x.broadcast_mul(cos)?.broadcast_add(&rotated.broadcast_mul(sin)?)?)
}
```

Gemma 4 uses two different RoPE configurations: sliding layers get standard RoPE with `theta=10,000` and full rotation, while global layers get proportional RoPE with `theta=1,000,000` and only 25% of dimensions rotated (`partial_rotary_factor=0.25`). Maybe I was mixing them up?

I rewrote the quantized model to use `candle_nn::rotary_emb::rope` directly. No change. Still garbled.

### Hypothesis 2: RmsNorm Offset

Gemma 3 applies RMS normalization with `(1 + weight) * x_normed` -- an offset of 1 added to the learned weight. I checked whether Gemma 4 did the same by looking at the GGUF converter source.

It doesn't. Gemma 4 uses raw `weight * x_normed`. But I was already using candle's standard `rms_norm`, which does exactly that. Dead end.

### Hypothesis 3: Attention Mask

I tried changing the mask pattern. Causal vs. sliding window, the exact boundary conditions. Nothing helped.

### The Breakthrough: Comparing Against a Known-Good Implementation

I was going in circles. I had too many hypotheses and no way to narrow them down. So I did the obvious thing I should have done first: I loaded the exact same GGUF file in llama-cpp-python and compared logits.

```python
from llama_cpp import Llama
llm = Llama(model_path="gemma-4-E4B-it-Q4_K_M.gguf")
# Extract logits for the same input tokens
```

The results were immediate and damning. For the same input, my top predicted token was completely different from llama.cpp's. The probability distributions weren't just slightly off -- they were in different universes.

This told me the computation was diverging somewhere through the 42 layers. Not a small numerical drift, but a fundamental architectural mistake.

### Root Cause #1: KV Sharing

I started reading the GGUF metadata more carefully:

```rust
num_kv_shared_layers: get_u32(&["gemma4.attention.shared_kv_layers"])
    .unwrap_or(0),
```

`shared_kv_layers=18`. I had been parsing this field but not doing anything with it.

What this means: the E4B model has 42 layers, but only the first 24 compute their own key and value projections. The last 18 layers *reuse* the K/V from layers 22 and 23 (the last sliding and full-attention layers before the shared section). Those layers still have their own Q projection and attention weights -- they just share the K/V cache.

Without this, 43% of the layers were computing K/V from their own weight matrices -- weight matrices that exist in the GGUF file but that llama.cpp ignores for those layers. The GGUF has K/V weights for all 42 layers, but you're not supposed to use them for the shared ones.

**This is worth emphasizing: just because a weight exists in the model file doesn't mean it should be used.**

The fix was structural. Each layer now knows whether it computes its own KV:

```rust
has_kv: layer_idx < n_layer_kv_from_start,
```

And shared layers look up the KV cache from their reference layer:

```rust
let (k, v) = if self.has_kv {
    // Compute K, V from this layer's weights
    let k = self.attention_wk.forward(x)?;
    let v = self.attention_wv.forward(x)?;
    // ... norm, RoPE, cache ...
    (k, v)
} else {
    // Reuse from the reference layer's cache
    match shared_kv {
        Some((k, v)) => (k.clone(), v.clone()),
        None => anyhow::bail!("Shared KV layer has no reference"),
    }
};
```

After this fix: the garbled multilingual output became **coherent English**. The model was forming complete sentences. But it was producing wrong answers, repeating itself, and sometimes generating long incoherent loops.

Progress, but not done.

### Root Cause #2: Attention Scale

Standard transformer attention computes:

```
attn = softmax(Q @ K^T / sqrt(d_k))
```

That `1/sqrt(d_k)` scaling factor prevents the dot products from growing too large, which would push softmax into saturation. For a head dimension of 256, the scale is `1/16 = 0.0625`.

But Gemma 4 uses `scale=1.0`. No scaling at all.

Why? Because Gemma 4 applies RMS normalization to both Q and K *before* the dot product. The vectors are already normalized, so the dot products naturally stay in a reasonable range. Adding the `1/sqrt(d)` scaling on top of that *flattens* the attention distribution catastrophically -- every position gets roughly equal attention weight, and the model can't focus on anything.

My code had:

```rust
let scale = 1.0 / (self.head_dim as f64).sqrt();
let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
```

The fix:

```rust
// Gemma 4 uses f_attention_scale = 1.0 (no scaling on QK).
// The Q/K norms already normalize the vectors, so no 1/sqrt(d) is needed.
let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
```

This was a one-line fix that dramatically improved output quality. The model was now giving correct answers to simple questions, but still occasionally producing slightly degraded output.

### Root Cause #3: V-Norm

This was the subtlest one. Gemma 4 applies RMS normalization to value vectors -- but *without a learned weight*. Most RMS norm implementations look like this:

```
y = x / sqrt(mean(x^2) + eps) * weight
```

V-norm is just:

```
y = x / sqrt(mean(x^2) + eps)
```

No learned weight. No bias. Just bare normalization. This is what the HuggingFace implementation calls `with_scale=False` and what llama.cpp does with a raw `ggml_rms_norm` call.

```rust
// V norm -- RMS norm without learned weight
let v = {
    let v_f32 = v.to_dtype(DType::F32)?;
    let sq = v_f32.sqr()?;
    let mean_sq = (sq.sum_keepdim(D::Minus1)? / self.head_dim as f64)?;
    let rms = (mean_sq + self.rms_norm_eps as f64)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(v.dtype())?
};
```

Without this, the model worked but with noticeably worse output quality -- more repetition, less coherence on longer responses, occasional factual errors.

### The Result

After all three fixes:

```
User: What is 2+2?
Model: 2 + 2 is **4**.

User: What is the capital of France?
Model: Paris.

User: Tell me a joke.
Model: Why don't scientists trust atoms?
       Because they make up everything!

User: Who are you?
Model: I am Gemma 4, a Large Language Model developed by Google DeepMind.
```

~5.7 tokens per second on CPU (Apple Silicon). Not blazing fast, but entirely usable for development and testing.

## What Gemma 4 Actually Is (And Why It's Hard to Implement)

After debugging all of this, I have a much deeper appreciation for what Gemma 4's architecture is doing. It is **not** a standard transformer. Here are the key differences:

**Four norms per layer, not two.** A standard transformer has pre-attention norm and pre-FFN norm. Gemma 4 adds post-attention norm and post-FFN norm. The forward pass is:

```
x = post_attention_norm(attention(input_layernorm(x))) + x
x = post_feedforward_norm(mlp(pre_feedforward_norm(x))) + x
```

**Dual RoPE.** Sliding-attention layers use standard RoPE with `theta=10,000` and full rotation across all head dimensions. Global-attention layers use proportional RoPE with `theta=1,000,000` and only rotate 25% of dimensions (`partial_rotary_factor=0.25`). The non-rotated dimensions act as position-independent features.

**Asymmetric head dimensions.** Sliding layers use 256-dim heads; global layers use 512-dim heads. The number of KV heads also differs: the E4B model uses 2 KV heads for sliding layers but may use a different count for global layers.

**Per-Layer Embeddings (PLE).** The E4B and E2B models have an additional sub-block per layer that combines a per-layer token embedding table with a projection of the main embedding. It's gated with GELU and has its own learned output scale:

```rust
let gated = gate.forward(&x)?;
let gated = Activation::GeluPytorchTanh.forward(&gated)?;
let gated = (gated * ple_input)?;
let projected = proj.forward(&gated)?;
let normed = post_norm.forward(&projected)?;
let x = (residual + normed)?;
```

**And the three features that broke me:**
1. No attention scaling -- Q/K are pre-normalized
2. V normalization without learned weight
3. KV sharing across half the layers

You cannot adapt a Llama implementation to run Gemma 4. You have to understand what each component is doing and why.

## Lessons Learned

**1. Unit tests can't catch architectural misunderstandings.**

All 44 tests passed. The streaming worked perfectly. The metrics tracked correctly. The API was fully OpenAI-compatible. But the model output was garbled. The tests verified that tensors flowed through the right shapes and that the API returned valid JSON -- they couldn't verify that the attention mechanism was computing the right thing.

**2. Compare against a reference implementation early.**

I should have loaded the same GGUF in llama-cpp-python and compared logits before writing a single line of attention code. Instead, I wrote the whole system first and then had to debug it backwards. Comparing logits token-by-token immediately told me *whether* the computation was correct and *how far off* it was.

**3. Read the metadata, all of it.**

The `shared_kv_layers=18` field was sitting right there in the GGUF metadata from the start. I parsed it but didn't use it. With novel architectures, every metadata field is potentially load-bearing.

**4. When weights exist in the file, ask whether they should be used.**

The GGUF file contains K/V projection weights for all 42 layers. A reasonable person would load all 42. But 18 of them are meant to be ignored at runtime. The weights are there because the training framework saved all parameters -- not because they're all used during inference.

**5. Architecture papers and implementation can diverge.**

The Gemma 4 technical report describes the architecture at a high level, but the implementation details -- V-norm without learned weight, attention scale of 1.0, KV sharing -- require reading the actual code in HuggingFace transformers or llama.cpp. The paper won't tell you that `scale=1.0` because Q and K are already RMS-normalized.

## The Stack

For anyone who wants to build something similar:

- **[candle](https://github.com/huggingface/candle)**: Rust tensor library from HuggingFace. Supports CPU, Metal, CUDA, and GGUF quantized tensors. The `QMatMul` type handles quantized matrix multiplication.
- **[Axum](https://github.com/tokio-rs/axum)**: HTTP framework. Its `Sse` type makes SSE streaming straightforward.
- **[tokenizers](https://github.com/huggingface/tokenizers)**: The same tokenizer library that powers HuggingFace's Python ecosystem, with Rust bindings.
- **[clap](https://github.com/clap-rs/clap)**: CLI argument parsing with derive macros.
- **[hf-hub](https://github.com/huggingface/hf-hub)**: Rust crate for downloading from HuggingFace Hub.

The full codebase is structured to be readable -- each component (attention, RoPE, MLP, MoE, KV cache) lives in its own file, and the quantized model implementation closely follows candle-transformers' patterns while adapting for Gemma 4's specifics.

---

*If you're building a local inference engine for a new model architecture, start by comparing logits against a known-good implementation. Don't wait until everything else is built. The infrastructure is the easy part. The hard part is getting the tensor math right -- and the only way to know if it's right is to check it against something that works.*
