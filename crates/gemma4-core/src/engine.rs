use std::path::Path;
use std::sync::mpsc;
use std::thread;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::chat_template::ChatMessage;
use crate::config::Gemma4Config;
use crate::kv_cache::KvCache;
use crate::loader;
use crate::sampling::{LogitsProcessor, SamplingParams};
use crate::sandbox::{Sandbox, SandboxLevel};
use crate::tokenizer::GemmaTokenizer;

#[derive(Debug)]
pub enum InferenceInput {
    Chat(Vec<ChatMessage>),
    Raw(String),
}

#[derive(Debug)]
pub struct InferenceRequest {
    pub id: String,
    pub input: InferenceInput,
    pub sampling: SamplingParams,
    pub response_tx: mpsc::Sender<InferenceEvent>,
    pub tools: Vec<crate::chat_template::ToolDef>,
    pub include_thinking: bool,
}

#[derive(Debug, Clone)]
pub enum InferenceEvent {
    Token(String),
    ThinkingToken(String),
    ToolCalls(Vec<crate::tool_parser::ParsedToolCall>),
    Usage(UsageStats),
    Done(FinishReason),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::ToolCalls => write!(f, "tool_calls"),
        }
    }
}

/// Which backend implementation to request when starting the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    /// Automatically pick the best available backend.
    Auto,
    /// Force the candle-based backend (safetensors or our GGUF decoder).
    Candle,
    /// Force the llama.cpp backend (requires the `llama-cpp` feature).
    LlamaCpp,
}

impl std::str::FromStr for BackendChoice {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "candle" => Ok(Self::Candle),
            "llama-cpp" | "llama_cpp" | "llamacpp" => Ok(Self::LlamaCpp),
            other => Err(anyhow::anyhow!("Unknown backend '{}'. Use: auto, candle, llama-cpp", other)),
        }
    }
}

/// Dispatch enum over the model backends.
#[allow(dead_code)]
enum ModelBackend {
    Safetensors(crate::model::GemmaTextModel),
    Quantized(crate::quantized_model::QuantizedGemmaModel),
    #[cfg(feature = "llama-cpp")]
    LlamaCpp(crate::llama_cpp_backend::backend::LlamaCppBackend),
}

impl ModelBackend {
    fn forward(
        &mut self,
        input_ids: &Tensor,
        cache: &mut KvCache,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Safetensors(m) => m.forward(input_ids, cache, seqlen_offset),
            Self::Quantized(m) => m.forward(input_ids, cache, seqlen_offset),
            #[cfg(feature = "llama-cpp")]
            Self::LlamaCpp(_) => {
                unreachable!("LlamaCpp backend does not use the candle forward path")
            }
        }
    }

    /// Returns `true` when this is the llama.cpp backend.
    #[allow(dead_code)]
    fn is_llama_cpp(&self) -> bool {
        #[cfg(feature = "llama-cpp")]
        if matches!(self, Self::LlamaCpp(_)) {
            return true;
        }
        false
    }
}

#[derive(Clone)]
pub struct EngineHandle {
    request_tx: mpsc::SyncSender<InferenceRequest>,
}

impl EngineHandle {
    pub fn send(&self, request: InferenceRequest) -> Result<()> {
        self.request_tx.try_send(request)
            .map_err(|e| anyhow::anyhow!("Engine queue full or disconnected: {}", e))
    }
}

pub fn device_from_string(s: &str) -> Result<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        "metal" => Ok(Device::new_metal(0)?),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::new_cuda(0)?),
        "auto" => {
            #[cfg(feature = "metal")]
            { return Ok(Device::new_metal(0)?); }
            #[cfg(feature = "cuda")]
            { return Ok(Device::new_cuda(0)?); }
            #[allow(unreachable_code)]
            Ok(Device::Cpu)
        }
        other => {
            #[cfg(feature = "cuda")]
            if let Some(idx_str) = other.strip_prefix("cuda:") {
                let idx: usize = idx_str.parse()
                    .map_err(|_| anyhow::anyhow!("Invalid CUDA device index: {}", idx_str))?;
                return Ok(Device::new_cuda(idx)?);
            }
            Err(anyhow::anyhow!("Unknown device: {}", other))
        }
    }
}

pub fn start_engine(model_path: &Path, device: Device, queue_depth: usize) -> Result<EngineHandle> {
    start_engine_with_backend(model_path, device, queue_depth, BackendChoice::Auto, None)
}

pub fn start_engine_with_backend(
    model_path: &Path,
    device: Device,
    queue_depth: usize,
    backend_choice: BackendChoice,
    sandbox_level: Option<SandboxLevel>,
) -> Result<EngineHandle> {
    // --- Create sandbox if requested ---
    let sandbox = match sandbox_level {
        Some(level) => {
            let session_id = uuid::Uuid::new_v4().to_string();
            let sb = Sandbox::new(level, &session_id)?;
            Some(sb)
        }
        None => None,
    };
    // --- Decide whether to use the llama.cpp backend ---
    #[cfg(feature = "llama-cpp")]
    let use_llama_cpp = match backend_choice {
        BackendChoice::LlamaCpp => {
            if !loader::is_gguf_file(model_path) {
                anyhow::bail!("--backend llama-cpp requires a .gguf model file");
            }
            true
        }
        BackendChoice::Auto => loader::is_gguf_file(model_path),
        BackendChoice::Candle => false,
    };

    #[cfg(not(feature = "llama-cpp"))]
    {
        if backend_choice == BackendChoice::LlamaCpp {
            anyhow::bail!(
                "--backend llama-cpp requested but the binary was not compiled with the \
                 `llama-cpp` feature. Rebuild with: cargo build --features llama-cpp-metal"
            );
        }
    }

    // --- llama.cpp fast path ---
    #[cfg(feature = "llama-cpp")]
    if use_llama_cpp {
        tracing::info!("Using llama.cpp backend for GGUF model");
        let llama = crate::llama_cpp_backend::backend::LlamaCppBackend::new(
            model_path,
            999, // offload all layers to GPU
            0,   // 0 = use model's training context length
        )?;

        let (request_tx, request_rx) = mpsc::sync_channel::<InferenceRequest>(queue_depth);
        thread::Builder::new()
            .name("inference-engine-llamacpp".to_string())
            .spawn(move || {
                engine_loop_llama_cpp(llama, request_rx, sandbox);
            })?;
        return Ok(EngineHandle { request_tx });
    }

    // --- Candle path (safetensors or our GGUF decoder) ---
    let (backend, tokenizer, config) = if loader::is_gguf_file(model_path) {
        // --- GGUF path via candle ---
        tracing::info!("Detected GGUF file: {} (using candle backend)", model_path.display());

        let gguf = crate::gguf_loader::GgufModel::load(model_path, &device)?;
        let text_config = gguf.config.clone();

        // Try to find tokenizer.json alongside the GGUF file
        let tokenizer_path = model_path.parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json");
        let tokenizer_path = if tokenizer_path.exists() {
            tokenizer_path
        } else {
            // Download tokenizer from HuggingFace (Gemma 4 E4B base model)
            tracing::info!("Tokenizer not found locally, downloading from HuggingFace...");
            loader::download_tokenizer("google/gemma-4-E4B-it", None)
                .context("Failed to download tokenizer. Place a tokenizer.json next to the GGUF file or ensure network access.")?
        };
        let eos_ids = vec![1u32, 106]; // <eos> and <turn|>
        let tokenizer = GemmaTokenizer::from_file(&tokenizer_path, eos_ids.clone())?;

        let q_model = crate::quantized_model::QuantizedGemmaModel::new(
            &text_config,
            &gguf,
            &device,
        )?;

        // Wrap the text config in a full Gemma4Config (no image/audio tokens for GGUF)
        let full_config = Gemma4Config {
            text_config,
            image_token_id: None,
            audio_token_id: None,
            eos_token_id: eos_ids,
        };

        (ModelBackend::Quantized(q_model), tokenizer, full_config)
    } else {
        // --- Safetensors path ---
        let dtype = match &device {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        };
        let loaded = loader::load_model(model_path, &device, dtype)?;
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = GemmaTokenizer::from_file(
            &tokenizer_path,
            loaded.config.eos_token_id.clone(),
        )?;
        let config = loaded.config;
        (ModelBackend::Safetensors(loaded.model), tokenizer, config)
    };

    // The candle backend does not yet support sandbox tool execution.
    if sandbox.is_some() {
        tracing::warn!("Sandbox is not yet supported with the candle backend; ignoring --sandbox");
    }
    drop(sandbox);

    let (request_tx, request_rx) = mpsc::sync_channel::<InferenceRequest>(queue_depth);

    thread::Builder::new()
        .name("inference-engine".to_string())
        .spawn(move || { engine_loop(backend, tokenizer, config, device, request_rx); })?;

    Ok(EngineHandle { request_tx })
}

fn engine_loop(
    mut model: ModelBackend,
    tokenizer: GemmaTokenizer,
    config: Gemma4Config,
    device: Device,
    request_rx: mpsc::Receiver<InferenceRequest>,
) {
    while let Ok(request) = request_rx.recv() {
        if let Err(e) = process_request(&mut model, &tokenizer, &config, &device, &request) {
            let _ = request.response_tx.send(InferenceEvent::Error(e.to_string()));
        }
    }
    tracing::info!("Inference engine shutting down");
}

fn process_request(
    model: &mut ModelBackend,
    tokenizer: &GemmaTokenizer,
    config: &Gemma4Config,
    device: &Device,
    request: &InferenceRequest,
) -> Result<()> {
    let prompt = match &request.input {
        InferenceInput::Chat(messages) => {
            let options = crate::chat_template::ChatFormatOptions {
                tools: request.tools.clone(),
                enable_thinking: request.include_thinking,
            };
            crate::chat_template::format_chat_prompt_with_options(messages, &options)
        }
        InferenceInput::Raw(text) => text.clone(),
    };
    let prompt_tokens = tokenizer.encode(&prompt)?;
    let prompt_len = prompt_tokens.len();
    tracing::debug!("Prompt: {} tokens", prompt_len);

    let mut cache = KvCache::new(&config.text_config.layer_types, config.text_config.sliding_window);
    let mut logits_processor = LogitsProcessor::new(request.sampling.seed);
    let mut think_parser = crate::think_parser::ThinkParser::new(request.include_thinking);

    // Prefill
    let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, &mut cache, 0)?;
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits, &request.sampling, &[])?;
    let mut generated_tokens: Vec<u32> = vec![next_token];

    // Check if first token is already EOS before entering decode loop
    if !tokenizer.is_eos(next_token) {
        let token_text = tokenizer.decode(&[next_token])?;
        for event in think_parser.feed(&token_text) {
            match event {
                crate::think_parser::ThinkEvent::Content(s) => {
                    let _ = request.response_tx.send(InferenceEvent::Token(s));
                }
                crate::think_parser::ThinkEvent::Thinking(s) => {
                    let _ = request.response_tx.send(InferenceEvent::ThinkingToken(s));
                }
            }
        }

        // Decode loop
        for step in 0..request.sampling.max_tokens {
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let seqlen_offset = prompt_len + step;
            let logits = model.forward(&input, &mut cache, seqlen_offset)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;
            next_token = logits_processor.sample(&logits, &request.sampling, &generated_tokens)?;
            generated_tokens.push(next_token);
            if tokenizer.is_eos(next_token) { break; }
            let token_text = tokenizer.decode(&[next_token])?;
            for event in think_parser.feed(&token_text) {
                match event {
                    crate::think_parser::ThinkEvent::Content(s) => {
                        let _ = request.response_tx.send(InferenceEvent::Token(s));
                    }
                    crate::think_parser::ThinkEvent::Thinking(s) => {
                        let _ = request.response_tx.send(InferenceEvent::ThinkingToken(s));
                    }
                }
            }
        }
    }

    // Flush any remaining buffered text from the think parser
    for event in think_parser.flush() {
        match event {
            crate::think_parser::ThinkEvent::Content(s) => {
                let _ = request.response_tx.send(InferenceEvent::Token(s));
            }
            crate::think_parser::ThinkEvent::Thinking(s) => {
                let _ = request.response_tx.send(InferenceEvent::ThinkingToken(s));
            }
        }
    }

    let mut finish_reason = if tokenizer.is_eos(next_token) { FinishReason::Stop } else { FinishReason::Length };

    // Collect full output text and check for tool calls
    let full_output = tokenizer.decode(&generated_tokens)?;
    if crate::tool_parser::has_tool_calls(&full_output) {
        let parsed = crate::tool_parser::parse_tool_calls(&full_output);
        if !parsed.is_empty() {
            let _ = request.response_tx.send(InferenceEvent::ToolCalls(parsed));
            finish_reason = FinishReason::ToolCalls;
        }
    }

    let _ = request.response_tx.send(InferenceEvent::Usage(UsageStats { prompt_tokens: prompt_len, completion_tokens: generated_tokens.len() }));
    let _ = request.response_tx.send(InferenceEvent::Done(finish_reason));
    Ok(())
}

// ---------------------------------------------------------------------------
// llama.cpp engine loop
// ---------------------------------------------------------------------------

#[cfg(feature = "llama-cpp")]
fn engine_loop_llama_cpp(
    llama: crate::llama_cpp_backend::backend::LlamaCppBackend,
    request_rx: mpsc::Receiver<InferenceRequest>,
    sandbox: Option<Sandbox>,
) {
    while let Ok(request) = request_rx.recv() {
        if let Err(e) = process_request_llama_cpp(&llama, &request, &sandbox) {
            let _ = request.response_tx.send(InferenceEvent::Error(e.to_string()));
        }
    }
    tracing::info!("llama.cpp inference engine shutting down");
}

#[cfg(feature = "llama-cpp")]
fn process_request_llama_cpp(
    llama: &crate::llama_cpp_backend::backend::LlamaCppBackend,
    request: &InferenceRequest,
    sandbox: &Option<Sandbox>,
) -> Result<()> {
    // Build the initial messages list.  When a sandbox is active we may
    // need to inject sandbox tool definitions *and* loop back after tool
    // execution, so we work with a mutable messages vec.
    let mut messages: Vec<ChatMessage> = match &request.input {
        InferenceInput::Chat(msgs) => msgs.clone(),
        InferenceInput::Raw(text) => {
            vec![ChatMessage {
                role: "user".to_string(),
                content: text.clone(),
                tool_calls: None,
                tool_call_id: None,
            }]
        }
    };

    // Merge sandbox tool definitions with any user-supplied tools
    let mut tools = request.tools.clone();
    if let Some(sb) = &sandbox {
        let sandbox_tools = Sandbox::tool_definitions(sb.level());
        for st in &sandbox_tools {
            if !tools.iter().any(|t| t.name == st.name) {
                tools.push(st.clone());
            }
        }
    }

    let tx = &request.response_tx;
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;

    // Maximum rounds of tool-call execution to prevent infinite loops
    const MAX_TOOL_ROUNDS: usize = 10;

    for round in 0..=MAX_TOOL_ROUNDS {
        let prompt = llama.format_prompt(&messages, &tools, request.include_thinking);

        let prompt_tokens = llama
            .tokenize(&prompt, false)
            .map(|t| t.len())
            .unwrap_or(0);
        total_prompt_tokens += prompt_tokens;

        let mut think_parser =
            crate::think_parser::ThinkParser::new(request.include_thinking);
        let mut all_text = String::new();
        let mut completion_tokens = 0usize;

        let _generated = llama.generate_streaming(
            &prompt,
            &request.sampling,
            |_token, text| {
                completion_tokens += 1;
                all_text.push_str(text);

                // Stream tokens to the caller
                for event in think_parser.feed(text) {
                    match event {
                        crate::think_parser::ThinkEvent::Content(s) => {
                            let _ = tx.send(InferenceEvent::Token(s));
                        }
                        crate::think_parser::ThinkEvent::Thinking(s) => {
                            let _ = tx.send(InferenceEvent::ThinkingToken(s));
                        }
                    }
                }
                true
            },
        )?;

        // Flush remaining think-parser buffer
        for event in think_parser.flush() {
            match event {
                crate::think_parser::ThinkEvent::Content(s) => {
                    let _ = tx.send(InferenceEvent::Token(s));
                }
                crate::think_parser::ThinkEvent::Thinking(s) => {
                    let _ = tx.send(InferenceEvent::ThinkingToken(s));
                }
            }
        }

        total_completion_tokens += completion_tokens;

        // Check for tool calls
        if crate::tool_parser::has_tool_calls(&all_text) {
            let parsed = crate::tool_parser::parse_tool_calls(&all_text);
            let sandbox_calls: Vec<_> = parsed
                .iter()
                .filter(|tc| Sandbox::is_sandbox_tool(&tc.name))
                .collect();

            if !sandbox_calls.is_empty() && sandbox.is_some() && round < MAX_TOOL_ROUNDS {
                let sb = sandbox.as_ref().unwrap();

                // Add the assistant message with tool calls to the
                // conversation
                let call_infos: Vec<crate::chat_template::ToolCallInfo> = parsed
                    .iter()
                    .map(|tc| crate::chat_template::ToolCallInfo {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    })
                    .collect();
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: crate::tool_parser::content_before_tool_calls(&all_text)
                        .unwrap_or_default(),
                    tool_calls: Some(call_infos),
                    tool_call_id: None,
                });

                // Execute each sandbox tool call and add tool response
                // messages
                for (i, tc) in parsed.iter().enumerate() {
                    if Sandbox::is_sandbox_tool(&tc.name) {
                        let result = match sb.dispatch_tool_call(&tc.name, &tc.arguments) {
                            Ok(r) => r,
                            Err(e) => format!("Error: {}", e),
                        };

                        tracing::info!(
                            "Sandbox tool {}({}) => {} bytes",
                            tc.name,
                            tc.arguments,
                            result.len()
                        );

                        messages.push(ChatMessage {
                            role: "tool".to_string(),
                            content: result,
                            tool_calls: None,
                            tool_call_id: Some(format!("call_{}", i)),
                        });
                    }
                }

                // Loop back for another generation round
                continue;
            }

            // No sandbox execution (or non-sandbox tool calls) -- report
            // to caller as usual
            if !parsed.is_empty() {
                let _ = tx.send(InferenceEvent::ToolCalls(parsed));
                let _ = tx.send(InferenceEvent::Usage(UsageStats {
                    prompt_tokens: total_prompt_tokens,
                    completion_tokens: total_completion_tokens,
                }));
                let _ = tx.send(InferenceEvent::Done(FinishReason::ToolCalls));
                return Ok(());
            }
        }

        // No tool calls (or all executed) -- we're done
        let finish_reason = if completion_tokens < request.sampling.max_tokens {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        let _ = tx.send(InferenceEvent::Usage(UsageStats {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
        }));
        let _ = tx.send(InferenceEvent::Done(finish_reason));
        return Ok(());
    }

    // If we got here, we hit the tool-call round limit
    tracing::warn!("Sandbox tool-call loop hit maximum rounds ({})", MAX_TOOL_ROUNDS);
    let _ = tx.send(InferenceEvent::Usage(UsageStats {
        prompt_tokens: total_prompt_tokens,
        completion_tokens: total_completion_tokens,
    }));
    let _ = tx.send(InferenceEvent::Done(FinishReason::Stop));
    Ok(())
}
