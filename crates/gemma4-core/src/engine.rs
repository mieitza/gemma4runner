use std::path::Path;
use std::sync::mpsc;
use std::thread;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::chat_template::ChatMessage;
use crate::config::Gemma4Config;
use crate::kv_cache::KvCache;
use crate::loader;
use crate::sampling::{LogitsProcessor, SamplingParams};
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

/// Dispatch enum over the two model backends (safetensors and quantized GGUF).
enum ModelBackend {
    Safetensors(crate::model::GemmaTextModel),
    Quantized(crate::quantized_model::QuantizedGemmaModel),
}

impl ModelBackend {
    fn forward(
        &self,
        input_ids: &Tensor,
        cache: &mut KvCache,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Safetensors(m) => m.forward(input_ids, cache, seqlen_offset),
            Self::Quantized(m) => m.forward(input_ids, cache, seqlen_offset),
        }
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
    let (backend, tokenizer, config) = if loader::is_gguf_file(model_path) {
        // --- GGUF path ---
        tracing::info!("Detected GGUF file: {}", model_path.display());

        let gguf = crate::gguf_loader::GgufModel::load(model_path, &device)?;
        let text_config = gguf.config.clone();

        // Tokenizer must be alongside the GGUF file (same directory)
        let tokenizer_path = model_path.parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json");
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

    let (request_tx, request_rx) = mpsc::sync_channel::<InferenceRequest>(queue_depth);

    thread::Builder::new()
        .name("inference-engine".to_string())
        .spawn(move || { engine_loop(backend, tokenizer, config, device, request_rx); })?;

    Ok(EngineHandle { request_tx })
}

fn engine_loop(
    model: ModelBackend,
    tokenizer: GemmaTokenizer,
    config: Gemma4Config,
    device: Device,
    request_rx: mpsc::Receiver<InferenceRequest>,
) {
    while let Ok(request) = request_rx.recv() {
        if let Err(e) = process_request(&model, &tokenizer, &config, &device, &request) {
            let _ = request.response_tx.send(InferenceEvent::Error(e.to_string()));
        }
    }
    tracing::info!("Inference engine shutting down");
}

fn process_request(
    model: &ModelBackend,
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
