use std::path::Path;
use std::sync::mpsc;
use std::thread;
use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::chat_template::{format_chat_prompt, ChatMessage};
use crate::config::Gemma4Config;
use crate::kv_cache::KvCache;
use crate::loader;
use crate::sampling::{LogitsProcessor, SamplingParams};
use crate::tokenizer::GemmaTokenizer;

#[derive(Debug)]
pub struct InferenceRequest {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    pub sampling: SamplingParams,
    pub response_tx: mpsc::Sender<InferenceEvent>,
}

#[derive(Debug, Clone)]
pub enum InferenceEvent {
    Token(String),
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
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { FinishReason::Stop => write!(f, "stop"), FinishReason::Length => write!(f, "length") }
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

pub fn start_engine(model_path: &Path, queue_depth: usize) -> Result<EngineHandle> {
    let device = Device::Cpu;
    let loaded = loader::load_model(model_path, &device)?;
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = GemmaTokenizer::from_file(&tokenizer_path, loaded.config.eos_token_id.clone())?;

    let (request_tx, request_rx) = mpsc::sync_channel::<InferenceRequest>(queue_depth);
    let config = loaded.config.clone();
    let model = loaded.model;

    thread::Builder::new()
        .name("inference-engine".to_string())
        .spawn(move || { engine_loop(model, tokenizer, config, device, request_rx); })?;

    Ok(EngineHandle { request_tx })
}

fn engine_loop(
    model: crate::model::GemmaTextModel,
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
    model: &crate::model::GemmaTextModel,
    tokenizer: &GemmaTokenizer,
    config: &Gemma4Config,
    device: &Device,
    request: &InferenceRequest,
) -> Result<()> {
    let prompt = format_chat_prompt(&request.messages);
    let prompt_tokens = tokenizer.encode(&prompt)?;
    let prompt_len = prompt_tokens.len();
    tracing::debug!("Prompt: {} tokens", prompt_len);

    let mut cache = KvCache::new(&config.text_config.layer_types, config.text_config.sliding_window);
    let mut logits_processor = LogitsProcessor::new(request.sampling.seed);

    // Prefill
    let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, &mut cache, 0)?;
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits, &request.sampling)?;
    let mut generated_tokens: Vec<u32> = vec![next_token];

    // Decode loop
    for step in 0..request.sampling.max_tokens {
        if tokenizer.is_eos(next_token) { break; }
        let token_text = tokenizer.decode(&[next_token])?;
        let _ = request.response_tx.send(InferenceEvent::Token(token_text));

        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let seqlen_offset = prompt_len + step + 1;
        let logits = model.forward(&input, &mut cache, seqlen_offset)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        next_token = logits_processor.sample(&logits, &request.sampling)?;
        generated_tokens.push(next_token);
    }

    let finish_reason = if tokenizer.is_eos(next_token) { FinishReason::Stop } else { FinishReason::Length };
    let _ = request.response_tx.send(InferenceEvent::Usage(UsageStats { prompt_tokens: prompt_len, completion_tokens: generated_tokens.len() }));
    let _ = request.response_tx.send(InferenceEvent::Done(finish_reason));
    Ok(())
}
