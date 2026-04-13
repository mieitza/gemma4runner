//! llama.cpp backend via the `llama-cpp-2` crate.
//!
//! Provides high-performance inference for GGUF models by delegating to
//! llama.cpp's optimised Metal / CUDA kernels.

#[cfg(feature = "llama-cpp")]
pub mod backend {
    use anyhow::{Context, Result};
    use std::num::NonZeroU32;
    use std::path::Path;

    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    #[allow(deprecated)]
    use llama_cpp_2::model::{AddBos, LlamaModel, Special};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::token::LlamaToken;

    use crate::chat_template::{ChatFormatOptions, ChatMessage, ToolDef};
    use crate::sampling::SamplingParams;

    /// Wraps a llama.cpp model + context for inference.
    ///
    /// Because `LlamaContext` borrows `LlamaModel` and neither is `Send`,
    /// this struct is designed to live on the dedicated inference thread
    /// that the engine already provides.
    pub struct LlamaCppBackend {
        backend: LlamaBackend,
        model: LlamaModel,
        n_ctx: u32,
    }

    impl LlamaCppBackend {
        /// Load a GGUF model from disk.
        ///
        /// * `model_path` - Path to the `.gguf` file.
        /// * `n_gpu_layers` - Number of layers to offload to GPU.
        ///   Use a large value (e.g. 999) to offload everything.
        /// * `n_ctx` - Context window size. `0` means use the model's training context length.
        pub fn new(model_path: &Path, n_gpu_layers: u32, n_ctx: u32) -> Result<Self> {
            let backend = LlamaBackend::init()
                .map_err(|e| anyhow::anyhow!("Failed to initialise llama.cpp backend: {}", e))?;

            let model_params = LlamaModelParams::default()
                .with_n_gpu_layers(n_gpu_layers);

            tracing::info!(
                "Loading GGUF model via llama.cpp: {} (gpu_layers={})",
                model_path.display(),
                n_gpu_layers,
            );

            let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
                .map_err(|e| anyhow::anyhow!("llama.cpp model load failed: {}", e))?;

            let effective_ctx = if n_ctx == 0 {
                model.n_ctx_train()
            } else {
                n_ctx
            };

            tracing::info!(
                "llama.cpp model loaded: vocab={}, embd={}, params={}, ctx={}",
                model.n_vocab(),
                model.n_embd(),
                model.n_params(),
                effective_ctx,
            );

            Ok(Self {
                backend,
                model,
                n_ctx: effective_ctx,
            })
        }

        /// Format a chat conversation into a prompt string.
        ///
        /// First tries to use the GGUF chat template embedded in the model.
        /// Falls back to our own Gemma 4 chat template if the GGUF template
        /// is unavailable.
        pub fn format_prompt(
            &self,
            messages: &[ChatMessage],
            tools: &[ToolDef],
            include_thinking: bool,
        ) -> String {
            // Try using our own chat template since we know it matches Gemma 4
            // and handles tools / thinking correctly.
            let options = ChatFormatOptions {
                tools: tools.to_vec(),
                enable_thinking: include_thinking,
            };
            crate::chat_template::format_chat_prompt_with_options(messages, &options)
        }

        /// Tokenize text using llama.cpp's built-in tokenizer.
        pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<LlamaToken>> {
            let bos = if add_bos { AddBos::Always } else { AddBos::Never };
            self.model
                .str_to_token(text, bos)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))
        }

        /// Convert a single token to its string representation.
        #[allow(deprecated)]
        pub fn token_to_string(&self, token: LlamaToken) -> Result<String> {
            self.model
                .token_to_str(token, Special::Tokenize)
                .map_err(|e| anyhow::anyhow!("Token-to-string failed: {}", e))
        }

        /// Check if a token is an end-of-generation token.
        pub fn is_eog(&self, token: LlamaToken) -> bool {
            self.model.is_eog_token(token)
        }

        /// Build a `LlamaSampler` chain from our `SamplingParams`.
        fn build_sampler(params: &SamplingParams) -> LlamaSampler {
            if params.temperature == 0.0 {
                return LlamaSampler::greedy();
            }

            let seed = params.seed.unwrap_or(0) as u32;
            let mut chain: Vec<LlamaSampler> = Vec::new();

            // Repetition / frequency / presence penalties
            if params.repetition_penalty != 1.0
                || params.frequency_penalty != 0.0
                || params.presence_penalty != 0.0
            {
                chain.push(LlamaSampler::penalties(
                    256, // penalty_last_n: look back window
                    params.repetition_penalty as f32,
                    params.frequency_penalty as f32,
                    params.presence_penalty as f32,
                ));
            }

            // Top-k
            if let Some(k) = params.top_k {
                chain.push(LlamaSampler::top_k(k as i32));
            }

            // Top-p
            if params.top_p < 1.0 {
                chain.push(LlamaSampler::top_p(params.top_p as f32, 1));
            }

            // Temperature
            chain.push(LlamaSampler::temp(params.temperature as f32));

            // Final distribution sampler
            chain.push(LlamaSampler::dist(seed));

            LlamaSampler::chain_simple(chain)
        }

        /// Run streaming generation.
        ///
        /// Calls `on_token` for each generated token string.  Return `false`
        /// from the callback to stop generation early.
        ///
        /// Returns the full list of generated token IDs.
        pub fn generate_streaming(
            &self,
            prompt_text: &str,
            params: &SamplingParams,
            mut on_token: impl FnMut(LlamaToken, &str) -> bool,
        ) -> Result<Vec<LlamaToken>> {
            // The prompt already contains <bos> from our chat template, so
            // don't let llama.cpp add another one.
            let prompt_tokens = self.tokenize(prompt_text, false)?;
            let prompt_len = prompt_tokens.len();
            tracing::debug!("llama.cpp prompt: {} tokens", prompt_len);

            // Create a fresh context for this request
            let ctx_size = std::cmp::max(self.n_ctx, (prompt_len + params.max_tokens) as u32);
            let n_threads = num_threads();
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(ctx_size))
                .with_n_batch(512)
                .with_n_threads(n_threads)
                .with_n_threads_batch(n_threads);

            let mut ctx = self
                .model
                .new_context(&self.backend, ctx_params)
                .map_err(|e| anyhow::anyhow!("Failed to create llama.cpp context: {}", e))?;

            // --- Prefill: feed the full prompt ---
            let mut batch = LlamaBatch::new(prompt_len.max(512), 1);
            for (i, &token) in prompt_tokens.iter().enumerate() {
                let is_last = i == prompt_len - 1;
                batch
                    .add(token, i as i32, &[0], is_last)
                    .context("Failed to add token to batch")?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Prefill decode failed: {}", e))?;

            // --- Decode loop ---
            let mut sampler = Self::build_sampler(params);
            let mut generated: Vec<LlamaToken> = Vec::new();
            let mut n_decoded = prompt_len;

            for _ in 0..params.max_tokens {
                let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(new_token);

                if self.is_eog(new_token) {
                    break;
                }

                #[allow(deprecated)]
                let text = self
                    .model
                    .token_to_str(new_token, Special::Tokenize)
                    .unwrap_or_default();

                generated.push(new_token);

                if !on_token(new_token, &text) {
                    break;
                }

                // Prepare next decode step
                batch.clear();
                batch
                    .add(new_token, n_decoded as i32, &[0], true)
                    .context("Failed to add token to batch")?;
                ctx.decode(&mut batch)
                    .map_err(|e| anyhow::anyhow!("Decode step failed: {}", e))?;
                n_decoded += 1;
            }

            Ok(generated)
        }
    }

    /// Pick a reasonable thread count for llama.cpp.
    fn num_threads() -> i32 {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        // Use half the available cores (physical cores on most machines)
        std::cmp::max(1, cpus / 2) as i32
    }
}
