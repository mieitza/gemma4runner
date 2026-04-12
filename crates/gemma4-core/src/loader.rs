use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::config::Gemma4Config;
use crate::model::GemmaTextModel;

pub struct LoadedModel {
    pub model: GemmaTextModel,
    pub config: Gemma4Config,
}

pub fn load_model(model_dir: &Path, device: &Device, dtype: DType) -> Result<LoadedModel> {
    let config_path = model_dir.join("config.json");
    let config: Gemma4Config = serde_json::from_reader(
        std::fs::File::open(&config_path)
            .with_context(|| format!("Failed to open {}", config_path.display()))?,
    ).context("Failed to parse config.json")?;

    let safetensor_files = find_safetensor_files(model_dir)?;
    tracing::info!("Loading {} safetensor file(s) from {}", safetensor_files.len(), model_dir.display());

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)
            .context("Failed to load safetensors")?
    };

    let model = GemmaTextModel::new(&config.text_config, vb)
        .context("Failed to build model from weights")?;

    tracing::info!("Model loaded: {} layers, hidden_size={}", config.text_config.num_hidden_layers, config.text_config.hidden_size);
    Ok(LoadedModel { model, config })
}

/// Resolve a model source: if it's a local path return it; if it's a HF model ID, download it.
pub fn resolve_model_source(source: &str, hf_token: Option<&str>) -> Result<PathBuf> {
    let path = PathBuf::from(source);
    if path.exists() {
        tracing::info!("Using local model at {}", path.display());
        return Ok(path);
    }
    if source.contains('/') {
        tracing::info!("Downloading model from HuggingFace: {}", source);
        return download_from_hub(source, hf_token);
    }
    anyhow::bail!("'{}' is not a valid model source. Provide a local path or HuggingFace model ID (e.g. google/gemma-4-E4B-it)", source);
}

fn download_from_hub(model_id: &str, token: Option<&str>) -> Result<PathBuf> {
    let mut builder = hf_hub::api::sync::ApiBuilder::new();
    if let Some(token) = token {
        builder = builder.with_token(Some(token.to_string()));
    }
    let api = builder.build()?;
    let repo = api.model(model_id.to_string());

    let config_path = repo.get("config.json").context("Failed to download config.json")?;
    let _tokenizer_path = repo.get("tokenizer.json").context("Failed to download tokenizer.json")?;

    let model_dir = config_path.parent().unwrap().to_path_buf();

    match repo.get("model.safetensors") {
        Ok(_) => { tracing::info!("Downloaded single model file"); }
        Err(_) => {
            let index_path = repo.get("model.safetensors.index.json")
                .context("No model.safetensors or index file found")?;
            let index: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index_path)?)?;
            if let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) {
                let files: std::collections::HashSet<String> = weight_map.values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                for file in &files {
                    tracing::info!("Downloading {}", file);
                    repo.get(file).with_context(|| format!("Failed to download {}", file))?;
                }
            }
        }
    }
    Ok(model_dir)
}

fn find_safetensor_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "safetensors").unwrap_or(false))
        .collect();
    anyhow::ensure!(!files.is_empty(), "No .safetensors files found in {}", dir.display());
    files.sort();
    Ok(files)
}
