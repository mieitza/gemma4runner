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
