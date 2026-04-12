use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use crate::config::Gemma4Config;

/// Vision encoder — wraps the SigLIP-based vision tower.
/// Currently a stub — will be connected when candle-transformers exports Gemma 4 vision.
pub struct VisionEncoder {
    _hidden_size: usize,
}

impl VisionEncoder {
    pub fn new(_config: &Gemma4Config, _vb: VarBuilder) -> Result<Self> {
        tracing::warn!("Vision encoder is not yet implemented — image inputs will be ignored");
        Ok(Self { _hidden_size: 0 })
    }

    /// Process images into text-space embeddings.
    /// Returns empty vec until vision support is fully implemented.
    pub fn encode(&self, _pixel_values: &[Tensor]) -> Result<Vec<Tensor>> {
        Ok(vec![])
    }

    pub fn is_available(&self) -> bool {
        false
    }
}

/// Audio encoder — wraps the Conformer-based audio model.
/// Currently a stub — will be connected when candle-transformers exports Gemma 4 audio.
pub struct AudioEncoder {
    _hidden_size: usize,
}

impl AudioEncoder {
    pub fn new(_config: &Gemma4Config, _vb: VarBuilder) -> Result<Option<Self>> {
        tracing::warn!("Audio encoder is not yet implemented — audio inputs will be ignored");
        Ok(None)
    }

    pub fn encode(&self, _audio_mel: &Tensor, _audio_mask: &Tensor) -> Result<Tensor> {
        anyhow::bail!("Audio encoding not yet implemented")
    }

    pub fn is_available(&self) -> bool {
        false
    }
}
