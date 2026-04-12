use std::path::Path;
use anyhow::Result;

pub struct GemmaTokenizer {
    inner: tokenizers::Tokenizer,
    eos_token_ids: Vec<u32>,
}

impl GemmaTokenizer {
    pub fn from_file(path: &Path, eos_token_ids: Vec<u32>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { inner, eos_token_ids })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let text = self.inner.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        Ok(text)
    }

    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
}
