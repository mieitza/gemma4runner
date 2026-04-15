use anyhow::Result;
use candle_core::Tensor;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub max_tokens: usize,
    pub seed: Option<u64>,
    pub repetition_penalty: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            // Google's recommended defaults for Gemma 4
            temperature: 1.0, top_p: 0.95, top_k: Some(64), max_tokens: 2048,
            seed: None, repetition_penalty: 1.0, frequency_penalty: 0.0, presence_penalty: 0.0,
        }
    }
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
}

impl LogitsProcessor {
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_os_rng(),
        };
        Self { rng }
    }

    pub fn sample(&mut self, logits: &Tensor, params: &SamplingParams, previous_tokens: &[u32]) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32)?.flatten_all()?;
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;

        // Apply repetition penalty
        if params.repetition_penalty != 1.0 {
            for &token_id in previous_tokens {
                if (token_id as usize) < logits_vec.len() {
                    let logit = &mut logits_vec[token_id as usize];
                    if *logit > 0.0 {
                        *logit /= params.repetition_penalty as f32;
                    } else {
                        *logit *= params.repetition_penalty as f32;
                    }
                }
            }
        }

        // Apply frequency and presence penalties
        if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
            let mut token_counts = std::collections::HashMap::new();
            for &t in previous_tokens {
                *token_counts.entry(t).or_insert(0u32) += 1;
            }
            for (&token_id, &count) in &token_counts {
                if (token_id as usize) < logits_vec.len() {
                    logits_vec[token_id as usize] -= params.frequency_penalty as f32 * count as f32;
                    logits_vec[token_id as usize] -= params.presence_penalty as f32;
                }
            }
        }

        // Greedy: temperature == 0 (applied after penalties)
        if params.temperature == 0.0 {
            let token = logits_vec.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32).unwrap();
            return Ok(token);
        }

        // Apply temperature
        if params.temperature != 1.0 {
            let inv_temp = 1.0 / params.temperature as f32;
            for l in logits_vec.iter_mut() { *l *= inv_temp; }
        }

        // Top-k filtering
        if let Some(k) = params.top_k {
            if k < logits_vec.len() {
                let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let threshold = indexed[k - 1].1;
                for l in logits_vec.iter_mut() {
                    if *l < threshold { *l = f32::NEG_INFINITY; }
                }
            }
        }

        // Softmax
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits_vec.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= sum; }

        // Top-p (nucleus) filtering
        if params.top_p < 1.0 {
            let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
            sorted_indices.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());
            let mut cumulative = 0.0f32;
            let mut cutoff_idx = sorted_indices.len();
            for (i, &idx) in sorted_indices.iter().enumerate() {
                cumulative += probs[idx];
                if cumulative > params.top_p as f32 { cutoff_idx = i + 1; break; }
            }
            let allowed: std::collections::HashSet<usize> = sorted_indices[..cutoff_idx].iter().copied().collect();
            for (i, p) in probs.iter_mut().enumerate() {
                if !allowed.contains(&i) { *p = 0.0; }
            }
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 { for p in probs.iter_mut() { *p /= sum; } }
        }

        let dist = WeightedIndex::new(&probs)?;
        let token = dist.sample(&mut self.rng) as u32;
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_greedy_sampling() {
        let logits = Tensor::new(&[1.0f32, 5.0, 2.0, 0.5], &Device::Cpu).unwrap();
        let params = SamplingParams { temperature: 0.0, ..Default::default() };
        let mut proc = LogitsProcessor::new(Some(42));
        let token = proc.sample(&logits, &params, &[]).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let params = SamplingParams { temperature: 1.0, ..Default::default() };
        let mut proc1 = LogitsProcessor::new(Some(42));
        let mut proc2 = LogitsProcessor::new(Some(42));
        assert_eq!(proc1.sample(&logits, &params, &[]).unwrap(), proc2.sample(&logits, &params, &[]).unwrap());
    }

    #[test]
    fn test_top_k_filters() {
        let logits = Tensor::new(&[1.0f32, 10.0, 2.0, 3.0], &Device::Cpu).unwrap();
        let params = SamplingParams { temperature: 1.0, top_k: Some(1), ..Default::default() };
        let mut proc = LogitsProcessor::new(Some(42));
        assert_eq!(proc.sample(&logits, &params, &[]).unwrap(), 1);
    }

    #[test]
    fn test_repetition_penalty_applied() {
        // Token 1 has the highest logit; with a large repetition penalty it should be suppressed.
        let logits = Tensor::new(&[1.0f32, 5.0, 2.0, 3.0], &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 100.0,
            ..Default::default()
        };
        let mut proc = LogitsProcessor::new(Some(42));
        // With token 1 in history and a huge penalty, token 1 should no longer win.
        let token = proc.sample(&logits, &params, &[1]).unwrap();
        assert_ne!(token, 1);
    }

    #[test]
    fn test_default_params() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.max_tokens, 2048);
        assert_eq!(params.repetition_penalty, 1.0);
    }
}
