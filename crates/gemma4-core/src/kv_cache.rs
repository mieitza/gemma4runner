use anyhow::Result;
use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct LayerKvCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
    pub sliding_window: Option<usize>,
    current_len: usize,
}

impl LayerKvCache {
    pub fn new(sliding_window: Option<usize>) -> Self {
        Self { key: None, value: None, sliding_window, current_len: 0 }
    }

    pub fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (self.key.take(), self.value.take()) {
            (Some(prev_k), Some(prev_v)) => {
                (Tensor::cat(&[&prev_k, key], 2)?, Tensor::cat(&[&prev_v, value], 2)?)
            }
            _ => (key.clone(), value.clone()),
        };
        let (k, v) = if let Some(window) = self.sliding_window {
            let seq_len = k.dim(2)?;
            if seq_len > window {
                let start = seq_len - window;
                (k.narrow(2, start, window)?, v.narrow(2, start, window)?)
            } else { (k, v) }
        } else { (k, v) };
        self.current_len = k.dim(2)?;
        self.key = Some(k.clone());
        self.value = Some(v.clone());
        Ok((k, v))
    }

    pub fn current_len(&self) -> usize { self.current_len }

    pub fn reset(&mut self) {
        self.key = None; self.value = None; self.current_len = 0;
    }
}

pub struct KvCache { layers: Vec<LayerKvCache> }

impl KvCache {
    pub fn new(layer_types: &[String], sliding_window: usize) -> Self {
        let layers = layer_types.iter().map(|t| {
            let window = if t == "sliding_attention" { Some(sliding_window) } else { None };
            LayerKvCache::new(window)
        }).collect();
        Self { layers }
    }
    pub fn layer_mut(&mut self, idx: usize) -> &mut LayerKvCache { &mut self.layers[idx] }
    pub fn reset(&mut self) { for l in &mut self.layers { l.reset(); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_global_cache_keeps_all() {
        let mut cache = LayerKvCache::new(None);
        let k1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let (k, _) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k.dim(2).unwrap(), 3);
        let k2 = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
        let (k, _) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k.dim(2).unwrap(), 8);
    }

    #[test]
    fn test_sliding_cache_evicts() {
        let mut cache = LayerKvCache::new(Some(4));
        let k1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let (k, _) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k.dim(2).unwrap(), 3);
        let k2 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let (k, _) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k.dim(2).unwrap(), 4);
    }

    #[test]
    fn test_kv_cache_multi_layer() {
        let types = vec!["sliding_attention".into(), "sliding_attention".into(), "full_attention".into()];
        let mut cache = KvCache::new(&types, 4);
        assert!(cache.layer_mut(0).sliding_window.is_some());
        assert!(cache.layer_mut(2).sliding_window.is_none());
    }
}
