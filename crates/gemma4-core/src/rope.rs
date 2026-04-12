use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct RotaryEmbedding { cos: Tensor, sin: Tensor }

impl RotaryEmbedding {
    pub fn new(dtype: DType, head_dim: usize, theta: f64, max_seq_len: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim).step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32)).collect();
        let inv_freq = Tensor::new(inv_freq, device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
        let inv_freq = inv_freq.unsqueeze(0)?;
        let freqs = positions.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
        Ok(Self { cos: emb.cos()?.to_dtype(dtype)?, sin: emb.sin()?.to_dtype(dtype)? })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        Ok((apply_rotary_emb(q, &cos, &sin)?, apply_rotary_emb(k, &cos, &sin)?))
    }
}

pub struct ProportionalRotaryEmbedding { cos: Tensor, sin: Tensor, rotary_dim: usize }

impl ProportionalRotaryEmbedding {
    pub fn new(dtype: DType, head_dim: usize, theta: f64, partial_rotary_factor: f64, max_seq_len: usize, device: &Device) -> Result<Self> {
        let rotary_dim = ((head_dim as f64 * partial_rotary_factor) as usize) & !1; // ensure even
        let inv_freq: Vec<f32> = (0..rotary_dim).step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / rotary_dim as f32)).collect();
        let inv_freq = Tensor::new(inv_freq, device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
        let inv_freq = inv_freq.unsqueeze(0)?;
        let freqs = positions.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
        Ok(Self { cos: emb.cos()?.to_dtype(dtype)?, sin: emb.sin()?.to_dtype(dtype)?, rotary_dim })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let head_dim = q.dim(3)?;
        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        if self.rotary_dim == head_dim {
            return Ok((apply_rotary_emb(q, &cos, &sin)?, apply_rotary_emb(k, &cos, &sin)?));
        }
        let q_rot = q.narrow(3, 0, self.rotary_dim)?;
        let q_pass = q.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?;
        let k_rot = k.narrow(3, 0, self.rotary_dim)?;
        let k_pass = k.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?;
        let q_rotated = apply_rotary_emb(&q_rot, &cos, &sin)?;
        let k_rotated = apply_rotary_emb(&k_rot, &cos, &sin)?;
        Ok((Tensor::cat(&[&q_rotated, &q_pass], 3)?, Tensor::cat(&[&k_rotated, &k_pass], 3)?))
    }
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
    Ok(x.broadcast_mul(cos)?.broadcast_add(&rotated.broadcast_mul(sin)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_output_shape() {
        let rope = RotaryEmbedding::new(DType::F32, 8, 10000.0, 128, &Device::Cpu).unwrap();
        let q = Tensor::zeros((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::zeros((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_rot.dims(), &[1, 2, 4, 8]);
        assert_eq!(k_rot.dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_proportional_rope_partial_rotation() {
        let rope = ProportionalRotaryEmbedding::new(DType::F32, 8, 1000000.0, 0.25, 128, &Device::Cpu).unwrap();
        let q = Tensor::ones((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 2, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_rot.dims(), &[1, 2, 4, 8]);
        assert_eq!(k_rot.dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_rope_offset() {
        let rope = RotaryEmbedding::new(DType::F32, 8, 10000.0, 128, &Device::Cpu).unwrap();
        let q = Tensor::ones((1, 2, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 2, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let (q0, _) = rope.apply(&q, &k, 0).unwrap();
        let (q5, _) = rope.apply(&q, &k, 5).unwrap();
        let diff = (q0 - q5).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff > 0.0);
    }
}
