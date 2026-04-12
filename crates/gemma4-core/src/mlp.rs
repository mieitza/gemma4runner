use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{linear_no_bias, Activation, Linear, Module, VarBuilder};

pub struct GemmaMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GemmaMlp {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj, act_fn: Activation::GeluPytorchTanh })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = self.act_fn.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let fused = (gate * up)?;
        let output = self.down_proj.forward(&fused)?;
        Ok(output)
    }
}
