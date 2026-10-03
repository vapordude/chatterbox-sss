use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct BasicTransformerBlock;

impl BasicTransformerBlock {
    pub fn new(_dim: usize, _num_attention_heads: usize, _attention_head_dim: usize, _dropout: f64, _activation_fn: &str, _vb: VarBuilder) -> Result<Self> { Ok(Self) }
    pub fn forward(&self, _hidden_states: &Tensor, _attention_mask: Option<&Tensor>, _timestep: Option<&Tensor>) -> Result<Tensor> { Ok(_hidden_states.clone()) }
}
