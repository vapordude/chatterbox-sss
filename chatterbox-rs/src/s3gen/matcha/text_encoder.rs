use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct TextEncoder;

impl TextEncoder {
    pub fn new(_n_vocab: usize, _n_spks: usize, _spk_emb_dim: usize, _vb: VarBuilder) -> Result<Self> { Ok(Self) }
    pub fn forward(&self, _x: &Tensor, _x_lengths: &Tensor, _spks: Option<&Tensor>) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((_x.clone(), _x.clone(), _x.clone()))
    }
}
