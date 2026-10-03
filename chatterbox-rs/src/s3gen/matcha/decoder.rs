use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct SinusoidalPosEmb;
pub struct Block1D;
pub struct ResnetBlock1D;
pub struct Downsample1D;
pub struct TimestepEmbedding;
pub struct Upsample1D;

impl SinusoidalPosEmb {
    pub fn new(_dim: usize) -> Self { Self }
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> { Ok(_x.clone()) }
}
impl Block1D {
    pub fn new(_dim_in: usize, _dim_out: usize, _vb: VarBuilder) -> Result<Self> { Ok(Self) }
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> { Ok(_x.clone()) }
}
impl ResnetBlock1D {
    pub fn new(_dim_in: usize, _dim_out: usize, _time_emb_dim: usize, _groups: usize, _vb: VarBuilder) -> Result<Self> { Ok(Self) }
    pub fn forward(&self, _x: &Tensor, _mask: &Tensor, _t: &Tensor) -> Result<Tensor> { Ok(_x.clone()) }
}
impl Downsample1D {
    pub fn new(_dim: usize, _vb: VarBuilder) -> Result<Self> { Ok(Self) }
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> { Ok(_x.clone()) }
}
impl TimestepEmbedding {
    pub fn new(_in_channels: usize, _time_embed_dim: usize, _act_fn: &str, _vb: VarBuilder) -> Result<Self> { Ok(Self) }
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> { Ok(_x.clone()) }
}
impl Upsample1D {
    pub fn new(_dim: usize, _use_conv_transpose: bool, _vb: VarBuilder) -> Result<Self> { Ok(Self) }
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> { Ok(_x.clone()) }
}
